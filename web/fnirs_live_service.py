from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from neurolabel.brain.fnirs import compute_window_metrics_sequence, series_from_rows
from neurolabel.ui.replay.brain3d.mesh_loader import load_brain_mesh
from neurolabel.ui.replay.brain3d.pfc_mapping import build_pfc_proxy_mapping
from neurolabel.ui.replay.brain3d.schemas import frame_to_json, mapping_to_json, mesh_to_json


@dataclass(frozen=True)
class LiveFrameEnvelope:
    sensor_now_ts: float
    display_ts: float
    display_delay_sec: float
    window: dict[str, Any]
    qc: dict[str, Any]


class FnirsLiveService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: set[queue.Queue] = set()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._connected = False
        self._streaming = False
        self._mock_mode = False
        self._session_id: str | None = None
        self._last_error: str | None = None
        self._sample_count = 0
        self._last_status_emit = 0.0
        self._display_delay_sec = 6.5

        self._rows: list[dict[str, float]] = []
        self._latest_sensor_ts = 0.0
        self._last_emitted_window_ts = float('-inf')

        self._mesh_json: dict[str, Any] | None = None
        self._mapping_json: dict[str, Any] | None = None

    # ---- public API -------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            sample_rate = 0.0
            if len(self._rows) > 1:
                duration = self._rows[-1]['timestamp'] - self._rows[0]['timestamp']
                if duration > 0:
                    sample_rate = (len(self._rows) - 1) / duration
            buffer_seconds = 0.0
            if len(self._rows) > 1:
                buffer_seconds = self._rows[-1]['timestamp'] - self._rows[0]['timestamp']
            return {
                'connected': self._connected,
                'streaming': self._streaming,
                'mock_mode': self._mock_mode,
                'session_id': self._session_id,
                'sample_count': self._sample_count,
                'sample_rate_est_hz': round(sample_rate, 2),
                'buffer_seconds': round(buffer_seconds, 2),
                'display_delay_sec': self._display_delay_sec,
                'last_error': self._last_error,
            }

    def connect(self, *, mock: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                raise RuntimeError('fNIRS live session is already running')
            self._stop_event.clear()
            self._connected = False
            self._streaming = False
            self._mock_mode = bool(mock)
            self._last_error = None
            self._session_id = f'fnirs-live-{uuid.uuid4().hex[:8]}'
            self._sample_count = 0
            self._rows = []
            self._latest_sensor_ts = 0.0
            self._last_emitted_window_ts = float('-inf')

            self._ensure_mesh_mapping()

            self._worker_thread = threading.Thread(
                target=self._worker_main,
                name='FnirsLiveServiceWorker',
                daemon=True,
            )
            self._worker_thread.start()
            self._broadcast({'type': 'fnirs_status', **self.status()})
            return self.status()

    def disconnect(self) -> dict[str, Any]:
        self._stop_event.set()
        self._broadcast({'type': 'fnirs_status', 'message': 'disconnect_requested', **self.status()})
        return self.status()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        # Ensure the neutral brain mesh is available even before a live connection starts.
        self._ensure_mesh_mapping()
        with self._lock:
            self._subscribers.add(q)
            q.put({'type': 'fnirs_status', **self.status()})
            if self._mesh_json is not None and self._mapping_json is not None:
                q.put({
                    'type': 'fnirs_mesh_init',
                    'mesh': self._mesh_json,
                    'mapping': self._mapping_json,
                    'display_delay_sec': self._display_delay_sec,
                    'viewer_defaults': {
                        'camera': 'isometric_front_left',
                        'colorscale': 'RdBu_r',
                        'symmetric_scale': True,
                    },
                    'disclaimer': 'Proxy prefrontal heatmap from sparse Mendi channels; not volumetric imaging.',
                })
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def process_csv_text(self, csv_text: str, *, filename: str | None = None) -> dict[str, Any]:
        from neurolabel.brain.fnirs.windows import load_fnirs_csv_text

        series = load_fnirs_csv_text(csv_text)
        seq = compute_window_metrics_sequence(series)
        self._ensure_mesh_mapping()
        return {
            'schema_version': 1,
            'source': {
                'device': 'mendi',
                'mode': 'fnirs',
                'stage': 'raw_optical_proxy',
                'filename': filename,
            },
            'mesh': self._mesh_json,
            'mapping': self._mapping_json,
            'display_delay_sec': self._display_delay_sec,
            'windows': [frame_to_json(_frame_from_metrics(m)) for m in seq.windows],
            'viewer_defaults': {
                'camera': 'isometric_front_left',
                'colorscale': 'RdBu_r',
                'symmetric_scale': True,
            },
            'disclaimer': 'Proxy prefrontal heatmap from sparse Mendi channels; not volumetric imaging.',
            'sample_rate_est_hz': seq.sample_rate_est_hz,
        }

    # ---- internal helpers -------------------------------------------
    def _ensure_mesh_mapping(self) -> None:
        with self._lock:
            if self._mesh_json is not None and self._mapping_json is not None:
                return
        mesh = load_brain_mesh(prefer_fsaverage=True)
        mapping = build_pfc_proxy_mapping(mesh)
        mesh_json = mesh_to_json(mesh)
        mapping_json = mapping_to_json(mapping)
        with self._lock:
            self._mesh_json = mesh_json
            self._mapping_json = mapping_json

    def _broadcast(self, payload: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # Drop the oldest message to keep live stream responsive.
                try:
                    q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(payload)
                except Exception:
                    pass

    def _append_sample(self, row: dict[str, float]) -> None:
        with self._lock:
            self._rows.append(row)
            self._sample_count += 1
            self._latest_sensor_ts = float(row['timestamp'])
            cutoff = self._latest_sensor_ts - 300.0
            while len(self._rows) > 2 and self._rows[0]['timestamp'] < cutoff:
                self._rows.pop(0)
            sample_count = self._sample_count
            now = time.time()
            should_emit_status = (now - self._last_status_emit) >= 1.0
            if should_emit_status:
                self._last_status_emit = now

        if should_emit_status:
            status = self.status()
            self._broadcast({'type': 'fnirs_sample_stats', **status})
            self._broadcast({'type': 'fnirs_status', **status})

    def _emit_delayed_heatmap_if_ready(self) -> None:
        with self._lock:
            rows = list(self._rows)
            latest_sensor_ts = float(self._latest_sensor_ts)
            last_emitted = float(self._last_emitted_window_ts)
            delay = float(self._display_delay_sec)

        if len(rows) < 8:
            return
        try:
            series = series_from_rows(rows)
            seq = compute_window_metrics_sequence(series, stride_sec=0.25)
        except Exception:
            return
        if not seq.windows:
            return

        display_cutoff = latest_sensor_ts - delay
        candidate = None
        for m in seq.windows:
            if m.timestamp <= display_cutoff and m.timestamp > last_emitted:
                candidate = m
        if candidate is None:
            return

        frame = _frame_from_metrics(candidate)
        envelope = {
            'type': 'fnirs_heatmap_frame',
            'sensor_now_ts': latest_sensor_ts,
            'display_ts': float(candidate.timestamp),
            'display_delay_sec': delay,
            'window': frame_to_json(frame),
            'qc': {
                'sample_rate_est_hz': seq.sample_rate_est_hz,
                'n_samples_buffered': len(rows),
                'window_n_samples': candidate.n_samples,
            },
        }
        with self._lock:
            if candidate.timestamp <= self._last_emitted_window_ts:
                return
            self._last_emitted_window_ts = float(candidate.timestamp)
        self._broadcast(envelope)

    def _worker_main(self) -> None:
        try:
            if self._mock_mode:
                self._run_mock_worker()
            else:
                asyncio.run(self._run_mendi_worker())
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._connected = False
                self._streaming = False
            self._broadcast({'type': 'fnirs_error', 'message': str(exc)})
        finally:
            with self._lock:
                self._connected = False
                self._streaming = False
            self._broadcast({'type': 'fnirs_status', **self.status()})

    def _run_mock_worker(self) -> None:
        with self._lock:
            self._connected = True
            self._streaming = True
        self._broadcast({'type': 'fnirs_status', **self.status()})

        start = time.time()
        sample_period = 1.0 / 11.0
        next_tick = start
        while not self._stop_event.is_set():
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            t = now - start
            # Simulated Mendi-like channels (left/right drift with opposite-phase modulations)
            ir_l = 20000 + 900 * (0.7 + 0.3 * __import__('math').sin(t * 0.6)) + 100 * __import__('math').sin(t * 2.3)
            red_l = 3000 + 300 * (0.4 + 0.6 * __import__('math').sin(t * 0.5 + 0.4))
            amb_l = -250 + 15 * __import__('math').sin(t * 3.0)
            ir_r = 19800 + 850 * (0.7 + 0.3 * __import__('math').sin(t * 0.6 + 1.2)) + 90 * __import__('math').sin(t * 2.1)
            red_r = 3050 + 280 * (0.4 + 0.6 * __import__('math').sin(t * 0.5 + 1.5))
            amb_r = -240 + 18 * __import__('math').sin(t * 2.7 + 0.2)
            ir_p = 180000 + 800 * __import__('math').sin(t * 6.0)
            red_p = 75000 + 450 * __import__('math').sin(t * 6.0 + 0.5)
            amb_p = -2300 + 50 * __import__('math').sin(t * 4.0)
            temp = 32.0 + 0.2 * __import__('math').sin(t * 0.1)
            self._append_sample({
                'timestamp': now,
                'ir_l': ir_l, 'red_l': red_l, 'amb_l': amb_l,
                'ir_r': ir_r, 'red_r': red_r, 'amb_r': amb_r,
                'ir_p': ir_p, 'red_p': red_p, 'amb_p': amb_p,
                'temp': temp,
            })
            self._emit_delayed_heatmap_if_ready()
            next_tick += sample_period

    async def _run_mendi_worker(self) -> None:
        from mendi.ble_client import MendiClient

        async with MendiClient() as mendi:
            with self._lock:
                self._connected = True
            self._broadcast({'type': 'fnirs_status', **self.status()})

            def _on_frame(_label: str, pkt: Any) -> None:
                self._append_sample({
                    'timestamp': time.time(),
                    'ir_l': float(pkt.ir_l), 'red_l': float(pkt.red_l), 'amb_l': float(pkt.amb_l),
                    'ir_r': float(pkt.ir_r), 'red_r': float(pkt.red_r), 'amb_r': float(pkt.amb_r),
                    'ir_p': float(pkt.ir_p), 'red_p': float(pkt.red_p), 'amb_p': float(pkt.amb_p),
                    'temp': float(pkt.temp),
                })

            mendi.on('frame', _on_frame)
            await mendi.start_streaming()
            with self._lock:
                self._streaming = True
            self._broadcast({'type': 'fnirs_status', **self.status()})

            try:
                while not self._stop_event.is_set() and mendi.is_connected:
                    self._emit_delayed_heatmap_if_ready()
                    await asyncio.sleep(0.25)
            finally:
                try:
                    if mendi.is_streaming:
                        await mendi.stop_streaming()
                except Exception:
                    pass


_service: FnirsLiveService | None = None


def get_fnirs_live_service() -> FnirsLiveService:
    global _service
    if _service is None:
        _service = FnirsLiveService()
    return _service


def _frame_from_metrics(m: Any):
    from neurolabel.ui.replay.brain3d.schemas import HeatmapFrame

    return HeatmapFrame(
        sec=float(m.sec),
        timestamp=float(m.timestamp),
        left_raw_score=float(m.left_raw_score),
        right_raw_score=float(m.right_raw_score),
        pulse_quality=float(getattr(m, 'pulse_quality', 0.0)),
        n_samples=int(getattr(m, 'n_samples', 0)),
        left_red_z=float(getattr(m, 'left_red_z', 0.0)),
        left_ir_z=float(getattr(m, 'left_ir_z', 0.0)),
        left_amb_z=float(getattr(m, 'left_amb_z', 0.0)),
        right_red_z=float(getattr(m, 'right_red_z', 0.0)),
        right_ir_z=float(getattr(m, 'right_ir_z', 0.0)),
        right_amb_z=float(getattr(m, 'right_amb_z', 0.0)),
    )
