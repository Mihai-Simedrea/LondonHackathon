"""FastAPI backend for NeuroLabel web dashboard."""

import sys
import os
import io
import csv
import json
import threading
import queue
from pathlib import Path
from contextlib import redirect_stdout
from types import SimpleNamespace

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

import asyncio  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.responses import StreamingResponse, FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from web.fnirs_live_service import get_fnirs_live_service  # noqa: E402

app = FastAPI(title="NeuroLabel Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────


class ConfigUpdate(BaseModel):
    device_mode: str | None = None
    game_backend: str | None = None


class FnirsCsvHeatmapRequest(BaseModel):
    csv_text: str
    filename: str | None = None


class FnirsLiveConnectRequest(BaseModel):
    mock: bool = False


# ── GET /api/config ─────────────────────────────────────────────────────


@app.get("/api/config")
def get_config():
    return {
        "device_mode": config.DEVICE_MODE,
        "game_backend": config.GAME_BACKEND,
    }


# ── POST /api/config ────────────────────────────────────────────────────


@app.post("/api/config")
def update_config(body: ConfigUpdate):
    if body.device_mode is not None:
        if body.device_mode not in ("eeg", "fnirs"):
            raise HTTPException(400, "device_mode must be 'eeg' or 'fnirs'")
        config.DEVICE_MODE = body.device_mode
        # Update derived path
        config.BRAIN_CSV = config.FNIRS_CSV if body.device_mode == "fnirs" else config.EEG_CSV

    if body.game_backend is not None:
        if body.game_backend not in ("velocity", "metadrive"):
            raise HTTPException(400, "game_backend must be 'velocity' or 'metadrive'")
        config.GAME_BACKEND = body.game_backend

    return get_config()


# ── GET /api/status ──────────────────────────────────────────────────────


@app.get("/api/status")
def get_status():
    """Check which pipeline artefacts exist on disk."""
    files = {
        "eeg_csv": config.EEG_CSV,
        "fnirs_csv": config.FNIRS_CSV,
        "game_jsonl": config.GAME_JSONL,
        "oc_scores_csv": config.OC_SCORES_CSV,
        "dataset_dirty": config.DATASET_DIRTY,
        "dataset_clean": config.DATASET_CLEAN,
        "model_dirty": config.MODEL_DIRTY,
        "model_clean": config.MODEL_CLEAN,
        "results_dirty": config.RESULTS_DIRTY,
        "results_clean": config.RESULTS_CLEAN,
    }
    return {name: Path(p).exists() for name, p in files.items()}


# ── POST /api/run/{step} ────────────────────────────────────────────────

VALID_STEPS = {"collect", "process", "train", "simulate", "demo"}

# Track running steps so we don't launch duplicates
_running_steps: dict[str, bool] = {}


def _sse_format(data: str) -> str:
    """Wrap a string as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def _emit_structured_data(step: str, output_queue: queue.Queue):
    """Emit structured JSON data after a pipeline step completes."""
    try:
        if step == "process":
            _emit_process_data(output_queue)
        elif step == "train":
            _emit_train_data(output_queue)
        elif step == "simulate":
            _emit_simulate_data(output_queue)
        elif step == "demo":
            for sub_step in ("process", "train", "simulate"):
                output_queue.put(json.dumps({"type": "step_marker", "name": sub_step, "status": "done"}))
                if sub_step == "process":
                    _emit_process_data(output_queue)
                elif sub_step == "train":
                    _emit_train_data(output_queue)
                elif sub_step == "simulate":
                    _emit_simulate_data(output_queue)
    except Exception as exc:
        output_queue.put(f"STRUCTURED_DATA_ERROR: {exc}")


def _emit_process_data(output_queue: queue.Queue):
    """Emit OC scores and dataset split counts."""
    oc_path = Path(config.OC_SCORES_CSV)
    if oc_path.exists():
        with open(oc_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                output_queue.put(json.dumps({
                    "type": "oc",
                    "sec": int(float(row.get("sec", row.get("second", 0)))),
                    "fatigue": float(row.get("fatigue", 0)),
                    "engagement": float(row.get("engagement", 0)),
                    "oc": float(row.get("oc", row.get("oc_score", 0))),
                }))

    dirty_count = 0
    clean_count = 0
    dirty_path = Path(config.DATASET_DIRTY)
    clean_path = Path(config.DATASET_CLEAN)
    if dirty_path.exists():
        with open(dirty_path, "r") as f:
            dirty_count = sum(1 for _ in f) - 1  # subtract header
    if clean_path.exists():
        with open(clean_path, "r") as f:
            clean_count = sum(1 for _ in f) - 1
    output_queue.put(json.dumps({"type": "split", "dirty": max(0, dirty_count), "clean": max(0, clean_count)}))


def _emit_train_data(output_queue: queue.Queue):
    """Emit tree structures and feature importances from trained models."""
    import joblib

    for label, model_path in [("dirty", config.MODEL_DIRTY), ("clean", config.MODEL_CLEAN)]:
        p = Path(model_path)
        if not p.exists():
            continue
        model = joblib.load(p)
        # Emit tree structures (first 50 trees)
        for idx, tree in enumerate(model.estimators_[:50]):
            t = tree.tree_
            output_queue.put(json.dumps({
                "type": "tree",
                "model": label,
                "idx": idx,
                "depth": int(t.max_depth),
                "node_count": int(t.node_count),
                "feature_indices": t.feature.tolist(),
                "thresholds": [round(v, 6) for v in t.threshold.tolist()],
                "children_left": t.children_left.tolist(),
                "children_right": t.children_right.tolist(),
            }))
        # Emit feature importances
        output_queue.put(json.dumps({
            "type": "importance",
            "model": label,
            "features": [round(v, 6) for v in model.feature_importances_.tolist()],
        }))


def _emit_simulate_data(output_queue: queue.Queue):
    """Emit per-run results and batch summary stats."""
    for label, results_path in [("dirty", config.RESULTS_DIRTY), ("clean", config.RESULTS_CLEAN)]:
        p = Path(results_path)
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)

        runs = data if isinstance(data, list) else data.get("runs", [])
        total_alive = 0.0
        total_reward = 0.0
        total_route = 0.0
        count = len(runs)

        for idx, run in enumerate(runs):
            crash_type = "none"
            if run.get("crash_vehicle"):
                crash_type = "vehicle"
            elif run.get("out_of_road"):
                crash_type = "road"
            output_queue.put(json.dumps({
                "type": "run",
                "model": label,
                "idx": idx,
                "alive": float(run.get("alive_time", run.get("steps", 0))),
                "reward": float(run.get("reward", run.get("total_reward", 0))),
                "route": float(run.get("route_completion", 0)),
                "crash_type": crash_type,
            }))
            total_alive += float(run.get("alive_time", run.get("steps", 0)))
            total_reward += float(run.get("reward", run.get("total_reward", 0)))
            total_route += float(run.get("route_completion", 0))

        if count > 0:
            output_queue.put(json.dumps({
                "type": "stats",
                "model": label,
                "avg_alive": round(total_alive / count, 2),
                "avg_reward": round(total_reward / count, 2),
                "avg_route": round(total_route / count, 4),
            }))


@app.post("/api/run/{step}")
def run_step(step: str):
    if step not in VALID_STEPS:
        raise HTTPException(400, f"Invalid step '{step}'. Must be one of {VALID_STEPS}")

    if _running_steps.get(step):
        raise HTTPException(409, f"Step '{step}' is already running")

    output_queue: queue.Queue[str | None] = queue.Queue()

    def _worker():
        _running_steps[step] = True
        try:
            import pipeline  # fresh import so config changes are visible

            args = SimpleNamespace()

            if step == "collect":
                args.synthetic = False
                args.record = True

            if step == "demo":
                args.synthetic = True
                args.dev = True

            # Capture stdout line-by-line via a custom writer
            class QueueWriter(io.TextIOBase):
                def __init__(self):
                    self._buf = ""

                def write(self, s):
                    self._buf += s
                    while "\n" in self._buf:
                        line, self._buf = self._buf.split("\n", 1)
                        output_queue.put(line)
                    return len(s)

                def flush(self):
                    if self._buf:
                        output_queue.put(self._buf)
                        self._buf = ""

            writer = QueueWriter()

            with redirect_stdout(writer):
                func = getattr(pipeline, f"cmd_{step}")
                func(args)

            writer.flush()
            _emit_structured_data(step, output_queue)
        except Exception as exc:
            output_queue.put(f"ERROR: {exc}")
        finally:
            _running_steps[step] = False
            output_queue.put(None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    def _event_stream():
        while True:
            item = output_queue.get()
            if item is None:
                yield _sse_format("[DONE]")
                break
            yield _sse_format(item)

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ── GET /api/results ─────────────────────────────────────────────────────


@app.get("/api/results")
def get_results():
    out = {}
    for label, path in [("dirty", config.RESULTS_DIRTY), ("clean", config.RESULTS_CLEAN)]:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                out[label] = json.load(f)
        else:
            out[label] = None
    return out


# ── fNIRS Heatmap / Live Bridge APIs ─────────────────────────────────────


@app.post("/api/fnirs/heatmap/from-csv")
def build_fnirs_heatmap_from_csv(body: FnirsCsvHeatmapRequest):
    service = get_fnirs_live_service()
    try:
        return service.process_csv_text(body.csv_text, filename=body.filename)
    except Exception as exc:
        raise HTTPException(400, f"Failed to parse/render fNIRS CSV: {exc}") from exc


@app.post("/api/fnirs/live/connect")
def connect_fnirs_live(body: FnirsLiveConnectRequest | None = None):
    service = get_fnirs_live_service()
    mock = bool(body.mock) if body is not None else False
    try:
        status = service.connect(mock=mock)
        return {"ok": True, **status}
    except Exception as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/fnirs/live/disconnect")
def disconnect_fnirs_live():
    service = get_fnirs_live_service()
    status = service.disconnect()
    return {"ok": True, **status}


@app.get("/api/fnirs/live/status")
def fnirs_live_status():
    service = get_fnirs_live_service()
    return service.status()


@app.get("/api/fnirs/live/stream")
def fnirs_live_stream():
    service = get_fnirs_live_service()
    sub = service.subscribe()

    def _event_stream():
        try:
            while True:
                try:
                    payload = sub.get(timeout=10.0)
                except queue.Empty:
                    # SSE keepalive comment
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            service.unsubscribe(sub)

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ── WebSocket: MetaDrive Game Streaming ─────────────────────────────────


@app.websocket("/ws/game")
async def game_websocket(ws: WebSocket):
    """Stream MetaDrive frames to browser, receive keyboard input."""
    await ws.accept()

    from web.game_service import GameStreamService

    game = GameStreamService(session_seconds=45.0)

    # Send loading status before blocking on MetaDrive init
    await ws.send_text(json.dumps({"type": "loading", "msg": "Starting MetaDrive..."}))

    # Run blocking start() in a thread to avoid freezing the event loop
    await asyncio.to_thread(game.start)

    if not game.alive:
        await ws.send_text(json.dumps({"type": "error", "msg": "Failed to start MetaDrive"}))
        await ws.close()
        return

    try:
        while game.alive:
            # Send latest frame as binary JPEG
            frame = game.get_frame()
            if frame is not None:
                await ws.send_bytes(frame)

            # Drain all pending keyboard messages (use latest state)
            stop_requested = False
            try:
                while True:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=0.005)
                    if msg.get("type") == "keys":
                        print(f"[ws] keys: up={msg.get('up')} down={msg.get('down')} left={msg.get('left')} right={msg.get('right')}", flush=True)
                        game.set_keys(msg)
                    elif msg.get("type") == "stop":
                        stop_requested = True
                        break
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            if stop_requested:
                break

            await asyncio.sleep(1 / 25)  # ~25fps send rate

        # Session ended
        try:
            await ws.send_text(json.dumps({"type": "session_end"}))
        except Exception:
            pass
    finally:
        game.stop()
        try:
            await ws.close()
        except Exception:
            pass


# ── Static files (frontend) ─────────────────────────────────────────────

_frontend_dist = Path(__file__).resolve().parent / "dist"
_frontend_dist.mkdir(parents=True, exist_ok=True)


@app.get("/fnirs-heatmap")
@app.get("/fnirs-heatmap/")
def fnirs_heatmap_page():
    index_file = _frontend_dist / "index.html"
    if not index_file.exists():
        raise HTTPException(404, "Frontend build not found. Run: cd web/frontend && npm run build")
    return FileResponse(index_file)


app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
