import '../fnirs/style.css';
import { FnirsBrainScene } from '../fnirs/brainScene';
import type {
  CsvHeatmapResponse,
  FnirsLiveEvent,
  HeatmapMappingPayload,
  HeatmapMeshPayload,
  HeatmapWindowPayload,
  LiveHeatmapFramePayload,
  LiveMeshInitPayload,
  LiveStatusPayload,
} from '../fnirs/types';

interface PageState {
  mesh: HeatmapMeshPayload | null;
  mapping: HeatmapMappingPayload | null;
  windows: HeatmapWindowPayload[];
  mode: 'idle' | 'csv' | 'live';
  playbackIndex: number;
  isPlaying: boolean;
  playbackTimer: number | null;
  displayDelaySec: number;
  sensorNowTs: number | null;
  displayTs: number | null;
  sampleRateEstHz: number | null;
  liveStatus: string;
  liveConnected: boolean;
  maxAbsScale: number;
}

function fmtTs(ts: number | null): string {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function initFnirsHeatmapPage() {
  document.body.classList.add('fnirs-heatmap-page');
  const bg = document.getElementById('bg-canvas');
  if (bg) (bg as HTMLElement).style.display = 'none';

  const app = document.getElementById('app');
  if (!app) return;

  app.innerHTML = `
    <div class="fnirs-page">
      <div class="card fnirs-header">
        <div class="fnirs-title">
          <h1>fNIRS HEATMAP</h1>
          <p>Standalone 3D prefrontal proxy heatmap for fNIRS data (CSV playback + live backend bridge).</p>
        </div>
        <div id="delay-badge" class="fnirs-badge">Displaying brain response with ~6–7s physiological delay (hemodynamic lag), not network latency.</div>
      </div>

      <div class="card fnirs-controls">
        <label class="fnirs-filebox">
          <input id="fnirs-file" type="file" accept=".csv,text/csv" />
        </label>
        <button id="fnirs-play" class="fnirs-btn" disabled>Play</button>
        <span id="fnirs-conn-dot" class="fnirs-conn-dot" aria-hidden="true"></span>
        <button id="fnirs-connect" class="fnirs-btn live">Connect fNIRS</button>
        <button id="fnirs-disconnect" class="fnirs-btn" disabled>Disconnect</button>
      </div>

      <div class="card fnirs-status-row">
        <div id="fnirs-status" class="fnirs-status">Load an fNIRS CSV or connect a local headset integration.</div>
        <div class="fnirs-legend">
          <div class="fnirs-legend-gradient"></div>
          <span>Cold ↔ Warm</span>
          <span>·</span>
          <span>ghost trail = delayed history</span>
        </div>
      </div>

      <div class="fnirs-layout">
        <div class="card fnirs-scene-card">
          <div class="fnirs-ruler-shell" aria-hidden="true">
            <div id="fnirs-ruler-track" class="fnirs-ruler-track">
              <div id="fnirs-ruler-ticks" class="fnirs-ruler-ticks"></div>
              <div id="fnirs-ruler-ticks-blur" class="fnirs-ruler-ticks-blur"></div>
              <div class="fnirs-ruler-marker show">
                <span class="fnirs-ruler-tag">SHOW</span>
                <span id="fnirs-ruler-show-delta" class="fnirs-ruler-delta">-7s</span>
              </div>
              <div class="fnirs-ruler-marker now">
                <span class="fnirs-ruler-tag">NOW</span>
                <span class="fnirs-ruler-delta">0s</span>
              </div>
            </div>
          </div>
          <div id="fnirs-scene" class="fnirs-scene"></div>
          <div class="fnirs-scene-overlay">
            <div class="fnirs-chip left"><span class="dot"></span><span>L PFC</span></div>
            <div class="fnirs-chip right"><span class="dot"></span><span>R PFC</span></div>
          </div>
        </div>

        <div class="fnirs-panel">
          <div class="card">
            <h3>Playback / Live</h3>
            <div class="fnirs-slider-wrap">
              <input id="fnirs-slider" class="fnirs-slider" type="range" min="0" max="0" value="0" step="1" disabled />
              <div class="fnirs-slider-row">
                <span id="slider-left">Displayed: —</span>
                <span id="slider-right">Window 0 / 0</span>
              </div>
              <div class="fnirs-delay-track" id="delay-track">
                <div class="fnirs-marker" id="marker-sensor" style="left: 0%"></div>
                <div class="fnirs-marker displayed" id="marker-display" style="left: 0%"></div>
                <div class="fnirs-marker-label" id="marker-sensor-label" style="left: 0%">Sensor Now</div>
                <div class="fnirs-marker-label" id="marker-display-label" style="left: 0%">Displayed</div>
              </div>
            </div>
          </div>

          <div class="card">
            <h3>Current Window</h3>
            <div class="fnirs-metric-grid">
              <div class="fnirs-metric">
                <span class="fnirs-metric-label">Left raw score</span>
                <div id="metric-left" class="fnirs-metric-value">—</div>
              </div>
              <div class="fnirs-metric">
                <span class="fnirs-metric-label">Right raw score</span>
                <div id="metric-right" class="fnirs-metric-value">—</div>
              </div>
              <div class="fnirs-metric">
                <span class="fnirs-metric-label">Sensor now</span>
                <div id="metric-sensor-ts" class="fnirs-metric-value">—</div>
              </div>
              <div class="fnirs-metric">
                <span class="fnirs-metric-label">Displayed ts</span>
                <div id="metric-display-ts" class="fnirs-metric-value">—</div>
              </div>
              <div class="fnirs-metric">
                <span class="fnirs-metric-label">Sample rate est</span>
                <div id="metric-sr" class="fnirs-metric-value">—</div>
              </div>
              <div class="fnirs-metric">
                <span class="fnirs-metric-label">Pulse quality</span>
                <div id="metric-pulse" class="fnirs-metric-value">—</div>
              </div>
            </div>
          </div>

          <div class="card">
            <h3>Notes</h3>
            <div class="fnirs-small" id="fnirs-notes">
              Proxy prefrontal heatmap from sparse fNIRS channels. This is not cross-sectional or volumetric brain imaging.
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  const sceneEl = document.getElementById('fnirs-scene') as HTMLElement;
  const brainScene = new FnirsBrainScene(sceneEl);
  let displayAssetReady = false;
  let displayAssetFailed = false;

  const fileInput = document.getElementById('fnirs-file') as HTMLInputElement;
  const playBtn = document.getElementById('fnirs-play') as HTMLButtonElement;
  const connectBtn = document.getElementById('fnirs-connect') as HTMLButtonElement;
  const disconnectBtn = document.getElementById('fnirs-disconnect') as HTMLButtonElement;
  const connDot = document.getElementById('fnirs-conn-dot') as HTMLElement;
  const slider = document.getElementById('fnirs-slider') as HTMLInputElement;
  const statusEl = document.getElementById('fnirs-status') as HTMLElement;
  const notesEl = document.getElementById('fnirs-notes') as HTMLElement;
  const sliderLeft = document.getElementById('slider-left') as HTMLElement;
  const sliderRight = document.getElementById('slider-right') as HTMLElement;
  const mLeft = document.getElementById('metric-left') as HTMLElement;
  const mRight = document.getElementById('metric-right') as HTMLElement;
  const mSensorTs = document.getElementById('metric-sensor-ts') as HTMLElement;
  const mDisplayTs = document.getElementById('metric-display-ts') as HTMLElement;
  const mSr = document.getElementById('metric-sr') as HTMLElement;
  const mPulse = document.getElementById('metric-pulse') as HTMLElement;
  const delayTrack = document.getElementById('delay-track') as HTMLElement;
  const markerSensor = document.getElementById('marker-sensor') as HTMLElement;
  const markerDisplay = document.getElementById('marker-display') as HTMLElement;
  const markerSensorLabel = document.getElementById('marker-sensor-label') as HTMLElement;
  const markerDisplayLabel = document.getElementById('marker-display-label') as HTMLElement;
  const rulerTrack = document.getElementById('fnirs-ruler-track') as HTMLElement;
  const rulerTicks = document.getElementById('fnirs-ruler-ticks') as HTMLElement;
  const rulerTicksBlur = document.getElementById('fnirs-ruler-ticks-blur') as HTMLElement;
  const rulerShowDelta = document.getElementById('fnirs-ruler-show-delta') as HTMLElement;

  const state: PageState = {
    mesh: null,
    mapping: null,
    windows: [],
    mode: 'idle',
    playbackIndex: 0,
    isPlaying: false,
    playbackTimer: null,
    displayDelaySec: 6.5,
    sensorNowTs: null,
    displayTs: null,
    sampleRateEstHz: null,
    liveStatus: 'Idle',
    liveConnected: false,
    maxAbsScale: 2.0,
  };

  let liveSource: EventSource | null = null;
  const useMockLive = new URLSearchParams(window.location.search).get('mockLive') === '1';
  if (useMockLive) {
    statusEl.textContent = 'Dev mode: Connect fNIRS will use a mock live stream (?mockLive=1).';
  }

  function setStatus(text: string, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle('fnirs-error', isError);
  }

  function syncLiveControlsUi() {
    connectBtn.classList.toggle('connected', state.liveConnected);
    connectBtn.textContent = state.liveConnected ? 'Disconnect' : 'Connect';
    connDot.classList.toggle('connected', state.liveConnected);
    connDot.setAttribute('aria-label', state.liveConnected ? 'fNIRS connected' : 'fNIRS disconnected');
    disconnectBtn.disabled = true; // hidden legacy control
  }

  function setNotes(text: string) {
    notesEl.textContent = text;
  }

  function currentWindow(): HeatmapWindowPayload | null {
    if (!state.windows.length) return null;
    const idx = clamp(state.playbackIndex, 0, state.windows.length - 1);
    return state.windows[idx] ?? null;
  }

  function ghostWindows(): HeatmapWindowPayload[] {
    const result: HeatmapWindowPayload[] = [];
    for (let i = 1; i <= 3; i++) {
      const idx = state.playbackIndex - i;
      if (idx >= 0 && state.windows[idx]) result.push(state.windows[idx]);
    }
    return result;
  }

  function recomputeScale() {
    const absVals: number[] = [];
    for (const w of state.windows) {
      absVals.push(Math.abs(w.left_raw_score), Math.abs(w.right_raw_score));
    }
    if (!absVals.length) {
      state.maxAbsScale = 1.5;
      return;
    }
    absVals.sort((a, b) => a - b);
    const q80 = absVals[Math.floor((absVals.length - 1) * 0.8)] ?? absVals[absVals.length - 1];
    const q90 = absVals[Math.floor((absVals.length - 1) * 0.9)] ?? absVals[absVals.length - 1];
    const maxSeen = absVals[absVals.length - 1] ?? q90;
    const robust = Math.max(q80 * 0.92, q90 * 0.82, maxSeen * 0.40);
    state.maxAbsScale = Math.min(2.2, Math.max(0.32, Math.ceil(robust * 20) / 20));
  }

  function updateRulerUi() {
    const pxPerSec = 14;
    const delayRounded = Math.round(state.displayDelaySec || 7);
    const showOffset = Math.round(clamp((state.displayDelaySec || 6.5) * pxPerSec, 72, 180));
    rulerTrack.style.setProperty('--show-offset-px', `${showOffset}px`);
    rulerTrack.style.setProperty('--tick-px-per-sec', `${pxPerSec}px`);
    rulerShowDelta.textContent = `-${delayRounded}s`;
    rulerTicks.classList.add('moving');
    rulerTicksBlur.classList.add('moving');
  }

  function applyMeshAndMappingIfReady() {
    if (displayAssetReady) {
      renderCurrentFrame();
      return;
    }
    if (displayAssetFailed && (!state.mesh || !state.mapping)) return;
    if (!state.mesh || !state.mapping) return;
    brainScene.setMesh(state.mesh, state.mapping);
    renderCurrentFrame();
  }

  function renderCurrentFrame() {
    const w = currentWindow();
    const ghosts = ghostWindows();
    brainScene.setFrame(w, ghosts, state.maxAbsScale);

    if (w) {
      const left = w.left_raw_score;
      const right = w.right_raw_score;
      mLeft.textContent = fmtNum(left, 2);
      mRight.textContent = fmtNum(right, 2);
      mLeft.className = `fnirs-metric-value ${left >= 0 ? 'hot' : 'cold'}`;
      mRight.className = `fnirs-metric-value ${right >= 0 ? 'hot' : 'cold'}`;
      mPulse.textContent = fmtNum(w.pulse_quality, 2);
      const displayedTs = state.displayTs ?? w.timestamp;
      const sensorTs = state.sensorNowTs ?? (displayedTs + state.displayDelaySec);
      mDisplayTs.textContent = fmtTs(displayedTs);
      mSensorTs.textContent = fmtTs(sensorTs);
      mSr.textContent = state.sampleRateEstHz ? `${fmtNum(state.sampleRateEstHz, 2)} Hz` : '—';
      sliderLeft.textContent = `Displayed: ${fmtTs(displayedTs)} (~-${fmtNum(state.displayDelaySec, 1)}s)`;
      sliderRight.textContent = `Window ${state.windows.length ? state.playbackIndex + 1 : 0} / ${state.windows.length}`;
      updateDelayTrack(sensorTs, displayedTs);
    } else {
      mLeft.textContent = '—';
      mRight.textContent = '—';
      mPulse.textContent = '—';
      mDisplayTs.textContent = '—';
      mSensorTs.textContent = '—';
      mSr.textContent = '—';
      sliderLeft.textContent = 'Displayed: —';
      sliderRight.textContent = 'Window 0 / 0';
      updateDelayTrack(null, null);
    }
    updateRulerUi();
  }

  function updateDelayTrack(sensorTs: number | null, displayTs: number | null) {
    if (!state.windows.length || sensorTs == null || displayTs == null) {
      markerSensor.style.left = '0%';
      markerDisplay.style.left = '0%';
      markerSensorLabel.style.left = '0%';
      markerDisplayLabel.style.left = '0%';
      delayTrack.style.setProperty('--delay-gap-pct', '0%');
      return;
    }
    const minTs = state.windows[0].timestamp;
    const maxTs = Math.max(state.windows[state.windows.length - 1].timestamp + state.displayDelaySec, sensorTs);
    const span = Math.max(1e-3, maxTs - minTs);
    const sensorPct = clamp(((sensorTs - minTs) / span) * 100, 0, 100);
    const displayPct = clamp(((displayTs - minTs) / span) * 100, 0, 100);
    markerSensor.style.left = `${sensorPct}%`;
    markerDisplay.style.left = `${displayPct}%`;
    markerSensorLabel.style.left = `${sensorPct}%`;
    markerDisplayLabel.style.left = `${displayPct}%`;
    delayTrack.style.setProperty('--delay-gap-pct', `${Math.max(0, sensorPct - displayPct)}%`);
  }

  function setWindows(windows: HeatmapWindowPayload[], mode: 'csv' | 'live') {
    state.windows = windows;
    state.mode = mode;
    state.playbackIndex = Math.max(0, windows.length - 1);
    slider.max = String(Math.max(0, windows.length - 1));
    slider.value = String(state.playbackIndex);
    slider.disabled = windows.length === 0 || mode === 'live';
    playBtn.disabled = windows.length === 0 || mode === 'live';
    recomputeScale();
    renderCurrentFrame();
  }

  function stopPlayback() {
    state.isPlaying = false;
    playBtn.textContent = 'Play';
    if (state.playbackTimer != null) {
      window.clearInterval(state.playbackTimer);
      state.playbackTimer = null;
    }
    updateRulerUi();
  }

  function startPlayback() {
    if (!state.windows.length || state.mode !== 'csv') return;
    stopPlayback();
    state.isPlaying = true;
    playBtn.textContent = 'Pause';
    updateRulerUi();
    state.playbackTimer = window.setInterval(() => {
      if (!state.isPlaying) return;
      state.playbackIndex = (state.playbackIndex + 1) % state.windows.length;
      slider.value = String(state.playbackIndex);
      state.displayTs = currentWindow()?.timestamp ?? null;
      state.sensorNowTs = state.displayTs != null ? state.displayTs + state.displayDelaySec : null;
      renderCurrentFrame();
    }, 650);
  }

  async function handleCsvFile(file: File) {
    stopPlayback();
    setStatus(`Loading ${file.name}...`);
    try {
      const csvText = await file.text();
      const res = await fetch('/api/fnirs/heatmap/from-csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ csv_text: csvText, filename: file.name }),
      });
      const data = (await res.json()) as CsvHeatmapResponse | { detail?: string };
      if (!res.ok) throw new Error((data as { detail?: string }).detail || 'CSV parse failed');
      const payload = data as CsvHeatmapResponse;
      state.displayDelaySec = payload.display_delay_sec || 6.5;
      state.sampleRateEstHz = payload.sample_rate_est_hz || null;
      state.mesh = payload.mesh;
      state.mapping = payload.mapping;
      applyMeshAndMappingIfReady();
      setWindows(payload.windows || [], 'csv');
      state.displayTs = currentWindow()?.timestamp ?? null;
      state.sensorNowTs = state.displayTs != null ? state.displayTs + state.displayDelaySec : null;
      setStatus(`CSV loaded`);
      setNotes(payload.disclaimer || 'Proxy prefrontal heatmap from sparse fNIRS channels.');
      renderCurrentFrame();
    } catch (err) {
      setStatus(`CSV load failed: ${String((err as Error)?.message || err)}`, true);
    }
  }

  async function refreshLiveStatus() {
    try {
      const res = await fetch('/api/fnirs/live/status');
      if (!res.ok) return;
      const s = (await res.json()) as LiveStatusPayload;
      state.liveConnected = !!s.connected;
      state.sampleRateEstHz = s.sample_rate_est_hz ?? state.sampleRateEstHz;
      state.displayDelaySec = s.display_delay_sec ?? state.displayDelaySec;
      syncLiveControlsUi();
    } catch {
      // ignore
    }
  }

  function ensureLiveStream() {
    if (liveSource) return;
    liveSource = new EventSource('/api/fnirs/live/stream');
    liveSource.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data) as FnirsLiveEvent;
        handleLiveEvent(payload);
      } catch {
        // ignore malformed line
      }
    };
    liveSource.onerror = () => {
      setStatus('Live stream disconnected or unavailable.', true);
      state.liveConnected = false;
      syncLiveControlsUi();
    };
  }

  function closeLiveStream() {
    if (liveSource) {
      liveSource.close();
      liveSource = null;
    }
  }

  function handleLiveEvent(payload: FnirsLiveEvent) {
    if (!payload || typeof payload !== 'object') return;
    if ((payload as LiveMeshInitPayload).type === 'fnirs_mesh_init') {
      const p = payload as LiveMeshInitPayload;
      state.mesh = p.mesh;
      state.mapping = p.mapping;
      state.displayDelaySec = p.display_delay_sec ?? state.displayDelaySec;
      applyMeshAndMappingIfReady();
      if (p.disclaimer) setNotes(p.disclaimer);
      updateRulerUi();
      return;
    }
    if ((payload as LiveHeatmapFramePayload).type === 'fnirs_heatmap_frame') {
      const p = payload as LiveHeatmapFramePayload;
      state.mode = 'live';
      state.displayDelaySec = p.display_delay_sec ?? state.displayDelaySec;
      state.sensorNowTs = p.sensor_now_ts;
      state.displayTs = p.display_ts;
      state.sampleRateEstHz = p.qc?.sample_rate_est_hz ?? state.sampleRateEstHz;

      const existingLast = state.windows[state.windows.length - 1];
      if (!existingLast || existingLast.timestamp !== p.window.timestamp) {
        state.windows = [...state.windows, p.window].slice(-240);
        state.playbackIndex = state.windows.length - 1;
        slider.value = String(state.playbackIndex);
        slider.max = String(Math.max(0, state.windows.length - 1));
        recomputeScale();
      }
      playBtn.disabled = true;
      slider.disabled = true;
      renderCurrentFrame();
      return;
    }
    if ((payload as LiveStatusPayload).type === 'fnirs_status' || (payload as LiveStatusPayload).type === 'fnirs_sample_stats') {
      const p = payload as LiveStatusPayload;
      state.liveConnected = !!p.connected;
      state.sampleRateEstHz = p.sample_rate_est_hz ?? state.sampleRateEstHz;
      state.displayDelaySec = p.display_delay_sec ?? state.displayDelaySec;
      syncLiveControlsUi();
      const statusText = p.last_error
        ? `Live error`
        : p.connected
          ? `Live connected${p.streaming ? ' • streaming' : ''}${p.mock_mode ? ' • mock' : ''}`
          : 'Live disconnected';
      setStatus(statusText, !!p.last_error);
      updateRulerUi();
      return;
    }
    if ((payload as { type?: string }).type === 'fnirs_error') {
      const p = payload as { type: 'fnirs_error'; message: string };
      setStatus(`Live error: ${p.message}`, true);
      return;
    }
  }

  fileInput.addEventListener('change', async () => {
    const f = fileInput.files?.[0];
    if (f) {
      await handleCsvFile(f);
    }
  });

  playBtn.addEventListener('click', () => {
    if (state.mode !== 'csv') return;
    if (state.isPlaying) stopPlayback();
    else startPlayback();
  });

  slider.addEventListener('input', () => {
    if (!state.windows.length) return;
    state.playbackIndex = Number(slider.value) || 0;
    state.displayTs = currentWindow()?.timestamp ?? null;
    state.sensorNowTs = state.displayTs != null ? state.displayTs + state.displayDelaySec : null;
    renderCurrentFrame();
  });

  connectBtn.addEventListener('click', async () => {
    if (state.liveConnected) {
      try {
        await fetch('/api/fnirs/live/disconnect', { method: 'POST' });
      } catch {
        // ignore
      }
      state.liveConnected = false;
      if (state.mode === 'live') state.mode = 'idle';
      setStatus('Live disconnected');
      syncLiveControlsUi();
      updateRulerUi();
      return;
    }
    ensureLiveStream();
    stopPlayback();
    // Start a fresh live buffer so scale/window counters are not polluted by CSV playback history.
    setWindows([], 'live');
    state.sensorNowTs = null;
    state.displayTs = null;
    renderCurrentFrame();
    updateRulerUi();
    setStatus('Connecting to fNIRS headset…');
    try {
      const res = await fetch('/api/fnirs/live/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mock: useMockLive }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Connect failed');
      state.mode = 'live';
      state.liveConnected = true;
      syncLiveControlsUi();
      setStatus(`Connecting fNIRS${useMockLive ? ' (mock)' : ''}…`);
      refreshLiveStatus();
    } catch (err) {
      setStatus(`Connect failed: ${String((err as Error)?.message || err)}`, true);
    }
  });

  disconnectBtn.addEventListener('click', async () => {
    try {
      await fetch('/api/fnirs/live/disconnect', { method: 'POST' });
    } catch {
      // ignore
    }
    state.liveConnected = false;
    syncLiveControlsUi();
    if (state.mode === 'live') {
      state.mode = 'idle';
    }
    setStatus('Live stream stopped.');
    updateRulerUi();
  });

  // Cleanup on navigation
  window.addEventListener('beforeunload', () => {
    stopPlayback();
    closeLiveStream();
    brainScene.dispose();
  });

  ensureLiveStream();
  brainScene
    .loadDisplayBrainAsset('/models/brain.glb', '/models/brain_fnirs_mapping.json')
    .then(() => {
      displayAssetReady = true;
      renderCurrentFrame();
    })
    .catch((err) => {
      displayAssetFailed = true;
      console.warn('Display brain asset load failed; falling back to backend mesh payload.', err);
      applyMeshAndMappingIfReady();
    });
  refreshLiveStatus();
  syncLiveControlsUi();
  renderCurrentFrame();
  updateRulerUi();
}
