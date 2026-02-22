/**
 * Game Stream — WebSocket client for MetaDrive frame streaming.
 * Receives JPEG frames → draws on canvas.
 * Sends keyboard state → server maps to MetaDrive actions.
 * 45-second session timer + Esc to stop early.
 */

const SESSION_SECONDS = 45;

export function startGameStream(
  canvas: HTMLCanvasElement,
  onEnd: () => void,
  onTick?: (remainingSeconds: number) => void,
  onLoading?: (msg: string) => void,
): { stop(): void } {
  const ctx = canvas.getContext('2d')!;
  const ws = new WebSocket(`ws://${location.host}/ws/game`);
  ws.binaryType = 'arraybuffer';

  let stopped = false;
  let timerId: ReturnType<typeof setInterval> | null = null;
  let remaining = SESSION_SECONDS;
  let firstFrame = false;

  function cleanup() {
    if (stopped) return;
    stopped = true;
    if (timerId !== null) clearInterval(timerId);
    clearInterval(keyPollId);
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('keyup', onKeyUp);
    try {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'stop' }));
        ws.close();
      }
    } catch { /* ignore */ }
    onEnd();
  }

  function startTimer() {
    if (timerId !== null) return; // already started
    remaining = SESSION_SECONDS;
    onTick?.(remaining);
    timerId = setInterval(() => {
      remaining--;
      onTick?.(remaining);
      if (remaining <= 0) {
        cleanup();
      }
    }, 1000);
  }

  // ── Frame rendering ──────────────────────────────────
  ws.onmessage = (e) => {
    if (stopped) return;

    if (e.data instanceof ArrayBuffer) {
      // Start timer on first frame (MetaDrive is ready)
      if (!firstFrame) {
        firstFrame = true;
        startTimer();
      }

      const blob = new Blob([e.data], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        if (canvas.width !== img.width) canvas.width = img.width;
        if (canvas.height !== img.height) canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
      };
      img.src = url;
    } else {
      // JSON message
      try {
        const msg = JSON.parse(e.data as string);
        if (msg.type === 'session_end' || msg.type === 'error') {
          cleanup();
        } else if (msg.type === 'loading') {
          onLoading?.(msg.msg || 'Loading...');
        }
      } catch { /* ignore */ }
    }
  };

  ws.onclose = () => {
    if (!stopped) cleanup();
  };

  ws.onerror = () => {
    if (!stopped) cleanup();
  };

  // ── Keyboard input ───────────────────────────────────
  const pressed = new Set<string>();

  function sendKeys() {
    if (ws.readyState !== WebSocket.OPEN || stopped) return;
    ws.send(JSON.stringify({
      type: 'keys',
      up: pressed.has('ArrowUp') || pressed.has('w'),
      down: pressed.has('ArrowDown') || pressed.has('s'),
      left: pressed.has('ArrowLeft') || pressed.has('a'),
      right: pressed.has('ArrowRight') || pressed.has('d'),
    }));
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      cleanup();
      return;
    }
    pressed.add(e.key);
    sendKeys();
  }

  function onKeyUp(e: KeyboardEvent) {
    pressed.delete(e.key);
    sendKeys();
  }

  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  // Continuous key polling — resend state every 100ms while keys are held
  const keyPollId = setInterval(() => {
    if (pressed.size > 0) sendKeys();
  }, 100);

  // Show loading state on open (timer starts on first frame, not here)
  ws.onopen = () => {
    onLoading?.('Starting MetaDrive...');
  };

  return {
    stop() {
      cleanup();
    },
  };
}
