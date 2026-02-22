import { StepVisualizer, SSEMessage, COLORS, createCanvas, drawGrid } from './types';

const CHANNEL_NAMES = ['Ch1', 'Ch2', 'Ch3', 'Ch4'];
const CHANNEL_OPACITIES = [1.0, 0.7, 0.5, 0.3];
const ACTION_LABELS = ['Accel', 'Brake', 'Left', 'Right'];

export function createCollectViz(): StepVisualizer {
  let canvas: HTMLCanvasElement | null = null;
  let ctx: CanvasRenderingContext2D | null = null;
  let animId = 0;
  let running = false;
  let completed = false;
  let startTime = 0;
  let frameCount = 0;
  let scanY = 0;

  // Waveform buffers — each channel stores recent sample values
  const BUFFER_LEN = 300;
  const waveBuffers: number[][] = CHANNEL_NAMES.map(() => new Array(BUFFER_LEN).fill(0));
  // Per-channel phase offsets for variety
  const phaseOffsets = [0, 1.2, 2.7, 4.1];
  // Per-channel frequency multipliers
  const freqMults = [1.0, 1.4, 0.7, 2.1];

  // Action distribution values (simulated, slowly grow)
  const actionValues = [0, 0, 0, 0];

  // Signal strength bars (0-1)
  const signalStrengths = [0.8, 0.6, 0.9, 0.5];

  function generateSample(channel: number, t: number): number {
    const phase = phaseOffsets[channel];
    const freq = freqMults[channel];
    const base = Math.sin(t * freq * 2 + phase) * 0.4;
    const harmonic = Math.sin(t * freq * 5.3 + phase * 2) * 0.15;
    const noise = (Math.random() - 0.5) * 0.2;
    const drift = Math.sin(t * 0.3 + phase) * 0.15;
    return base + harmonic + noise + drift;
  }

  function updateSimulation(t: number) {
    // Push new samples into waveform buffers
    for (let ch = 0; ch < 4; ch++) {
      waveBuffers[ch].shift();
      waveBuffers[ch].push(generateSample(ch, t));
    }

    // Slowly grow action distribution bars
    for (let i = 0; i < actionValues.length; i++) {
      const target = 0.3 + Math.sin(t * 0.5 + i * 1.5) * 0.3 + Math.random() * 0.05;
      actionValues[i] += (target - actionValues[i]) * 0.02;
    }

    // Fluctuate signal strengths
    for (let i = 0; i < signalStrengths.length; i++) {
      signalStrengths[i] = 0.5 + Math.sin(t * 0.7 + i * 2) * 0.3 + Math.random() * 0.1;
      signalStrengths[i] = Math.max(0.1, Math.min(1.0, signalStrengths[i]));
    }
  }

  function draw(t: number) {
    if (!ctx || !canvas) return;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;

    // Clear
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    // Grid
    drawGrid(ctx, w, h, 30);

    const topH = h * 0.55;
    const bottomH = h - topH;

    // --- TOP: Oscilloscope waveforms ---
    drawWaveforms(ctx, w, topH, t);

    // --- BOTTOM LEFT: Action distribution ---
    const bottomLeftW = w * 0.5;
    drawActionBars(ctx, 0, topH, bottomLeftW, bottomH);

    // --- BOTTOM RIGHT: Status indicators ---
    drawStatus(ctx, bottomLeftW, topH, w - bottomLeftW, bottomH, t);

    // --- Scan line effect ---
    if (running && !completed) {
      scanY = (scanY + 0.5) % h;
      ctx.fillStyle = 'rgba(0, 240, 255, 0.03)';
      ctx.fillRect(0, scanY - 1, w, 2);
      ctx.fillStyle = 'rgba(0, 240, 255, 0.06)';
      ctx.fillRect(0, scanY, w, 1);
    }

    // --- COMPLETE badge ---
    if (completed) {
      drawCompleteBadge(ctx, w, h);
    }
  }

  function drawWaveforms(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    _t: number
  ) {
    const channelH = h / 4;
    const labelW = 36;

    for (let ch = 0; ch < 4; ch++) {
      const yCenter = channelH * ch + channelH / 2;
      const amplitude = channelH * 0.35;

      // Channel label
      ctx.fillStyle = `rgba(0, 240, 255, ${CHANNEL_OPACITIES[ch]})`;
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(CHANNEL_NAMES[ch], 6, yCenter);

      // Waveform trace
      ctx.beginPath();
      ctx.strokeStyle = `rgba(0, 240, 255, ${CHANNEL_OPACITIES[ch]})`;
      ctx.lineWidth = 1.2;

      const buffer = waveBuffers[ch];
      const drawW = w - labelW;
      for (let i = 0; i < buffer.length; i++) {
        const x = labelW + (i / (buffer.length - 1)) * drawW;
        const y = yCenter + buffer[i] * amplitude;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Glow effect for the trace
      ctx.strokeStyle = `rgba(0, 240, 255, ${CHANNEL_OPACITIES[ch] * 0.2})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i = 0; i < buffer.length; i++) {
        const x = labelW + (i / (buffer.length - 1)) * drawW;
        const y = yCenter + buffer[i] * amplitude;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Separator line between channels
      if (ch < 3) {
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(labelW, channelH * (ch + 1));
        ctx.lineTo(w, channelH * (ch + 1));
        ctx.stroke();
      }
    }

    // Bottom border of waveform area
    ctx.strokeStyle = COLORS.gridBright;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h);
    ctx.lineTo(w, h);
    ctx.stroke();
  }

  function drawActionBars(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number
  ) {
    const pad = 16;
    const barH = 14;
    const gap = 8;
    const labelW = 50;

    // Title
    ctx.fillStyle = COLORS.muted;
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('ACTION DISTRIBUTION', x + pad, y + pad);

    const startY = y + pad + 18;
    const maxBarW = w - pad * 2 - labelW;

    for (let i = 0; i < ACTION_LABELS.length; i++) {
      const by = startY + i * (barH + gap);

      // Label
      ctx.fillStyle = COLORS.text;
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(ACTION_LABELS[i], x + pad, by + barH / 2);

      // Bar background
      ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.fillRect(x + pad + labelW, by, maxBarW, barH);

      // Bar fill
      const barW = actionValues[i] * maxBarW;
      ctx.fillStyle = COLORS.accent;
      ctx.globalAlpha = 0.7;
      ctx.fillRect(x + pad + labelW, by, barW, barH);

      // Glow on bar
      ctx.fillStyle = COLORS.accentGlow;
      ctx.globalAlpha = 1.0;
      ctx.fillRect(x + pad + labelW, by, barW, barH);
      ctx.globalAlpha = 1.0;

      // Percentage
      ctx.fillStyle = COLORS.muted;
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(
        `${Math.round(actionValues[i] * 100)}%`,
        x + pad + labelW + barW + 6,
        by + barH / 2
      );
    }
  }

  function drawStatus(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    t: number
  ) {
    const pad = 16;

    // RECORDING indicator
    const recY = y + pad;
    if (running && !completed) {
      const pulse = Math.sin(t * 4) * 0.5 + 0.5;
      const recColor = pulse > 0.5 ? COLORS.red : COLORS.accent;
      ctx.fillStyle = recColor;
      ctx.font = 'bold 13px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';

      // Dot
      ctx.beginPath();
      ctx.arc(x + pad + 6, recY + 7, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillText('RECORDING', x + pad + 16, recY);
    } else if (completed) {
      ctx.fillStyle = COLORS.green;
      ctx.font = 'bold 13px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('COMPLETE', x + pad + 16, recY);
    } else {
      ctx.fillStyle = COLORS.muted;
      ctx.font = 'bold 13px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('STANDBY', x + pad + 16, recY);
    }

    // Frame counter
    const frameY = recY + 26;
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`Frames: ${frameCount}`, x + pad, frameY);

    // Elapsed time
    const elapsedY = frameY + 22;
    let elapsed = '00:00';
    if (running || completed) {
      const secs = Math.floor(((completed ? Date.now() : Date.now()) - startTime) / 1000);
      const m = Math.floor(secs / 60);
      const s = secs % 60;
      elapsed = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
    ctx.fillText(`Time: ${elapsed}`, x + pad, elapsedY);

    // Signal strength bars
    const sigY = elapsedY + 30;
    ctx.fillStyle = COLORS.muted;
    ctx.font = '10px monospace';
    ctx.textBaseline = 'top';
    ctx.fillText('SIGNAL', x + pad, sigY);

    const barW = 8;
    const barGap = 4;
    const maxBarH = h - (sigY - y) - pad - 20;
    const barBaseY = sigY + 16 + maxBarH;

    for (let i = 0; i < 4; i++) {
      const bx = x + pad + i * (barW + barGap);
      const bh = signalStrengths[i] * maxBarH;

      // Bar background
      ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.fillRect(bx, barBaseY - maxBarH, barW, maxBarH);

      // Bar fill
      const strength = signalStrengths[i];
      if (strength > 0.7) ctx.fillStyle = COLORS.green;
      else if (strength > 0.4) ctx.fillStyle = COLORS.accent;
      else ctx.fillStyle = COLORS.red;

      ctx.fillRect(bx, barBaseY - bh, barW, bh);
    }
  }

  function drawCompleteBadge(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number
  ) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, w, h);

    const badgeW = 180;
    const badgeH = 40;
    const bx = (w - badgeW) / 2;
    const by = (h - badgeH) / 2;

    ctx.fillStyle = 'rgba(0, 230, 118, 0.15)';
    ctx.strokeStyle = COLORS.green;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(bx, by, badgeW, badgeH, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = COLORS.green;
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('COLLECTION COMPLETE', w / 2, h / 2);
  }

  function loop() {
    if (!running && !completed) return;
    const t = (Date.now() - startTime) / 1000;

    if (running && !completed) {
      updateSimulation(t);
      // Auto-increment frame count for simulation feel
      if (Math.random() > 0.7) frameCount++;
    }

    draw(t);

    if (running && !completed) {
      animId = requestAnimationFrame(loop);
    }
  }

  return {
    mount(container: HTMLElement) {
      const result = createCanvas(container);
      canvas = result.canvas;
      ctx = result.ctx;

      // Initial draw
      draw(0);
    },

    onStart() {
      running = true;
      completed = false;
      startTime = Date.now();
      frameCount = 0;
      scanY = 0;

      // Reset buffers
      for (let ch = 0; ch < 4; ch++) {
        waveBuffers[ch].fill(0);
      }
      actionValues.fill(0);

      loop();
    },

    onData(_msg: SSEMessage) {
      // Structured SSE messages — could update frame count or status
      if (_msg.type === 'progress' && typeof _msg.frame === 'number') {
        frameCount = _msg.frame as number;
      }
    },

    onText(text: string) {
      // Parse stdout lines like "Recording frame 42..."
      const match = text.match(/frame\s+(\d+)/i);
      if (match) {
        frameCount = parseInt(match[1], 10);
      }
    },

    onComplete() {
      completed = true;
      running = false;
      cancelAnimationFrame(animId);

      // Final draw with complete badge
      if (ctx && canvas) {
        draw((Date.now() - startTime) / 1000);
      }
    },

    unmount() {
      running = false;
      completed = false;
      cancelAnimationFrame(animId);

      if (canvas) {
        const ro = (canvas as any)._ro as ResizeObserver | undefined;
        if (ro) ro.disconnect();
        canvas.remove();
        canvas = null;
      }
      ctx = null;
    },
  };
}
