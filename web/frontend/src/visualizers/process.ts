import {
  StepVisualizer,
  SSEMessage,
  COLORS,
  createCanvas,
  drawGrid,
  lerp,
} from './types';

interface OCPoint {
  sec: number;
  oc: number;
}

export function createProcessViz(): StepVisualizer {
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let container: HTMLElement;
  let animId: number | null = null;
  let startTime = 0;

  // State
  const ocHistory: OCPoint[] = [];
  let currentOC = 0;
  let displayOC = 0;
  let dirtyTarget = 0;
  let cleanTarget = 0;
  let dirtyDisplay = 0;
  let cleanDisplay = 0;
  let statusText = '';
  let completed = false;
  let completedFlashAlpha = 0;

  const OC_THRESHOLD = 0.6;

  function drawBrainSilhouette(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    w: number,
    h: number
  ) {
    ctx.beginPath();
    // Start from bottom-left of brain
    const left = cx - w / 2;
    const right = cx + w / 2;
    const top = cy - h / 2;
    const bottom = cy + h / 2;

    // Bottom (brain stem area) — flatter
    ctx.moveTo(cx, bottom);
    ctx.bezierCurveTo(
      left + w * 0.1, bottom,
      left, bottom - h * 0.15,
      left, cy + h * 0.1
    );

    // Left side going up
    ctx.bezierCurveTo(
      left - w * 0.02, cy - h * 0.1,
      left + w * 0.05, top + h * 0.3,
      left + w * 0.1, top + h * 0.2
    );

    // Top — cerebral folds (3 bumps)
    // First bump (frontal)
    ctx.bezierCurveTo(
      left + w * 0.15, top + h * 0.05,
      left + w * 0.25, top - h * 0.02,
      cx - w * 0.15, top + h * 0.03
    );
    // Second bump (parietal)
    ctx.bezierCurveTo(
      cx - w * 0.05, top - h * 0.05,
      cx + w * 0.05, top - h * 0.05,
      cx + w * 0.15, top + h * 0.03
    );
    // Third bump (occipital)
    ctx.bezierCurveTo(
      right - w * 0.25, top - h * 0.02,
      right - w * 0.15, top + h * 0.05,
      right - w * 0.1, top + h * 0.2
    );

    // Right side going down
    ctx.bezierCurveTo(
      right - w * 0.05, top + h * 0.3,
      right + w * 0.02, cy - h * 0.1,
      right, cy + h * 0.05
    );

    // Cerebellum indent (back-bottom)
    ctx.bezierCurveTo(
      right, cy + h * 0.2,
      right - w * 0.05, bottom - h * 0.1,
      right - w * 0.15, bottom - h * 0.05
    );

    // Small cerebellum bump
    ctx.bezierCurveTo(
      right - w * 0.2, bottom + h * 0.02,
      cx + w * 0.1, bottom + h * 0.01,
      cx, bottom
    );

    ctx.closePath();
  }

  function render() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const now = performance.now();
    const elapsed = (now - startTime) / 1000;

    // Smooth interpolation
    displayOC = lerp(displayOC, currentOC, 0.08);
    dirtyDisplay = lerp(dirtyDisplay, dirtyTarget, 0.1);
    cleanDisplay = lerp(cleanDisplay, cleanTarget, 0.1);

    if (completed && completedFlashAlpha > 0) {
      completedFlashAlpha = Math.max(0, completedFlashAlpha - 0.015);
    }

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    // Grid
    drawGrid(ctx, w, h, 30);

    // Layout: left 60%, right 40%
    const leftW = w * 0.6;
    const rightX = leftW;
    const rightW = w * 0.4;

    // ====== LEFT SIDE: Brain + OC Timeline ======

    // -- Brain silhouette --
    const brainCx = leftW / 2;
    const brainAreaH = h * 0.5;
    const brainCy = brainAreaH * 0.5 + 20;
    const brainW = Math.min(leftW * 0.5, brainAreaH * 0.55);
    const brainH = brainW * 0.85;

    // Glow effect
    const glowPulse = 0.5 + 0.5 * Math.sin(elapsed * 2);
    const glowAlpha = 0.1 + glowPulse * 0.15 * displayOC;
    ctx.save();
    ctx.shadowColor = COLORS.accent;
    ctx.shadowBlur = 20 + glowPulse * 15;

    // Filled brain with OC opacity
    drawBrainSilhouette(ctx, brainCx, brainCy, brainW, brainH);
    ctx.fillStyle = `rgba(0, 240, 255, ${displayOC * 0.6 + glowAlpha * 0.2})`;
    ctx.fill();

    // Brain outline
    drawBrainSilhouette(ctx, brainCx, brainCy, brainW, brainH);
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();

    // OC score label below brain
    ctx.fillStyle = COLORS.text;
    ctx.font = 'bold 18px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`OC: ${displayOC.toFixed(2)}`, brainCx, brainCy + brainH / 2 + 30);

    // "Neural Filtering" label
    ctx.fillStyle = COLORS.muted;
    ctx.font = '12px monospace';
    ctx.fillText('NEURAL FILTERING', brainCx, 16);

    // -- OC Timeline chart (bottom half of left side) --
    const chartPadLeft = 50;
    const chartPadRight = 20;
    const chartPadTop = 10;
    const chartPadBottom = 30;
    const chartX = chartPadLeft;
    const chartY = brainAreaH + chartPadTop;
    const chartW = leftW - chartPadLeft - chartPadRight;
    const chartH = h - brainAreaH - chartPadTop - chartPadBottom - 30; // 30 for status text

    // Chart background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(chartX, chartY, chartW, chartH);

    // Chart grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const gy = chartY + (chartH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(chartX, gy);
      ctx.lineTo(chartX + chartW, gy);
      ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = COLORS.muted;
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = 1 - i * 0.25;
      const gy = chartY + (chartH * i) / 4;
      ctx.fillText(val.toFixed(1), chartX - 5, gy + 4);
    }

    // Threshold line (red dashed)
    const thresholdY = chartY + chartH * (1 - OC_THRESHOLD);
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(chartX, thresholdY);
    ctx.lineTo(chartX + chartW, thresholdY);
    ctx.stroke();
    ctx.restore();

    // Threshold label
    ctx.fillStyle = COLORS.red;
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('CUTOFF 0.6', chartX + chartW - 60, thresholdY - 5);

    // Plot OC history
    if (ocHistory.length > 1) {
      const minSec = ocHistory[0].sec;
      const maxSec = Math.max(ocHistory[ocHistory.length - 1].sec, minSec + 10);
      const secRange = maxSec - minSec;

      // Line
      ctx.beginPath();
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      for (let i = 0; i < ocHistory.length; i++) {
        const px = chartX + ((ocHistory[i].sec - minSec) / secRange) * chartW;
        const py = chartY + chartH * (1 - ocHistory[i].oc);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Points with color coding
      for (let i = 0; i < ocHistory.length; i++) {
        const px = chartX + ((ocHistory[i].sec - minSec) / secRange) * chartW;
        const py = chartY + chartH * (1 - ocHistory[i].oc);
        const above = ocHistory[i].oc >= OC_THRESHOLD;

        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = above ? COLORS.green : COLORS.red;
        ctx.fill();

        // Glow
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = above
          ? 'rgba(0, 230, 118, 0.2)'
          : 'rgba(255, 59, 92, 0.2)';
        ctx.fill();
      }

      // X-axis labels (time)
      ctx.fillStyle = COLORS.muted;
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      const tickCount = Math.min(6, Math.floor(secRange / 5) + 1);
      for (let i = 0; i <= tickCount; i++) {
        const sec = minSec + (secRange * i) / tickCount;
        const px = chartX + (i / tickCount) * chartW;
        ctx.fillText(`${Math.round(sec)}s`, px, chartY + chartH + 15);
      }
    }

    // X-axis label
    ctx.fillStyle = COLORS.muted;
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('TIME (seconds)', chartX + chartW / 2, chartY + chartH + 27);

    // ====== RIGHT SIDE: Filtering Stats ======
    const rcx = rightX + rightW / 2;

    // "DATA FILTERING" header
    ctx.fillStyle = COLORS.muted;
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('DATA FILTERING', rcx, 30);

    // Dirty counter
    const dirtyY = 80;
    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px monospace';
    ctx.fillText('DIRTY', rcx, dirtyY);
    ctx.fillStyle = 'rgba(255, 59, 92, 0.7)';
    ctx.font = 'bold 32px monospace';
    ctx.fillText(`${Math.round(dirtyDisplay)}`, rcx, dirtyY + 35);
    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px monospace';
    ctx.fillText('rows', rcx, dirtyY + 52);

    // Clean counter
    const cleanY = 180;
    ctx.fillStyle = COLORS.accent;
    ctx.font = '11px monospace';
    ctx.fillText('CLEAN', rcx, cleanY);
    ctx.fillStyle = COLORS.accent;
    ctx.font = 'bold 32px monospace';
    ctx.fillText(`${Math.round(cleanDisplay)}`, rcx, cleanY + 35);
    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px monospace';
    ctx.fillText('rows', rcx, cleanY + 52);

    // Vertical progress bar (clean/dirty ratio)
    const barW = 40;
    const barH = h * 0.35;
    const barX = rcx - barW / 2;
    const barY = cleanY + 75;
    const total = Math.round(dirtyDisplay) + Math.round(cleanDisplay);
    const cleanRatio = total > 0 ? Math.round(cleanDisplay) / total : 0;

    // Bar background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.fillRect(barX, barY, barW, barH);

    // Dirty portion (top)
    const dirtyH = barH * (1 - cleanRatio);
    ctx.fillStyle = 'rgba(255, 59, 92, 0.3)';
    ctx.fillRect(barX, barY, barW, dirtyH);

    // Clean portion (bottom)
    ctx.fillStyle = 'rgba(0, 240, 255, 0.3)';
    ctx.fillRect(barX, barY + dirtyH, barW, barH - dirtyH);

    // Bar border
    ctx.strokeStyle = COLORS.gridBright;
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barY, barW, barH);

    // Ratio label
    ctx.fillStyle = COLORS.text;
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(
      `${(cleanRatio * 100).toFixed(1)}%`,
      rcx,
      barY + barH + 20
    );
    ctx.fillStyle = COLORS.muted;
    ctx.font = '10px monospace';
    ctx.fillText('clean', rcx, barY + barH + 34);

    // Threshold label
    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px monospace';
    ctx.fillText(`Threshold: ${OC_THRESHOLD.toFixed(2)}`, rcx, barY + barH + 55);

    // Completion flash
    if (completed) {
      const filteredOut = total > 0
        ? ((Math.round(dirtyDisplay) - Math.round(cleanDisplay)) / Math.round(dirtyDisplay) * 100)
        : 0;
      ctx.fillStyle = COLORS.green;
      ctx.font = 'bold 13px monospace';
      ctx.fillText(
        `FILTERED: ${filteredOut > 0 ? filteredOut.toFixed(1) : '0.0'}%`,
        rcx,
        barY + barH + 78
      );
    }

    // Completion flash overlay
    if (completedFlashAlpha > 0) {
      ctx.fillStyle = `rgba(0, 240, 255, ${completedFlashAlpha * 0.3})`;
      ctx.fillRect(0, 0, w, h);
    }

    // Status text at bottom
    if (statusText) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(statusText, 10, h - 10);
    }

    animId = requestAnimationFrame(render);
  }

  return {
    mount(el: HTMLElement) {
      container = el;
      const result = createCanvas(el);
      canvas = result.canvas;
      ctx = result.ctx;
    },

    onStart() {
      startTime = performance.now();
      completed = false;
      completedFlashAlpha = 0;
      ocHistory.length = 0;
      currentOC = 0;
      displayOC = 0;
      dirtyTarget = 0;
      cleanTarget = 0;
      dirtyDisplay = 0;
      cleanDisplay = 0;
      statusText = '';
      animId = requestAnimationFrame(render);
    },

    onData(msg: SSEMessage) {
      if (msg.type === 'oc') {
        const sec = msg.sec as number;
        const oc = msg.oc as number;
        currentOC = oc;
        ocHistory.push({ sec, oc });
      } else if (msg.type === 'split') {
        dirtyTarget = msg.dirty as number;
        cleanTarget = msg.clean as number;
      }
    },

    onText(text: string) {
      statusText = text.trim();
    },

    onComplete() {
      completed = true;
      completedFlashAlpha = 1;
    },

    unmount() {
      if (animId !== null) {
        cancelAnimationFrame(animId);
        animId = null;
      }
      if (canvas) {
        const ro = (canvas as any)._ro as ResizeObserver | undefined;
        if (ro) ro.disconnect();
        canvas.remove();
      }
    },
  };
}
