import {
  StepVisualizer,
  SSEMessage,
  COLORS,
  createCanvas,
  drawGrid,
  lerp,
} from './types';

interface StepDef {
  label: string;
  key: string;
}

const STEPS: StepDef[] = [
  { label: 'Synthetic', key: 'synthetic' },
  { label: 'Process', key: 'process' },
  { label: 'Train (Dirty)', key: 'train_dirty' },
  { label: 'Train (Clean)', key: 'train_clean' },
  { label: 'Simulate', key: 'simulate' },
  { label: 'Results', key: 'results' },
];

interface MiniVizState {
  // Synthetic
  dataStreamChars: string[];
  streamOffset: number;

  // Process
  ocScores: number[];

  // Train
  treeCount: number;
  treeTotal: number;
  treeGrowth: number; // 0-1 for icon animation

  // Simulate
  dirtyProgress: number;
  dirtyTotal: number;
  cleanProgress: number;
  cleanTotal: number;

  // Results
  dirtyStats: { avg_alive: number; avg_reward: number; route_completion: number };
  cleanStats: { avg_alive: number; avg_reward: number; route_completion: number };
  improvement: number;
}

interface ResultsAnim {
  startTime: number;
  counterValue: number;
  barWidths: number[];
  glowPhase: number;
  opacity: number;
  offsetY: number;
}

export function createDemoViz(): StepVisualizer {
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let container: HTMLElement;
  let animId: number | null = null;
  let startTime = 0;

  let currentStepIndex = -1;
  let completedSteps = new Set<number>();
  let statusText = '';
  let showResults = false;
  let resultsAnim: ResultsAnim | null = null;

  const miniViz: MiniVizState = {
    dataStreamChars: [],
    streamOffset: 0,
    ocScores: [],
    treeCount: 0,
    treeTotal: 500,
    treeGrowth: 0,
    dirtyProgress: 0,
    dirtyTotal: 20,
    cleanProgress: 0,
    cleanTotal: 20,
    dirtyStats: { avg_alive: 0, avg_reward: 0, route_completion: 0 },
    cleanStats: { avg_alive: 0, avg_reward: 0, route_completion: 0 },
    improvement: 0,
  };

  // Generate random stream chars for synthetic animation
  function refreshStreamChars() {
    const chars = '0123456789ABCDEF.-+';
    miniViz.dataStreamChars = Array.from({ length: 80 }, () =>
      chars[Math.floor(Math.random() * chars.length)]
    );
  }
  refreshStreamChars();

  function mapStepName(name: string): number {
    const map: Record<string, number> = {
      synthetic: 0,
      collect: 0,
      process: 1,
      train_dirty: 2,
      train_clean: 3,
      simulate: 4,
      results: 5,
    };
    // Also handle "train" with context
    if (name === 'train') {
      // If dirty is done, it's clean; otherwise dirty
      return completedSteps.has(2) ? 3 : 2;
    }
    return map[name] ?? -1;
  }

  function drawTimeline(w: number, t: number) {
    const y = 40;
    const padX = 50;
    const stepSpacing = (w - padX * 2) / (STEPS.length - 1);

    // Draw connecting lines
    for (let i = 0; i < STEPS.length - 1; i++) {
      const x1 = padX + i * stepSpacing;
      const x2 = padX + (i + 1) * stepSpacing;
      const completed = completedSteps.has(i);

      ctx.beginPath();
      ctx.strokeStyle = completed ? COLORS.accent : COLORS.muted;
      ctx.lineWidth = 2;
      if (!completed) {
        ctx.setLineDash([4, 4]);
      } else {
        ctx.setLineDash([]);
      }
      ctx.moveTo(x1, y);
      ctx.lineTo(x2, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw step circles and labels
    for (let i = 0; i < STEPS.length; i++) {
      const x = padX + i * stepSpacing;
      const isActive = i === currentStepIndex;
      const isCompleted = completedSteps.has(i);
      const isPending = !isActive && !isCompleted;

      // Circle
      let radius = 8;
      if (isActive) {
        radius = 8 + 2 * Math.sin(t * 3);
      }

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);

      if (isCompleted) {
        ctx.fillStyle = COLORS.accent;
        ctx.fill();
        // Checkmark
        ctx.strokeStyle = COLORS.bg;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 4, y);
        ctx.lineTo(x - 1, y + 3);
        ctx.lineTo(x + 4, y - 3);
        ctx.stroke();
      } else if (isActive) {
        // Pulsing glow
        ctx.fillStyle = COLORS.accentDim;
        ctx.fill();
        ctx.strokeStyle = COLORS.accent;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Glow ring
        const glowAlpha = 0.3 + 0.2 * Math.sin(t * 3);
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 240, 255, ${glowAlpha})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      } else {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.fill();
        ctx.strokeStyle = COLORS.muted;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = isActive ? COLORS.accent : isPending ? COLORS.muted : COLORS.text;
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(STEPS[i].label, x, y + 25);
    }
  }

  function drawMiniVizSynthetic(x: number, y: number, w: number, h: number, t: number) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();

    // Title
    ctx.fillStyle = COLORS.accent;
    ctx.font = '13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Generating Synthetic Data...', x + w / 2, y + 20);

    // Flowing data stream
    const cols = 20;
    const rows = 6;
    const cellW = w / cols;
    const cellH = 18;
    const startY = y + 35;

    miniViz.streamOffset += 0.05;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = (r * cols + c + Math.floor(miniViz.streamOffset * 10)) % miniViz.dataStreamChars.length;
        const alpha = 0.2 + 0.5 * Math.sin(t * 2 + r * 0.5 + c * 0.3);
        ctx.fillStyle = `rgba(0, 240, 255, ${alpha.toFixed(2)})`;
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(miniViz.dataStreamChars[idx], x + c * cellW + cellW / 2, startY + r * cellH);
      }
    }

    ctx.restore();
  }

  function drawMiniVizProcess(x: number, y: number, w: number, h: number, t: number) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();

    // Title
    ctx.fillStyle = COLORS.accent;
    ctx.font = '13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Computing OC Scores', x + w / 2, y + 20);

    // Simple line chart of OC scores
    if (miniViz.ocScores.length > 1) {
      const padL = 30;
      const padR = 10;
      const padT = 35;
      const padB = 15;
      const chartW = w - padL - padR;
      const chartH = h - padT - padB;
      const maxVal = Math.max(...miniViz.ocScores, 1);

      ctx.beginPath();
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      for (let i = 0; i < miniViz.ocScores.length; i++) {
        const px = x + padL + (i / Math.max(miniViz.ocScores.length - 1, 1)) * chartW;
        const py = y + padT + chartH - (miniViz.ocScores[i] / maxVal) * chartH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Glow under the line
      ctx.lineTo(x + padL + chartW, y + padT + chartH);
      ctx.lineTo(x + padL, y + padT + chartH);
      ctx.closePath();
      ctx.fillStyle = COLORS.accentGlow;
      ctx.fill();
    }

    ctx.restore();
  }

  function drawMiniVizTrain(x: number, y: number, w: number, h: number, t: number) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();

    const isDirty = currentStepIndex === 2;
    const label = isDirty ? 'Training Dirty Model' : 'Training Clean Model';

    // Title
    ctx.fillStyle = COLORS.accent;
    ctx.font = '13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label, x + w / 2, y + 20);

    // Tree counter
    const count = miniViz.treeCount;
    const total = miniViz.treeTotal;
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px monospace';
    ctx.fillText(`Building forest: ${count}/${total}`, x + w / 2, y + h / 2 - 5);

    // Progress bar
    const barW = w * 0.6;
    const barH = 10;
    const barX = x + (w - barW) / 2;
    const barY = y + h / 2 + 15;
    const progress = total > 0 ? count / total : 0;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = COLORS.accent;
    ctx.fillRect(barX, barY, barW * progress, barH);

    // Simple tree icon
    const treeX = x + w / 2;
    const treeY = y + h / 2 + 50;
    const scale = 0.5 + 0.5 * Math.min(miniViz.treeGrowth, 1);

    ctx.fillStyle = '#00e676';
    // Triangle canopy
    ctx.beginPath();
    ctx.moveTo(treeX, treeY - 20 * scale);
    ctx.lineTo(treeX - 12 * scale, treeY);
    ctx.lineTo(treeX + 12 * scale, treeY);
    ctx.closePath();
    ctx.fill();

    // Trunk
    ctx.fillStyle = '#8d6e63';
    ctx.fillRect(treeX - 3 * scale, treeY, 6 * scale, 10 * scale);

    ctx.restore();
  }

  function drawMiniVizSimulate(x: number, y: number, w: number, h: number, t: number) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();

    // Title
    ctx.fillStyle = COLORS.accent;
    ctx.font = '13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Running Simulations', x + w / 2, y + 20);

    const barW = w * 0.5;
    const barH = 14;
    const barX = x + (w - barW) / 2;

    // Dirty model progress
    const dirtyY = y + 50;
    ctx.fillStyle = COLORS.muted;
    ctx.font = '12px monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Dirty:', barX - 10, dirtyY + 11);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.fillRect(barX, dirtyY, barW, barH);
    const dirtyProg = miniViz.dirtyTotal > 0 ? miniViz.dirtyProgress / miniViz.dirtyTotal : 0;
    ctx.fillStyle = COLORS.muted;
    ctx.fillRect(barX, dirtyY, barW * dirtyProg, barH);

    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'left';
    ctx.fillText(`${miniViz.dirtyProgress}/${miniViz.dirtyTotal}`, barX + barW + 10, dirtyY + 11);

    // Clean model progress
    const cleanY = y + 80;
    ctx.fillStyle = COLORS.accent;
    ctx.font = '12px monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Clean:', barX - 10, cleanY + 11);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.fillRect(barX, cleanY, barW, barH);
    const cleanProg = miniViz.cleanTotal > 0 ? miniViz.cleanProgress / miniViz.cleanTotal : 0;
    ctx.fillStyle = COLORS.accent;
    ctx.fillRect(barX, cleanY, barW * cleanProg, barH);

    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'left';
    ctx.fillText(`${miniViz.cleanProgress}/${miniViz.cleanTotal}`, barX + barW + 10, cleanY + 11);

    ctx.restore();
  }

  function drawMiniViz(w: number, h: number, t: number) {
    const vizX = 20;
    const vizY = 85;
    const vizW = w - 40;
    const vizH = h - 130;

    if (currentStepIndex < 0 || showResults) return;

    switch (currentStepIndex) {
      case 0:
        drawMiniVizSynthetic(vizX, vizY, vizW, vizH, t);
        break;
      case 1:
        drawMiniVizProcess(vizX, vizY, vizW, vizH, t);
        break;
      case 2:
      case 3:
        drawMiniVizTrain(vizX, vizY, vizW, vizH, t);
        break;
      case 4:
        drawMiniVizSimulate(vizX, vizY, vizW, vizH, t);
        break;
    }
  }

  function drawResultsReveal(w: number, h: number, t: number) {
    if (!resultsAnim) return;

    const elapsed = (Date.now() - resultsAnim.startTime) / 1000;

    // Entrance animation: opacity and slide up
    const entranceDuration = 0.5;
    const entranceT = Math.min(elapsed / entranceDuration, 1);
    const easeOut = 1 - Math.pow(1 - entranceT, 3);
    resultsAnim.opacity = easeOut;
    resultsAnim.offsetY = 20 * (1 - easeOut);

    ctx.save();
    ctx.globalAlpha = resultsAnim.opacity;
    ctx.translate(0, resultsAnim.offsetY);

    const centerX = w / 2;
    const revealY = 100;

    // Improvement counter (counts up over 1.5s, starting after entrance)
    const counterDelay = 0.3;
    const counterDuration = 1.5;
    const counterElapsed = Math.max(elapsed - counterDelay, 0);
    const counterT = Math.min(counterElapsed / counterDuration, 1);
    const easeOutCounter = 1 - Math.pow(1 - counterT, 3);
    resultsAnim.counterValue = miniViz.improvement * easeOutCounter;

    // Glow pulse on improvement number
    const glowStart = counterDelay + counterDuration;
    const glowElapsed = Math.max(elapsed - glowStart, 0);
    const pulseCount = 3;
    const pulseDuration = 0.4;
    const totalPulseDur = pulseCount * pulseDuration;
    let glowAlpha = 0;
    if (glowElapsed < totalPulseDur) {
      const pulsePhase = (glowElapsed % pulseDuration) / pulseDuration;
      glowAlpha = 0.6 * Math.sin(pulsePhase * Math.PI);
    }

    // Draw improvement number
    const sign = resultsAnim.counterValue >= 0 ? '+' : '';
    const impText = `${sign}${resultsAnim.counterValue.toFixed(1)}%`;

    // Glow behind text
    if (glowAlpha > 0) {
      ctx.shadowColor = COLORS.accent;
      ctx.shadowBlur = 30 * glowAlpha;
    }

    ctx.fillStyle = COLORS.accent;
    ctx.font = 'bold 36px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(impText, centerX, revealY);
    ctx.shadowBlur = 0;

    ctx.fillStyle = COLORS.muted;
    ctx.font = '12px monospace';
    ctx.fillText('Overall Improvement', centerX, revealY + 18);

    // Side-by-side comparison bars
    const metrics = [
      { label: 'Avg Alive', dirty: miniViz.dirtyStats.avg_alive, clean: miniViz.cleanStats.avg_alive },
      { label: 'Avg Reward', dirty: miniViz.dirtyStats.avg_reward, clean: miniViz.cleanStats.avg_reward },
      { label: 'Route Comp.', dirty: miniViz.dirtyStats.route_completion, clean: miniViz.cleanStats.route_completion },
    ];

    const barStartY = revealY + 45;
    const barMaxW = w * 0.3;
    const barH = 16;
    const barGap = 35;
    const barDelay = 0.8; // bars start growing after this
    const barDuration = 1.0;
    const barStagger = 0.2;

    for (let i = 0; i < metrics.length; i++) {
      const m = metrics[i];
      const my = barStartY + i * barGap;

      // Bar animation timing
      const bElapsed = Math.max(elapsed - barDelay - i * barStagger, 0);
      const bT = Math.min(bElapsed / barDuration, 1);
      const bEase = 1 - Math.pow(1 - bT, 3);

      // Determine max for normalization
      const maxVal = Math.max(Math.abs(m.dirty), Math.abs(m.clean), 0.001);

      // Label
      ctx.fillStyle = COLORS.text;
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(m.label, centerX, my - 3);

      // Dirty bar (left-aligned from center)
      const dirtyW = (Math.abs(m.dirty) / maxVal) * barMaxW * bEase;
      ctx.fillStyle = COLORS.muted;
      ctx.fillRect(centerX - 10 - dirtyW, my, dirtyW, barH);

      // Dirty value
      ctx.fillStyle = COLORS.muted;
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(
        (m.dirty * bEase).toFixed(1),
        centerX - 10 - dirtyW - 5,
        my + 12
      );

      // Clean bar (right-aligned from center)
      const cleanW = (Math.abs(m.clean) / maxVal) * barMaxW * bEase;
      ctx.fillStyle = COLORS.accent;
      ctx.fillRect(centerX + 10, my, cleanW, barH);

      // Clean value
      ctx.fillStyle = COLORS.accent;
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(
        (m.clean * bEase).toFixed(1),
        centerX + 10 + cleanW + 5,
        my + 12
      );
    }

    // Legend
    const legendY = barStartY + metrics.length * barGap + 10;
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    ctx.fillStyle = COLORS.muted;
    ctx.fillRect(centerX - 60, legendY, 10, 10);
    ctx.fillText('Dirty', centerX - 35, legendY + 9);

    ctx.fillStyle = COLORS.accent;
    ctx.fillRect(centerX + 20, legendY, 10, 10);
    ctx.fillText('Clean', centerX + 45, legendY + 9);

    ctx.restore();
  }

  function draw() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const t = (Date.now() - startTime) / 1000;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);
    drawGrid(ctx, w, h);

    // Timeline
    drawTimeline(w, t);

    // Status text
    if (statusText) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(statusText, w / 2, h - 10);
    }

    if (showResults) {
      drawResultsReveal(w, h, t);
    } else {
      drawMiniViz(w, h, t);
    }

    animId = requestAnimationFrame(draw);
  }

  return {
    mount(el: HTMLElement) {
      container = el;
      const c = createCanvas(el);
      canvas = c.canvas;
      ctx = c.ctx;
    },

    onStart() {
      startTime = Date.now();
      currentStepIndex = 0;
      completedSteps.clear();
      showResults = false;
      resultsAnim = null;
      animId = requestAnimationFrame(draw);
    },

    onData(msg: SSEMessage) {
      const msgType = msg.type as string;

      // Step markers
      if (msgType === 'step_marker') {
        const name = msg.name as string;
        const status = msg.status as string;
        const idx = mapStepName(name);
        if (idx >= 0) {
          if (status === 'start') {
            currentStepIndex = idx;
            // Reset mini-viz state for new step
            if (idx === 0) refreshStreamChars();
            if (idx === 1) miniViz.ocScores = [];
            if (idx === 2 || idx === 3) {
              miniViz.treeCount = 0;
              miniViz.treeGrowth = 0;
            }
          } else if (status === 'done') {
            completedSteps.add(idx);
          }
        }
        return;
      }

      // OC scores for process step
      if (msgType === 'oc' && typeof msg.score === 'number') {
        miniViz.ocScores.push(msg.score as number);
      }

      // Tree building for train steps
      if (msgType === 'tree') {
        miniViz.treeCount = (msg.count as number) || miniViz.treeCount + 1;
        miniViz.treeTotal = (msg.total as number) || miniViz.treeTotal;
        miniViz.treeGrowth = miniViz.treeTotal > 0 ? miniViz.treeCount / miniViz.treeTotal : 0;
      }

      // Simulation run progress
      if (msgType === 'run') {
        const model = msg.model as string;
        if (model === 'dirty' || model === 'distracted') {
          miniViz.dirtyProgress = (msg.index as number) || miniViz.dirtyProgress + 1;
          miniViz.dirtyTotal = (msg.total as number) || miniViz.dirtyTotal;
        } else if (model === 'clean' || model === 'normal') {
          miniViz.cleanProgress = (msg.index as number) || miniViz.cleanProgress + 1;
          miniViz.cleanTotal = (msg.total as number) || miniViz.cleanTotal;
        }
      }

      // Stats / final results
      if (msgType === 'stats') {
        const model = msg.model as string;
        const stats = {
          avg_alive: (msg.avg_alive as number) || 0,
          avg_reward: (msg.avg_reward as number) || 0,
          route_completion: (msg.route_completion as number) || 0,
        };
        if (model === 'dirty' || model === 'distracted') {
          miniViz.dirtyStats = stats;
        } else if (model === 'clean' || model === 'normal') {
          miniViz.cleanStats = stats;
        }

        // Compute improvement
        if (miniViz.dirtyStats.avg_reward !== 0) {
          miniViz.improvement =
            ((miniViz.cleanStats.avg_reward - miniViz.dirtyStats.avg_reward) /
              Math.abs(miniViz.dirtyStats.avg_reward)) *
            100;
        }
      }

      // Split messages for process step
      if (msgType === 'split') {
        // just an indicator that processing is happening, no special handling needed
      }
    },

    onText(text: string) {
      statusText = text.trim().slice(0, 80);
    },

    onComplete() {
      // Mark all steps as completed
      for (let i = 0; i < STEPS.length; i++) {
        completedSteps.add(i);
      }
      currentStepIndex = 5; // Results step

      // Trigger results reveal
      showResults = true;
      resultsAnim = {
        startTime: Date.now(),
        counterValue: 0,
        barWidths: [0, 0, 0],
        glowPhase: 0,
        opacity: 0,
        offsetY: 20,
      };
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
