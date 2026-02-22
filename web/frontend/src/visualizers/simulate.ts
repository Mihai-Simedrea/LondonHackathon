import { StepVisualizer, SSEMessage, COLORS, createCanvas, drawGrid, lerp } from './types';

interface RunResult {
  alive: number;
  reward: number;
  route: number;
  crash_type: string | null;
  flashTime: number; // timestamp of arrival for flash animation
}

interface ModelStats {
  avg_alive: number;
  avg_reward: number;
  avg_route: number;
}

interface AnimatedStats {
  avg_alive: number;
  avg_reward: number;
  avg_route: number;
}

const GRID_COLS = 4;
const GRID_ROWS = 5;
const TILE_GAP = 4;
const HEADER_HEIGHT = 36;
const STATS_HEIGHT = 80;
const COLUMN_GAP = 20;
const FLASH_DURATION = 600; // ms

export function createSimulateViz(): StepVisualizer {
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let container: HTMLElement;
  let animId: number | null = null;

  const dirtyRuns: (RunResult | null)[] = new Array(20).fill(null);
  const cleanRuns: (RunResult | null)[] = new Array(20).fill(null);

  const dirtyStats: ModelStats = { avg_alive: 0, avg_reward: 0, avg_route: 0 };
  const cleanStats: ModelStats = { avg_alive: 0, avg_reward: 0, avg_route: 0 };

  // Animated display values that smoothly interpolate toward targets
  const dirtyAnimStats: AnimatedStats = { avg_alive: 0, avg_reward: 0, avg_route: 0 };
  const cleanAnimStats: AnimatedStats = { avg_alive: 0, avg_reward: 0, avg_route: 0 };

  let improvementTarget = 0;
  let improvementDisplay = 0;
  let showImprovement = false;
  let completed = false;
  let statusText = '';

  function computeRunningAvg(runs: (RunResult | null)[]): ModelStats {
    const filled = runs.filter((r): r is RunResult => r !== null);
    if (filled.length === 0) return { avg_alive: 0, avg_reward: 0, avg_route: 0 };
    return {
      avg_alive: filled.reduce((s, r) => s + r.alive, 0) / filled.length,
      avg_reward: filled.reduce((s, r) => s + r.reward, 0) / filled.length,
      avg_route: filled.reduce((s, r) => s + r.route, 0) / filled.length,
    };
  }

  function drawTile(
    x: number, y: number, w: number, h: number,
    run: RunResult | null, now: number
  ) {
    // Background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.fillRect(x, y, w, h);

    if (!run) {
      // Empty tile — just border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      return;
    }

    // Flash animation
    const elapsed = now - run.flashTime;
    const flashAlpha = elapsed < FLASH_DURATION
      ? Math.max(0, 1 - elapsed / FLASH_DURATION) * 0.6
      : 0;

    // Border color based on result
    const survived = run.route > 0.5 && !run.crash_type;
    const borderColor = survived ? COLORS.green : COLORS.red;

    // Draw border
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

    // Flash overlay
    if (flashAlpha > 0) {
      ctx.fillStyle = `rgba(0, 240, 255, ${flashAlpha})`;
      ctx.fillRect(x, y, w, h);
    }

    // Road stripe (vertical center line)
    const roadX = x + w / 2;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(roadX, y + 4);
    ctx.lineTo(roadX, y + h - 14);
    ctx.stroke();
    ctx.setLineDash([]);

    // Car icon (small rect on the road)
    const carY = y + 6 + (h - 24) * (1 - run.route);
    ctx.fillStyle = survived ? COLORS.green : COLORS.red;
    ctx.fillRect(roadX - 3, carY, 6, 8);

    // Crash X marker
    if (run.crash_type) {
      ctx.strokeStyle = COLORS.red;
      ctx.lineWidth = 1.5;
      const cx = roadX;
      const cy = carY + 4;
      ctx.beginPath();
      ctx.moveTo(cx - 4, cy - 4);
      ctx.lineTo(cx + 4, cy + 4);
      ctx.moveTo(cx + 4, cy - 4);
      ctx.lineTo(cx - 4, cy + 4);
      ctx.stroke();
    }

    // Route completion bar at bottom
    const barH = 3;
    const barY = y + h - barH - 2;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.fillRect(x + 2, barY, w - 4, barH);
    ctx.fillStyle = survived ? COLORS.green : COLORS.red;
    ctx.fillRect(x + 2, barY, (w - 4) * run.route, barH);

    // Alive text
    ctx.fillStyle = COLORS.muted;
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${run.alive}`, x + w / 2, y + h - 8);
  }

  function drawColumn(
    label: string, runs: (RunResult | null)[],
    colX: number, colW: number, gridY: number, isCyan: boolean, now: number
  ) {
    // Header
    ctx.font = 'bold 13px monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = isCyan ? COLORS.accent : COLORS.muted;
    ctx.fillText(label, colX + colW / 2, gridY - 10);

    // Count completed
    const done = runs.filter(r => r !== null).length;
    ctx.font = '10px monospace';
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(`${done}/20 complete`, colX + colW / 2, gridY - 0);

    const tileStartY = gridY + 8;

    // Tile sizing
    const tileW = (colW - (GRID_COLS + 1) * TILE_GAP) / GRID_COLS;
    const availH = container.clientHeight - tileStartY - STATS_HEIGHT - 20;
    const tileH = (availH - (GRID_ROWS + 1) * TILE_GAP) / GRID_ROWS;

    for (let row = 0; row < GRID_ROWS; row++) {
      for (let col = 0; col < GRID_COLS; col++) {
        const idx = row * GRID_COLS + col;
        const tx = colX + TILE_GAP + col * (tileW + TILE_GAP);
        const ty = tileStartY + TILE_GAP + row * (tileH + TILE_GAP);
        drawTile(tx, ty, tileW, tileH, runs[idx], now);
      }
    }
  }

  function drawStatsBar(w: number, h: number) {
    const barY = h - STATS_HEIGHT;

    // Separator line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, barY);
    ctx.lineTo(w - 20, barY);
    ctx.stroke();

    const midX = w / 2;

    // Dirty stats (left)
    const dirtyDone = dirtyRuns.filter(r => r !== null).length;
    if (dirtyDone > 0) {
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(
        `Avg Alive: ${Math.round(dirtyAnimStats.avg_alive)} frames (${(dirtyAnimStats.avg_alive / 60).toFixed(1)}s)  |  Reward: ${dirtyAnimStats.avg_reward.toFixed(1)}  |  Route: ${(dirtyAnimStats.avg_route * 100).toFixed(0)}%`,
        midX / 2, barY + 22
      );
    }

    // Clean stats (right)
    const cleanDone = cleanRuns.filter(r => r !== null).length;
    if (cleanDone > 0) {
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = COLORS.accent;
      ctx.fillText(
        `Avg Alive: ${Math.round(cleanAnimStats.avg_alive)} frames (${(cleanAnimStats.avg_alive / 60).toFixed(1)}s)  |  Reward: ${cleanAnimStats.avg_reward.toFixed(1)}  |  Route: ${(cleanAnimStats.avg_route * 100).toFixed(0)}%`,
        midX + midX / 2, barY + 22
      );
    }

    // Improvement percentage
    if (showImprovement) {
      const sign = improvementDisplay >= 0 ? '+' : '';
      const text = `${sign}${improvementDisplay.toFixed(1)}% improvement`;

      ctx.font = 'bold 18px monospace';
      ctx.textAlign = 'center';

      // Glow
      ctx.shadowColor = COLORS.accent;
      ctx.shadowBlur = 15;
      ctx.fillStyle = COLORS.accent;
      ctx.fillText(text, midX, barY + 55);
      ctx.shadowBlur = 0;
    }

    // Status text
    if (statusText) {
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(statusText, midX, barY + 72);
    }
  }

  function render() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const now = performance.now();

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background grid
    drawGrid(ctx, w, h, 30);

    // Two columns
    const colW = (w - COLUMN_GAP - 20) / 2;
    const leftX = 10;
    const rightX = leftX + colW + COLUMN_GAP;
    const gridY = HEADER_HEIGHT + 14;

    drawColumn('DIRTY MODEL', dirtyRuns, leftX, colW, gridY, false, now);
    drawColumn('CLEAN MODEL', cleanRuns, rightX, colW, gridY, true, now);

    // Column divider
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(w / 2, gridY - 14);
    ctx.lineTo(w / 2, h - STATS_HEIGHT);
    ctx.stroke();
    ctx.setLineDash([]);

    // Smoothly interpolate animated stats
    const lerpSpeed = 0.08;
    dirtyAnimStats.avg_alive = lerp(dirtyAnimStats.avg_alive, dirtyStats.avg_alive, lerpSpeed);
    dirtyAnimStats.avg_reward = lerp(dirtyAnimStats.avg_reward, dirtyStats.avg_reward, lerpSpeed);
    dirtyAnimStats.avg_route = lerp(dirtyAnimStats.avg_route, dirtyStats.avg_route, lerpSpeed);
    cleanAnimStats.avg_alive = lerp(cleanAnimStats.avg_alive, cleanStats.avg_alive, lerpSpeed);
    cleanAnimStats.avg_reward = lerp(cleanAnimStats.avg_reward, cleanStats.avg_reward, lerpSpeed);
    cleanAnimStats.avg_route = lerp(cleanAnimStats.avg_route, cleanStats.avg_route, lerpSpeed);

    if (showImprovement) {
      improvementDisplay = lerp(improvementDisplay, improvementTarget, 0.04);
    }

    // Stats bar
    drawStatsBar(w, h);

    // Title
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = COLORS.text;
    ctx.fillText('SIMULATION  //  MISSION CONTROL', w / 2, 18);

    animId = requestAnimationFrame(render);
  }

  return {
    mount(cont: HTMLElement) {
      container = cont;
      const c = createCanvas(container);
      canvas = c.canvas;
      ctx = c.ctx;
    },

    onStart() {
      // Reset state
      dirtyRuns.fill(null);
      cleanRuns.fill(null);
      Object.assign(dirtyStats, { avg_alive: 0, avg_reward: 0, avg_route: 0 });
      Object.assign(cleanStats, { avg_alive: 0, avg_reward: 0, avg_route: 0 });
      Object.assign(dirtyAnimStats, { avg_alive: 0, avg_reward: 0, avg_route: 0 });
      Object.assign(cleanAnimStats, { avg_alive: 0, avg_reward: 0, avg_route: 0 });
      improvementTarget = 0;
      improvementDisplay = 0;
      showImprovement = false;
      completed = false;
      statusText = '';
      animId = requestAnimationFrame(render);
    },

    onData(msg: SSEMessage) {
      if (msg.type === 'run') {
        const model = msg.model as string;
        const idx = msg.idx as number;
        const run: RunResult = {
          alive: msg.alive as number,
          reward: msg.reward as number,
          route: msg.route as number,
          crash_type: (msg.crash_type as string) || null,
          flashTime: performance.now(),
        };

        const runs = model === 'clean' ? cleanRuns : dirtyRuns;
        const stats = model === 'clean' ? cleanStats : dirtyStats;

        if (idx >= 0 && idx < runs.length) {
          runs[idx] = run;
        }

        // Recompute running averages
        const avg = computeRunningAvg(runs);
        Object.assign(stats, avg);
      }

      if (msg.type === 'stats') {
        const model = msg.model as string;
        const stats = model === 'clean' ? cleanStats : dirtyStats;
        stats.avg_alive = msg.avg_alive as number;
        stats.avg_reward = msg.avg_reward as number;
        stats.avg_route = msg.avg_route as number;
      }

      if (msg.type === 'batch') {
        const model = msg.model as string;
        const stats = model === 'clean' ? cleanStats : dirtyStats;
        if (typeof msg.avg === 'number') {
          stats.avg_alive = msg.avg as number;
        }
        statusText = `${model} model: ${msg.done}/20 runs complete`;
      }
    },

    onText(text: string) {
      // Show progress text
      if (text.trim()) {
        statusText = text.trim().slice(0, 80);
      }
    },

    onComplete() {
      completed = true;
      // Compute improvement based on route completion
      const dirtyDone = dirtyRuns.filter(r => r !== null).length;
      const cleanDone = cleanRuns.filter(r => r !== null).length;
      if (dirtyDone > 0 && cleanDone > 0 && dirtyStats.avg_route > 0) {
        improvementTarget = ((cleanStats.avg_route - dirtyStats.avg_route) / dirtyStats.avg_route) * 100;
        showImprovement = true;
      }
      statusText = 'Simulation complete';
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
