/**
 * Shared interface for all step visualizers.
 * Each step (collect, process, train, simulate, demo) implements this.
 */

export interface SSEMessage {
  type: string;
  [key: string]: unknown;
}

export interface StepVisualizer {
  /** Create DOM elements inside container. Canvas, labels, etc. */
  mount(container: HTMLElement): void;

  /** Called when the step starts running. Begin animations. */
  onStart(): void;

  /** Called with each structured SSE message from the backend. */
  onData(msg: SSEMessage): void;

  /** Called with raw text lines (stdout) from the backend. */
  onText?(text: string): void;

  /** Called when the step completes. Show final state. */
  onComplete(): void;

  /** Remove all DOM elements and stop animations. */
  unmount(): void;
}

/** Color constants matching the app theme */
export const COLORS = {
  bg: '#0a0a0a',
  text: '#e0e0e0',
  accent: '#00f0ff',
  accentDim: 'rgba(0, 240, 255, 0.3)',
  accentGlow: 'rgba(0, 240, 255, 0.15)',
  grid: 'rgba(255, 255, 255, 0.04)',
  gridBright: 'rgba(255, 255, 255, 0.08)',
  red: '#ff3b5c',
  green: '#00e676',
  muted: 'rgba(255, 255, 255, 0.3)',
} as const;

/** Create a full-size canvas inside a container, return ctx */
export function createCanvas(container: HTMLElement): {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
} {
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  container.appendChild(canvas);

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = container.clientWidth * dpr;
    canvas.height = container.clientHeight * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
  };
  resize();

  const ro = new ResizeObserver(resize);
  ro.observe(container);
  (canvas as any)._ro = ro; // store for cleanup

  const ctx = canvas.getContext('2d')!;
  return { canvas, ctx };
}

/** Draw subtle grid lines on canvas (oscilloscope style) */
export function drawGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  spacing = 30
) {
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= w; x += spacing) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y <= h; y += spacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
}

/** Lerp between two values */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
