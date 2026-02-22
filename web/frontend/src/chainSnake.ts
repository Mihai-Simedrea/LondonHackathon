/**
 * Chain Snake — subtle pipeline progress indicator.
 * 5 nodes (one per pipeline stage), activated one at a time
 * with damped-overshoot glow. Small, right-aligned, matte labels.
 */

const NODE_RADIUS = 5;
const LABELS = ['Collect', 'Process', 'Train', 'Simulate', 'Compare'];

interface ChainNode {
  x: number;
  y: number;
  label: string;
  activatedAt: number; // ms timestamp, -1 if not yet
  /** 0→1 progress of the fill-line coming FROM the previous node */
  fillProgress: number;
}

export function initChain(canvas: HTMLCanvasElement): {
  start(): void;
  activateNext(): void;
} {
  const ctx = canvas.getContext('2d')!;
  let nodes: ChainNode[] = [];
  let animating = false;
  let activeCount = 0;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    layoutNodes(rect.width, rect.height);
  }

  function layoutNodes(w: number, h: number) {
    const prev = nodes.map((n) => ({ activatedAt: n.activatedAt, fillProgress: n.fillProgress }));
    nodes = [];

    const count = LABELS.length;
    const totalChainHeight = (count - 1) * 48; // 48px between nodes
    const startY = (h - totalChainHeight) / 2;
    const nodeX = w - 22;

    for (let i = 0; i < count; i++) {
      nodes.push({
        x: nodeX,
        y: startY + i * 48,
        label: LABELS[i],
        activatedAt: prev[i]?.activatedAt ?? -1,
        fillProgress: prev[i]?.fillProgress ?? 0,
      });
    }
  }

  function render(time: number) {
    if (!animating) return;
    requestAnimationFrame(render);

    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    // ── Connecting lines ──
    for (let i = 0; i < nodes.length - 1; i++) {
      const a = nodes[i];
      const b = nodes[i + 1];

      // Dim baseline line
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Animated fill line (progresses when next node activates)
      if (b.activatedAt >= 0) {
        // Animate fill progress toward 1
        b.fillProgress = Math.min(1, b.fillProgress + 0.04);
      }
      if (b.fillProgress > 0) {
        const midX = a.x + (b.x - a.x) * b.fillProgress;
        const midY = a.y + (b.y - a.y) * b.fillProgress;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(midX, midY);
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // ── Nodes + labels ──
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];

      if (node.activatedAt < 0) {
        drawInactive(node);
      } else {
        const age = (time - node.activatedAt) / 1000;
        drawActive(node, age);
      }

      // Label — small white matte text to the left of the node
      const isActive = node.activatedAt >= 0;
      const labelAlpha = isActive ? 0.5 : 0.18;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = `rgba(255, 255, 255, ${labelAlpha})`;
      ctx.fillText(node.label, node.x - NODE_RADIUS - 10, node.y);
    }
  }

  function drawInactive(node: ChainNode) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, NODE_RADIUS, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawActive(node: ChainNode, age: number) {
    // Boon-style 3-phase glow: strong burst → gradual fade → settled
    // 2s initial delay before glow begins, then 3s ease-out cubic
    const DELAY = 1.2;
    const DURATION = 3.0;
    if (age < DELAY) {
      drawInactive(node);
      return;
    }
    const t = Math.min(1, (age - DELAY) / DURATION);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
    const glow = 1 - eased; // 1 → 0 over 3s

    // Scale: small initial pulse that settles
    const r = NODE_RADIUS * (1 + glow * 0.35);

    // Settled fill alpha (stays visible after glow fades)
    const fillAlpha = 0.65 + glow * 0.35;

    ctx.save();

    // Outer glow layer (Boon: 60px shadow fading out)
    if (glow > 0.02) {
      ctx.shadowColor = `rgba(0, 240, 255, ${glow * 0.5})`;
      ctx.shadowBlur = 40 * glow + 2;
    }

    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(0, 240, 255, ${fillAlpha})`;
    ctx.fill();

    // Double fill during strong glow phase (Boon 0-30%: extra intensity)
    if (glow > 0.7) {
      ctx.shadowBlur = 20 * glow;
      ctx.fill();
    }

    ctx.restore();

    // Ring — fades from bright to subtle
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(0, 240, 255, ${0.25 + glow * 0.35})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  window.addEventListener('resize', resize);
  resize();

  return {
    start() {
      animating = true;
      activeCount = 0;
      for (const node of nodes) {
        node.activatedAt = -1;
        node.fillProgress = 0;
      }
      requestAnimationFrame(render);
      // Activate first node
      this.activateNext();
    },

    activateNext() {
      if (activeCount < nodes.length) {
        nodes[activeCount].activatedAt = performance.now();
        activeCount++;
      }
    },
  };
}
