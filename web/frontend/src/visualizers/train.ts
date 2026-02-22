import { StepVisualizer, SSEMessage, COLORS, createCanvas, drawGrid, lerp } from './types';

const FEATURE_NAMES = [
  'dist_left', 'dist_right', 'heading_diff', 'speed', 'steering',
  'yaw_rate', 'lateral_offset', 'lidar_front', 'lidar_front_right',
  'lidar_right', 'lidar_rear_right', 'lidar_rear', 'lidar_rear_left',
  'lidar_left', 'lidar_front_left', 'navi_forward_dist', 'navi_side_dist',
  'navi_lane_radius', 'navi_direction', 'min_front_dist', 'danger_score',
  'heading_alignment',
];

const GRID_COLS = 10;
const GRID_ROWS = 5;
const MAX_RENDERED = GRID_COLS * GRID_ROWS; // 50
const TOTAL_TREES = 500;

interface TreeNode {
  id: number;
  x: number;
  y: number;
  depth: number;
  isLeaf: boolean;
  parentId: number | null;
}

interface TreeData {
  nodes: TreeNode[];
  maxDepth: number;
  animStartTime: number;
}

export function createTrainViz(): StepVisualizer {
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let animId: number | null = null;
  let container: HTMLElement;

  // State
  const trees: TreeData[] = [];
  let treeCount = 0;
  let modelLabel = 'DIRTY MODEL';
  let cvAccuracy: number | null = null;
  let statusText = '';

  // Feature importance: current displayed values and target values (for smooth animation)
  const importanceCurrent = new Float64Array(FEATURE_NAMES.length);
  const importanceTarget = new Float64Array(FEATURE_NAMES.length);

  function buildTreeLayout(
    childrenLeft: number[],
    childrenRight: number[],
    cellW: number,
    cellH: number,
  ): TreeData {
    const n = childrenLeft.length;
    if (n === 0) return { nodes: [], maxDepth: 0, animStartTime: performance.now() };

    // Compute depth of each node via BFS from root (node 0)
    const depth = new Int32Array(n).fill(-1);
    depth[0] = 0;
    const queue = [0];
    let maxDepth = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const nd = queue[qi];
      const d = depth[nd];
      if (d > maxDepth) maxDepth = d;
      const l = childrenLeft[nd];
      const r = childrenRight[nd];
      if (l !== -1 && l < n) { depth[l] = d + 1; queue.push(l); }
      if (r !== -1 && r < n) { depth[r] = d + 1; queue.push(r); }
    }

    // Compute subtree widths for x positioning
    const subtreeWidth = new Float64Array(n).fill(1);
    // Process in reverse BFS order (leaves first)
    for (let i = queue.length - 1; i >= 0; i--) {
      const nd = queue[i];
      const l = childrenLeft[nd];
      const r = childrenRight[nd];
      let w = 0;
      if (l !== -1 && l < n) w += subtreeWidth[l];
      if (r !== -1 && r < n) w += subtreeWidth[r];
      if (w > 0) subtreeWidth[nd] = w;
    }

    // Assign x positions
    const padX = 4;
    const padTop = 6;
    const padBottom = 4;
    const usableW = cellW - padX * 2;
    const usableH = cellH - padTop - padBottom;
    const levelH = maxDepth > 0 ? usableH / maxDepth : 0;

    const nodeX = new Float64Array(n);
    const nodeY = new Float64Array(n);

    // Recursive x assignment
    function assignX(nd: number, leftBound: number, rightBound: number) {
      const midX = (leftBound + rightBound) / 2;
      nodeX[nd] = midX;
      nodeY[nd] = padTop + depth[nd] * levelH;

      const l = childrenLeft[nd];
      const r = childrenRight[nd];
      if (l !== -1 && l < n && r !== -1 && r < n) {
        const totalW = subtreeWidth[l] + subtreeWidth[r];
        const splitX = leftBound + (subtreeWidth[l] / totalW) * (rightBound - leftBound);
        assignX(l, leftBound, splitX);
        assignX(r, splitX, rightBound);
      } else if (l !== -1 && l < n) {
        assignX(l, leftBound, rightBound);
      } else if (r !== -1 && r < n) {
        assignX(r, leftBound, rightBound);
      }
    }

    assignX(0, padX, padX + usableW);

    // Build node array
    const nodes: TreeNode[] = [];
    for (let i = 0; i < queue.length; i++) {
      const nd = queue[i];
      const l = childrenLeft[nd];
      const r = childrenRight[nd];
      const isLeaf = (l === -1 || l >= n) && (r === -1 || r >= n);
      // Find parent
      let parentId: number | null = null;
      if (nd !== 0) {
        // Search for parent (the node whose left or right child is nd)
        for (let j = 0; j < n; j++) {
          if (childrenLeft[j] === nd || childrenRight[j] === nd) {
            parentId = j;
            break;
          }
        }
      }
      nodes.push({
        id: nd,
        x: nodeX[nd],
        y: nodeY[nd],
        depth: depth[nd],
        isLeaf,
        parentId,
      });
    }

    return { nodes, maxDepth, animStartTime: performance.now() };
  }

  function drawTree(tree: TreeData, ox: number, oy: number, now: number) {
    const elapsed = now - tree.animStartTime;
    const levelDelay = 100; // ms per level

    // Build a map from id to node for fast parent lookups
    const nodeMap = new Map<number, TreeNode>();
    for (const node of tree.nodes) {
      nodeMap.set(node.id, node);
    }

    for (const node of tree.nodes) {
      const levelTime = elapsed - node.depth * levelDelay;
      if (levelTime < 0) continue;
      const alpha = Math.min(1, levelTime / 200); // 200ms fade in

      // Draw edge from parent
      if (node.parentId !== null) {
        const parent = nodeMap.get(node.parentId);
        if (parent) {
          const parentLevelTime = elapsed - parent.depth * levelDelay;
          if (parentLevelTime > 0) {
            const edgeAlpha = alpha * 0.6;
            ctx.strokeStyle = `rgba(0, 240, 255, ${edgeAlpha})`;
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            // Animate edge drawing from parent to child
            const edgeProgress = Math.min(1, levelTime / 150);
            const fromX = ox + parent.x;
            const fromY = oy + parent.y;
            const toX = ox + lerp(parent.x, node.x, edgeProgress);
            const toY = oy + lerp(parent.y, node.y, edgeProgress);
            ctx.moveTo(fromX, fromY);
            ctx.lineTo(toX, toY);
            ctx.stroke();
          }
        }
      }

      // Draw node
      const nx = ox + node.x;
      const ny = oy + node.y;
      const radius = node.isLeaf ? 2 : 2.5;
      const brightness = node.isLeaf ? 1.0 : 0.7;
      ctx.fillStyle = `rgba(0, 240, 255, ${alpha * brightness})`;
      ctx.beginPath();
      ctx.arc(nx, ny, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function render() {
    const w = container.clientWidth;
    const h = container.clientHeight;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    // Draw background grid
    drawGrid(ctx, w, h, 30);

    const forestH = h * 0.7;
    const bottomH = h * 0.3;
    const bottomY = forestH;

    const now = performance.now();

    // --- Forest area ---
    // Title
    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('FOREST GENESIS', 12, 18);

    // Separator line
    ctx.strokeStyle = COLORS.gridBright;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, forestH);
    ctx.lineTo(w, forestH);
    ctx.stroke();

    // Tree grid
    const cellW = Math.floor(w / GRID_COLS);
    const cellH = Math.floor((forestH - 26) / GRID_ROWS);

    for (let i = 0; i < Math.min(trees.length, MAX_RENDERED); i++) {
      const col = i % GRID_COLS;
      const row = Math.floor(i / GRID_COLS);
      const ox = col * cellW;
      const oy = 26 + row * cellH;

      // Draw subtle cell border
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 0.5;
      ctx.strokeRect(ox, oy, cellW, cellH);

      // Draw tree
      drawTree(trees[i], ox, oy, now);
    }

    // Draw empty cells as dim dots for unfilled slots
    for (let i = trees.length; i < MAX_RENDERED; i++) {
      const col = i % GRID_COLS;
      const row = Math.floor(i / GRID_COLS);
      const cx = col * cellW + cellW / 2;
      const cy = 26 + row * cellH + cellH / 2;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // --- Bottom Left: Feature Importance ---
    const importW = w * 0.6;

    ctx.fillStyle = COLORS.muted;
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('FEATURE IMPORTANCE (top 10)', 12, bottomY + 16);

    // Sort features by importance, take top 10
    const indexed = Array.from(importanceCurrent).map((v, i) => ({ name: FEATURE_NAMES[i], value: v }));
    indexed.sort((a, b) => b.value - a.value);
    const top10 = indexed.slice(0, 10);
    const maxImp = Math.max(0.001, ...top10.map(f => f.value));

    const barStartX = 120;
    const barMaxW = importW - barStartX - 20;
    const barH = 12;
    const barGap = 3;
    const barY0 = bottomY + 26;

    for (let i = 0; i < top10.length; i++) {
      const y = barY0 + i * (barH + barGap);
      const feat = top10[i];

      // Feature name
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(feat.name, barStartX - 6, y + barH - 2);

      // Bar background
      ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.fillRect(barStartX, y, barMaxW, barH);

      // Bar fill
      const barW = (feat.value / maxImp) * barMaxW;
      const gradient = ctx.createLinearGradient(barStartX, 0, barStartX + barW, 0);
      gradient.addColorStop(0, COLORS.accentDim);
      gradient.addColorStop(1, COLORS.accent);
      ctx.fillStyle = gradient;
      ctx.fillRect(barStartX, y, barW, barH);

      // Value label
      ctx.fillStyle = COLORS.muted;
      ctx.font = '9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(feat.value.toFixed(3), barStartX + barW + 4, y + barH - 2);
    }

    // --- Bottom Right: Progress ---
    const progX = importW;
    const progW = w - importW;
    const progCenterX = progX + progW / 2;
    const progCenterY = bottomY + bottomH / 2;

    // Circular progress ring
    const ringRadius = Math.min(progW, bottomH) * 0.25;
    const progress = Math.min(1, treeCount / TOTAL_TREES);
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + progress * Math.PI * 2;

    // Background ring
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(progCenterX, progCenterY - 8, ringRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Progress ring
    if (progress > 0) {
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.arc(progCenterX, progCenterY - 8, ringRadius, startAngle, endAngle);
      ctx.stroke();
      ctx.lineCap = 'butt';

      // Glow effect
      ctx.shadowColor = COLORS.accent;
      ctx.shadowBlur = 8;
      ctx.strokeStyle = COLORS.accentDim;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(progCenterX, progCenterY - 8, ringRadius, startAngle, endAngle);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Counter text inside ring
    ctx.fillStyle = COLORS.text;
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${treeCount}`, progCenterX, progCenterY - 12);
    ctx.font = '10px monospace';
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(`/ ${TOTAL_TREES}`, progCenterX, progCenterY + 4);

    // Model label
    ctx.fillStyle = COLORS.accent;
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(modelLabel, progCenterX, progCenterY + ringRadius + 14);

    // CV accuracy
    if (cvAccuracy !== null) {
      ctx.fillStyle = COLORS.green;
      ctx.font = '11px monospace';
      ctx.fillText(`CV: ${cvAccuracy.toFixed(3)}`, progCenterX, progCenterY + ringRadius + 28);
    }

    ctx.textBaseline = 'alphabetic';

    // Status text
    if (statusText) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = '9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(statusText, 12, h - 6);
    }

    // Lerp importance values toward target
    for (let i = 0; i < FEATURE_NAMES.length; i++) {
      importanceCurrent[i] = lerp(importanceCurrent[i], importanceTarget[i], 0.08);
    }

    animId = requestAnimationFrame(render);
  }

  return {
    mount(el: HTMLElement) {
      container = el;
      const result = createCanvas(container);
      canvas = result.canvas;
      ctx = result.ctx;
    },

    onStart() {
      trees.length = 0;
      treeCount = 0;
      cvAccuracy = null;
      importanceCurrent.fill(0);
      importanceTarget.fill(0);
      statusText = '';
      modelLabel = 'DIRTY MODEL';
      animId = requestAnimationFrame(render);
    },

    onData(msg: SSEMessage) {
      if (msg.type === 'tree') {
        const childrenLeft = msg.children_left as number[];
        const childrenRight = msg.children_right as number[];
        const model = msg.model as string;

        modelLabel = model === 'clean' ? 'CLEAN MODEL' : 'DIRTY MODEL';

        const idx = msg.idx as number;
        treeCount = Math.max(treeCount, idx + 1);

        // Only render trees in the visible grid
        if (trees.length < MAX_RENDERED && childrenLeft && childrenRight) {
          const cellW = Math.floor(container.clientWidth / GRID_COLS);
          const forestH = container.clientHeight * 0.7;
          const cellH = Math.floor((forestH - 26) / GRID_ROWS);
          const treeData = buildTreeLayout(childrenLeft, childrenRight, cellW, cellH);
          trees.push(treeData);
        }
      } else if (msg.type === 'importance') {
        const features = msg.features as number[];
        if (features) {
          for (let i = 0; i < Math.min(features.length, FEATURE_NAMES.length); i++) {
            importanceTarget[i] = features[i];
          }
        }
        const model = msg.model as string;
        if (model) {
          modelLabel = model === 'clean' ? 'CLEAN MODEL' : 'DIRTY MODEL';
        }
      } else if (msg.type === 'tree_count') {
        treeCount = msg.count as number;
        const model = msg.model as string;
        if (model) {
          modelLabel = model === 'clean' ? 'CLEAN MODEL' : 'DIRTY MODEL';
        }
      } else if (msg.type === 'cv_accuracy') {
        cvAccuracy = msg.accuracy as number;
      }
    },

    onText(text: string) {
      statusText = text.trim().slice(0, 100);
    },

    onComplete() {
      statusText = 'Training complete';
      // Snap importance to final values
      for (let i = 0; i < FEATURE_NAMES.length; i++) {
        importanceCurrent[i] = importanceTarget[i];
      }
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
