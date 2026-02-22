import './style.css';
import { initScene } from './scene';
import { initChain } from './chainSnake';
import { startGameStream } from './gameStream';
import { initFnirsHeatmapPage } from './pages/fnirsHeatmap';

const isFnirsHeatmapPage = window.location.pathname.startsWith('/fnirs-heatmap');

if (isFnirsHeatmapPage) {
  initFnirsHeatmapPage();
} else {
  // 3D neuron background
  const bgCanvas = document.getElementById('bg-canvas') as HTMLCanvasElement;
  initScene(bgCanvas);

  // Minimal UI
  const app = document.getElementById('app')!;
  app.innerHTML = `
    <div class="header" id="header">
      <h1>NEUROLABEL</h1>
      <p>brain-filtered ai training</p>
    </div>
    <button class="start-btn" id="start-btn">Start</button>
  `;

  // Chain canvas
  const chainCanvas = document.createElement('canvas');
  chainCanvas.id = 'chain-canvas';
  document.body.appendChild(chainCanvas);

  const chain = initChain(chainCanvas);

  // Start button → launch game session
  const startBtn = document.getElementById('start-btn')!;
  startBtn.addEventListener('click', () => {
    // Hide button
    startBtn.style.opacity = '0';
    startBtn.style.pointerEvents = 'none';

    // Fire first chain node (Collect) — no setActivity, neurons stay idle
    chain.start();

    // Blur background
    bgCanvas.classList.add('blurred');

    // Fade out header
    document.getElementById('header')!.classList.add('fade-out');

    // Create game container
    const gameContainer = document.createElement('div');
    gameContainer.className = 'game-container';

    const gameCanvas = document.createElement('canvas');
    gameCanvas.width = 800;
    gameCanvas.height = 600;
    gameContainer.appendChild(gameCanvas);

    // Timer overlay
    const timerEl = document.createElement('div');
    timerEl.className = 'game-timer';
    timerEl.textContent = '';
    gameContainer.appendChild(timerEl);

    // Loading overlay
    const loadingEl = document.createElement('div');
    loadingEl.className = 'game-loading';
    loadingEl.textContent = 'Starting MetaDrive...';
    gameContainer.appendChild(loadingEl);

    document.body.appendChild(gameContainer);

    // Fade in game
    requestAnimationFrame(() => {
      gameContainer.classList.add('visible');
    });

    // Start game stream
    const stream = startGameStream(
      gameCanvas,
      // onEnd: game session finished
      () => {
        // Fade out game
        gameContainer.classList.remove('visible');
        setTimeout(() => {
          gameContainer.remove();
        }, 600);

        // Advance chain: Process → Train → Simulate → Compare
        chain.activateNext(); // → Process
        runPipeline(chain);
      },
      // onTick: update countdown timer
      (remaining) => {
        timerEl.textContent = `${remaining}s`;
        loadingEl.style.display = 'none';
      },
      // onLoading: show loading status
      (msg) => {
        loadingEl.textContent = msg;
      },
    );

    // Expose stop for debugging
    (window as any).__gameStream = stream;
  });
}

async function runPipeline(chain: { activateNext(): void }) {
  try {
    const res = await fetch('/api/run/demo', { method: 'POST' });
    if (!res.body) return;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      const lines = text.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === '"[DONE]"') continue;

        try {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object' && parsed.type === 'step_marker') {
            chain.activateNext();
          }
        } catch {
          // text line, ignore
        }
      }
    }

    // Pipeline finished → activate last node (Compare)
    chain.activateNext();
  } catch {
    // pipeline error, silent
  }
}
