/**
 * FLIP-based card expansion/collapse animations using the Web Animations API.
 *
 * Expand: card grows to fill the entire viewport, siblings slide offscreen.
 * Collapse: exact inverse — card shrinks back, siblings slide in.
 *
 * All sibling animations use `transform` + `opacity` only (GPU-composited).
 * The target card animates position/size properties (needs layout, but it's
 * the only element doing so — acceptable for one element).
 */

interface SavedStyles {
  position: string;
  top: string;
  left: string;
  width: string;
  height: string;
  margin: string;
  transform: string;
  opacity: string;
}

interface ExpandState {
  stepId: string;
  /** Rects measured before layout was frozen */
  measurements: Map<HTMLElement, DOMRect>;
  appRect: DOMRect;
  /** Original inline styles on #app */
  appOriginalStyles: Record<string, string>;
  /** Original inline styles on every tracked child */
  childOriginalStyles: Map<HTMLElement, SavedStyles>;
  /** Extra elements that were hidden (results container, steps-section, etc.) */
  hiddenElements: HTMLElement[];
  /** Constrained expansion box */
  boxLeft: number;
  boxTop: number;
  boxWidth: number;
  boxHeight: number;
  /** Gradient fade overlay element */
  fadeOverlay: HTMLElement | null;
  /** Index of next sibling that peeks through */
  nextSiblingIdx: number;
}

let state: ExpandState | null = null;

// ─── helpers ──────────────────────────────────────────────────────────

function saveStyles(el: HTMLElement): SavedStyles {
  return {
    position: el.style.position,
    top: el.style.top,
    left: el.style.left,
    width: el.style.width,
    height: el.style.height,
    margin: el.style.margin,
    transform: el.style.transform,
    opacity: el.style.opacity,
  };
}

function restoreStyles(el: HTMLElement, s: SavedStyles) {
  el.style.position = s.position;
  el.style.top = s.top;
  el.style.left = s.left;
  el.style.width = s.width;
  el.style.height = s.height;
  el.style.margin = s.margin;
  el.style.transform = s.transform;
  el.style.opacity = s.opacity;
}

// ─── expand ───────────────────────────────────────────────────────────

export async function expandStep(stepId: string): Promise<void> {
  const app = document.getElementById('app')!;
  const header = app.querySelector('.header') as HTMLElement;
  const configEl = app.querySelector('.config-section') as HTMLElement;
  const stepsSection = app.querySelector('.steps-section') as HTMLElement;
  const resultsContainer = document.getElementById('results-container') as HTMLElement;
  const wrappers = [...app.querySelectorAll('.step-wrapper')] as HTMLElement[];
  const target = app.querySelector(`[data-wrapper="${stepId}"]`) as HTMLElement;

  if (!target) return;

  // ── 1. Measure ──────────────────────────────────────────────────────
  const appRect = app.getBoundingClientRect();
  const measurements = new Map<HTMLElement, DOMRect>();

  // Elements that participate in the animation
  const animatedChildren: HTMLElement[] = [];
  if (header) animatedChildren.push(header);
  if (configEl) animatedChildren.push(configEl);
  animatedChildren.push(...wrappers);

  for (const el of animatedChildren) {
    measurements.set(el, el.getBoundingClientRect());
  }

  // ── 2. Save original styles ─────────────────────────────────────────
  const appOriginalStyles: Record<string, string> = {
    position: app.style.position,
    height: app.style.height,
    width: app.style.width,
    overflow: app.style.overflow,
    padding: app.style.padding,
    maxWidth: app.style.maxWidth,
    margin: app.style.margin,
  };

  const childOriginalStyles = new Map<HTMLElement, SavedStyles>();
  for (const el of animatedChildren) {
    childOriginalStyles.set(el, saveStyles(el));
  }

  // ── 3. Hide non-animated siblings ───────────────────────────────────
  const hiddenElements: HTMLElement[] = [];
  if (resultsContainer) {
    resultsContainer.style.display = 'none';
    hiddenElements.push(resultsContainer);
  }

  // ── 4. Lock #app to fill the viewport ───────────────────────────────
  // We need the app to become viewport-sized so the card can fill it.
  // Use fixed positioning so it truly covers the screen edge-to-edge.
  app.style.position = 'fixed';
  app.style.top = '0';
  app.style.left = '0';
  app.style.width = '100vw';
  app.style.height = '100vh';
  app.style.overflow = 'hidden';
  app.style.padding = '0';
  app.style.maxWidth = 'none';
  app.style.margin = '0';

  // ── 5. Freeze children to absolute at measured positions ────────────
  // Positions are relative to the original appRect (since app was centered).
  // Now app is at (0,0), so we offset by the original app position.
  for (const el of animatedChildren) {
    const rect = measurements.get(el)!;
    el.style.position = 'absolute';
    el.style.top = `${rect.top}px`;
    el.style.left = `${rect.left}px`;
    el.style.width = `${rect.width}px`;
    el.style.height = `${rect.height}px`;
    el.style.margin = '0';
  }

  // Hide stepsSection's own layout (gap, flex) since children are absolute
  if (stepsSection) {
    stepsSection.style.display = 'contents';
  }

  // ── 6. Animate ──────────────────────────────────────────────────────
  const targetIdx = wrappers.indexOf(target);
  const targetRect = measurements.get(target)!;

  const animations: Animation[] = [];
  const duration = 500;
  const easing = 'cubic-bezier(0.33, 1, 0.68, 1)';

  // Constrained expansion box
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const firstWrapperRect = measurements.get(wrappers[0])!;
  const boxLeft = 0.3 * vw;
  const boxTop = firstWrapperRect.top;
  const boxWidth = 0.4 * vw;
  const boxHeight = 0.9 * vh - boxTop;
  const nextSiblingIdx = targetIdx + 1;

  // Target: from measured position → constrained box
  animations.push(
    target.animate(
      [
        {
          top: `${targetRect.top}px`,
          left: `${targetRect.left}px`,
          width: `${targetRect.width}px`,
          height: `${targetRect.height}px`,
        },
        {
          top: `${boxTop}px`,
          left: `${boxLeft}px`,
          width: `${boxWidth}px`,
          height: `${boxHeight}px`,
        },
      ],
      { duration, easing, fill: 'forwards' }
    )
  );

  // Header + config → slide up past top edge
  const topEls: HTMLElement[] = [];
  if (header) topEls.push(header);
  if (configEl) topEls.push(configEl);

  for (const el of topEls) {
    const r = measurements.get(el)!;
    animations.push(
      el.animate(
        [
          { transform: 'translateY(0)', opacity: '1' },
          { transform: `translateY(-${r.bottom + 40}px)`, opacity: '0' },
        ],
        { duration, easing, fill: 'forwards' }
      )
    );
  }

  // Wrappers above target → slide up
  for (let i = 0; i < targetIdx; i++) {
    const r = measurements.get(wrappers[i])!;
    animations.push(
      wrappers[i].animate(
        [
          { transform: 'translateY(0)', opacity: '1' },
          { transform: `translateY(-${r.bottom + 40}px)`, opacity: '0' },
        ],
        { duration, easing, fill: 'forwards' }
      )
    );
  }

  // Next sibling below target → slide to peek position (top at 0.9vh)
  if (nextSiblingIdx < wrappers.length) {
    const r = measurements.get(wrappers[nextSiblingIdx])!;
    const peekY = 0.9 * vh - r.top;
    animations.push(
      wrappers[nextSiblingIdx].animate(
        [
          { transform: 'translateY(0)', opacity: '1' },
          { transform: `translateY(${peekY}px)`, opacity: '0.6' },
        ],
        { duration, easing, fill: 'forwards' }
      )
    );
  }

  // Remaining wrappers below → slide fully offscreen
  for (let i = targetIdx + 2; i < wrappers.length; i++) {
    const r = measurements.get(wrappers[i])!;
    animations.push(
      wrappers[i].animate(
        [
          { transform: 'translateY(0)', opacity: '1' },
          { transform: `translateY(${window.innerHeight - r.top + 40}px)`, opacity: '0' },
        ],
        { duration, easing, fill: 'forwards' }
      )
    );
  }

  // Compact content of target → fade out (first half of animation)
  const compact = target.querySelector('.step-compact') as HTMLElement;
  if (compact) {
    animations.push(
      compact.animate([{ opacity: '1' }, { opacity: '0' }], {
        duration: 250,
        fill: 'forwards',
      })
    );
  }

  // ── 7. Wait for all animations ──────────────────────────────────────
  await Promise.all(animations.map((a) => a.finished));

  // ── 8. Cancel animations and snap to final state ────────────────────
  // CRITICAL: cancel first, then set inline styles. fill:forwards overrides
  // inline styles, so we must remove the animations before snapping.
  animations.forEach((a) => a.cancel());

  // Snap target to constrained box
  target.style.top = `${boxTop}px`;
  target.style.left = `${boxLeft}px`;
  target.style.width = `${boxWidth}px`;
  target.style.height = `${boxHeight}px`;

  // Hide compact content
  if (compact) compact.style.opacity = '0';

  // Hide siblings (they're offscreen anyway, but prevent any flicker)
  for (const el of topEls) {
    el.style.opacity = '0';
  }
  for (let i = 0; i < wrappers.length; i++) {
    if (i === targetIdx) continue;
    if (i === nextSiblingIdx && nextSiblingIdx < wrappers.length) {
      // Next sibling peeks — snap to peek position with partial opacity
      const r = measurements.get(wrappers[i])!;
      wrappers[i].style.transform = `translateY(${0.9 * vh - r.top}px)`;
      wrappers[i].style.opacity = '0.6';
    } else {
      wrappers[i].style.opacity = '0';
    }
  }

  // ── 9. Create fade overlay ──────────────────────────────────────────
  const fadeOverlay = document.createElement('div');
  fadeOverlay.className = 'expand-fade-overlay';
  fadeOverlay.style.top = `${0.9 * vh}px`;
  app.appendChild(fadeOverlay);

  // ── 10. Store state for collapse ────────────────────────────────────
  state = {
    stepId,
    measurements,
    appRect,
    appOriginalStyles,
    childOriginalStyles,
    hiddenElements,
    boxLeft,
    boxTop,
    boxWidth,
    boxHeight,
    fadeOverlay,
    nextSiblingIdx,
  };
}

// ─── collapse ─────────────────────────────────────────────────────────

export async function collapseStep(): Promise<void> {
  if (!state) return;

  const {
    measurements, appOriginalStyles, childOriginalStyles, hiddenElements,
    boxLeft, boxTop, boxWidth, boxHeight, nextSiblingIdx: nextIdx, fadeOverlay,
  } = state;

  const app = document.getElementById('app')!;
  const header = app.querySelector('.header') as HTMLElement;
  const configEl = app.querySelector('.config-section') as HTMLElement;
  const stepsSection = app.querySelector('.steps-section') as HTMLElement;
  const wrappers = [...app.querySelectorAll('.step-wrapper')] as HTMLElement[];
  const target = app.querySelector(`[data-wrapper="${state.stepId}"]`) as HTMLElement;

  if (!target) {
    state = null;
    return;
  }

  const targetIdx = wrappers.indexOf(target);
  const targetRect = measurements.get(target)!;

  // ── 1. Set elements to their expanded-end positions ─────────────────
  // Target is at constrained box
  target.style.top = `${boxTop}px`;
  target.style.left = `${boxLeft}px`;
  target.style.width = `${boxWidth}px`;
  target.style.height = `${boxHeight}px`;
  target.style.opacity = '1';

  const compact = target.querySelector('.step-compact') as HTMLElement;
  if (compact) compact.style.opacity = '0';

  // Siblings at their displaced positions
  const topEls: HTMLElement[] = [];
  if (header) topEls.push(header);
  if (configEl) topEls.push(configEl);

  for (const el of topEls) {
    const r = measurements.get(el)!;
    el.style.transform = `translateY(-${r.bottom + 40}px)`;
    el.style.opacity = '0';
  }
  for (let i = 0; i < targetIdx; i++) {
    const r = measurements.get(wrappers[i])!;
    wrappers[i].style.transform = `translateY(-${r.bottom + 40}px)`;
    wrappers[i].style.opacity = '0';
  }
  // Next sibling: at peek position
  if (nextIdx < wrappers.length) {
    const r = measurements.get(wrappers[nextIdx])!;
    wrappers[nextIdx].style.transform = `translateY(${0.9 * window.innerHeight - r.top}px)`;
    wrappers[nextIdx].style.opacity = '0.6';
  }
  // Other below-wrappers: fully offscreen
  for (let i = Math.max(targetIdx + 1, nextIdx + 1); i < wrappers.length; i++) {
    const r = measurements.get(wrappers[i])!;
    wrappers[i].style.transform = `translateY(${window.innerHeight - r.top + 40}px)`;
    wrappers[i].style.opacity = '0';
  }

  // ── 2. Animate reverse ──────────────────────────────────────────────
  const animations: Animation[] = [];
  const duration = 500;
  const easing = 'cubic-bezier(0.33, 1, 0.68, 1)';

  // Target: from constrained box → original rect
  animations.push(
    target.animate(
      [
        {
          top: `${boxTop}px`,
          left: `${boxLeft}px`,
          width: `${boxWidth}px`,
          height: `${boxHeight}px`,
        },
        {
          top: `${targetRect.top}px`,
          left: `${targetRect.left}px`,
          width: `${targetRect.width}px`,
          height: `${targetRect.height}px`,
        },
      ],
      { duration, easing, fill: 'forwards' }
    )
  );

  // Siblings (except next peek): slide back to original positions
  const otherSiblings = [
    ...topEls,
    ...wrappers.filter((_, i) => i !== targetIdx && i !== nextIdx),
  ];

  for (const el of otherSiblings) {
    animations.push(
      el.animate(
        [
          { transform: el.style.transform, opacity: '0' },
          { transform: 'translateY(0)', opacity: '1' },
        ],
        { duration, easing, fill: 'forwards' }
      )
    );
  }

  // Next sibling: from peek position back to original
  if (nextIdx < wrappers.length) {
    const el = wrappers[nextIdx];
    animations.push(
      el.animate(
        [
          { transform: el.style.transform, opacity: '0.6' },
          { transform: 'translateY(0)', opacity: '1' },
        ],
        { duration, easing, fill: 'forwards' }
      )
    );
  }

  // Compact: fade back in
  if (compact) {
    animations.push(
      compact.animate([{ opacity: '0' }, { opacity: '1' }], {
        duration: 300,
        delay: 200,
        easing,
        fill: 'forwards',
      })
    );
  }

  // ── 3. Wait & clean up ──────────────────────────────────────────────
  await Promise.all(animations.map((a) => a.finished));
  animations.forEach((a) => a.cancel());

  // Remove fade overlay
  if (fadeOverlay && fadeOverlay.parentNode) {
    fadeOverlay.parentNode.removeChild(fadeOverlay);
  }

  // Restore compact
  if (compact) compact.style.opacity = '';

  // Restore steps-section
  if (stepsSection) stepsSection.style.display = '';

  // Restore all child styles
  for (const [el, s] of childOriginalStyles) {
    restoreStyles(el, s);
  }

  // Restore app styles
  for (const [key, val] of Object.entries(appOriginalStyles)) {
    (app.style as any)[key] = val;
  }
  // Restore app-specific properties we set that might not be in original
  app.style.top = '';
  app.style.left = '';

  // Show hidden elements
  for (const el of hiddenElements) {
    el.style.display = '';
  }

  state = null;
}

// ─── queries ──────────────────────────────────────────────────────────

export function isExpanded(): boolean {
  return state !== null;
}

export function getExpandedStepId(): string | null {
  return state?.stepId ?? null;
}
