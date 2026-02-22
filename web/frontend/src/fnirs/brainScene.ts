import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import type {
  DisplayBrainMappingPayload,
  HeatmapMappingPayload,
  HeatmapMeshPayload,
  HeatmapWindowPayload,
} from './types';

const DEFAULT_NEUTRAL = new THREE.Color(0x4a6385);
const MID_NEUTRAL = new THREE.Color(0x6688b6);
const COLD_COLOR = new THREE.Color(0x2f78ff);
const HOT_COLOR = new THREE.Color(0xff6d52);
const GHOST_TINT = new THREE.Color(0xcde6ff);
const HEAT_SENSITIVITY_GAIN = 1.9;
const HEAT_GAMMA = 0.58;
const HEAT_DEAD_ZONE = 0.03;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function divergingColor(value: number, maxAbs: number): THREE.Color {
  const scaled = clamp((value / Math.max(0.01, maxAbs)) * HEAT_SENSITIVITY_GAIN, -1, 1);
  const mag = Math.abs(scaled);
  const deZoned = mag <= HEAT_DEAD_ZONE ? 0 : (mag - HEAT_DEAD_ZONE) / (1 - HEAT_DEAD_ZONE);
  const v = Math.sign(scaled) * clamp(deZoned, 0, 1);
  const t = Math.pow(Math.abs(v), HEAT_GAMMA);
  return MID_NEUTRAL.clone().lerp(v < 0 ? COLD_COLOR : HOT_COLOR, t);
}

function cloneWindow(window: HeatmapWindowPayload | null): HeatmapWindowPayload | null {
  return window ? { ...window } : null;
}

function lerpMaybe(a: number | undefined, b: number | undefined, t: number): number | undefined {
  if (a === undefined && b === undefined) return undefined;
  if (a === undefined) return b;
  if (b === undefined) return a;
  return lerp(a, b, t);
}

function lerpWindow(
  current: HeatmapWindowPayload | null,
  target: HeatmapWindowPayload | null,
  t: number
): HeatmapWindowPayload | null {
  if (!target) return null;
  if (!current) return cloneWindow(target);
  return {
    ...target,
    sec: lerp(current.sec, target.sec, t),
    timestamp: lerp(current.timestamp, target.timestamp, t),
    left_raw_score: lerp(current.left_raw_score, target.left_raw_score, t),
    right_raw_score: lerp(current.right_raw_score, target.right_raw_score, t),
    pulse_quality: lerp(current.pulse_quality, target.pulse_quality, t),
    n_samples: lerp(current.n_samples, target.n_samples, t),
    left_red_z: lerpMaybe(current.left_red_z, target.left_red_z, t),
    left_ir_z: lerpMaybe(current.left_ir_z, target.left_ir_z, t),
    left_amb_z: lerpMaybe(current.left_amb_z, target.left_amb_z, t),
    right_red_z: lerpMaybe(current.right_red_z, target.right_red_z, t),
    right_ir_z: lerpMaybe(current.right_ir_z, target.right_ir_z, t),
    right_amb_z: lerpMaybe(current.right_amb_z, target.right_amb_z, t),
  };
}

export class FnirsBrainScene {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private clock = new THREE.Clock();
  private elapsed = 0;
  private rafId = 0;
  private root = new THREE.Group();
  private mesh: THREE.Mesh | null = null;
  private ghostMeshes: THREE.Mesh[] = [];
  private anchorMarkers: THREE.Mesh[] = [];
  private meshPayload: HeatmapMeshPayload | null = null;
  private mappingPayload: HeatmapMappingPayload | null = null;
  private currentScaleMaxAbs = 2.0;
  private targetScaleMaxAbs = 2.0;
  private displayScaleMaxAbs = 2.0;
  private targetCurrentWindow: HeatmapWindowPayload | null = null;
  private displayCurrentWindow: HeatmapWindowPayload | null = null;
  private targetGhostWindows: Array<HeatmapWindowPayload | null> = [null, null, null];
  private displayGhostWindows: Array<HeatmapWindowPayload | null> = [null, null, null];
  private assetModeActive = false;
  private gltfLoader = new GLTFLoader();
  private stlLoader = new STLLoader();
  private baseRotX = 0.35;
  private baseRotY = 1.28;
  private baseRotZ = 0.55;
  private autoRotateEnabled = true;
  private userYaw = 0;
  private userYawTarget = 0;
  private draggingRotate = false;
  private dragStartX = 0;
  private dragStartYaw = 0;
  private interactResumeAt = 0;

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x000000, 0);
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = null;
    this.scene.add(this.root);
    this.container.style.touchAction = 'none';

    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    this.camera.position.set(4.05, 2.45, 2.0);
    this.camera.lookAt(0, -0.01, -0.03);
    this.applyDebugOverridesFromUrl();

    const key = new THREE.DirectionalLight(0xffffff, 1.26);
    key.position.set(2.9, 3.2, 2.0);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xc0dcff, 0.98);
    fill.position.set(-2.7, 1.45, 2.25);
    this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xe4efff, 0.72);
    rim.position.set(-2.0, -1.15, -2.45);
    this.scene.add(rim);
    const hemi = new THREE.HemisphereLight(0xe3f1ff, 0x2b3646, 1.02);
    this.scene.add(hemi);

    window.addEventListener('resize', this.onResize);
    this.container.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('pointercancel', this.onPointerUp);
    this.onResize();
    this.animate();
  }

  dispose() {
    cancelAnimationFrame(this.rafId);
    window.removeEventListener('resize', this.onResize);
    this.container.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('pointercancel', this.onPointerUp);
    this.renderer.dispose();
    this.container.innerHTML = '';
  }

  async loadDisplayBrainAsset(assetUrl: string, mappingUrl: string): Promise<boolean> {
    const mappingRes = await fetch(mappingUrl, { cache: 'no-cache' });
    if (!mappingRes.ok) {
      throw new Error(`Failed to load brain mapping (${mappingRes.status})`);
    }
    const mappingJson = (await mappingRes.json()) as DisplayBrainMappingPayload;
    const mappingPayload: HeatmapMappingPayload = {
      roi_mask: mappingJson.roi_mask,
      left_weights: mappingJson.left_weights,
      right_weights: mappingJson.right_weights,
      anchors: mappingJson.anchors ?? {},
    };

    let geometry: THREE.BufferGeometry | null = null;
    try {
      geometry = await this.loadGltfGeometry(assetUrl);
    } catch (err) {
      const stlUrl = assetUrl.replace(/\.glb(\?.*)?$/i, '.stl$1');
      geometry = await this.loadStlGeometry(stlUrl).catch(() => {
        throw err;
      });
    }

    const pos = geometry.getAttribute('position');
    if (!pos || pos.itemSize !== 3) {
      throw new Error('Brain asset is missing a valid position attribute');
    }
    if (typeof mappingJson.vertex_count === 'number' && mappingJson.vertex_count !== pos.count) {
      throw new Error(`Brain asset vertex count mismatch (asset=${pos.count}, mapping=${mappingJson.vertex_count})`);
    }

    this.assetModeActive = true;
    this.meshPayload = null;
    this.configureSceneGeometry(geometry, mappingPayload);
    return true;
  }

  setMesh(meshPayload: HeatmapMeshPayload, mappingPayload: HeatmapMappingPayload) {
    if (this.assetModeActive) {
      return;
    }
    this.meshPayload = meshPayload;
    const vertices = new Float32Array(meshPayload.vertices.flat());
    const indices = new Uint32Array(meshPayload.faces.flat());
    const baseGeometry = new THREE.BufferGeometry();
    baseGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    baseGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.configureSceneGeometry(baseGeometry, mappingPayload);
  }

  setFrame(current: HeatmapWindowPayload | null, ghosts: HeatmapWindowPayload[] = [], maxAbs = 2.0) {
    this.targetScaleMaxAbs = Math.max(0.35, maxAbs);
    this.targetCurrentWindow = cloneWindow(current);
    this.targetGhostWindows = [ghosts[0] ?? null, ghosts[1] ?? null, ghosts[2] ?? null].map((w) => cloneWindow(w));

    if (this.targetCurrentWindow && !this.displayCurrentWindow) {
      this.displayCurrentWindow = cloneWindow(this.targetCurrentWindow);
    }
    for (let i = 0; i < 3; i++) {
      if (this.targetGhostWindows[i] && !this.displayGhostWindows[i]) {
        this.displayGhostWindows[i] = cloneWindow(this.targetGhostWindows[i]);
      }
    }
  }

  private async loadGltfGeometry(assetUrl: string): Promise<THREE.BufferGeometry> {
    const gltf = await new Promise<unknown>((resolve, reject) => {
      this.gltfLoader.load(
        assetUrl,
        (v) => resolve(v),
        undefined,
        (e) => reject(e || new Error('GLB load failed'))
      );
    }) as { scene: THREE.Object3D };

    let chosenGeometry: THREE.BufferGeometry | null = null;
    let bestCount = -1;
    gltf.scene.traverse((obj) => {
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        const geo = mesh.geometry as THREE.BufferGeometry;
        const pos = geo.getAttribute('position');
        const count = pos ? pos.count : 0;
        if (count > bestCount) {
          bestCount = count;
          chosenGeometry = geo;
        }
      }
    });
    const selectedGeometry = chosenGeometry as THREE.BufferGeometry | null;
    if (!selectedGeometry) throw new Error('GLB contains no mesh geometry');
    return this.prepareGeometry(selectedGeometry.clone());
  }

  private async loadStlGeometry(assetUrl: string): Promise<THREE.BufferGeometry> {
    const geo = await new Promise<THREE.BufferGeometry>((resolve, reject) => {
      this.stlLoader.load(
        assetUrl,
        (g) => resolve(g),
        undefined,
        (e) => reject(e || new Error('STL load failed'))
      );
    });
    return this.prepareGeometry(geo.clone());
  }

  private prepareGeometry(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
    let g = geometry;
    const pos = g.getAttribute('position');
    if (!pos || pos.itemSize !== 3) {
      throw new Error('Geometry has no valid position attribute');
    }
    if (!g.index) {
      const idx = new Uint32Array(pos.count);
      for (let i = 0; i < pos.count; i++) idx[i] = i;
      g.setIndex(new THREE.BufferAttribute(idx, 1));
    }
    g.computeVertexNormals();
    return g;
  }

  private configureSceneGeometry(baseGeometryInput: THREE.BufferGeometry, mappingPayload: HeatmapMappingPayload) {
    this.mappingPayload = mappingPayload;
    this.root.clear();
    this.mesh = null;
    this.ghostMeshes = [];
    this.anchorMarkers = [];

    const baseGeometry = this.prepareGeometry(baseGeometryInput.clone());
    const pos = baseGeometry.getAttribute('position');
    const vertexCount = pos.count;
    baseGeometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3));

    const baseMat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      shininess: 12,
      specular: new THREE.Color(0x121a26),
      transparent: true,
      opacity: 0.98,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(baseGeometry, baseMat);
    this.root.add(this.mesh);

    const shell = new THREE.Mesh(
      baseGeometry.clone(),
      new THREE.MeshBasicMaterial({ color: 0x3a5276, transparent: true, opacity: 0.04, side: THREE.BackSide })
    );
    shell.scale.setScalar(1.006);
    this.root.add(shell);

    for (let i = 0; i < 3; i++) {
      const g = baseGeometry.clone();
      g.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3));
      const m = new THREE.MeshLambertMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.0,
        side: THREE.FrontSide,
        depthWrite: false,
      });
      const ghost = new THREE.Mesh(g, m);
      ghost.scale.setScalar(1.0015 + i * 0.002);
      this.ghostMeshes.push(ghost);
      this.root.add(ghost);
    }

    this.root.scale.setScalar(1.24);
    this.root.position.set(0.0, -0.02, -0.12);
    this.root.rotation.set(this.baseRotX, this.baseRotY + this.userYaw, this.baseRotZ);

    this.displayCurrentWindow = null;
    this.targetCurrentWindow = null;
    this.displayGhostWindows = [null, null, null];
    this.targetGhostWindows = [null, null, null];
    this.applyWindowToMesh(this.mesh, null, 0.98);
    this.ghostMeshes.forEach((g) => this.applyWindowToMesh(g, null, 0.0, true));
  }

  private applyWindowToMesh(target: THREE.Mesh | null, window: HeatmapWindowPayload | null, opacity: number, ghost = false) {
    if (!target || !this.mappingPayload) return;
    const mat = target.material as THREE.Material & { opacity?: number };
    if (!window) {
      if (typeof mat.opacity === 'number') mat.opacity = ghost ? 0 : 0.98;
      if (!ghost) {
        const geometry = target.geometry as THREE.BufferGeometry;
        const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
        const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
        const n = this.mappingPayload.left_weights.length;
        const roiMask = this.mappingPayload.roi_mask;
        for (let i = 0; i < n; i++) {
          const x = posAttr.getX(i);
          const y = posAttr.getY(i);
          const z = posAttr.getZ(i);
          const foldShade = 0.035 * Math.sin(7.4 * x + 5.9 * z) + 0.028 * Math.sin(10.2 * y - 6.3 * x);
          const c = DEFAULT_NEUTRAL.clone();
          if (roiMask[i]) c.lerp(MID_NEUTRAL, 0.22);
          c.offsetHSL(0, 0, foldShade);
          colorAttr.setXYZ(i, c.r, c.g, c.b);
        }
        colorAttr.needsUpdate = true;
      }
      return;
    }
    if (typeof mat.opacity === 'number') mat.opacity = opacity;

    const geometry = target.geometry as THREE.BufferGeometry;
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    const n = this.mappingPayload.left_weights.length;
    const roiMask = this.mappingPayload.roi_mask;
    const lw = this.mappingPayload.left_weights;
    const rw = this.mappingPayload.right_weights;
    const left = window.left_raw_score;
    const right = window.right_raw_score;
    const leftRedZ = window.left_red_z ?? left;
    const leftIrZ = window.left_ir_z ?? left;
    const leftAmbZ = window.left_amb_z ?? 0;
    const rightRedZ = window.right_red_z ?? right;
    const rightIrZ = window.right_ir_z ?? right;
    const rightAmbZ = window.right_amb_z ?? 0;
    const leftBalance = clamp((leftRedZ - leftIrZ) * 0.35, -1.5, 1.5);
    const rightBalance = clamp((rightRedZ - rightIrZ) * 0.35, -1.5, 1.5);
    const leftAmbientTension = clamp(leftAmbZ * 0.18, -1.0, 1.0);
    const rightAmbientTension = clamp(rightAmbZ * 0.18, -1.0, 1.0);

    for (let i = 0; i < n; i++) {
      if (!roiMask[i]) {
        colorAttr.setXYZ(i, DEFAULT_NEUTRAL.r, DEFAULT_NEUTRAL.g, DEFAULT_NEUTRAL.b);
        continue;
      }
      const x = posAttr.getX(i);
      const y = posAttr.getY(i);
      const z = posAttr.getZ(i);
      const lDetail =
        lw[i] * (
          0.22 * leftBalance * (0.65 * y + 0.35 * z)
          + 0.12 * leftAmbientTension * Math.sin(8.0 * x + 5.7 * z + 2.3 * y)
        );
      const rDetail =
        rw[i] * (
          0.22 * rightBalance * (0.65 * y + 0.35 * z)
          + 0.12 * rightAmbientTension * Math.sin(-8.2 * x + 5.4 * z + 2.1 * y)
        );
      const value = lw[i] * left + rw[i] * right + lDetail + rDetail;
      const c = divergingColor(value, this.currentScaleMaxAbs);
      const strength = Math.pow(clamp(Math.abs(value) / Math.max(0.01, this.currentScaleMaxAbs), 0, 1), 0.52);
      c.lerp(DEFAULT_NEUTRAL, 0.06 + 0.18 * (1 - strength));
      if (ghost) c.lerp(GHOST_TINT, 0.25);
      colorAttr.setXYZ(i, c.r, c.g, c.b);
    }
    colorAttr.needsUpdate = true;
  }

  private onResize = () => {
    const width = Math.max(280, this.container.clientWidth);
    const height = Math.max(280, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  };

  private onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0) return;
    this.draggingRotate = true;
    this.dragStartX = ev.clientX;
    this.dragStartYaw = this.userYawTarget;
    this.interactResumeAt = performance.now() + 3200;
    this.container.classList.add('fnirs-dragging');
    try {
      this.container.setPointerCapture(ev.pointerId);
    } catch {
      // ignore unsupported capture contexts
    }
  };

  private onPointerMove = (ev: PointerEvent) => {
    if (!this.draggingRotate) return;
    const dx = ev.clientX - this.dragStartX;
    const yawDelta = dx * 0.0048;
    this.userYawTarget = clamp(this.dragStartYaw + yawDelta, -Math.PI, Math.PI);
    this.interactResumeAt = performance.now() + 3200;
  };

  private onPointerUp = (ev: PointerEvent) => {
    if (!this.draggingRotate) return;
    this.draggingRotate = false;
    this.interactResumeAt = performance.now() + 3200;
    this.container.classList.remove('fnirs-dragging');
    try {
      this.container.releasePointerCapture(ev.pointerId);
    } catch {
      // ignore
    }
  };

  private applyDebugOverridesFromUrl() {
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    const getNum = (key: string): number | null => {
      const raw = params.get(key);
      if (raw === null || raw === '') return null;
      const n = Number(raw);
      return Number.isFinite(n) ? n : null;
    };
    const rx = getNum('rx');
    const ry = getNum('ry');
    const rz = getNum('rz');
    if (rx !== null) this.baseRotX = rx;
    if (ry !== null) this.baseRotY = ry;
    if (rz !== null) this.baseRotZ = rz;
    const auto = params.get('auto');
    if (auto === '0' || auto === 'false') this.autoRotateEnabled = false;
  }

  private animate = () => {
    this.rafId = requestAnimationFrame(this.animate);
    const dt = Math.min(this.clock.getDelta(), 0.1);
    this.elapsed += dt;
    const t = this.elapsed;

    const smoothFast = 1 - Math.exp(-dt * 10.5);
    const smoothScale = 1 - Math.exp(-dt * 6.5);
    const smoothYaw = 1 - Math.exp(-dt * 12.0);
    this.displayScaleMaxAbs = lerp(this.displayScaleMaxAbs, this.targetScaleMaxAbs, smoothScale);
    this.currentScaleMaxAbs = this.displayScaleMaxAbs;
    this.userYaw = lerp(this.userYaw, this.userYawTarget, smoothYaw);

    this.displayCurrentWindow = lerpWindow(this.displayCurrentWindow, this.targetCurrentWindow, smoothFast);
    for (let i = 0; i < 3; i++) {
      this.displayGhostWindows[i] = lerpWindow(this.displayGhostWindows[i], this.targetGhostWindows[i], smoothFast);
    }

    this.applyWindowToMesh(this.mesh, this.displayCurrentWindow, 0.98);
    this.ghostMeshes.forEach((g, idx) => {
      const ghost = this.displayGhostWindows[idx] ?? null;
      this.applyWindowToMesh(g, ghost, 0.0, true);
    });

    const autoEnabled = this.autoRotateEnabled && performance.now() > this.interactResumeAt;
    const autoX = autoEnabled ? 0.008 * Math.sin(t * 0.15) : 0;
    const autoY = autoEnabled ? 0.016 * Math.sin(t * 0.18) : 0;
    this.root.rotation.x = this.baseRotX + autoX;
    this.root.rotation.y = this.baseRotY + this.userYaw + autoY;
    this.root.rotation.z = this.baseRotZ;
    this.anchorMarkers.forEach((m, i) => {
      const s = 1 + 0.15 * Math.sin(t * 2.2 + i);
      m.scale.setScalar(s);
    });
    this.renderer.render(this.scene, this.camera);
  };
}
