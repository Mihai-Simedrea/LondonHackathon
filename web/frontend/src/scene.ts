import * as THREE from 'three';

interface Particle {
  edgeIdx: number;
  progress: number;
  speed: number;
  mesh: THREE.Mesh;
}

interface Edge {
  from: number;
  to: number;
}

let renderer: THREE.WebGLRenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let nodes: THREE.Mesh[] = [];
let edges: Edge[] = [];
let edgeLines: THREE.LineSegments;
let particles: Particle[] = [];
let nodePositions: THREE.Vector3[] = [];

let currentActivity: 'idle' | 'running' | 'done' = 'idle';
let targetParticleSpeed = 0.002;
let targetNodeOpacity = 0.15;
let targetEdgeOpacity = 0.05;
let pulseTimer = 0;
let jitterAmount = 0;

// Expansion dimming state
let dimFactor = 0;
let targetDimFactor = 0;
let currentStepId: string | undefined;
let orbitTimeOffset = 0;
let lastTime = 0;

const NODE_COUNT = 50;
const PARTICLE_COUNT = 200;
const SPHERE_RADIUS = 5;
const EDGE_DISTANCE = 3;

export function initScene(canvas: HTMLCanvasElement) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 8);

  // Create nodes
  const nodeGeo = new THREE.IcosahedronGeometry(0.08, 1);
  const nodeMat = new THREE.MeshBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.15,
  });

  for (let i = 0; i < NODE_COUNT; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = SPHERE_RADIUS * Math.cbrt(Math.random());
    const pos = new THREE.Vector3(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi)
    );
    nodePositions.push(pos);

    const mesh = new THREE.Mesh(nodeGeo, nodeMat.clone());
    mesh.position.copy(pos);
    scene.add(mesh);
    nodes.push(mesh);
  }

  // Create edges between nearby nodes
  const edgePositions: number[] = [];
  for (let i = 0; i < NODE_COUNT; i++) {
    for (let j = i + 1; j < NODE_COUNT; j++) {
      if (nodePositions[i].distanceTo(nodePositions[j]) < EDGE_DISTANCE) {
        edges.push({ from: i, to: j });
        edgePositions.push(
          nodePositions[i].x, nodePositions[i].y, nodePositions[i].z,
          nodePositions[j].x, nodePositions[j].y, nodePositions[j].z
        );
      }
    }
  }

  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
  const edgeMat = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.05,
  });
  edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
  scene.add(edgeLines);

  // Create particles
  if (edges.length > 0) {
    const particleGeo = new THREE.SphereGeometry(0.025, 4, 4);
    const particleMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.6,
    });

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const mesh = new THREE.Mesh(particleGeo, particleMat.clone());
      const edgeIdx = Math.floor(Math.random() * edges.length);
      const progress = Math.random();
      particles.push({
        edgeIdx,
        progress,
        speed: 0.001 + Math.random() * 0.002,
        mesh,
      });
      scene.add(mesh);
    }
  }

  window.addEventListener('resize', onResize);
  animate(0);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate(time: number) {
  requestAnimationFrame(animate);

  const t = time * 0.001;
  const dt = lastTime > 0 ? t - lastTime : 0.016;
  lastTime = t;

  // Lerp dimFactor toward target
  dimFactor += (targetDimFactor - dimFactor) * 0.05;

  // Smooth camera orbit — slow to near-stop when dimmed
  const baseOrbitSpeed = 0.08;
  const orbitSpeed = baseOrbitSpeed * (1 - dimFactor * 0.9);
  // Accumulate orbit time with variable speed so position doesn't jump
  orbitTimeOffset += dt * orbitSpeed;
  camera.position.x = 8 * Math.sin(orbitTimeOffset);
  camera.position.z = 8 * Math.cos(orbitTimeOffset);
  camera.position.y = 2 * Math.sin(orbitTimeOffset * 0.5);

  // Camera jitter for running state
  if (jitterAmount > 0) {
    camera.position.x += (Math.random() - 0.5) * jitterAmount;
    camera.position.y += (Math.random() - 0.5) * jitterAmount;
  }

  camera.lookAt(0, 0, 0);

  // Smooth transitions — apply dim factor to reduce opacities
  const dimMul = 1 - dimFactor * 0.5;
  const edgeMat = edgeLines.material as THREE.LineBasicMaterial;
  const dimmedEdgeTarget = targetEdgeOpacity * dimMul;
  edgeMat.opacity += (dimmedEdgeTarget - edgeMat.opacity) * 0.05;

  // Step-specific node color tinting
  let stepNodeColor: THREE.Color | null = null;
  let stepParticleColor: THREE.Color | null = null;
  if (currentStepId && dimFactor > 0.01) {
    switch (currentStepId) {
      case 'collect':
        stepNodeColor = new THREE.Color(0xffaa44);
        stepParticleColor = new THREE.Color(0x44ddcc);
        break;
      case 'process':
        stepNodeColor = new THREE.Color(0xff66aa);
        stepParticleColor = new THREE.Color(0xaa66ff);
        break;
      case 'train':
        stepNodeColor = new THREE.Color(0x44ff88);
        stepParticleColor = new THREE.Color(0x88ff44);
        break;
      case 'simulate':
        stepNodeColor = new THREE.Color(0x6688ff);
        stepParticleColor = new THREE.Color(0xff8866);
        break;
      case 'demo':
        stepNodeColor = new THREE.Color(0x00f0ff);
        stepParticleColor = new THREE.Color(0x00f0ff);
        break;
    }
  }

  const defaultColor = new THREE.Color(0x00f0ff);

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const mat = node.material as THREE.MeshBasicMaterial;

    let nodeTarget = targetNodeOpacity * dimMul;

    // Step-specific effects on nodes
    if (currentStepId && dimFactor > 0.01) {
      switch (currentStepId) {
        case 'process': {
          // Brain-like pulse — waves from center outward
          const dist = nodePositions[i].length() / SPHERE_RADIUS;
          const wave = Math.sin(t * 3 - dist * 6) * 0.5 + 0.5;
          nodeTarget *= 0.5 + wave * 0.5;
          break;
        }
        case 'train': {
          // Nodes flash in sequence
          const seq = ((t * 2 + i * 0.3) % (nodes.length * 0.3)) / (nodes.length * 0.3);
          const flash = Math.max(0, 1 - Math.abs(seq - (i / nodes.length)) * nodes.length * 0.3);
          nodeTarget *= 0.4 + flash * 0.6;
          break;
        }
        case 'simulate': {
          // Split nodes visually — left side dimmer, right side brighter
          const side = nodePositions[i].x > 0 ? 1.3 : 0.7;
          nodeTarget *= side;
          break;
        }
        case 'demo': {
          // Full brightness, all nodes glow
          nodeTarget = Math.max(nodeTarget, 0.6);
          break;
        }
      }
    }

    mat.opacity += (nodeTarget - mat.opacity) * 0.05;

    // Color tinting
    if (stepNodeColor && dimFactor > 0.01) {
      mat.color.lerp(stepNodeColor, 0.03);
    } else {
      mat.color.lerp(defaultColor, 0.03);
    }
  }

  // Pulse effect
  if (pulseTimer > 0) {
    pulseTimer -= 0.016;
    const pulseIntensity = Math.max(0, pulseTimer) * 2;
    edgeMat.opacity = Math.min(0.4, edgeMat.opacity + pulseIntensity * 0.1);
    for (const node of nodes) {
      const mat = node.material as THREE.MeshBasicMaterial;
      mat.opacity = Math.min(0.8, mat.opacity + pulseIntensity * 0.15);
    }
  }

  // Animate particles along edges
  // Step-specific speed modifiers
  let particleSpeedMul = 1;
  if (currentStepId && dimFactor > 0.01) {
    switch (currentStepId) {
      case 'train':
        particleSpeedMul = 2.0;
        break;
      case 'demo':
        particleSpeedMul = 2.5;
        break;
      case 'process':
        particleSpeedMul = 0.6;
        break;
    }
  }

  for (const p of particles) {
    p.progress += p.speed * (targetParticleSpeed / 0.002) * particleSpeedMul;
    if (p.progress >= 1) {
      p.progress = 0;
      if (edges.length > 0) {
        p.edgeIdx = Math.floor(Math.random() * edges.length);
      }
    }

    const edge = edges[p.edgeIdx];
    if (edge) {
      const from = nodePositions[edge.from];
      const to = nodePositions[edge.to];
      p.mesh.position.lerpVectors(from, to, p.progress);
    }

    // Apply dim + step coloring to particles
    const pMat = p.mesh.material as THREE.MeshBasicMaterial;
    const baseParticleOpacity = 0.6;
    const dimmedParticleOpacity = currentStepId === 'demo' && dimFactor > 0.01
      ? baseParticleOpacity
      : baseParticleOpacity * dimMul;
    pMat.opacity += (dimmedParticleOpacity - pMat.opacity) * 0.05;

    if (stepParticleColor && dimFactor > 0.01) {
      pMat.color.lerp(stepParticleColor, 0.03);
    } else {
      pMat.color.lerp(defaultColor, 0.03);
    }
  }

  // Jitter smooth decay
  if (currentActivity !== 'running') {
    jitterAmount *= 0.95;
  }

  renderer.render(scene, camera);
}

export function setExpanded(expanded: boolean, stepId?: string) {
  targetDimFactor = expanded ? 1 : 0;
  currentStepId = expanded ? stepId : undefined;
}

export function setActivity(state: 'idle' | 'running' | 'done') {
  currentActivity = state;

  switch (state) {
    case 'idle':
      targetParticleSpeed = 0.002;
      targetNodeOpacity = 0.15;
      targetEdgeOpacity = 0.05;
      jitterAmount = 0;
      break;
    case 'running':
      targetParticleSpeed = 0.01;
      targetNodeOpacity = 0.4;
      targetEdgeOpacity = 0.15;
      jitterAmount = 0.04;
      break;
    case 'done':
      pulseTimer = 1.5;
      targetParticleSpeed = 0.002;
      targetNodeOpacity = 0.15;
      targetEdgeOpacity = 0.05;
      jitterAmount = 0;
      break;
  }
}
