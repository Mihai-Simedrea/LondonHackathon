"""Psychedelic visual distraction overlay for MetaDrive human play sessions.

Hooks into MetaDrive's Panda3D engine to display GPU-rendered procedural effects
(kaleidoscope flowers, fractals, spirals, sacred geometry, moire patterns)
as transparent overlays on top of the driving scene.  Effects are scheduled
randomly and ramp up in intensity over time to create genuine distraction —
producing variation in OC scores for the NeuroLabel pipeline.
"""

import glob
import math
import random
import time
from pathlib import Path

from panda3d.core import (
    CardMaker,
    NodePath,
    Shader,
    Texture,
    TransparencyAttrib,
    LVecBase4f,
)

SHADERS_DIR = Path(__file__).parent / "shaders"
ASSETS_DIR = Path(__file__).parent / "assets"

# ── Effect catalogue ─────────────────────────────────────────
EFFECTS = [
    {
        "name": "kaleidoscope",
        "shader": "kaleidoscope.frag.glsl",
        "weight": 3,            # higher = more likely to be picked
        "duration": (4, 8),     # seconds
        "extra_inputs": {"segments": 6.0},
    },
    {
        "name": "mandelbrot",
        "shader": "mandelbrot.frag.glsl",
        "weight": 2,
        "duration": (5, 10),
        "extra_inputs": {},
    },
    {
        "name": "spiral",
        "shader": "spiral.frag.glsl",
        "weight": 2,
        "duration": (3, 7),
        "extra_inputs": {},
    },
    {
        "name": "sacred_geometry",
        "shader": "sacred_geometry.frag.glsl",
        "weight": 2,
        "duration": (4, 8),
        "extra_inputs": {},
    },
    {
        "name": "moire",
        "shader": "moire.frag.glsl",
        "weight": 1,
        "duration": (3, 6),
        "extra_inputs": {},
    },
]

SCREEN_EFFECTS = [
    {"name": "dim",       "type_val": 0.0, "weight": 2, "duration": (1, 3)},
    {"name": "flicker",   "type_val": 1.0, "weight": 2, "duration": (0.5, 2)},
    {"name": "inversion", "type_val": 2.0, "weight": 1, "duration": (1, 3)},
    {"name": "chromatic", "type_val": 3.0, "weight": 1, "duration": (2, 4)},
]

FADE_IN_SEC = 0.8
FADE_OUT_SEC = 1.0


class _ActiveEffect:
    """Tracks a single running overlay effect."""

    __slots__ = ("node", "start", "duration", "fade_in", "fade_out",
                 "extra_inputs", "peak_intensity", "is_texture",
                 "rotate_speed", "pulse_speed", "base_scale")

    def __init__(self, node, duration, peak_intensity=0.6,
                 fade_in=FADE_IN_SEC, fade_out=FADE_OUT_SEC,
                 extra_inputs=None, is_texture=False,
                 rotate_speed=0.0, pulse_speed=0.0, base_scale=1.0):
        self.node = node
        self.start = time.time()
        self.duration = duration
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.peak_intensity = peak_intensity
        self.extra_inputs = extra_inputs or {}
        self.is_texture = is_texture
        self.rotate_speed = rotate_speed
        self.pulse_speed = pulse_speed
        self.base_scale = base_scale

    @property
    def elapsed(self):
        return time.time() - self.start

    @property
    def finished(self):
        return self.elapsed >= self.duration

    @property
    def fade(self):
        e = self.elapsed
        # Fade in
        if e < self.fade_in:
            return e / self.fade_in
        # Fade out
        remaining = self.duration - e
        if remaining < self.fade_out:
            return max(0.0, remaining / self.fade_out)
        return 1.0


class PsychedelicOverlay:
    """Manages psychedelic visual distractions during MetaDrive human play.

    Usage::

        env = MetaDriveEnv(config)
        obs, info = env.reset()
        overlay = PsychedelicOverlay(env)
        overlay.start()

        while playing:
            obs, reward, terminated, truncated, info = env.step(action)
            overlay.tick()          # updates effects each frame
            # ... recording logic ...

        overlay.cleanup()
    """

    def __init__(self, env, *,
                 min_gap=5.0,
                 max_gap=15.0,
                 max_concurrent=3,
                 ramp_seconds=120.0,
                 base_intensity=0.25,
                 max_intensity=0.7):
        self.engine = env.engine
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.max_concurrent = max_concurrent
        self.ramp_seconds = ramp_seconds
        self.base_intensity = base_intensity
        self.max_intensity = max_intensity

        self._active: list[_ActiveEffect] = []
        self._next_spawn = 0.0
        self._session_start = 0.0
        self._started = False
        self._rng = random.Random()

        # Pre-compile shaders
        self._shaders: dict[str, Shader] = {}
        self._vert_path = str(SHADERS_DIR / "overlay_vert.glsl")
        for eff in EFFECTS:
            frag_path = str(SHADERS_DIR / eff["shader"])
            try:
                shader = Shader.load(
                    Shader.SL_GLSL,
                    vertex=self._vert_path,
                    fragment=frag_path,
                )
                self._shaders[eff["name"]] = shader
            except Exception as exc:
                print(f"[overlay] Warning: could not compile {eff['name']} shader: {exc}")

        # Screen effects shader
        frag_path = str(SHADERS_DIR / "screen_effects.frag.glsl")
        try:
            self._screen_shader = Shader.load(
                Shader.SL_GLSL,
                vertex=self._vert_path,
                fragment=frag_path,
            )
        except Exception as exc:
            print(f"[overlay] Warning: could not compile screen_effects shader: {exc}")
            self._screen_shader = None

        # Load texture assets (animated via rotation/scale/fade)
        self._textures: list[str] = []
        if ASSETS_DIR.exists():
            for ext in ("*.png", "*.jpg"):
                self._textures.extend(
                    str(p) for p in sorted(ASSETS_DIR.glob(ext))
                )
        if self._textures:
            print(f"[overlay] Loaded {len(self._textures)} texture assets")

    # ── Public API ────────────────────────────────────────────

    def start(self):
        """Begin the distraction session."""
        self._session_start = time.time()
        self._next_spawn = self._session_start + self._rng.uniform(3.0, 8.0)
        self._started = True
        print("[overlay] Psychedelic overlay started")

    def tick(self):
        """Call once per frame to update / spawn / expire effects."""
        if not self._started:
            return

        now = time.time()
        session_elapsed = now - self._session_start

        # Intensity ramps up over time
        ramp = min(1.0, session_elapsed / self.ramp_seconds)
        current_intensity = self.base_intensity + ramp * (self.max_intensity - self.base_intensity)

        # Update active effects
        for eff in self._active:
            if eff.is_texture:
                # Animate texture overlays: rotate, pulse scale, fade alpha
                e = eff.elapsed
                angle = e * eff.rotate_speed * 60.0  # degrees
                pulse = 1.0 + 0.3 * math.sin(e * eff.pulse_speed)
                s = eff.base_scale * pulse
                eff.node.setR(angle)
                eff.node.setScale(s)
                alpha = current_intensity * eff.peak_intensity * eff.fade
                eff.node.setColorScale(1, 1, 1, alpha)
            else:
                eff.node.setShaderInput("time", session_elapsed)
                eff.node.setShaderInput("intensity", current_intensity * eff.peak_intensity)
                eff.node.setShaderInput("fade", eff.fade)
                for k, v in eff.extra_inputs.items():
                    eff.node.setShaderInput(k, v)

        # Remove finished effects
        expired = [e for e in self._active if e.finished]
        for eff in expired:
            eff.node.removeNode()
            self._active.remove(eff)

        # Spawn new effects
        if now >= self._next_spawn and len(self._active) < self.max_concurrent:
            self._spawn_effect(current_intensity)
            # As session progresses, gaps get shorter
            gap_scale = 1.0 - ramp * 0.5  # gaps shrink to 50% at full ramp
            gap = self._rng.uniform(self.min_gap * gap_scale, self.max_gap * gap_scale)
            self._next_spawn = now + gap

    def cleanup(self):
        """Remove all overlay nodes."""
        for eff in self._active:
            eff.node.removeNode()
        self._active.clear()
        self._started = False
        print("[overlay] Psychedelic overlay cleaned up")

    # ── Internal ──────────────────────────────────────────────

    def _spawn_effect(self, intensity):
        """Randomly pick and spawn an effect."""
        roll = self._rng.random()
        if roll < 0.5 and self._shaders:
            self._spawn_shader_effect(intensity)
        elif roll < 0.8 and self._textures:
            self._spawn_texture_effect(intensity)
        elif self._screen_shader:
            self._spawn_screen_effect(intensity)
        elif self._shaders:
            self._spawn_shader_effect(intensity)

    def _spawn_shader_effect(self, intensity):
        """Spawn a procedural shader overlay (kaleidoscope, mandelbrot, etc.)."""
        available = [e for e in EFFECTS if e["name"] in self._shaders]
        if not available:
            return

        weights = [e["weight"] for e in available]
        choice = self._rng.choices(available, weights=weights, k=1)[0]

        shader = self._shaders[choice["name"]]
        duration = self._rng.uniform(*choice["duration"])

        # Create a full-screen transparent quad on render2d
        node = self._make_fullscreen_quad(f"effect_{choice['name']}_{id(self)}")
        node.setShader(shader)
        node.setShaderInput("time", 0.0)
        node.setShaderInput("intensity", intensity)
        node.setShaderInput("fade", 0.0)
        for k, v in choice.get("extra_inputs", {}).items():
            node.setShaderInput(k, float(v))

        # Random position offset for variety
        ox = self._rng.uniform(-0.3, 0.3)
        oy = self._rng.uniform(-0.3, 0.3)
        scale = self._rng.uniform(0.6, 1.2)
        node.setPos(ox, 0, oy)
        node.setScale(scale)

        active = _ActiveEffect(
            node=node,
            duration=duration,
            peak_intensity=self._rng.uniform(0.4, 0.8),
            extra_inputs=choice.get("extra_inputs", {}),
        )
        self._active.append(active)

    def _spawn_texture_effect(self, intensity):
        """Spawn an animated texture overlay — rotates, pulses, and fades."""
        if not self._textures:
            return

        tex_path = self._rng.choice(self._textures)
        duration = self._rng.uniform(4, 9)

        # Load texture via Panda3D
        tex = self.engine.loader.loadTexture(tex_path)
        if tex is None:
            return

        # Create quad with the texture
        cm = CardMaker(f"tex_{Path(tex_path).stem}")
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)  # centered, will scale
        node = self.engine.aspect2d.attachNewNode(cm.generate())
        node.setTexture(tex)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setBin("fixed", 45)

        # Random position, scale, and animation params
        ox = self._rng.uniform(-0.6, 0.6)
        oy = self._rng.uniform(-0.5, 0.5)
        base_scale = self._rng.uniform(0.4, 1.0)
        node.setPos(ox, 0, oy)
        node.setScale(base_scale)
        node.setColorScale(1, 1, 1, 0)  # start invisible

        active = _ActiveEffect(
            node=node,
            duration=duration,
            peak_intensity=self._rng.uniform(0.3, 0.7),
            is_texture=True,
            rotate_speed=self._rng.uniform(0.3, 1.5) * self._rng.choice([-1, 1]),
            pulse_speed=self._rng.uniform(1.0, 3.0),
            base_scale=base_scale,
        )
        self._active.append(active)

    def _spawn_screen_effect(self, intensity):
        """Spawn a screen-wide effect (dim, flicker, inversion, chromatic)."""
        if not self._screen_shader:
            return

        weights = [e["weight"] for e in SCREEN_EFFECTS]
        choice = self._rng.choices(SCREEN_EFFECTS, weights=weights, k=1)[0]

        duration = self._rng.uniform(*choice["duration"])

        node = self._make_fullscreen_quad(f"screen_{choice['name']}_{id(self)}")
        node.setShader(self._screen_shader)
        node.setShaderInput("time", 0.0)
        node.setShaderInput("intensity", intensity)
        node.setShaderInput("fade", 0.0)
        node.setShaderInput("effect_type", choice["type_val"])

        active = _ActiveEffect(
            node=node,
            duration=duration,
            peak_intensity=self._rng.uniform(0.5, 1.0),
            fade_in=0.3,
            fade_out=0.5,
        )
        self._active.append(active)

    def _make_fullscreen_quad(self, name):
        """Create a transparent full-screen quad attached to render2d."""
        cm = CardMaker(name)
        cm.setFrame(-1, 1, -1, 1)
        node = self.engine.render2d.attachNewNode(cm.generate())
        node.setTransparency(TransparencyAttrib.MAlpha)
        # Render on top of 3D scene but below UI
        node.setBin("fixed", 50)
        return node
