#!/usr/bin/env python3
from __future__ import annotations
"""
NEUROLABEL Visualizer — Side-by-side replay of dirty vs clean model runs.

Renders two scaled-down game panels showing how each model drives,
with click-to-cycle and keyboard controls.

Requirements:
    pip install pygame
"""

import pygame
import random
import sys
import math

from neurolabel.config.schema import Settings
from neurolabel.ui.replay.serialization import load_summary

# ─────────────────────────────────────────
# Colors
# ─────────────────────────────────────────
C_BG      = (10,  10,  16)
C_ROAD    = (20,  20,  32)
C_KERB    = (28,  28,  42)
C_STRIPE  = (45,  45,  65)
C_EDGE    = (55,  55,  80)
C_BORDER  = (26,  26,  46)
C_WHITE   = (255, 255, 255)
C_DIM     = (102, 102, 128)
C_PLAYER  = (0,   215, 255)
C_OBS     = [(255, 65, 85), (255, 155, 20), (170, 65, 255)]
C_GREEN   = (0,   255, 136)
C_RED     = (255, 65,  85)

# Window
WIN_W, WIN_H = 960, 720
PANEL_W      = WIN_W // 2
HEADER_H     = 110
FOOTER_H     = 72
GAME_PAD_X   = 40
GAME_PAD_TOP = 8
GAME_PAD_BOT = 48

# Scaled game area within each panel
GAME_W = PANEL_W - GAME_PAD_X * 2
GAME_H = WIN_H - HEADER_H - FOOTER_H - GAME_PAD_TOP - GAME_PAD_BOT
GAME_Y = HEADER_H + GAME_PAD_TOP

# Scaled road geometry
LANE_COUNT   = 3
ROAD_MARGIN  = 30
ROAD_LEFT_L  = ROAD_MARGIN
ROAD_RIGHT_L = GAME_W - ROAD_MARGIN
LANE_W       = (ROAD_RIGHT_L - ROAD_LEFT_L) // LANE_COUNT
LANE_CENTERS = [ROAD_LEFT_L + LANE_W * i + LANE_W // 2 for i in range(LANE_COUNT)]

# Scaled car sizes
CAR_W, CAR_H = 28, 46
OBS_W, OBS_H = 28, 44

# Player Y position (fixed near bottom of game area)
PLAYER_Y = GAME_H - 90

# Road speed for stripe animation
ROAD_SPEED = 5.0

FPS = 60
REPLAY_SPEED = 2
AUTO_ADVANCE_DELAY = 60  # frames (1 second at 60fps)


# ─────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────

def draw_car(surf, cx, cy, w, h, color, is_player):
    """Draw a car with glow, body, and detail — matches car_game.py style."""
    rx, ry = cx - w // 2, cy - h // 2

    # Glow
    gs = 14
    glow = pygame.Surface((w + gs * 2, h + gs * 2), pygame.SRCALPHA)
    pygame.draw.rect(glow, (*color, 40), (gs, gs, w, h), border_radius=7)
    surf.blit(glow, (rx - gs, ry - gs))

    # Body
    pygame.draw.rect(surf, color, (rx, ry, w, h), border_radius=7)

    ww = w - 8
    dark = (0, 30, 45) if is_player else (35, 8, 8)

    if is_player:
        pygame.draw.rect(surf, dark, (cx - ww // 2, ry + 7, ww, 11), border_radius=3)
        pygame.draw.rect(surf, dark, (cx - ww // 2, ry + h - 18, ww, 10), border_radius=3)
        pygame.draw.ellipse(surf, (210, 255, 255), (rx + 3, ry + 4, 7, 4))
        pygame.draw.ellipse(surf, (210, 255, 255), (rx + w - 10, ry + 4, 7, 4))
        pygame.draw.ellipse(surf, (255, 55, 55), (rx + 3, ry + h - 8, 7, 4))
        pygame.draw.ellipse(surf, (255, 55, 55), (rx + w - 10, ry + h - 8, 7, 4))
    else:
        pygame.draw.rect(surf, dark, (cx - ww // 2, ry + 7, ww, 10), border_radius=3)
        pygame.draw.rect(surf, dark, (cx - ww // 2, ry + h - 17, ww, 9), border_radius=3)
        pygame.draw.ellipse(surf, (255, 195, 100), (rx + 3, ry + h - 8, 7, 4))
        pygame.draw.ellipse(surf, (255, 195, 100), (rx + w - 10, ry + h - 8, 7, 4))


def draw_road(surf):
    """Draw road background, edges and kerbs on a game-area surface."""
    surf.fill(C_BG)

    # Kerbs
    pygame.draw.rect(surf, C_KERB, (0, 0, ROAD_LEFT_L, GAME_H))
    pygame.draw.rect(surf, C_KERB, (ROAD_RIGHT_L, 0, GAME_W - ROAD_RIGHT_L, GAME_H))

    # Road surface
    pygame.draw.rect(surf, C_ROAD, (ROAD_LEFT_L, 0, ROAD_RIGHT_L - ROAD_LEFT_L, GAME_H))

    # Edge lines
    pygame.draw.line(surf, C_EDGE, (ROAD_LEFT_L, 0), (ROAD_LEFT_L, GAME_H), 2)
    pygame.draw.line(surf, C_EDGE, (ROAD_RIGHT_L, 0), (ROAD_RIGHT_L, GAME_H), 2)


def draw_stripes(surf, offset):
    """Draw animated lane-divider stripes."""
    stripe_h   = 28
    stripe_gap  = 62
    total       = stripe_h + stripe_gap

    for li in range(1, LANE_COUNT):
        sx = ROAD_LEFT_L + LANE_W * li
        y  = (offset % total) - stripe_gap
        while y < GAME_H + stripe_h:
            pygame.draw.rect(surf, C_STRIPE, (sx - 2, int(y), 4, stripe_h))
            y += total


def text_centered(screen, font, text, color, cx, cy):
    """Render text centered at (cx, cy)."""
    surf = font.render(text, True, color)
    screen.blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))
    return surf


def text_at(screen, font, text, color, x, y):
    """Render text at (x, y) top-left."""
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))
    return surf


# ─────────────────────────────────────────
# Panel replay state
# ─────────────────────────────────────────

class PanelState:
    """Tracks the replay state for one panel."""

    def __init__(self, results, label):
        self.results       = results
        self.label         = label
        self.run_idx       = 0
        self.frame_i       = 0
        self.playing       = True
        self.stripe_offset = 0.0
        self.finished      = False
        self.pause_timer   = 0

    @property
    def current_run(self):
        return self.results["runs"][self.run_idx]

    @property
    def current_frame(self):
        frames = self.current_run["frames"]
        idx = min(self.frame_i, len(frames) - 1)
        return frames[idx]

    @property
    def alive_time(self):
        return self.current_run["alive_time"]

    @property
    def num_runs(self):
        return len(self.results["runs"])

    def advance(self):
        if not self.playing:
            return False
        if self.finished:
            self.pause_timer += 1
            if self.pause_timer >= AUTO_ADVANCE_DELAY:
                self._next_random()
            return False
        frames = self.current_run["frames"]
        for _ in range(REPLAY_SPEED):
            if self.frame_i < len(frames) - 1:
                self.frame_i += 1
                self.stripe_offset += ROAD_SPEED
            else:
                self.finished = True
                self.pause_timer = 0
                break
        return not self.finished

    def start_run(self, idx):
        self.run_idx       = idx
        self.frame_i       = 0
        self.playing       = True
        self.finished      = False
        self.stripe_offset = 0.0
        self.pause_timer   = 0

    def start_random(self):
        idx = random.randint(0, self.num_runs - 1)
        self.start_run(idx)

    def _next_random(self):
        self.start_random()


# ─────────────────────────────────────────
# Panel rendering
# ─────────────────────────────────────────

def render_panel(screen, panel, x_offset, fonts):
    """Render one panel (header + road + cars + footer) at x_offset."""

    game_x = x_offset + GAME_PAD_X

    # ── Header ──
    # Model label
    label_y = 52
    text_centered(screen, fonts["label"], panel.label, C_DIM, x_offset + PANEL_W // 2, label_y)

    # Big avg number
    avg = panel.results["avg_alive"]
    avg_seconds = avg / FPS
    avg_text = f"{avg:.0f} frames"
    text_centered(screen, fonts["hero"], avg_text, C_WHITE, x_offset + PANEL_W // 2, label_y + 30)
    sec_text = f"({avg_seconds:.1f}s avg survival)"
    text_centered(screen, fonts["tiny"], sec_text, C_DIM, x_offset + PANEL_W // 2, label_y + 52)

    # ── Game surface ──
    game_surf = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
    draw_road(game_surf)
    draw_stripes(game_surf, panel.stripe_offset)

    frame = panel.current_frame

    # Player car
    player_lane = frame["lane"]
    px = LANE_CENTERS[player_lane]
    draw_car(game_surf, px, PLAYER_Y, CAR_W, CAR_H, C_PLAYER, True)

    # Obstacle cars
    for obs in frame["obs"]:
        obs_lane = obs[0]
        obs_progress = obs[1]
        oy = int(-OBS_H + (PLAYER_Y + OBS_H) * obs_progress)
        ox = LANE_CENTERS[obs_lane]
        color = C_OBS[obs_lane % len(C_OBS)]
        draw_car(game_surf, ox, oy, OBS_W, OBS_H, color, False)

    # Blit game surface
    screen.blit(game_surf, (game_x, GAME_Y))

    # Border around game area
    pygame.draw.rect(screen, C_BORDER, (game_x - 1, GAME_Y - 1, GAME_W + 2, GAME_H + 2), 1)

    # ── Footer under game ──
    footer_y = GAME_Y + GAME_H + 10

    # Frame counter + status
    if panel.finished:
        status_text = f"CRASHED at {panel.frame_i}f"
        status_color = C_RED
    else:
        status_text = f"frame {panel.frame_i}"
        status_color = C_DIM

    text_centered(screen, fonts["small"], status_text, status_color,
                  x_offset + PANEL_W // 2, footer_y)

    # Run index
    run_text = f"Run {panel.run_idx + 1}/{panel.num_runs}  |  alive: {panel.alive_time}f"
    text_centered(screen, fonts["tiny"], run_text, C_DIM,
                  x_offset + PANEL_W // 2, footer_y + 20)


# ─────────────────────────────────────────
# Mock data generation
# ─────────────────────────────────────────

def generate_mock_results(num_runs=50, frames_per_run=400):
    """Create simple mock replay data for testing."""
    runs = []
    alive_times = []

    for _ in range(num_runs):
        alive_time = random.randint(120, 500)
        alive_times.append(alive_time)
        num_frames = min(frames_per_run, alive_time)

        frames = []
        player_lane = 1
        obs_list = []
        spawn_counter = 0

        for fi in range(num_frames):
            spawn_counter += 1

            new_obs = []
            for ol, op in obs_list:
                np_ = op + 0.012
                if np_ < 1.15:
                    new_obs.append([ol, round(np_, 3)])
            obs_list = new_obs

            if spawn_counter >= 50:
                spawn_counter = 0
                lane = random.randint(0, 2)
                obs_list.append([lane, 0.0])

            decision = 0
            for ol, op in obs_list:
                if ol == player_lane and 0.5 < op < 0.75:
                    if player_lane > 0:
                        decision = -1
                    elif player_lane < 2:
                        decision = 1
                    else:
                        decision = -1
                    break

            new_lane = max(0, min(2, player_lane + decision))
            player_lane = new_lane

            frames.append({
                "lane": player_lane,
                "obs": [list(o) for o in obs_list],
                "decision": decision,
            })

        runs.append({
            "alive_time": alive_time,
            "seed": random.randint(0, 99999),
            "frames": frames,
        })

    avg_alive = sum(alive_times) / len(alive_times)
    std_alive = (sum((t - avg_alive) ** 2 for t in alive_times) / len(alive_times)) ** 0.5

    return {
        "avg_alive": round(avg_alive, 1),
        "std_alive": round(std_alive, 1),
        "runs": runs,
    }


# ─────────────────────────────────────────
# Main visualizer
# ─────────────────────────────────────────

def run_visualizer(dirty_results, clean_results):
    """Launch side-by-side comparison visualization."""
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("NEUROLABEL")
    clock = pygame.time.Clock()

    try:
        fonts = {
            "brand": pygame.font.SysFont("Courier New", 13),
            "label": pygame.font.SysFont("Courier New", 15, bold=True),
            "hero":  pygame.font.SysFont("Courier New", 24, bold=True),
            "small": pygame.font.SysFont("Courier New", 16, bold=True),
            "tiny":  pygame.font.SysFont("Courier New", 12),
            "pct":   pygame.font.SysFont("Courier New", 22, bold=True),
            "keys":  pygame.font.SysFont("Courier New", 11),
        }
    except Exception:
        fonts = {k: pygame.font.SysFont(None, v) for k, v in {
            "brand": 13, "label": 15, "hero": 24, "small": 16,
            "tiny": 12, "pct": 22, "keys": 11,
        }.items()}

    left_panel  = PanelState(dirty_results, "DIRTY MODEL")
    right_panel = PanelState(clean_results, "CLEAN MODEL")

    left_panel.start_run(0)
    right_panel.start_run(0)

    # Compute improvement
    dirty_avg = dirty_results["avg_alive"]
    clean_avg = clean_results["avg_alive"]
    if dirty_avg > 0:
        improvement = ((clean_avg - dirty_avg) / dirty_avg) * 100
    else:
        improvement = 0.0

    running = True

    while running:
        clock.tick(FPS)

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    left_panel.start_random()
                    right_panel.start_random()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, _ = event.pos
                if mx < PANEL_W:
                    left_panel.start_random()
                else:
                    right_panel.start_random()

        # ── Update ──
        left_panel.advance()
        right_panel.advance()

        # ── Draw ──
        screen.fill(C_BG)

        # Title - small, understated
        text_centered(screen, fonts["brand"], "NEUROLABEL", C_DIM, WIN_W // 2, 16)

        # Divider line
        pygame.draw.line(screen, C_BORDER, (PANEL_W, 30), (PANEL_W, WIN_H - FOOTER_H), 1)

        # Panels
        render_panel(screen, left_panel, 0, fonts)
        render_panel(screen, right_panel, PANEL_W, fonts)

        # ── Improvement percentage between panels ──
        mid_y = GAME_Y + GAME_H // 2
        if improvement >= 0:
            pct_text = f"+{improvement:.1f}%"
            pct_color = C_GREEN
        else:
            pct_text = f"{improvement:.1f}%"
            pct_color = C_RED

        # Dark pill behind percentage
        pct_surf = fonts["pct"].render(pct_text, True, pct_color)
        pw, ph = pct_surf.get_width() + 16, pct_surf.get_height() + 8
        pill_x = PANEL_W - pw // 2
        pill_y = mid_y - ph // 2
        pill = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(pill, (10, 10, 16, 220), (0, 0, pw, ph), border_radius=4)
        pygame.draw.rect(pill, (*pct_color, 50), (0, 0, pw, ph), 1, border_radius=4)
        screen.blit(pill, (pill_x, pill_y))
        screen.blit(pct_surf, (pill_x + 8, pill_y + 4))

        # ── Footer ──
        footer_y = WIN_H - FOOTER_H
        pygame.draw.line(screen, C_BORDER, (0, footer_y), (WIN_W, footer_y), 1)

        # Controls
        keys_text = "[SPACE] next run    [ESC] quit"
        text_centered(screen, fonts["keys"], keys_text, C_DIM, WIN_W // 2, footer_y + 20)

        # Tagline
        text_centered(screen, fonts["keys"], "Better data, better AI", C_DIM,
                      WIN_W // 2, footer_y + 42)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    mock_dirty = generate_mock_results()
    mock_clean = generate_mock_results()
    run_visualizer(mock_dirty, mock_clean)


def run_from_results(settings: Settings, *, dirty_results=None, clean_results=None) -> None:
    """Compatibility helper used by the package CLI/orchestration layer."""
    if dirty_results is not None and clean_results is not None:
        run_visualizer(dirty_results, clean_results)
        return

    dirty_summary = load_summary(settings.paths.results_dirty)
    clean_summary = load_summary(settings.paths.results_clean)

    dirty_mock = generate_mock_results(num_runs=len(dirty_summary.get("alive_times", [5])))
    clean_mock = generate_mock_results(num_runs=len(clean_summary.get("alive_times", [5])))

    for key in ("avg_alive", "std_alive"):
        if key in dirty_summary:
            dirty_mock[key] = dirty_summary[key]
        if key in clean_summary:
            clean_mock[key] = clean_summary[key]

    run_visualizer(dirty_mock, clean_mock)
