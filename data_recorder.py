#!/usr/bin/env python3
"""
VELOCITY — Data Recorder
Plays the game exactly like car_game.py but records game state
every second (60 frames) to a JSONL file for training data.
"""

import pygame
import json
import time
import sys
import random

import config
from game_engine import (
    GameState, WIDTH, HEIGHT, FPS,
    LANE_COUNT, ROAD_LEFT, ROAD_RIGHT, LANE_WIDTH, LANE_CENTERS,
    ROAD_SPEED, SPAWN_INTERVAL, CAR_W, CAR_H, OBS_W, OBS_H,
)

# ─────────────────────────────────────────
# Colors (same as car_game.py)
# ─────────────────────────────────────────
C_BG     = (10,  10,  16)
C_ROAD   = (20,  20,  30)
C_KERB   = (28,  28,  40)
C_STRIPE = (45,  45,  65)
C_EDGE   = (55,  55,  80)
C_WHITE  = (255, 255, 255)
C_DIM    = (160, 160, 180)
C_PLAYER = (0,   215, 255)
C_GLOW   = (0,   100, 160)
C_OBS    = [(255, 65, 85), (255, 155, 20), (170, 65, 255)]
C_REC    = (255, 50, 50)

# ─────────────────────────────────────────
# Drawing helpers (reused from car_game.py)
# ─────────────────────────────────────────

def draw_car(surf, cx, cy, w, h, color, is_player):
    rx, ry = cx - w // 2, cy - h // 2

    gs = 20
    glow = pygame.Surface((w + gs*2, h + gs*2), pygame.SRCALPHA)
    pygame.draw.rect(glow, (*color, 40), (gs, gs, w, h), border_radius=10)
    surf.blit(glow, (rx - gs, ry - gs))

    pygame.draw.rect(surf, color, (rx, ry, w, h), border_radius=9)

    ww = w - 10
    dark = (0, 30, 45) if is_player else (35, 8, 8)

    if is_player:
        pygame.draw.rect(surf, dark, (cx - ww//2, ry + 10,      ww, 15), border_radius=4)
        pygame.draw.rect(surf, dark, (cx - ww//2, ry + h - 25,  ww, 13), border_radius=4)
        pygame.draw.ellipse(surf, (210, 255, 255), (rx + 4,      ry + 5,  10, 6))
        pygame.draw.ellipse(surf, (210, 255, 255), (rx + w - 14, ry + 5,  10, 6))
        pygame.draw.ellipse(surf, (255, 55, 55),   (rx + 4,      ry + h - 11, 10, 6))
        pygame.draw.ellipse(surf, (255, 55, 55),   (rx + w - 14, ry + h - 11, 10, 6))
    else:
        pygame.draw.rect(surf, dark, (cx - ww//2, ry + 10,      ww, 13), border_radius=4)
        pygame.draw.rect(surf, dark, (cx - ww//2, ry + h - 23,  ww, 11), border_radius=4)
        pygame.draw.ellipse(surf, (255, 195, 100), (rx + 4,      ry + h - 11, 10, 6))
        pygame.draw.ellipse(surf, (255, 195, 100), (rx + w - 14, ry + h - 11, 10, 6))


class Stripe:
    def __init__(self, x, y):
        self.x, self.y = x, float(y)

    def update(self):
        self.y += ROAD_SPEED
        if self.y > HEIGHT + 40:
            self.y -= HEIGHT + 80

    def draw(self, surf):
        pygame.draw.rect(surf, C_STRIPE, (self.x - 2, int(self.y), 4, 38))


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("VELOCITY — Recording")
    clock = pygame.time.Clock()

    try:
        font_score = pygame.font.SysFont("Courier New", 38, bold=True)
        font_title = pygame.font.SysFont("Courier New", 52, bold=True)
        font_sub   = pygame.font.SysFont("Courier New", 17)
    except Exception:
        font_score = pygame.font.SysFont(None, 38)
        font_title = pygame.font.SysFont(None, 52)
        font_sub   = pygame.font.SysFont(None, 17)

    stripes = [
        Stripe(lx, y)
        for lx in [ROAD_LEFT + LANE_WIDTH, ROAD_LEFT + LANE_WIDTH * 2]
        for y in range(0, HEIGHT + 100, 90)
    ]

    # Assign random colors to obstacles for rendering
    obs_colors = {}
    color_rng = random.Random()

    # Clear old data from previous runs
    config.GAME_JSONL.parent.mkdir(exist_ok=True)
    open(config.GAME_JSONL, "w").close()

    game_state = None
    state = "title"
    best = 0
    blink = 0
    key_cd = 0

    # Recording state
    recording_file = None
    second_counter = 0
    frame_in_second = 0
    last_decision = 0       # last decision in this second (for JSONL)
    pending_decision = 0    # decision to apply this frame (consumed after one step)

    def start_game():
        nonlocal game_state, recording_file, second_counter, frame_in_second
        nonlocal last_decision, pending_decision, obs_colors, key_cd
        seed = int(time.time() * 1000) % (2**31)
        game_state = GameState(seed=seed)
        obs_colors = {}
        key_cd = 0
        second_counter = 0
        frame_in_second = 0
        last_decision = 0
        pending_decision = 0
        recording_file = open(config.GAME_JSONL, "a")
        print(f"[recorder] Recording to {config.GAME_JSONL}")

    def stop_recording(final_score):
        nonlocal recording_file
        if recording_file:
            recording_file.close()
            recording_file = None
            print(f"[recorder] Stopped. Score: {final_score}, seconds recorded: {second_counter}")

    def write_record():
        nonlocal second_counter
        if not recording_file or not game_state:
            return
        obs_list = [[o.lane, round(o.y / HEIGHT, 4)] for o in game_state.obstacles]
        record = {
            "t": round(time.time(), 2),
            "sec": second_counter,
            "lane": game_state.player.lane,
            "obs": obs_list,
            "decision": last_decision,
            "score": game_state.score,
            "alive": game_state.alive,
        }
        recording_file.write(json.dumps(record) + "\n")
        recording_file.flush()
        second_counter += 1

    pygame.key.set_repeat(0, 0)

    while True:
        clock.tick(FPS)
        blink += 1

        # ── Events ──────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if state == "playing":
                    stop_recording(game_state.score if game_state else 0)
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if state == "playing":
                        stop_recording(game_state.score if game_state else 0)
                    pygame.quit()
                    sys.exit()
                if state == "title":
                    start_game()
                    state = "playing"
                elif state == "playing" and key_cd <= 0:
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        pending_decision = -1
                        last_decision = -1
                        key_cd = 6
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        pending_decision = 1
                        last_decision = 1
                        key_cd = 6
                elif state == "dead":
                    start_game()
                    state = "playing"

        if key_cd > 0:
            key_cd -= 1

        # ── Update ──────────────────────
        for s in stripes:
            s.update()

        if state == "playing" and game_state:
            # Step the game engine — consume pending decision (apply once)
            game_state.step(pending_decision)
            pending_decision = 0

            # Assign colors to new obstacles
            for o in game_state.obstacles:
                if id(o) not in obs_colors:
                    obs_colors[id(o)] = color_rng.choice(C_OBS)

            frame_in_second += 1

            # Record every 60 frames (1 second)
            if frame_in_second >= FPS:
                write_record()
                frame_in_second = 0
                last_decision = 0  # reset for next second

            # Check death
            if not game_state.alive:
                # Write final record on death
                write_record()
                if game_state.score > best:
                    best = game_state.score
                stop_recording(game_state.score)
                state = "dead"

        # ── Draw ────────────────────────
        screen.fill(C_BG)

        # Kerbs
        pygame.draw.rect(screen, C_KERB, (0, 0, ROAD_LEFT, HEIGHT))
        pygame.draw.rect(screen, C_KERB, (ROAD_RIGHT, 0, WIDTH - ROAD_RIGHT, HEIGHT))

        # Road
        pygame.draw.rect(screen, C_ROAD, (ROAD_LEFT, 0, ROAD_RIGHT - ROAD_LEFT, HEIGHT))

        # Edges
        pygame.draw.line(screen, C_EDGE, (ROAD_LEFT,  0), (ROAD_LEFT,  HEIGHT), 2)
        pygame.draw.line(screen, C_EDGE, (ROAD_RIGHT, 0), (ROAD_RIGHT, HEIGHT), 2)

        # Lane stripes
        for s in stripes:
            s.draw(screen)

        # Obstacles + player
        if game_state and state in ("playing", "dead"):
            for o in game_state.obstacles:
                color = obs_colors.get(id(o), C_OBS[0])
                draw_car(screen, int(o.x), int(o.y), OBS_W, OBS_H, color, False)
            draw_car(screen, int(game_state.player.x), int(game_state.player.y),
                     CAR_W, CAR_H, C_PLAYER, True)

        # ── HUD ─────────────────────────
        if state == "playing" and game_state:
            sc = font_score.render(f"{game_state.score:05d}", True, C_WHITE)
            screen.blit(sc, (WIDTH // 2 - sc.get_width() // 2, 16))

            # Recording indicator
            if blink % 60 < 45:
                pygame.draw.circle(screen, C_REC, (WIDTH - 30, 25), 7)
            rec_txt = font_sub.render("REC", True, C_REC)
            screen.blit(rec_txt, (WIDTH - 60, 17))

        # ── Overlays ────────────────────
        if state in ("title", "dead"):
            dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 155))
            screen.blit(dim, (0, 0))

            label = "VELOCITY" if state == "title" else "GAME OVER"
            t = font_title.render(label, True, C_PLAYER)
            screen.blit(t, (WIDTH//2 - t.get_width()//2, HEIGHT//2 - 90))

            if state == "dead" and game_state:
                sc_txt = font_score.render(f"{game_state.score:05d}", True, C_WHITE)
                screen.blit(sc_txt, (WIDTH//2 - sc_txt.get_width()//2, HEIGHT//2 - 20))
                if game_state.score >= best and game_state.score > 0:
                    nb = font_sub.render("new best", True, C_PLAYER)
                    screen.blit(nb, (WIDTH//2 - nb.get_width()//2, HEIGHT//2 + 22))

            if blink % 60 < 42:
                hint = "press any key" if state == "title" else "press any key to retry"
                h = font_sub.render(hint, True, C_DIM)
                screen.blit(h, (WIDTH//2 - h.get_width()//2, HEIGHT//2 + 55))

            if state == "title":
                ctrl = font_sub.render("← →  or  A D  to switch lanes", True, C_DIM)
                screen.blit(ctrl, (WIDTH//2 - ctrl.get_width()//2, HEIGHT//2 + 90))
                rec_note = font_sub.render("Recording mode — data saved to JSONL", True, C_REC)
                screen.blit(rec_note, (WIDTH//2 - rec_note.get_width()//2, HEIGHT//2 + 120))

        pygame.display.flip()


if __name__ == "__main__":
    main()
