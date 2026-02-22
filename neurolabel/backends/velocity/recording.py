#!/usr/bin/env python3
from __future__ import annotations
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
from neurolabel.backends.velocity.render import (
    C_WHITE,
    C_DIM,
    C_PLAYER,
    C_OBS,
    C_REC,
    draw_car,
    draw_road_background,
    make_default_stripes,
)
from game_engine import (
    GameState, WIDTH, HEIGHT, FPS,
    LANE_COUNT, LANE_CENTERS,
    SPAWN_INTERVAL, CAR_W, CAR_H, OBS_W, OBS_H,
)


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

    stripes = make_default_stripes()

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
    record_counter = 0
    frame_in_tick = 0
    RECORD_INTERVAL = 6     # record every 6 frames (~10 Hz at 60 fps)
    last_decision = 0       # last decision in this tick (for JSONL)
    pending_decision = 0    # decision to apply this frame (consumed after one step)

    def start_game():
        nonlocal game_state, recording_file, record_counter, frame_in_tick
        nonlocal last_decision, pending_decision, obs_colors, key_cd
        seed = int(time.time() * 1000) % (2**31)
        game_state = GameState(seed=seed)
        obs_colors = {}
        key_cd = 0
        record_counter = 0
        frame_in_tick = 0
        last_decision = 0
        pending_decision = 0
        recording_file = open(config.GAME_JSONL, "a")
        print(f"[recorder] Recording to {config.GAME_JSONL} (~{FPS // RECORD_INTERVAL} Hz)")

    def stop_recording(final_score):
        nonlocal recording_file
        if recording_file:
            recording_file.close()
            recording_file = None
            print(f"[recorder] Stopped. Score: {final_score}, records: {record_counter}")

    def write_record():
        nonlocal record_counter
        if not recording_file or not game_state:
            return
        obs_list = [[o.lane, round(o.y / HEIGHT, 4)] for o in game_state.obstacles]
        record = {
            "t": round(time.time(), 2),
            "sec": record_counter,
            "lane": game_state.player.lane,
            "obs": obs_list,
            "decision": last_decision,
            "score": game_state.score,
            "alive": game_state.alive,
        }
        recording_file.write(json.dumps(record) + "\n")
        recording_file.flush()
        record_counter += 1

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

            frame_in_tick += 1

            # Record every RECORD_INTERVAL frames (~10 Hz)
            if frame_in_tick >= RECORD_INTERVAL:
                write_record()
                frame_in_tick = 0
                last_decision = 0  # reset for next tick

            # Check death
            if not game_state.alive:
                # Write final record on death
                write_record()
                if game_state.score > best:
                    best = game_state.score
                stop_recording(game_state.score)
                state = "dead"

        # ── Draw ────────────────────────
        draw_road_background(screen)

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
