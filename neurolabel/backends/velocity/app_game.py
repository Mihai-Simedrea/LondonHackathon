#!/usr/bin/env python3
from __future__ import annotations
"""
VELOCITY — 2D car dodge game
Avoid incoming traffic. Constant speed. Simple.

Requirements:
    pip install pygame
"""

import pygame
import random
import sys

from neurolabel.backends.velocity.render import (
    C_WHITE,
    C_DIM,
    C_PLAYER,
    C_OBS,
    draw_car,
    draw_road_background,
    make_default_stripes,
)
from game_engine import (
    Player, Obstacle,
    WIDTH, HEIGHT, FPS,
    LANE_COUNT, LANE_CENTERS,
    SPAWN_INTERVAL, CAR_W, CAR_H, OBS_W, OBS_H,
)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("VELOCITY")
    clock  = pygame.time.Clock()

    try:
        font_score = pygame.font.SysFont("Courier New", 38, bold=True)
        font_title = pygame.font.SysFont("Courier New", 52, bold=True)
        font_sub   = pygame.font.SysFont("Courier New", 17)
    except Exception:
        font_score = pygame.font.SysFont(None, 38)
        font_title = pygame.font.SysFont(None, 52)
        font_sub   = pygame.font.SysFont(None, 17)

    stripes = make_default_stripes()

    obs_colors = {}  # id(obstacle) -> color

    def reset():
        obs_colors.clear()
        return Player(), [], 0, 0

    player, obstacles, score, spawn_t = reset()
    best   = 0
    state  = "playing"  # title | playing | dead
    blink  = 0
    key_cd = 0

    pygame.key.set_repeat(0, 0)

    while True:
        clock.tick(FPS)
        blink += 1

        # ── Events ──────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if state == "title":
                    state = "playing"
                elif state == "playing" and key_cd <= 0:
                    if event.key in (pygame.K_LEFT,  pygame.K_a): player.move(-1); key_cd = 6
                    if event.key in (pygame.K_RIGHT, pygame.K_d): player.move( 1); key_cd = 6
                elif state == "dead":
                    player, obstacles, score, spawn_t = reset()
                    state = "playing"

        if key_cd > 0:
            key_cd -= 1

        # ── Update ──────────────────────
        for s in stripes:
            s.update()

        if state == "playing":
            score  += 1
            spawn_t += 1
            player.update()

            if spawn_t >= SPAWN_INTERVAL:
                spawn_t = 0
                o = Obstacle(random.randint(0, LANE_COUNT - 1))
                obs_colors[id(o)] = random.choice(C_OBS)
                obstacles.append(o)

            for o in obstacles:
                o.update()
            obstacles = [o for o in obstacles if not o.gone()]

            px, py, pw, ph = player.rect()
            if any(px < ox + ow and px + pw > ox and py < oy + oh and py + ph > oy
                   for ox, oy, ow, oh in (o.rect() for o in obstacles)):
                if score > best:
                    best = score
                state = "dead"

        # ── Draw ────────────────────────
        draw_road_background(screen)

        # Lane stripes
        for s in stripes:
            s.draw(screen)

        # Obstacles + player
        for o in obstacles:
            color = obs_colors.get(id(o), C_OBS[0])
            draw_car(screen, int(o.x), int(o.y), OBS_W, OBS_H, color, False)
        if state in ("playing", "dead"):
            draw_car(screen, int(player.x), int(player.y), CAR_W, CAR_H, C_PLAYER, True)

        # ── HUD ─────────────────────────
        if state == "playing":
            # Score — top center, small and unobtrusive
            sc = font_score.render(f"{score:05d}", True, C_WHITE)
            screen.blit(sc, (WIDTH // 2 - sc.get_width() // 2, 16))

        # ── Overlays ────────────────────
        if state in ("title", "dead"):
            dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 155))
            screen.blit(dim, (0, 0))

            label = "VELOCITY" if state == "title" else "GAME OVER"
            t = font_title.render(label, True, C_PLAYER)
            screen.blit(t, (WIDTH//2 - t.get_width()//2, HEIGHT//2 - 90))

            if state == "dead":
                sc_txt = font_score.render(f"{score:05d}", True, C_WHITE)
                screen.blit(sc_txt, (WIDTH//2 - sc_txt.get_width()//2, HEIGHT//2 - 20))
                if score >= best and score > 0:
                    nb = font_sub.render("new best", True, C_PLAYER)
                    screen.blit(nb, (WIDTH//2 - nb.get_width()//2, HEIGHT//2 + 22))

            if blink % 60 < 42:
                hint = "press any key" if state == "title" else "press any key to retry"
                h = font_sub.render(hint, True, C_DIM)
                screen.blit(h, (WIDTH//2 - h.get_width()//2, HEIGHT//2 + 55))

            if state == "title":
                ctrl = font_sub.render("← →  or  A D  to switch lanes", True, C_DIM)
                screen.blit(ctrl, (WIDTH//2 - ctrl.get_width()//2, HEIGHT//2 + 90))

        pygame.display.flip()


if __name__ == "__main__":
    main()
