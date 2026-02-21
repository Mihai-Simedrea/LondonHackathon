#!/usr/bin/env python3
"""
VELOCITY — 2D car dodge game
Avoid incoming traffic. Constant speed. Simple.

Requirements:
    pip install pygame
"""

import pygame
import random
import sys

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
WIDTH, HEIGHT = 480, 720
FPS           = 60

LANE_COUNT    = 3
ROAD_LEFT     = 90
ROAD_RIGHT    = 390
LANE_WIDTH    = (ROAD_RIGHT - ROAD_LEFT) // LANE_COUNT
LANE_CENTERS  = [ROAD_LEFT + LANE_WIDTH * i + LANE_WIDTH // 2 for i in range(LANE_COUNT)]

ROAD_SPEED    = 7.0   # constant — never changes
SPAWN_INTERVAL = 55   # frames between obstacle spawns (constant)

CAR_W, CAR_H  = 38, 64
OBS_W, OBS_H  = 38, 60

# Colors
C_BG        = (10,  10,  16)
C_ROAD      = (20,  20,  30)
C_KERB      = (28,  28,  40)
C_STRIPE    = (45,  45,  65)
C_EDGE      = (55,  55,  80)
C_WHITE     = (255, 255, 255)
C_DIM       = (160, 160, 180)
C_PLAYER    = (0,   215, 255)
C_GLOW      = (0,   100, 160)
C_OBS       = [(255, 65, 85), (255, 155, 20), (170, 65, 255)]

# ─────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────

def draw_car(surf, cx, cy, w, h, color, is_player):
    rx, ry = cx - w // 2, cy - h // 2

    # Soft glow behind car
    gs = 20
    glow = pygame.Surface((w + gs*2, h + gs*2), pygame.SRCALPHA)
    pygame.draw.rect(glow, (*color, 40), (gs, gs, w, h), border_radius=10)
    surf.blit(glow, (rx - gs, ry - gs))

    # Body
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


# ─────────────────────────────────────────
# Game objects
# ─────────────────────────────────────────

class Player:
    def __init__(self):
        self.lane     = 1
        self.x        = float(LANE_CENTERS[1])
        self.y        = float(HEIGHT - 120)
        self.target_x = self.x

    def move(self, d):
        nl = self.lane + d
        if 0 <= nl < LANE_COUNT:
            self.lane     = nl
            self.target_x = float(LANE_CENTERS[nl])

    def update(self):
        self.x += (self.target_x - self.x) * 0.20

    def rect(self):
        return pygame.Rect(int(self.x) - CAR_W//2 + 5, int(self.y) - CAR_H//2 + 5,
                           CAR_W - 10, CAR_H - 10)

    def draw(self, surf):
        draw_car(surf, int(self.x), int(self.y), CAR_W, CAR_H, C_PLAYER, True)


class Obstacle:
    def __init__(self, lane):
        self.x     = float(LANE_CENTERS[lane])
        self.y     = float(-OBS_H)
        self.color = random.choice(C_OBS)

    def update(self):
        self.y += ROAD_SPEED

    def gone(self):
        return self.y > HEIGHT + OBS_H

    def rect(self):
        return pygame.Rect(int(self.x) - OBS_W//2 + 5, int(self.y) - OBS_H//2 + 5,
                           OBS_W - 10, OBS_H - 10)

    def draw(self, surf):
        draw_car(surf, int(self.x), int(self.y), OBS_W, OBS_H, self.color, False)


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

    stripes = [
        Stripe(lx, y)
        for lx in [ROAD_LEFT + LANE_WIDTH, ROAD_LEFT + LANE_WIDTH * 2]
        for y in range(0, HEIGHT + 100, 90)
    ]

    def reset():
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
                obstacles.append(Obstacle(random.randint(0, LANE_COUNT - 1)))

            for o in obstacles:
                o.update()
            obstacles = [o for o in obstacles if not o.gone()]

            if any(player.rect().colliderect(o.rect()) for o in obstacles):
                if score > best:
                    best = score
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
        for o in obstacles:
            o.draw(screen)
        if state in ("playing", "dead"):
            player.draw(screen)

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