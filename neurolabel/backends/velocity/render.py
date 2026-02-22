from __future__ import annotations

"""Shared pygame rendering primitives for full-size Velocity UIs."""

import pygame

from game_engine import HEIGHT, WIDTH, ROAD_LEFT, ROAD_RIGHT, LANE_WIDTH, ROAD_SPEED

# Shared palette (car_game.py + data_recorder.py)
C_BG = (10, 10, 16)
C_ROAD = (20, 20, 30)
C_KERB = (28, 28, 40)
C_STRIPE = (45, 45, 65)
C_EDGE = (55, 55, 80)
C_WHITE = (255, 255, 255)
C_DIM = (160, 160, 180)
C_PLAYER = (0, 215, 255)
C_GLOW = (0, 100, 160)
C_OBS = [(255, 65, 85), (255, 155, 20), (170, 65, 255)]
C_REC = (255, 50, 50)


def draw_car(surf, cx, cy, w, h, color, is_player):
    """Draw the full-size Velocity car sprite (shared by game + recorder)."""
    rx, ry = cx - w // 2, cy - h // 2

    gs = 20
    glow = pygame.Surface((w + gs * 2, h + gs * 2), pygame.SRCALPHA)
    pygame.draw.rect(glow, (*color, 40), (gs, gs, w, h), border_radius=10)
    surf.blit(glow, (rx - gs, ry - gs))

    pygame.draw.rect(surf, color, (rx, ry, w, h), border_radius=9)

    ww = w - 10
    dark = (0, 30, 45) if is_player else (35, 8, 8)

    if is_player:
        pygame.draw.rect(surf, dark, (cx - ww // 2, ry + 10, ww, 15), border_radius=4)
        pygame.draw.rect(surf, dark, (cx - ww // 2, ry + h - 25, ww, 13), border_radius=4)
        pygame.draw.ellipse(surf, (210, 255, 255), (rx + 4, ry + 5, 10, 6))
        pygame.draw.ellipse(surf, (210, 255, 255), (rx + w - 14, ry + 5, 10, 6))
        pygame.draw.ellipse(surf, (255, 55, 55), (rx + 4, ry + h - 11, 10, 6))
        pygame.draw.ellipse(surf, (255, 55, 55), (rx + w - 14, ry + h - 11, 10, 6))
    else:
        pygame.draw.rect(surf, dark, (cx - ww // 2, ry + 10, ww, 13), border_radius=4)
        pygame.draw.rect(surf, dark, (cx - ww // 2, ry + h - 23, ww, 11), border_radius=4)
        pygame.draw.ellipse(surf, (255, 195, 100), (rx + 4, ry + h - 11, 10, 6))
        pygame.draw.ellipse(surf, (255, 195, 100), (rx + w - 14, ry + h - 11, 10, 6))


class Stripe:
    """Animated lane stripe for the full-size Velocity road renderer."""

    def __init__(self, x, y):
        self.x, self.y = x, float(y)

    def update(self):
        self.y += ROAD_SPEED
        if self.y > HEIGHT + 40:
            self.y -= HEIGHT + 80

    def draw(self, surf):
        pygame.draw.rect(surf, C_STRIPE, (self.x - 2, int(self.y), 4, 38))


def make_default_stripes() -> list[Stripe]:
    """Build the default lane-divider stripe set used by Velocity screens."""
    return [
        Stripe(lx, y)
        for lx in [ROAD_LEFT + LANE_WIDTH, ROAD_LEFT + LANE_WIDTH * 2]
        for y in range(0, HEIGHT + 100, 90)
    ]


def draw_road_background(screen) -> None:
    """Draw the full-size Velocity road background (kerbs, road, edges)."""
    screen.fill(C_BG)
    pygame.draw.rect(screen, C_KERB, (0, 0, ROAD_LEFT, HEIGHT))
    pygame.draw.rect(screen, C_KERB, (ROAD_RIGHT, 0, WIDTH - ROAD_RIGHT, HEIGHT))
    pygame.draw.rect(screen, C_ROAD, (ROAD_LEFT, 0, ROAD_RIGHT - ROAD_LEFT, HEIGHT))
    pygame.draw.line(screen, C_EDGE, (ROAD_LEFT, 0), (ROAD_LEFT, HEIGHT), 2)
    pygame.draw.line(screen, C_EDGE, (ROAD_RIGHT, 0), (ROAD_RIGHT, HEIGHT), 2)

