"""
Pure game logic for VELOCITY — no pygame dependency.
Used by simulator, data recorder, and synthetic data generator.
"""

import random

# ─────────────────────────────────────────
# Constants (mirrored from car_game.py)
# ─────────────────────────────────────────
WIDTH, HEIGHT = 480, 720
FPS = 60

LANE_COUNT = 3
ROAD_LEFT = 90
ROAD_RIGHT = 390
LANE_WIDTH = (ROAD_RIGHT - ROAD_LEFT) // LANE_COUNT
LANE_CENTERS = [ROAD_LEFT + LANE_WIDTH * i + LANE_WIDTH // 2 for i in range(LANE_COUNT)]

ROAD_SPEED = 7.0
SPAWN_INTERVAL = 55

CAR_W, CAR_H = 38, 64
OBS_W, OBS_H = 38, 60


# ─────────────────────────────────────────
# Game objects
# ─────────────────────────────────────────

class Player:
    def __init__(self):
        self.lane = 1
        self.x = float(LANE_CENTERS[1])
        self.y = float(HEIGHT - 120)
        self.target_x = self.x

    def move(self, d):
        """Move lane. d=-1 for left, d=1 for right."""
        nl = self.lane + d
        if 0 <= nl < LANE_COUNT:
            self.lane = nl
            self.target_x = float(LANE_CENTERS[nl])

    def update(self):
        self.x += (self.target_x - self.x) * 0.20

    def rect(self):
        """Return (x, y, w, h) tuple for collision detection."""
        return (int(self.x) - CAR_W // 2 + 5, int(self.y) - CAR_H // 2 + 5,
                CAR_W - 10, CAR_H - 10)


class Obstacle:
    def __init__(self, lane, rng=None):
        self.lane = lane
        self.x = float(LANE_CENTERS[lane])
        self.y = float(-OBS_H)

    def update(self):
        self.y += ROAD_SPEED

    def gone(self):
        return self.y > HEIGHT + OBS_H

    def rect(self):
        """Return (x, y, w, h) tuple for collision detection."""
        return (int(self.x) - OBS_W // 2 + 5, int(self.y) - OBS_H // 2 + 5,
                OBS_W - 10, OBS_H - 10)


class GameState:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.player = Player()
        self.obstacles = []
        self.score = 0
        self.spawn_timer = 0
        self.alive = True
        self.frame = 0

    def step(self, decision=0):
        """Advance game by one frame. decision: -1=left, 0=stay, 1=right."""
        if not self.alive:
            return

        if decision == -1:
            self.player.move(-1)
        elif decision == 1:
            self.player.move(1)

        self.player.update()
        self.score += 1
        self.spawn_timer += 1
        self.frame += 1

        # Spawn obstacles
        if self.spawn_timer >= SPAWN_INTERVAL:
            self.spawn_timer = 0
            self.obstacles.append(Obstacle(self.rng.randint(0, LANE_COUNT - 1)))

        # Update obstacles
        for o in self.obstacles:
            o.update()
        self.obstacles = [o for o in self.obstacles if not o.gone()]

        # Check collisions
        if self._check_collision():
            self.alive = False

    def _check_collision(self):
        px, py, pw, ph = self.player.rect()
        for o in self.obstacles:
            ox, oy, ow, oh = o.rect()
            if (px < ox + ow and px + pw > ox and py < oy + oh and py + ph > oy):
                return True
        return False

    def encode(self):
        """Encode current state as dict for dataset/replay."""
        obs_list = [[o.lane, o.y / HEIGHT] for o in self.obstacles]
        return {
            "lane": self.player.lane,
            "obs": obs_list,
            "alive": self.alive,
            "score": self.score,
            "frame": self.frame
        }

    def get_nearest_obstacles(self):
        """Return normalized distance to nearest obstacle in each lane. 1.0 = no obstacle."""
        distances = [1.0] * LANE_COUNT
        player_y = self.player.y
        for o in self.obstacles:
            if o.y < player_y:  # Only obstacles ahead (above player on screen)
                norm_dist = (player_y - o.y) / HEIGHT
                if norm_dist < distances[o.lane]:
                    distances[o.lane] = norm_dist
        return distances
