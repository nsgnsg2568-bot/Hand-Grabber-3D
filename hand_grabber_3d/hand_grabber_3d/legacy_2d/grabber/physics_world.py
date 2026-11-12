"""2D rigid-body world (pymunk). Screen coordinates: y grows downward, so gravity is +y."""
import math
import random
from dataclasses import dataclass

import pymunk

# BGR colours (OpenCV order)
PALETTE = [
    (80, 80, 255), (60, 180, 255), (70, 220, 255), (110, 230, 140),
    (230, 200, 70), (255, 130, 90), (230, 100, 200), (190, 120, 255),
]


@dataclass
class Item:
    body: pymunk.Body
    shape: pymunk.Shape
    kind: str
    color: tuple


class PhysicsWorld:
    def __init__(self, width, height, cfg):
        self.w, self.h, self.cfg = width, height, cfg
        self.space = pymunk.Space()
        self.space.gravity = (0, cfg.gravity)
        self.space.iterations = 20
        self.items = []
        self._spawn_timer = 0.0
        self._build_walls()

    # ------------------------------------------------------------------ setup
    def _build_walls(self):
        t = 40  # wall half-thickness (segments are drawn with a radius)
        static = self.space.static_body
        segs = [
            ((-t, self.h + t), (self.w + t, self.h + t)),      # floor
            ((-t, -4000), (-t, self.h + t)),                    # left wall
            ((self.w + t, -4000), (self.w + t, self.h + t)),    # right wall
        ]
        for a, b in segs:
            s = pymunk.Segment(static, a, b, t)
            s.friction = 0.8
            s.elasticity = 0.4
            self.space.add(s)

    # --------------------------------------------------------------- spawning
    def spawn(self, x=None, y=None):
        kind = random.choice(["ball", "ball", "box", "box", "tri"])
        color = random.choice(PALETTE)
        x = random.uniform(70, self.w - 70) if x is None else x
        y = -70 if y is None else y

        if kind == "ball":
            r = random.randint(18, 32)
            mass = r * r * 0.02
            body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, r))
            shape = pymunk.Circle(body, r)
            shape.elasticity = 0.75
        elif kind == "box":
            s = random.randint(38, 70)
            mass = s * s * 0.02
            body = pymunk.Body(mass, pymunk.moment_for_box(mass, (s, s)))
            shape = pymunk.Poly.create_box(body, (s, s))
            shape.elasticity = 0.35
        else:
            s = random.randint(28, 44)
            verts = [(-s, s * 0.6), (s, s * 0.6), (0, -s * 1.1)]
            mass = s * s * 0.03
            body = pymunk.Body(mass, pymunk.moment_for_poly(mass, verts))
            shape = pymunk.Poly(body, verts)
            shape.elasticity = 0.3

        shape.friction = 0.7
        body.position = (x, y)
        body.angle = random.uniform(0, math.tau)
        body.velocity = (random.uniform(-80, 80), random.uniform(0, 120))
        body.angular_velocity = random.uniform(-3, 3)
        self.space.add(body, shape)
        self.items.append(Item(body, shape, kind, color))

        while len(self.items) > self.cfg.max_objects:
            old = self.items.pop(0)
            self.space.remove(old.body, old.shape)

    def reset(self):
        for it in self.items:
            self.space.remove(it.body, it.shape)
        self.items.clear()

    def toggle_gravity(self):
        on = self.space.gravity[1] == 0
        self.space.gravity = (0, self.cfg.gravity if on else 0)
        return on

    # ------------------------------------------------------------------ query
    def pick(self, point, radius, exclude=()):
        """Closest dynamic body within `radius` of `point` (None if nothing is close)."""
        best, best_d = None, 1e9
        for info in self.space.point_query(point, radius, pymunk.ShapeFilter()):
            body = info.shape.body
            if body.body_type != pymunk.Body.DYNAMIC or body in exclude:
                continue
            if info.distance < best_d:
                best, best_d = body, info.distance
        return best

    # ------------------------------------------------------------------- step
    def update(self, dt):
        self._spawn_timer += dt
        if self._spawn_timer >= self.cfg.spawn_interval:
            self._spawn_timer = 0.0
            if len(self.items) < self.cfg.max_objects:
                self.spawn()

        steps = max(1, int(round(dt / (1 / 120))))
        h = dt / steps
        for _ in range(steps):
            self.space.step(h)
