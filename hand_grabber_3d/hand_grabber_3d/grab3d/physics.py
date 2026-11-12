"""Bullet physics world (Panda3D). Metres, camera at the origin, +Y forward, +Z up.

The 'play volume' is bounded by invisible planes: a floor, two side planes that coincide with the
camera's left/right frustum edges (so nothing ever leaves the picture sideways), a near plane (the
'glass of the screen') and a far plane.
"""
import math
import random

import numpy as np
from panda3d.bullet import (BulletBoxShape, BulletCylinderShape, BulletPlaneShape,
                            BulletRigidBodyNode, BulletSphereShape, BulletWorld, ZUp)
from panda3d.core import BitMask32, Vec3

from .geometry import sdf_box, sdf_cyl, sdf_sphere
from .hands import COLLIDER_R, COLLIDER_W
from .panda_assets import make_material

PALETTE = [  # RGB, slightly desaturated "painted toy" colours
    (0.90, 0.25, 0.22), (0.96, 0.62, 0.15), (0.97, 0.82, 0.20), (0.30, 0.72, 0.38),
    (0.20, 0.60, 0.85), (0.32, 0.38, 0.85), (0.72, 0.36, 0.80), (0.92, 0.92, 0.90),
]
# density kg/m^3, friction, restitution (Bullet multiplies both by the surface's values)
MATERIALS = {
    "box": (650.0, 0.65, 0.22),   # wooden block
    "ball": (320.0, 0.85, 0.78),  # rubber ball
    "cyl": (520.0, 0.50, 0.28),   # wooden puck / can
}


def _v3(a):
    return Vec3(float(a[0]), float(a[1]), float(a[2]))


class Item:
    """A dynamic rigid body + its visual. Implements the interface used by GrabController."""

    def __init__(self, render, kind, dims, node, nodepath, visual, mass):
        self.render, self.kind, self.dims = render, kind, np.asarray(dims, dtype=np.float64)
        self.node, self.np, self.visual, self.mass = node, nodepath, visual, mass
        self.hl = 1.0
        self.blob = None

    # -- state
    def pos(self):
        p = self.np.getPos(self.render)
        return np.array([p.getX(), p.getY(), p.getZ()])

    def rot(self):
        m = self.np.getMat(self.render)
        return np.array([[m.getCell(i, j) for j in range(3)] for i in range(3)]).T   # columns = local axes

    def lin_vel(self):
        v = self.node.getLinearVelocity()
        return np.array([v.getX(), v.getY(), v.getZ()])

    def ang_vel(self):
        v = self.node.getAngularVelocity()
        return np.array([v.getX(), v.getY(), v.getZ()])

    def set_vel(self, v=None, w=None):
        if v is not None:
            self.node.setLinearVelocity(_v3(v))
        if w is not None:
            self.node.setAngularVelocity(_v3(w))

    # -- distance to the surface (negative inside)
    def sdf_local(self, q):
        if self.kind == "box":
            return sdf_box(q, self.dims)
        if self.kind == "ball":
            return sdf_sphere(q, self.dims[0])
        return sdf_cyl(q, self.dims[0], self.dims[1])

    def sdf(self, point):
        return self.sdf_local(self.rot().T @ (np.asarray(point, dtype=np.float64) - self.pos()))

    def radius(self):
        return float(self.dims.max()) if self.kind != "cyl" else float(max(self.dims[0], self.dims[1]))

    def highlight(self, level):
        if abs(level - self.hl) > 1e-3:
            self.hl = level
            self.visual.setColorScale(level, level, level, 1.0)


class PhysicsWorld:
    def __init__(self, render, cfg, intr, models, textures):
        self.render, self.cfg, self.intr = render, cfg, intr
        self.models, self.textures = models, textures
        th = math.radians(cfg.tilt_deg)
        self.up = np.array([0.0, math.sin(th), math.cos(th)])       # world "up" seen from the camera
        self.bullet = BulletWorld()
        self.gravity_on = True
        self.bullet.setGravity(_v3(-cfg.gravity * self.up))
        self.items = []
        self._t = 0.0
        self.floor_h = cfg.floor_depth
        self._floor = None
        self._build_static()

    # ------------------------------------------------------------------ static world
    def _plane(self, normal, const, name, friction, restitution):
        node = BulletRigidBodyNode(name)
        node.addShape(BulletPlaneShape(_v3(normal), float(const)))
        node.setFriction(friction)
        node.setRestitution(restitution)
        npath = self.render.attachNewNode(node)
        npath.setCollideMask(BitMask32.allOn())
        self.bullet.attachRigidBody(node)
        return npath, node

    def _build_static(self):
        t = self.intr.tan_h
        n = math.sqrt(1.0 + t * t)
        self._plane((1.0 / n, t / n, 0.0), 0.0, "wall_left", 0.2, 0.6)     # frustum left edge
        self._plane((-1.0 / n, t / n, 0.0), 0.0, "wall_right", 0.2, 0.6)   # frustum right edge
        self._plane((0.0, 1.0, 0.0), self.cfg.near_wall, "wall_near", 0.2, 0.6)
        self._plane((0.0, -1.0, 0.0), -self.cfg.far_wall, "wall_far", 0.2, 0.6)
        self.set_floor(self.cfg.floor_depth)

    def set_floor(self, h):
        self.floor_h = float(h)
        if self._floor is not None:
            npath, node = self._floor
            self.bullet.removeRigidBody(node)
            npath.removeNode()
        self._floor = self._plane(self.up, -self.floor_h, "floor", 1.0, 1.0)

    def toggle_gravity(self):
        self.gravity_on = not self.gravity_on
        g = -self.cfg.gravity * self.up if self.gravity_on else np.zeros(3)
        self.bullet.setGravity(_v3(g))
        return self.gravity_on

    # ------------------------------------------------------------------ objects
    def spawn(self, kind=None, pos=None):
        rng = random
        kind = kind or rng.choices(["box", "ball", "cyl"], weights=[0.45, 0.35, 0.20])[0]
        color = rng.choice(PALETTE)
        density, friction, restitution = MATERIALS[kind]

        if kind == "box":
            s = rng.uniform(0.02, 0.037)
            dims = np.array([s, s * (rng.choice([1.0, 1.0, 1.5, 2.0])), s * (rng.choice([1.0, 1.0, 1.5]))])
            shape = BulletBoxShape(_v3(dims))
            shape.setMargin(0.002)
            volume = 8.0 * dims.prod()
            scale = tuple(2.0 * dims)
        elif kind == "ball":
            r = rng.uniform(0.022, 0.036)
            dims = np.array([r, r, r])
            shape = BulletSphereShape(r)
            volume = 4.0 / 3.0 * math.pi * r ** 3
            scale = (r, r, r)
        else:
            r, hh = rng.uniform(0.018, 0.03), rng.uniform(0.02, 0.04)
            dims = np.array([r, hh, 0.0])
            shape = BulletCylinderShape(r, 2.0 * hh, ZUp)
            shape.setMargin(0.002)
            volume = math.pi * r * r * 2.0 * hh
            scale = (r, r, hh)

        mass = density * volume
        node = BulletRigidBodyNode(f"item_{kind}")
        node.addShape(shape)
        node.setMass(mass)
        node.setFriction(friction)
        node.setRestitution(restitution)
        node.setLinearDamping(0.02)
        node.setAngularDamping(0.25 if kind == "ball" else 0.1)
        node.setDeactivationEnabled(False)

        if pos is None:
            y = rng.uniform(0.55, 1.05)
            half_w = y * self.intr.tan_h * 0.85
            pos = (rng.uniform(-half_w, half_w), y, y * self.intr.tan_v + 0.10 + rng.uniform(0.0, 0.12))
        npath = self.render.attachNewNode(node)
        npath.setPos(float(pos[0]), float(pos[1]), float(pos[2]))
        npath.setHpr(rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360))
        npath.setCollideMask(BitMask32.allOn())
        self.bullet.attachRigidBody(node)

        visual = self.models[kind].copyTo(npath)
        visual.setScale(*scale)
        visual.setTexture(self.textures["paint" if kind == "box" else "stripe"], 1)
        visual.setMaterial(make_material(color, 0.30 if kind == "box" else 0.55, 28 if kind == "box" else 60), 1)
        visual.setColor(color[0], color[1], color[2], 1.0)
        item = Item(self.render, kind, dims, node, npath, visual, mass)
        self.items.append(item)
        return item

    def remove(self, item):
        if item in self.items:
            self.items.remove(item)
        self.bullet.removeRigidBody(item.node)
        if item.blob is not None:
            item.blob.removeNode()
        item.np.removeNode()

    def reset(self):
        for it in list(self.items):
            self.remove(it)

    def _prune(self, protect):
        while len(self.items) > self.cfg.max_objects:
            victim = next((it for it in self.items if it not in protect), None)
            if victim is None:
                break
            self.remove(victim)

    # ------------------------------------------------------------------ step
    def update(self, dt, protect=()):
        self._t += dt
        if self._t >= self.cfg.spawn_interval:
            self._t = 0.0
            if len(self.items) < self.cfg.max_objects:
                self.spawn()
        self._prune(protect)
        self.bullet.doPhysics(min(dt, 1.0 / 30.0), 10, 1.0 / 240.0)
        for it in list(self.items):                           # safety net for anything that escaped
            if np.linalg.norm(it.pos()) > 8.0:
                self.remove(it)


class HandColliders:
    """Every hand is a cloud of kinematic spheres (fingertips, joints, bones, palm).
    They shove objects around with real contact forces, so you can bat a ball with the back of your hand."""

    def __init__(self, render, bullet):
        self.render, self.bullet = render, bullet
        self.pools = {}

    def _create(self, label, pts, now):
        nodes, paths = [], []
        for r, p in zip(COLLIDER_R, pts):
            node = BulletRigidBodyNode(f"hand_{label}")
            node.addShape(BulletSphereShape(float(r)))
            node.setKinematic(True)
            node.setFriction(1.0)
            node.setRestitution(0.1)
            node.setDeactivationEnabled(False)
            node.setCollisionResponse(False)
            npath = self.render.attachNewNode(node)
            npath.setPos(float(p[0]), float(p[1]), float(p[2]))
            npath.setCollideMask(BitMask32.allOn())
            self.bullet.attachRigidBody(node)
            nodes.append(node)
            paths.append(npath)
        return {"nodes": nodes, "paths": paths, "on": False, "enable_at": now + 0.25, "present": True}

    @staticmethod
    def _response(pool, flag):
        if pool["on"] != flag:
            for node in pool["nodes"]:
                node.setCollisionResponse(flag)
            pool["on"] = flag

    def update(self, hands, hidden, now):
        seen = set()
        for h in hands:
            seen.add(h.label)
            pts = COLLIDER_W @ h.p3
            pool = self.pools.get(h.label)
            if pool is None:
                pool = self.pools[h.label] = self._create(h.label, pts, now)
            elif not pool["present"]:                          # hand re-entered: teleport silently first
                self._response(pool, False)
                pool["present"] = True
                pool["enable_at"] = now + 0.2
            for npath, p in zip(pool["paths"], pts):
                npath.setPos(float(p[0]), float(p[1]), float(p[2]))
            self._response(pool, h.label not in hidden and now >= pool["enable_at"])
        for label, pool in self.pools.items():
            if label not in seen:
                self._response(pool, False)
                pool["present"] = False
