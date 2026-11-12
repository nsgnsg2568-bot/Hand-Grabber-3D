"""Turns hand data into physics: pinch-grab / throw, plus open-hand fingertip pushers."""
from collections import deque

import pymunk
from pymunk import Vec2d

from .hand_tracker import FINGERTIPS

OFFSCREEN = (-5000, -5000)


class _Hold:
    def __init__(self, body, offset):
        self.body = body
        self.offset = offset


class _Tip:
    def __init__(self, space, radius):
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = OFFSCREEN
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = 0.9
        self.shape.elasticity = 0.2
        space.add(self.body, self.shape)
        self.active = False


class HandInteraction:
    def __init__(self, world, cfg):
        self.world, self.cfg = world, cfg
        self.holds = {}       # hand label -> _Hold
        self.history = {}     # hand label -> deque[(t, (x, y))] of palm positions
        self.tips = {}        # (label, fingertip idx) -> _Tip
        self.grabs = 0
        self.best_throw = 0.0
        self.last_throw = 0.0

    # --------------------------------------------------------------- helpers
    def held_bodies(self):
        return {h.body for h in self.holds.values()}

    def reset(self):
        self.holds.clear()

    def _clamp(self, v, limit):
        n = v.length
        return v * (limit / n) if n > limit else v

    def _palm_velocity(self, label, now):
        hist = self.history.get(label)
        if not hist or len(hist) < 2:
            return Vec2d(0, 0)
        recent = [(t, p) for t, p in hist if now - t <= 0.15]
        if len(recent) < 2:
            return Vec2d(0, 0)
        (t0, p0), (t1, p1) = recent[0], recent[-1]
        span = t1 - t0
        if span < 0.03:
            return Vec2d(0, 0)
        return Vec2d(p1[0] - p0[0], p1[1] - p0[1]) / span

    def _release(self, label, now, throw=True):
        hold = self.holds.pop(label, None)
        if hold is None or hold.body.space is None:
            return
        if throw:
            v = self._clamp(self._palm_velocity(label, now) * self.cfg.throw_scale, self.cfg.max_speed)
            hold.body.velocity = v
            self.last_throw = v.length
            self.best_throw = max(self.best_throw, self.last_throw)

    # ---------------------------------------------------------------- tips
    def _update_tip(self, key, target, dt):
        tip = self.tips.get(key)
        if tip is None:
            tip = self.tips[key] = _Tip(self.world.space, self.cfg.fingertip_radius)
        if target is None:
            if tip.active:
                tip.body.position = OFFSCREEN
                tip.body.velocity = (0, 0)
                tip.active = False
            return
        if not tip.active:
            tip.body.position = target
            tip.body.velocity = (0, 0)
            tip.active = True
        else:
            v = (Vec2d(*target) - tip.body.position) / max(dt, 1e-3)
            tip.body.velocity = self._clamp(v, 3500)

    # ---------------------------------------------------------------- update
    def update(self, hands, now, dt):
        labels = {h.label for h in hands}

        # hands that vanished: drop what they held (with their last velocity)
        for label in list(self.holds):
            if label not in labels:
                self._release(label, now)
        for label in list(self.history):
            if label not in labels:
                del self.history[label]

        for hand in hands:
            label = hand.label
            hist = self.history.setdefault(label, deque(maxlen=10))
            hist.append((now, hand.palm))
            hold = self.holds.get(label)

            if hand.pinching:
                if hold is not None and hold.body.space is None:   # object got recycled
                    self.holds.pop(label, None)
                    hold = None
                if hold is None:
                    body = self.world.pick(hand.pinch_point, self.cfg.grab_radius,
                                           exclude=self.held_bodies())
                    if body is not None:
                        offset = body.position - Vec2d(*hand.pinch_point)
                        hold = self.holds[label] = _Hold(body, offset)
                        self.grabs += 1
                if hold is not None:
                    target = Vec2d(*hand.pinch_point) + hold.offset
                    vel = (target - hold.body.position) * self.cfg.grab_gain
                    hold.body.velocity = self._clamp(vel, self.cfg.max_speed)
                    hold.body.angular_velocity *= 0.02 ** dt   # calm the spinning while held
            elif hold is not None:
                self._release(label, now)

            # open hand -> fingertips act as small solid pushers; pinching -> hide them
            for idx in FINGERTIPS:
                target = None if hand.pinching else hand.points[idx]
                self._update_tip((label, idx), target, dt)

        # tips of hands that left the frame
        for (label, idx) in list(self.tips):
            if label not in labels:
                self._update_tip((label, idx), None, dt)
