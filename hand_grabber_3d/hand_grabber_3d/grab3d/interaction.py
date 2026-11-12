"""Pinch -> grab -> carry (with wrist rotation) -> throw. Pure Python + numpy: works with any items
that implement pos(), rot(), ang_vel(), set_vel(v, w), sdf(point), sdf_local(q)."""
from collections import deque

import numpy as np

from .geometry import fit_velocity, pick_nearest, rot_log


class _Hold:
    def __init__(self, item, grab_local, r_rel):
        self.item = item
        self.grab_local = grab_local      # grab point in the item's local frame
        self.r_rel = r_rel                # item orientation relative to the hand


class GrabController:
    GRIP_DEPTH = -0.010                   # the pinch point sits this deep inside the surface

    def __init__(self, world, cfg):
        self.world, self.cfg = world, cfg
        self.holds = {}                   # hand label -> _Hold
        self.hist = {}                    # hand label -> deque[(t, palm)]
        self.candidates = {}              # hand label -> item that a pinch would grab right now
        self.released_at = {}             # hand label -> time of the last release
        self.grabs = 0
        self.last_throw = 0.0
        self.best_throw = 0.0

    # ------------------------------------------------------------------ queries
    def held_items(self):
        return {h.item for h in self.holds.values()}

    def hidden_labels(self, now, cooldown=0.45):
        """Hands whose fingertip colliders must stay inactive (holding something / just let go)."""
        hid = set(self.holds)
        for label, t in self.released_at.items():
            if now - t < cooldown:
                hid.add(label)
        return hid

    def reset(self):
        self.holds.clear()
        self.candidates.clear()

    # -------------------------------------------------- called when a NEW camera result arrives
    def on_hands(self, hands, t):
        labels = {h.label for h in hands}
        for label in list(self.holds):
            if label not in labels:
                self._release(label, t)
        for label in list(self.hist):
            if label not in labels:
                del self.hist[label]

        for h in hands:
            hist = self.hist.setdefault(h.label, deque(maxlen=12))
            hist.append((t, h.palm.copy()))
            hold = self.holds.get(h.label)
            if hold is not None and hold.item not in self.world.items:      # item was recycled
                self.holds.pop(h.label, None)
                hold = None
            if h.pinching:
                if hold is None:
                    item = pick_nearest(self.world.items, h.pinch_pt, self.cfg.grab_radius,
                                        exclude=self.held_items())
                    if item is not None:
                        self._grab(h, item)
            elif hold is not None:
                self._release(h.label, t)

    def _grab(self, hand, item):
        pos, R = item.pos(), item.rot()
        local = R.T @ (hand.pinch_pt - pos)
        if item.sdf_local(local) > self.GRIP_DEPTH:      # pinch point is outside: slide it onto the surface
            lo, hi = 0.0, 1.0
            for _ in range(18):
                mid = 0.5 * (lo + hi)
                if item.sdf_local(local * mid) < self.GRIP_DEPTH:
                    lo = mid
                else:
                    hi = mid
            local = local * lo
        self.holds[hand.label] = _Hold(item, local, hand.R.T @ R)
        self.grabs += 1

    def _hand_velocity(self, label, now):
        hist = self.hist.get(label)
        if not hist:
            return np.zeros(3)
        recent = [(t, p) for t, p in hist if now - t <= 0.14]
        if len(recent) < 2 or recent[-1][0] - recent[0][0] < 0.03:
            return np.zeros(3)
        ts, ps = zip(*recent)
        return fit_velocity(ts, ps)

    def _release(self, label, now):
        hold = self.holds.pop(label, None)
        self.released_at[label] = now
        if hold is None:
            return
        v = self._hand_velocity(label, now)
        v[1] *= self.cfg.depth_vel_scale                 # depth speed is the noisiest component
        v = v * self.cfg.throw_scale
        speed = float(np.linalg.norm(v))
        if speed > self.cfg.max_speed:
            v *= self.cfg.max_speed / speed
            speed = self.cfg.max_speed
        hold.item.set_vel(v, hold.item.ang_vel())
        self.last_throw = speed
        self.best_throw = max(self.best_throw, speed)

    # ------------------------------------------------------------ called EVERY render frame
    def drive(self, hands, dt):
        by_label = {h.label: h for h in hands}
        cfg = self.cfg
        for label, hold in list(self.holds.items()):
            hand = by_label.get(label)
            if hand is None or hold.item not in self.world.items:
                continue
            item = hold.item
            pos, R = item.pos(), item.rot()
            r = R @ hold.grab_local
            err = hand.pinch_pt - (pos + r)
            if cfg.rotate_with_hand:
                w = rot_log((hand.R @ hold.r_rel) @ R.T) * cfg.rot_gain
                n = float(np.linalg.norm(w))
                if n > 25.0:
                    w *= 25.0 / n
            else:
                w = np.zeros(3)
            v = err * cfg.grab_gain - np.cross(w, r)        # so that the GRAB POINT follows the pinch
            n = float(np.linalg.norm(v))
            if n > cfg.max_speed:
                v *= cfg.max_speed / n
            item.set_vel(v, w)

        self.candidates = {}
        held = self.held_items()
        for h in hands:
            if h.label not in self.holds:
                c = pick_nearest(self.world.items, h.pinch_pt, cfg.grab_radius, exclude=held)
                if c is not None:
                    self.candidates[h.label] = c
