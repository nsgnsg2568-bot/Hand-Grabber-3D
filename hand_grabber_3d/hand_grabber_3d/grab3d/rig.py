"""Per-render-frame smoothing of hand data (the camera delivers ~30 Hz, we render at 60 Hz).
Kinematic hand colliders must move continuously, otherwise they would shove objects every other frame."""
import math

from .hands import Hand3D


class HandRig:
    def __init__(self, rate=30.0):
        self.rate = rate
        self._state = {}

    def update(self, hands, dt):
        a = 1.0 - math.exp(-self.rate * dt)
        out, seen = [], set()
        for h in hands:
            seen.add(h.label)
            st = self._state.get(h.label)
            if st is None:
                st = self._state[h.label] = h.copy()
            else:
                st.p3 += (h.p3 - st.p3) * a
                st.px = h.px
                st.R = h.R
                st.pinch_ratio = h.pinch_ratio
                st.pinching = h.pinching
                st.depth = h.depth
                st.t = h.t
            out.append(st)
        for k in list(self._state):
            if k not in seen:
                del self._state[k]
        return out
