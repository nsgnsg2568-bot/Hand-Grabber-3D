"""Hand data structure + the 'sphere skeleton' used for hand colliders and the hand occluder."""
from dataclasses import dataclass

import numpy as np

FINGERTIPS = (4, 8, 12, 16, 20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


@dataclass
class Hand3D:
    label: str
    px: np.ndarray        # (21, 2) image pixels
    p3: np.ndarray        # (21, 3) metres, Panda axes, camera at the origin
    R: np.ndarray         # (3, 3) hand orientation, columns = hand axes in world coordinates
    pinch_ratio: float
    pinching: bool
    depth: float          # estimated distance of the hand from the camera (m)
    t: float              # capture time (perf_counter seconds)

    @property
    def pinch_pt(self):
        return (self.p3[4] + self.p3[8]) * 0.5

    @property
    def palm(self):
        return self.p3[[0, 5, 9, 13, 17]].mean(axis=0)

    def copy(self):
        return Hand3D(self.label, self.px.copy(), self.p3.copy(), self.R.copy(),
                      self.pinch_ratio, self.pinching, self.depth, self.t)


def _build_spec():
    spec = []
    for i in range(21):
        if i in FINGERTIPS:
            r = 0.0085
        elif i == 0:
            r = 0.026
        elif i in (5, 9, 13, 17):
            r = 0.012
        else:
            r = 0.0095
        spec.append(((i,), r))
    for a, b in [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                 (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 1)]:
        spec.append(((a, b), 0.0095))
    for idx in [(0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17), (1, 5)]:
        spec.append((idx, 0.02))
    spec.append(((0, 5, 9, 13, 17), 0.032))
    W = np.zeros((len(spec), 21))
    for k, (idx, _) in enumerate(spec):
        W[k, list(idx)] = 1.0 / len(idx)
    R = np.array([r for _, r in spec])
    return W, R


# collider_points = COLLIDER_W @ p3   (n, 3);   radii = COLLIDER_R (n,)
COLLIDER_W, COLLIDER_R = _build_spec()
