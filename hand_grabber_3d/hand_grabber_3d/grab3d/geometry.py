"""Pure-numpy geometry: pinhole camera, monocular depth from hand size, rotations, distance fields.

World convention (same as Panda3D): camera at the origin, +X right, +Y forward (into the scene), +Z up.
MediaPipe convention: x right, y DOWN, z away from the camera.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    hfov_deg: float

    @property
    def tan_h(self):
        return math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def f(self):
        return (self.width / 2.0) / self.tan_h

    @property
    def tan_v(self):
        return (self.height / 2.0) / self.f

    @property
    def vfov_deg(self):
        return math.degrees(2.0 * math.atan(self.tan_v))

    @property
    def cx(self):
        return self.width / 2.0

    @property
    def cy(self):
        return self.height / 2.0


def mp_to_panda(v):
    """MediaPipe axes (x right, y down, z away) -> Panda axes (x right, y forward, z up)."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


_I, _J = np.triu_indices(21, 1)


def estimate_depth(px, wl_xy, f):
    """Distance (m) of a hand from the camera.

    Every pair of landmarks gives d_px = f * d_world / Z. Solve for 1/Z in the least-squares sense
    using the metric hand model (world landmarks) and the measured pixel distances.
    """
    dpx = np.linalg.norm(px[_I] - px[_J], axis=1)
    dw = np.linalg.norm(wl_xy[_I] - wl_xy[_J], axis=1)
    denom = f * float(np.sum(dw * dw))
    if denom < 1e-12:
        return None
    s = float(np.sum(dpx * dw)) / denom
    if s <= 1e-9:
        return None
    return 1.0 / s


def backproject(px, depth, intr):
    """Pixels + per-point depth (distance along the camera axis) -> 3D points in Panda axes."""
    px = np.asarray(px, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    x = (px[:, 0] - intr.cx) * depth / intr.f
    z = -(px[:, 1] - intr.cy) * depth / intr.f
    return np.stack([x, depth, z], axis=1)


def project(p3, intr):
    """Panda-axes 3D points -> pixels (inverse of backproject)."""
    p3 = np.asarray(p3, dtype=np.float64)
    u = intr.cx + intr.f * p3[:, 0] / p3[:, 1]
    v = intr.cy - intr.f * p3[:, 2] / p3[:, 1]
    return np.stack([u, v], axis=1)


def hand_frame(wl_p):
    """Orientation of the hand as a 3x3 matrix whose COLUMNS are the hand axes in world coordinates.
    y = wrist -> knuckles, z = palm normal, x = y cross z."""
    palm_c = wl_p[[5, 9, 13, 17]].mean(axis=0)
    y = palm_c - wl_p[0]
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y = y / ny
    z = np.cross(wl_p[5] - wl_p[17], y)
    nz = np.linalg.norm(z)
    if nz < 1e-6:
        return None
    z = z / nz
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def orthonormalize(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rot_log(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-7:
        return np.zeros(3)
    if abs(ang - math.pi) < 1e-3:
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
        return axis * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return w * ang


def rot_exp(w):
    """Rotation vector -> rotation matrix (Rodrigues)."""
    w = np.asarray(w, dtype=np.float64)
    ang = float(np.linalg.norm(w))
    if ang < 1e-9:
        return np.eye(3)
    k = w / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def pinch_ratio(p3):
    """thumb-tip <-> index-tip distance divided by palm length. Depth (Y) is down-weighted: it is noisier."""
    w = np.array([1.0, 0.5, 1.0])
    d = (p3[4] - p3[8]) * w
    palm = (p3[9] - p3[0]) * w
    return float(np.linalg.norm(d) / max(np.linalg.norm(palm), 1e-3))


def fit_velocity(ts, ps):
    """Least-squares slope of position over time (robust throw velocity)."""
    ts = np.asarray(ts, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    if len(ts) < 2:
        return np.zeros(3)
    tm = ts - ts.mean()
    den = float(np.sum(tm * tm))
    if den < 1e-9:
        return np.zeros(3)
    return (tm[:, None] * (ps - ps.mean(axis=0))).sum(axis=0) / den


# ---------------------------------------------------------------- signed distance fields (local frame)
def sdf_box(p, half):
    q = np.abs(p) - half
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(q.max()), 0.0))


def sdf_sphere(p, r):
    return float(np.linalg.norm(p) - r)


def sdf_cyl(p, r, hh):
    d = np.array([math.hypot(p[0], p[1]) - r, abs(p[2]) - hh])
    return float(min(max(d[0], d[1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0)))


def pick_nearest(items, point, radius, exclude=()):
    """Item whose surface is closest to `point` (and closer than `radius`), or None."""
    best, best_d = None, radius
    for it in items:
        if it in exclude:
            continue
        d = it.sdf(point)
        if d < best_d:
            best, best_d = it, d
    return best
