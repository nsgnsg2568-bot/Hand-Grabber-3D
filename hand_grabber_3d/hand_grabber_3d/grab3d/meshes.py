"""Procedural meshes as numpy arrays (no Panda3D needed, so they are unit-testable)."""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    v: np.ndarray    # (N, 3)
    n: np.ndarray    # (N, 3)
    uv: np.ndarray   # (N, 2)
    tri: np.ndarray  # (M, 3) int, counter-clockwise when seen from outside


def _fix_winding(v, n, tri):
    """Flip any triangle whose geometric normal disagrees with its vertex normals."""
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    avg = n[tri[:, 0]] + n[tri[:, 1]] + n[tri[:, 2]]
    flip = (fn * avg).sum(axis=1) < 0
    tri = tri.copy()
    t1 = tri[flip, 1].copy()
    tri[flip, 1] = tri[flip, 2]
    tri[flip, 2] = t1
    return tri


def sphere_mesh(r=1.0, rings=16, segs=32):
    th = np.linspace(0, np.pi, rings + 1)
    ph = np.linspace(0, 2 * np.pi, segs + 1)
    T, P = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
    v = n * r
    uv = np.stack([(P / (2 * np.pi)).ravel(), (1 - T / np.pi).ravel()], 1)
    idx = np.arange((rings + 1) * (segs + 1)).reshape(rings + 1, segs + 1)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tri = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    return Mesh(v, n, uv, _fix_winding(v, n, tri))


def cylinder_mesh(r=1.0, hh=1.0, segs=36):
    ang = np.linspace(0, 2 * np.pi, segs + 1)
    c, s = np.cos(ang), np.sin(ang)
    V, N, UV = [], [], []
    # side
    for j in range(segs + 1):
        V += [[r * c[j], r * s[j], hh], [r * c[j], r * s[j], -hh]]
        N += [[c[j], s[j], 0.0], [c[j], s[j], 0.0]]
        UV += [[j / segs, 1.0], [j / segs, 0.0]]
    tri = []
    for j in range(segs):
        a = 2 * j
        tri += [[a, a + 1, a + 2], [a + 2, a + 1, a + 3]]
    base = len(V)
    # caps
    for z, nz in ((hh, 1.0), (-hh, -1.0)):
        center = len(V)
        V.append([0.0, 0.0, z]); N.append([0.0, 0.0, nz]); UV.append([0.5, 0.5])
        for j in range(segs + 1):
            V.append([r * c[j], r * s[j], z]); N.append([0.0, 0.0, nz])
            UV.append([0.5 + 0.5 * c[j], 0.5 + 0.5 * s[j]])
        for j in range(segs):
            tri.append([center, center + 1 + j, center + 2 + j])
    v, n, uv = np.array(V), np.array(N), np.array(UV)
    tri = np.array(tri, dtype=np.int64)
    return Mesh(v, n, uv, _fix_winding(v, n, tri))


def _axis_samples(h, r):
    """Sample positions along one axis: dense inside the rounded edge, sparse on the flat part."""
    pos = [h - r + r * math.tan(math.radians(a)) for a in (0, 15, 30, 45)]
    return np.array(sorted([-x for x in pos] + pos))


def rounded_box_mesh(half=(0.5, 0.5, 0.5), r=0.1):
    """A box with rounded edges (catches highlights like a real toy block)."""
    H = np.array(half, dtype=np.float64)
    r = min(r, float(H.min()) * 0.95)
    samples = [_axis_samples(H[k], r) for k in range(3)]
    V, N, UV, T = [], [], [], []
    base = 0
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for sg in (1.0, -1.0):
            sb, sc = samples[b], samples[c]
            nb, nc = len(sb), len(sc)
            Pb, Pc = np.meshgrid(sb, sc, indexing="ij")
            P = np.zeros((nb, nc, 3))
            P[..., a] = sg * H[a]
            P[..., b] = Pb
            P[..., c] = Pc
            inner = np.clip(P, -(H - r), H - r)
            d = P - inner
            nrm = d / np.linalg.norm(d, axis=-1, keepdims=True)
            vert = inner + nrm * r
            uv = np.stack([(Pb / H[b] + 1) / 2, (Pc / H[c] + 1) / 2], -1)
            idx = base + np.arange(nb * nc).reshape(nb, nc)
            i00, i10 = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
            i11, i01 = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
            T.append(np.stack([i00, i10, i11], 1))
            T.append(np.stack([i00, i11, i01], 1))
            V.append(vert.reshape(-1, 3)); N.append(nrm.reshape(-1, 3)); UV.append(uv.reshape(-1, 2))
            base += nb * nc
    v, n, uv = np.concatenate(V), np.concatenate(N), np.concatenate(UV)
    tri = np.concatenate(T).astype(np.int64)
    return Mesh(v, n, uv, _fix_winding(v, n, tri))


def quad_mesh(x0, x1, y0, y1, z=0.0, uv_scale=(1.0, 1.0)):
    """Flat quad in the XY plane facing +Z."""
    v = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=np.float64)
    n = np.tile([0.0, 0.0, 1.0], (4, 1))
    uv = np.array([[0, 0], [uv_scale[0], 0], [uv_scale[0], uv_scale[1]], [0, uv_scale[1]]], dtype=np.float64)
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh(v, n, uv, _fix_winding(v, n, tri))
