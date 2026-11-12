"""Pure-math tests (no webcam / Panda3D needed).   Run:  python tests/test_core.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grab3d import geometry as G
from grab3d import meshes as M
from grab3d import textures as T
from grab3d.filters import OneEuro, OneEuroVec


def fake_hand(rng):
    """A rough 21-point hand in metres (MediaPipe axes), ~19 cm long."""
    pts = np.zeros((21, 3))
    pts[0] = [0, 0.09, 0]
    for f, x in zip((5, 9, 13, 17), (-0.03, -0.01, 0.01, 0.03)):
        pts[f] = [x, 0.0, rng.normal(0, 0.004)]
        for k in range(1, 4):
            pts[f + k] = [x, -0.03 * k, rng.normal(0, 0.01)]
    for k in range(1, 5):
        pts[k] = [-0.04 - 0.02 * k, 0.07 - 0.025 * k, -0.01]
    return pts - pts.mean(axis=0)


def test_depth_roundtrip():
    intr = G.Intrinsics(1280, 720, 62.0)
    rng = np.random.default_rng(0)
    wl = fake_hand(rng)
    for Z in (0.3, 0.5, 0.8, 1.2):
        center = np.array([0.05, Z, -0.02])                  # Panda axes
        p_panda = G.mp_to_panda(wl) + center                 # true 3D points
        px = G.project(p_panda, intr)
        px_noisy = px + rng.normal(0, 0.7, px.shape)         # 0.7 px landmark noise
        est = G.estimate_depth(px_noisy, wl[:, :2], intr.f)
        assert abs(est - Z) / Z < 0.04, (Z, est)
        # back-projecting with the true per-point depth must land exactly on the pixels
        p3 = G.backproject(px, p_panda[:, 1], intr)
        assert np.allclose(p3, p_panda, atol=1e-9)
    print("depth/projection OK")


def test_mp_to_panda_is_rotation():
    A = G.mp_to_panda(np.eye(3))
    assert abs(np.linalg.det(A) - 1) < 1e-12
    print("axis conversion OK")


def test_rotations():
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = rng.normal(0, 1, 3)
        w *= rng.uniform(0.0, 3.0) / (np.linalg.norm(w) + 1e-9)
        R = G.rot_exp(w)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.allclose(G.rot_exp(G.rot_log(R)), R, atol=1e-6)
    R = G.rot_exp(np.array([0, 0, math.pi]))
    assert np.allclose(G.rot_exp(G.rot_log(R)), R, atol=1e-4)
    M3 = G.orthonormalize(G.rot_exp([0.3, 0.2, 0.1]) + 0.05 * rng.normal(size=(3, 3)))
    assert abs(np.linalg.det(M3) - 1) < 1e-9
    print("rotations OK")


def test_hand_frame_follows_rotation():
    rng = np.random.default_rng(2)
    wl = G.mp_to_panda(fake_hand(rng))
    R0 = G.hand_frame(wl)
    Rz = G.rot_exp([0.2, -0.4, 0.7])
    R1 = G.hand_frame(wl @ Rz.T)
    assert np.allclose(R1, Rz @ R0, atol=1e-9)
    print("hand frame OK")


def test_velocity_fit():
    ts = np.linspace(0, 0.12, 6)
    ps = np.stack([1.5 * ts + 0.2, -0.5 * ts, 0.0 * ts], 1)
    v = G.fit_velocity(ts, ps)
    assert np.allclose(v, [1.5, -0.5, 0.0], atol=1e-9)
    print("velocity fit OK")


def test_sdf():
    assert abs(G.sdf_box(np.array([0.5, 0, 0]), np.array([0.25, 0.25, 0.25])) - 0.25) < 1e-12
    assert G.sdf_box(np.zeros(3), np.array([0.25, 0.25, 0.25])) < 0
    assert abs(G.sdf_sphere(np.array([0, 0.3, 0]), 0.1) - 0.2) < 1e-12
    assert abs(G.sdf_cyl(np.array([0.3, 0, 0]), 0.1, 0.2) - 0.2) < 1e-12
    assert G.sdf_cyl(np.zeros(3), 0.1, 0.2) < 0
    print("sdf OK")


def _check_mesh(name, m):
    a, b, c = m.v[m.tri[:, 0]], m.v[m.tri[:, 1]], m.v[m.tri[:, 2]]
    fn = np.cross(b - a, c - a)
    area = np.linalg.norm(fn, axis=1)
    ok = area > 1e-12
    avg = m.n[m.tri[:, 0]] + m.n[m.tri[:, 1]] + m.n[m.tri[:, 2]]
    assert np.all((fn[ok] * avg[ok]).sum(1) > 0), name + ": inverted triangles"
    assert np.allclose(np.linalg.norm(m.n, axis=1), 1, atol=1e-6), name + ": normals not unit"
    assert m.tri.max() < len(m.v)
    print(f"mesh {name} OK ({len(m.v)} verts, {len(m.tri)} tris)")


def test_meshes():
    _check_mesh("sphere", M.sphere_mesh())
    _check_mesh("cylinder", M.cylinder_mesh())
    box = M.rounded_box_mesh((0.5, 0.5, 0.5), 0.12)
    _check_mesh("box", box)
    assert abs(np.abs(box.v).max() - 0.5) < 1e-9            # outer size is exactly the requested size
    _check_mesh("quad", M.quad_mesh(-1, 1, 0, 2))
    # a point on the closed surface must be at the right distance: sphere radius 1
    s = M.sphere_mesh(1.0)
    assert np.allclose(np.linalg.norm(s.v, axis=1), 1.0)


def test_textures():
    for arr in (T.paint_texture(), T.stripe_texture(), T.floor_texture(True), T.floor_texture(False),
                T.backdrop_texture(), T.blob_texture()):
        assert arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] in (3, 4)
    print("textures OK")


def test_filters():
    rng = np.random.default_rng(3)
    f = OneEuro(1.0, 0.0)
    out = [f(1.0 + rng.normal(0, 0.05), i / 30) for i in range(200)]
    assert np.std(out[50:]) < 0.03
    fv = OneEuroVec(1.0, 0.0)
    for i in range(10):
        r = fv(np.ones(63), i / 30)
    assert r.shape == (63,)
    print("filters OK")


# ------------------------------------------------------------------ grab / carry / throw simulation
class FakeItem:
    def __init__(self, pos, half=0.03):
        self.p = np.array(pos, float)
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.half = np.array([half] * 3)

    def pos(self): return self.p.copy()
    def rot(self): return self.R.copy()
    def ang_vel(self): return self.w.copy()
    def set_vel(self, v=None, w=None):
        if v is not None: self.v = np.array(v, float)
        if w is not None: self.w = np.array(w, float)
    def sdf_local(self, q): return G.sdf_box(np.asarray(q), self.half)
    def sdf(self, point): return self.sdf_local(self.R.T @ (np.asarray(point) - self.p))
    def step(self, dt):
        self.p += self.v * dt
        self.R = G.rot_exp(self.w * dt) @ self.R


class FakeWorld:
    def __init__(self, items): self.items = items


def make_hand(pinch_pt, pinching, R=None, t=0.0, label="Right"):
    from grab3d.hands import Hand3D
    pinch_pt = np.asarray(pinch_pt, float)
    p3 = np.zeros((21, 3)); p3[:] = pinch_pt
    p3[4] = pinch_pt + [-0.005, 0, 0]; p3[8] = pinch_pt + [0.005, 0, 0]
    p3[0] = pinch_pt + [0, 0, -0.09]
    return Hand3D(label, np.zeros((21, 2)), p3, np.eye(3) if R is None else R, 0.1, pinching, 0.6, t)


def test_grab_carry_throw():
    from grab3d.config import Config
    from grab3d.interaction import GrabController
    cfg = Config()
    item = FakeItem([0.0, 0.6, 0.0])
    world = FakeWorld([item])
    gc = GrabController(world, cfg)

    # open hand near the object: it is a grab candidate, nothing held
    hand = make_hand([0.05, 0.6, 0.0], False)
    gc.drive([hand], 1 / 60)
    assert gc.candidates.get("Right") is item and not gc.holds

    # pinch -> grab
    t = 0.0
    hand = make_hand([0.05, 0.6, 0.0], True, t=t)
    gc.on_hands([hand], t)
    assert "Right" in gc.holds and gc.grabs == 1

    # carry: hand moves in +X at 1 m/s for 1 s; the item must follow
    dt = 1 / 60
    for k in range(60):
        t += dt
        pos = np.array([0.05 + 1.0 * (k + 1) * dt, 0.6, 0.0])
        hand = make_hand(pos, True, t=t)
        gc.on_hands([hand], t)
        gc.drive([hand], dt)
        item.step(dt)
    gp = item.p + item.R @ gc.holds["Right"].grab_local
    assert np.linalg.norm(gp - hand.pinch_pt) < 0.03, np.linalg.norm(gp - hand.pinch_pt)

    # release while moving at 1 m/s -> thrown at ~ throw_scale * 1 m/s
    t += dt
    hand = make_hand(hand.pinch_pt + np.array([0.0167, 0, 0]), False, t=t)
    gc.on_hands([hand], t)
    assert not gc.holds
    assert abs(item.v[0] - 1.0 * cfg.throw_scale) < 0.25, item.v
    assert gc.best_throw > 1.0
    assert "Right" in gc.hidden_labels(t)          # colliders stay off right after a release
    print("grab/carry/throw OK  (thrown at %.2f m/s)" % np.linalg.norm(item.v))


def test_wrist_rotation_is_followed():
    from grab3d.config import Config
    from grab3d.interaction import GrabController
    cfg = Config()
    item = FakeItem([0.0, 0.6, 0.0])
    gc = GrabController(FakeWorld([item]), cfg)
    gc.on_hands([make_hand([0, 0.6, 0], True, t=0)], 0.0)
    Rt = G.rot_exp([0.0, 0.0, 0.9])                  # user turns the wrist by 0.9 rad about Z
    dt, t = 1 / 60, 0.0
    for _ in range(120):
        t += dt
        h = make_hand([0, 0.6, 0], True, R=Rt, t=t)
        gc.on_hands([h], t); gc.drive([h], dt); item.step(dt)
    err = np.linalg.norm(G.rot_log(Rt @ item.R.T))
    assert err < 0.05, err
    print("wrist rotation OK")


def test_nothing_to_grab_far_away():
    from grab3d.config import Config
    from grab3d.interaction import GrabController
    item = FakeItem([0.0, 0.6, 0.0])
    gc = GrabController(FakeWorld([item]), Config())
    gc.on_hands([make_hand([0.0, 0.6, 0.3], True, t=0)], 0.0)    # 30 cm away
    assert not gc.holds
    print("far pinch ignored OK")


def test_hand_lost_releases():
    from grab3d.config import Config
    from grab3d.interaction import GrabController
    item = FakeItem([0.0, 0.6, 0.0])
    gc = GrabController(FakeWorld([item]), Config())
    gc.on_hands([make_hand([0, 0.6, 0], True, t=0)], 0.0)
    assert gc.holds
    gc.on_hands([], 0.1)
    assert not gc.holds
    print("hand lost -> release OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL CORE TESTS PASSED")
