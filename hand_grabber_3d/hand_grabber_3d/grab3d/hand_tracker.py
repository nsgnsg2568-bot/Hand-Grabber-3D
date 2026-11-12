"""MediaPipe Hands -> metric 3D hands (Hand3D).

Pipeline per frame:
 1. 21 image landmarks (pixels) + 21 'world' landmarks (a metric hand model, metres).
 2. Depth of the hand from the camera: least squares over all landmark pairs (geometry.estimate_depth).
 3. Each landmark is back-projected through the pinhole camera at depth Z + its relative depth.
 4. One-Euro smoothing, pinch detection with hysteresis, and a hand orientation matrix.
"""
import sys

# MediaPipe optionally imports TensorFlow. This app never needs it, and a broken TensorFlow install
# (DLL errors) must not stop us, so we hide it. (A hidden module raises ModuleNotFoundError, which MediaPipe handles.)
sys.modules.setdefault("tensorflow", None)

import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except (ImportError, AttributeError) as e:  # pragma: no cover
    raise RuntimeError(
        "Could not load mediapipe.solutions.hands. Install the pinned version:\n"
        "    pip install mediapipe==0.10.14\n(Python 3.10-3.12)\n"
        f"Original error: {e!r}"
    ) from e

from .filters import OneEuro, OneEuroVec
from .geometry import (backproject, estimate_depth, hand_frame, mp_to_panda, orthonormalize,
                       pinch_ratio)
from .hands import Hand3D


class _State:
    def __init__(self):
        self.zf = OneEuro(min_cutoff=0.7, beta=1.5, d_cutoff=1.0)
        self.pf = OneEuroVec(min_cutoff=1.6, beta=7.0, d_cutoff=1.0)
        self.R = None
        self.ratio = None
        self.pinching = False
        self.last_Z = None


def _fallback_world(px, zpx):
    """If the world landmarks are missing: build a pseudo-metric hand from image landmarks."""
    pts = np.column_stack([px[:, 0], px[:, 1], zpx])
    palm = float(np.linalg.norm(pts[9] - pts[0])) + 1e-6
    return (pts - pts.mean(axis=0)) * (0.095 / palm)


class HandTracker:
    def __init__(self, cfg, intr):
        self.cfg, self.intr = cfg, intr
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.detection_conf,
            min_tracking_confidence=cfg.tracking_conf,
        )
        self._state = {}

    def close(self):
        self._hands.close()

    def process(self, frame_bgr, t):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks or not res.multi_handedness:
            self._state.clear()
            return []

        worlds = getattr(res, "multi_hand_world_landmarks", None)
        cfg, out, seen = self.cfg, [], set()
        for k, (lms, handed) in enumerate(zip(res.multi_hand_landmarks, res.multi_handedness)):
            label = handed.classification[0].label
            if label in seen:                              # both hands got the same label
                label += "2"
            seen.add(label)
            st = self._state.setdefault(label, _State())

            px = np.array([[l.x * w, l.y * h] for l in lms.landmark], dtype=np.float64)
            if worlds and k < len(worlds):
                wl = np.array([[l.x, l.y, l.z] for l in worlds[k].landmark], dtype=np.float64)
            else:
                wl = _fallback_world(px, np.array([l.z for l in lms.landmark]) * w)
            wl = wl * cfg.hand_scale

            z_raw = estimate_depth(px, wl[:, :2], self.intr.f)
            if z_raw is None:
                z_raw = st.last_Z or 0.6
            z = float(np.clip(st.zf(float(np.clip(z_raw, 0.12, 3.0)), t), 0.12, 3.0))
            st.last_Z = z

            dz = cfg.depth_sign * (wl[:, 2] - wl[:, 2].mean())
            depth = np.maximum(z + dz, 0.05)
            p3 = st.pf(backproject(px, depth, self.intr).ravel(), t).reshape(21, 3)

            wl_p = mp_to_panda(wl * np.array([1.0, 1.0, cfg.depth_sign]))
            r_new = hand_frame(wl_p)
            if r_new is not None:
                st.R = r_new if st.R is None else orthonormalize(0.65 * st.R + 0.35 * r_new)
            if st.R is None:
                st.R = np.eye(3)

            ratio = pinch_ratio(p3)
            st.ratio = ratio if st.ratio is None else 0.5 * st.ratio + 0.5 * ratio
            st.pinching = st.ratio < (cfg.pinch_off if st.pinching else cfg.pinch_on)

            out.append(Hand3D(label, px, p3, st.R.copy(), float(st.ratio), bool(st.pinching), z, t))

        for k in list(self._state):
            if k not in seen:
                del self._state[k]
        return out
