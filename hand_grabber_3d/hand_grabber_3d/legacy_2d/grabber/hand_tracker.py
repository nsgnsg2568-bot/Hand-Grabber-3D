"""MediaPipe Hands wrapper: landmarks -> pinch state, smoothed pinch point and palm centre."""
import math
import sys
from dataclasses import dataclass

sys.modules.setdefault("tensorflow", None)   # broken TensorFlow must not break MediaPipe

import cv2

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except (ImportError, AttributeError) as e:  # pragma: no cover
    raise RuntimeError(
        "Could not load mediapipe.solutions.hands. Install the pinned version:\n"
        "    pip install mediapipe==0.10.14\n"
        "(use Python 3.10-3.12)"
    ) from e

from .filters import OneEuro2D

HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9
FINGERTIPS = (4, 8, 12, 16, 20)


@dataclass
class Hand:
    label: str            # "Left" / "Right" (unique per frame)
    points: list          # 21 x (x, y) in pixels
    pinch_ratio: float
    pinch_point: tuple    # smoothed midpoint between thumb tip and index tip
    palm: tuple           # smoothed middle-finger knuckle (stable for velocity estimation)
    pinching: bool


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class HandTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.detection_conf,
            min_tracking_confidence=cfg.tracking_conf,
        )
        self._pinch_state = {}
        self._pinch_filters = {}
        self._palm_filters = {}

    def close(self):
        self._hands.close()

    def process(self, frame_bgr, t):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)

        hands, seen = [], set()
        if res.multi_hand_landmarks and res.multi_handedness:
            for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label
                if label in seen:                       # both hands got the same label
                    label += "2"
                seen.add(label)

                pts = [(lm.x * w, lm.y * h) for lm in lms.landmark]
                size = max(_dist(pts[WRIST], pts[MIDDLE_MCP]), 1.0)
                ratio = _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / size

                prev = self._pinch_state.get(label, False)
                pinching = ratio < (self.cfg.pinch_off if prev else self.cfg.pinch_on)
                self._pinch_state[label] = pinching

                raw_pinch = ((pts[THUMB_TIP][0] + pts[INDEX_TIP][0]) / 2,
                             (pts[THUMB_TIP][1] + pts[INDEX_TIP][1]) / 2)
                pf = self._pinch_filters.setdefault(label, OneEuro2D())
                palf = self._palm_filters.setdefault(label, OneEuro2D(min_cutoff=2.0, beta=0.01))
                hands.append(Hand(label, pts, ratio, pf(raw_pinch, t), palf(pts[MIDDLE_MCP], t), pinching))

        # forget state of hands that left the frame
        for store in (self._pinch_state, self._pinch_filters, self._palm_filters):
            for k in list(store):
                if k not in seen:
                    del store[k]
        return hands
