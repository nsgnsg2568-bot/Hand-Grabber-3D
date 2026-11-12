"""Webcam capture + hand tracking on a background thread, so rendering stays at 60 FPS."""
import threading
import time
from dataclasses import dataclass

import cv2


@dataclass
class Packet:
    seq: int
    frame: object        # mirrored BGR frame (numpy)
    hands: list          # list[Hand3D]
    t: float             # capture time (perf_counter)
    brightness: float    # mean luminance 0..1 (used to match the virtual lighting to your room)


class CameraWorker(threading.Thread):
    def __init__(self, cap, tracker_factory):
        super().__init__(daemon=True)
        self.cap = cap
        self.tracker_factory = tracker_factory
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._pkt = None
        self.error = None

    def latest(self):
        with self._lock:
            return self._pkt

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            tracker = self.tracker_factory()
        except Exception as e:  # noqa: BLE001
            self.error = e
            return
        seq, fails = 0, 0
        try:
            while not self._stop_evt.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    fails += 1
                    if fails > 100:
                        self.error = RuntimeError("Camera stopped delivering frames.")
                        return
                    time.sleep(0.01)
                    continue
                fails = 0
                frame = cv2.flip(frame, 1)                 # mirror: feels like a mirror
                t = time.perf_counter()
                hands = tracker.process(frame, t)
                small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
                seq += 1
                pkt = Packet(seq, frame, hands, t, float(small.mean()) / 255.0)
                with self._lock:
                    self._pkt = pkt
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            try:
                tracker.close()
            except Exception:  # noqa: BLE001
                pass
