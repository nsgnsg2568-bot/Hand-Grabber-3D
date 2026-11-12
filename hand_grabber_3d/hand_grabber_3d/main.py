"""Virtual Hand-Grabber 3D
Your real hands, tracked in 3D, interact with virtual objects that obey real physics
(gravity, bounce, friction, stacking, spin, shadows).   Run:  python main.py   (python main.py --help)
"""
import argparse
import os
import sys

sys.modules.setdefault("tensorflow", None)   # a broken TensorFlow install must not break MediaPipe

import cv2


def parse_args():
    from grab3d.config import Config
    d = Config()
    ap = argparse.ArgumentParser(description="Virtual Hand-Grabber 3D")
    ap.add_argument("--camera", type=int, default=d.camera_index, help="webcam index (try 1 if 0 is wrong)")
    ap.add_argument("--width", type=int, default=d.width)
    ap.add_argument("--height", type=int, default=d.height)
    ap.add_argument("--fov", type=float, default=d.hfov_deg, help="horizontal field of view of your webcam in degrees")
    ap.add_argument("--hand-scale", type=float, default=d.hand_scale, help="scale for bigger/smaller hands")
    ap.add_argument("--depth-sign", type=float, default=d.depth_sign, choices=[1.0, -1.0],
                    help="-1 flips the finger-depth direction if it looks inverted")
    ap.add_argument("--complexity", type=int, choices=[0, 1], default=d.model_complexity)
    ap.add_argument("--pinch-on", type=float, default=d.pinch_on)
    ap.add_argument("--pinch-off", type=float, default=d.pinch_off)
    ap.add_argument("--tilt", type=float, default=d.tilt_deg, help="webcam tilt-up angle in degrees (gravity direction)")
    ap.add_argument("--floor", type=float, default=d.floor_depth, help="floor depth below the camera axis (m)")
    ap.add_argument("--near", type=float, default=d.near_wall)
    ap.add_argument("--max-objects", type=int, default=d.max_objects)
    ap.add_argument("--spawn", type=float, default=d.spawn_interval, help="seconds between spawned objects")
    ap.add_argument("--shadows", choices=["real", "blob", "off"], default=d.shadows)
    ap.add_argument("--no-shaders", action="store_true", help="fixed-function rendering for very old GPUs")
    ap.add_argument("--studio", action="store_true", help="start in the virtual studio instead of AR")
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--no-rotate", action="store_true", help="held objects do not follow your wrist rotation")
    ap.add_argument("--no-msaa", action="store_true")
    return ap.parse_args()


def open_camera(cfg):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY       # DirectShow starts fast on Windows
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    from grab3d.config import Config
    args = parse_args()
    cfg = Config(camera_index=args.camera, width=args.width, height=args.height, hfov_deg=args.fov,
                 hand_scale=args.hand_scale, depth_sign=args.depth_sign, model_complexity=args.complexity,
                 pinch_on=args.pinch_on, pinch_off=args.pinch_off, tilt_deg=args.tilt,
                 floor_depth=args.floor, near_wall=args.near, max_objects=args.max_objects,
                 spawn_interval=args.spawn, shadows=args.shadows, shaders=not args.no_shaders,
                 studio=args.studio, occlusion=not args.no_occlusion, show_skeleton=not args.no_skeleton,
                 rotate_with_hand=not args.no_rotate, msaa=not args.no_msaa)

    cap = open_camera(cfg)
    ok, frame = cap.read()
    if not ok:
        print(f"Could not read from camera {cfg.camera_index}. Close other apps using the webcam, "
              f"or try:  python main.py --camera 1")
        sys.exit(1)
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    print(f"Camera running at {w}x{h}  (assumed horizontal FOV {cfg.hfov_deg:.0f} deg)")

    from panda3d.core import loadPrcFileData
    ww = min(w, 1280)
    wh = int(round(ww * h / w))
    for line in (f"win-size {ww} {wh}", "window-title Virtual Hand-Grabber 3D", "sync-video 1",
                 "audio-library-name null", "textures-power-2 none", "textures-auto-power-2 0",
                 "show-frame-rate-meter 0", "notify-level warning"):
        loadPrcFileData("", line)
    if cfg.msaa:
        loadPrcFileData("", "framebuffer-multisample 1")
        loadPrcFileData("", "multisamples 4")

    from grab3d.app import HandGrabberApp
    from grab3d.camera_worker import CameraWorker
    from grab3d.geometry import Intrinsics
    from grab3d.hand_tracker import HandTracker

    intr = Intrinsics(w, h, cfg.hfov_deg)
    worker = CameraWorker(cap, lambda: HandTracker(cfg, intr))
    worker.start()
    app = HandGrabberApp(cfg, intr, worker, frame)
    try:
        app.run()
    finally:
        worker.stop()
        cap.release()


if __name__ == "__main__":
    main()
