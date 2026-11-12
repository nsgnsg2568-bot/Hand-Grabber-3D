"""Virtual Hand-Grabber (Physics Sandbox)
Pinch your fingers in front of the webcam to pick up falling blocks and throw them.

Run:  python main.py            (see  python main.py --help  for options)
"""
import argparse
import os
import sys
import time

import cv2

from grabber.config import Config


def parse_args():
    ap = argparse.ArgumentParser(description="Virtual Hand-Grabber physics sandbox")
    d = Config()
    ap.add_argument("--camera", type=int, default=d.camera_index, help="webcam index (try 1 if 0 is wrong)")
    ap.add_argument("--width", type=int, default=d.width)
    ap.add_argument("--height", type=int, default=d.height)
    ap.add_argument("--max-objects", type=int, default=d.max_objects)
    ap.add_argument("--gravity", type=float, default=d.gravity)
    ap.add_argument("--complexity", type=int, choices=[0, 1], default=d.model_complexity,
                    help="hand model: 0 = faster, 1 = more accurate")
    ap.add_argument("--pinch-on", type=float, default=d.pinch_on, help="pinch ratio to start grabbing")
    ap.add_argument("--pinch-off", type=float, default=d.pinch_off, help="pinch ratio to release")
    ap.add_argument("--no-skeleton", action="store_true", help="hide the hand skeleton overlay")
    return ap.parse_args()


def open_camera(cfg):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY   # DirectShow = fast start on Windows
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    args = parse_args()
    cfg = Config(camera_index=args.camera, width=args.width, height=args.height,
                 max_objects=args.max_objects, gravity=args.gravity,
                 model_complexity=args.complexity, pinch_on=args.pinch_on,
                 pinch_off=args.pinch_off, show_skeleton=not args.no_skeleton)

    # imported here so a missing dependency gives a readable message
    from grabber.hand_tracker import HandTracker
    from grabber.interaction import HandInteraction
    from grabber.physics_world import PhysicsWorld
    from grabber.renderer import Renderer

    cap = open_camera(cfg)
    ok, frame = cap.read()
    if not ok:
        print(f"Could not read from camera {cfg.camera_index}. Close other apps using the webcam "
              f"or try:  python main.py --camera 1")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"Camera running at {w}x{h}")

    world = PhysicsWorld(w, h, cfg)
    tracker = HandTracker(cfg)
    interaction = HandInteraction(world, cfg)
    renderer = Renderer(cfg)
    for _ in range(6):                      # start with a few objects on screen
        world.spawn(y=-40 * _)

    win = cfg.window_name
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))
    fullscreen, fps, last, fails = False, 30.0, time.perf_counter(), 0
    os.makedirs("screenshots", exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 60:
                    print("Camera stopped delivering frames.")
                    break
                continue
            fails = 0
            frame = cv2.flip(frame, 1)      # mirror view so it feels like looking in a mirror

            now = time.perf_counter()
            dt = min(max(now - last, 1e-3), 0.05)
            last = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

            hands = tracker.process(frame, now)
            interaction.update(hands, now, dt)
            world.update(dt)
            cv2.imshow(win, renderer.draw(frame, world, hands, interaction, fps))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord("r"):
                interaction.reset()
                world.reset()
            elif key == ord(" "):
                for _ in range(5):
                    world.spawn()
            elif key == ord("g"):
                print("Gravity", "ON" if world.toggle_gravity() else "OFF (zero-g)")
            elif key == ord("k"):
                renderer.show_skeleton = not renderer.show_skeleton
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("s"):
                path = os.path.join("screenshots", f"shot_{int(time.time())}.png")
                cv2.imwrite(path, renderer.draw(frame, world, hands, interaction, fps))
                print("Saved", path)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
