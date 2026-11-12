# Virtual Hand-Grabber 3D

Your real hands, tracked in **3D**, pick up and throw virtual blocks, balls and cans that obey real physics:
gravity, bounce, friction, spin, stacking, soft shadows. The webcam picture is the live background, so the
objects really look like they are in your room.

## Setup (Windows, inside your activated environment)

```
install.bat        (or: pip install -r requirements.txt)
run.bat            (or: python main.py)
```

Python 3.10 - 3.12. New dependency compared with the 2D version: `panda3d` (renderer + Bullet physics; it ships
Windows wheels, so no compiler is needed). `mediapipe==0.10.14` stays pinned.

## Controls

| Action | How |
|---|---|
| Grab | Pinch thumb + index; the faint white sphere shows your reach, the nearest object glows |
| Carry / rotate | Move your hand; **turn your wrist and the object turns with it** |
| Throw | Swing your hand and open the pinch |
| Bat / push | Open hand: fingers, knuckles and palm are solid |
| Depth cue | A line + ring drops from your pinch point to the floor: line the ring up with an object's shadow |
| Spawn 4 / Reset | `SPACE` / `R` |
| Zero gravity | `G` |
| AR <-> virtual studio | `B` (studio draws your 3D hand as a solid model) |
| Hand occlusion on/off | `O` |
| Skeleton on/off | `K` |
| Move the floor up / down | `[` / `]` |
| Screenshot | `S` (saved in `screenshots/`) |
| Quit | `Q` / `Esc` |

## First-run calibration (30 seconds)

1. **Field of view**: `python main.py --fov 70` (laptop webcams are usually 60-78 deg). If an object held at
   arm's length looks too big/small compared with your hand, tune this and `--hand-scale` (e.g. 0.9 for smaller hands).
2. **Floor**: press `[` / `]` until the grid sits where you want a table to be.
3. **Tilted webcam**: laptop screens lean back, so the camera looks slightly up. `--tilt 10` makes gravity and the
   floor match the real world.
4. If a hand pointing at the camera looks inside-out in depth, run with `--depth-sign -1`.
5. Pinch too strict / too twitchy: `--pinch-on 0.35 --pinch-off 0.5`.

## How it works

| Stage | File | Idea |
|---|---|---|
| Capture + tracking (own thread) | `camera_worker.py`, `hand_tracker.py` | MediaPipe gives 21 landmarks in pixels and 21 *metric* landmarks of an average hand. |
| Monocular depth | `geometry.estimate_depth` | Pixel distance between every landmark pair = focal length x real distance / depth. Solving for depth by least squares over all 210 pairs tells how far the hand is from the camera, in metres. |
| Lifting to 3D | `geometry.backproject` | Each landmark is back-projected through the pinhole camera, so it lands exactly on the right pixel *and* at a metric depth. |
| Smoothing | `filters.py`, `rig.py` | One-Euro filters (still = steady, fast = no lag). Hand colliders move continuously at 60 Hz. |
| Pinch | `hand_tracker.py` | 3D thumb-index distance / palm length, with hysteresis. |
| Physics | `physics.py` | Bullet rigid bodies in metres (real g = 9.81). The play volume is bounded by planes that coincide with the camera frustum, so nothing leaves the picture. Each hand is ~46 kinematic spheres. |
| Grab / throw | `interaction.py` | A held object is steered so that its grab point follows the pinch, and its orientation follows your wrist. On release it gets your palm's least-squares velocity. |
| Looks | `app.py`, `hand_view.py` | Webcam card at the back, key light with shadow map, ambient light that follows the brightness of your room, invisible hand-shaped depth occluders so objects can go *behind* your fingers, procedural textures (rounded blocks, striped balls so you can see them spin). |

The virtual camera has the same field of view as your webcam and sits at the origin, so virtual objects stay
pixel-aligned with the video no matter what the depth estimate is; the depth estimate only decides scale,
occlusion and physical reach.

## Troubleshooting

- **Black screen / no hand**: close apps using the webcam, or `--camera 1`.
- **Weird or no shadows / GPU errors**: `--shadows blob` (cheap soft discs), or `--no-shaders` for very old GPUs.
- **Objects pop in front of / behind your hand incorrectly**: press `O` to turn the occlusion off.
- **Low FPS**: `--complexity 0`, `--no-msaa`, `--width 960 --height 540`, `--max-objects 14`.
- **Can't reach objects**: they live 0.4 - 1.35 m from the camera. Reach out ~50-70 cm, or `--near 0.35`.
- **Throws too weak/strong**: `throw_scale` in `grab3d/config.py`.

## Tests

`python tests/test_core.py` runs the webcam-free math tests: depth recovery (< 4 % error with pixel noise),
projection round-trips, rotation maths, mesh winding, signed-distance fields, filters and a full simulated
grab -> carry -> wrist-rotate -> throw.

## The old 2D version

Kept in `legacy_2d/` (needs `pymunk`): `cd legacy_2d` then `python main.py`.

## Ideas to level up

- Replace the pinch rule with a small classifier trained on your own landmark data (pinch / fist / point / open).
- Learn a better depth estimator: regress depth from landmark geometry using a depth camera or a ruler for ground truth.
- Two-hand gestures: stretch an object, spawn on a clap, throw with spin from wrist angular velocity.
