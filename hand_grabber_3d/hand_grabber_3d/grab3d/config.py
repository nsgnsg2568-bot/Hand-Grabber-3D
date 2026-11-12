from dataclasses import dataclass


@dataclass
class Config:
    # ---- camera ----
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    hfov_deg: float = 62.0          # horizontal field of view of your webcam (typical laptop: 60-70)

    # ---- hand tracking ----
    model_complexity: int = 1       # 0 = faster, 1 = more accurate
    detection_conf: float = 0.6
    tracking_conf: float = 0.5
    hand_scale: float = 1.0         # >1 if your hands are bigger than average (it also changes the depth estimate)
    depth_sign: float = 1.0         # set to -1 if finger depth looks inverted
    # pinch ratio = distance(thumb tip, index tip) / distance(wrist, middle knuckle)
    pinch_on: float = 0.30
    pinch_off: float = 0.45

    # ---- world (metres, camera at the origin, +Y forward, +Z up, +X right) ----
    gravity: float = 9.81
    tilt_deg: float = 0.0           # how far your webcam is tilted UP from horizontal (laptops: ~5-15)
    floor_depth: float = 0.14       # floor is this far BELOW the camera axis ([ and ] change it live)
    near_wall: float = 0.40         # invisible glass wall "at the screen"
    far_wall: float = 1.35
    max_objects: int = 24
    spawn_interval: float = 1.6

    # ---- grabbing / throwing ----
    grab_radius: float = 0.055      # how close the pinch point must be to an object's surface (m)
    grab_gain: float = 22.0         # how stiffly a held object follows the pinch
    rot_gain: float = 10.0          # how stiffly it follows your wrist rotation
    rotate_with_hand: bool = True
    throw_scale: float = 1.2
    depth_vel_scale: float = 0.5    # depth speed is noisy, so it is damped on release
    max_speed: float = 6.0          # m/s safety clamp

    # ---- look ----
    shadows: str = "real"           # "real" | "blob" | "off"
    studio: bool = False            # start in the virtual studio instead of the camera background
    occlusion: bool = True          # your hand hides virtual objects that are behind it
    show_skeleton: bool = True
    msaa: bool = True
    shaders: bool = True            # False = old fixed-function pipeline (no real shadows)
