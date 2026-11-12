from dataclasses import dataclass


@dataclass
class Config:
    # --- camera ---
    camera_index: int = 0
    width: int = 1280
    height: int = 720

    # --- hand tracking ---
    model_complexity: int = 1          # 0 = fastest, 1 = more accurate
    detection_conf: float = 0.6
    tracking_conf: float = 0.5
    # pinch = (thumb tip <-> index tip distance) / (wrist <-> middle knuckle distance)
    pinch_on: float = 0.30             # ratio below this -> pinch starts
    pinch_off: float = 0.45            # ratio above this -> pinch ends (hysteresis)

    # --- physics ---
    gravity: float = 1100.0            # pixels / s^2
    max_objects: int = 35
    spawn_interval: float = 1.1        # seconds between auto-spawned objects

    # --- grabbing / throwing ---
    grab_radius: float = 80.0          # how close the pinch must be to an object (px)
    grab_gain: float = 18.0            # how stiffly a held object follows the hand
    throw_scale: float = 1.15          # multiplies hand velocity on release
    max_speed: float = 4500.0          # px / s safety clamp
    fingertip_radius: float = 13.0     # open-hand "pushers" on fingertips

    # --- display ---
    show_skeleton: bool = True
    window_name: str = "Virtual Hand-Grabber"
