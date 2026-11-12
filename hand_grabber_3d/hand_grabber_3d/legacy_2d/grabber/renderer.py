"""Draws the AR overlay: dimmed camera feed + physics objects + hand skeleton + HUD."""
import math

import cv2
import numpy as np
import pymunk

from .hand_tracker import HAND_CONNECTIONS


def _darker(c, f=0.55):
    return tuple(int(v * f) for v in c)


def _text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Renderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.show_skeleton = cfg.show_skeleton

    def draw(self, frame, world, hands, interaction, fps):
        img = cv2.convertScaleAbs(frame, alpha=0.65, beta=0)   # dim so objects pop
        held = interaction.held_bodies()

        # ---- objects
        for it in world.items:
            body, shape = it.body, it.shape
            is_held = body in held
            fill = it.color
            edge = (255, 255, 255) if is_held else _darker(fill)
            thick = 4 if is_held else 2
            if isinstance(shape, pymunk.Circle):
                c = (int(body.position.x), int(body.position.y))
                r = int(shape.radius)
                cv2.circle(img, c, r, fill, -1, cv2.LINE_AA)
                cv2.circle(img, c, r, edge, thick, cv2.LINE_AA)
                tip = (int(c[0] + math.cos(body.angle) * r * 0.8), int(c[1] + math.sin(body.angle) * r * 0.8))
                cv2.line(img, c, tip, edge, 2, cv2.LINE_AA)           # shows rotation
                cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), max(2, r // 6), (255, 255, 255), -1, cv2.LINE_AA)
            else:
                verts = [body.local_to_world(v) for v in shape.get_vertices()]
                pts = np.array([[int(v.x), int(v.y)] for v in verts], np.int32)
                cv2.fillPoly(img, [pts], fill, cv2.LINE_AA)
                cv2.polylines(img, [pts], True, edge, thick, cv2.LINE_AA)

        # ---- hands
        for hand in hands:
            if self.show_skeleton:
                p = [(int(x), int(y)) for x, y in hand.points]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, p[a], p[b], (235, 235, 235), 2, cv2.LINE_AA)
                for q in p:
                    cv2.circle(img, q, 3, (60, 60, 255), -1, cv2.LINE_AA)
            pp = (int(hand.pinch_point[0]), int(hand.pinch_point[1]))
            if hand.pinching:
                cv2.circle(img, pp, 14, (80, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(img, pp, 18, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(img, pp, int(self.cfg.grab_radius), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(img, pp, 4, (255, 255, 255), -1, cv2.LINE_AA)

        # ---- HUD
        h = img.shape[0]
        _text(img, f"FPS {fps:4.0f}   hands {len(hands)}   objects {len(world.items)}", (14, 28))
        _text(img, f"grabs {interaction.grabs}   last throw {interaction.last_throw:5.0f} px/s   "
                   f"best {interaction.best_throw:5.0f} px/s", (14, 54), 0.55, (120, 255, 160))
        _text(img, "pinch = grab | open hand = push | SPACE spawn | R reset | G gravity | K skeleton | S shot | F fullscreen | Q quit",
              (14, h - 14), 0.5, (230, 230, 230))
        if not hands:
            _text(img, "Show your hand to the camera", (img.shape[1] // 2 - 170, h // 2), 0.9, (0, 220, 255), 2)
        return img
