"""Everything you SEE of the hands:
 * AR mode:     invisible 'occluder' spheres that only write depth, so virtual objects behind your real hand
                are hidden by it (the hand also casts shadows onto the virtual objects).
 * Studio mode: the same spheres are drawn as a solid skin-coloured 3D hand.
 * a pinch cursor, the grab radius, a skeleton, and a drop-line + ring on the floor (a depth cue)."""
import math

import numpy as np
from panda3d.core import ColorWriteAttrib, LineSegs, TransparencyAttrib

from .hands import COLLIDER_R, COLLIDER_W, HAND_CONNECTIONS
from .panda_assets import enum, make_material

SKIN = (0.93, 0.74, 0.63)


class HandView:
    def __init__(self, render, sphere_model, cfg):
        self.render, self.sphere_model, self.cfg = render, sphere_model, cfg
        self.root = render.attachNewNode("hand_view")
        self.pools = {}        # label -> {"paths": [...], "visible": bool}
        self.cursors = {}      # label -> (core, halo)
        self.studio = cfg.studio
        self.occlusion = cfg.occlusion
        self.skeleton = cfg.show_skeleton
        self._skin_mat = make_material(SKIN, 0.15, 12.0)

        self.overlay = render.attachNewNode("hand_overlay")
        self._style_unlit(self.overlay)

    # ------------------------------------------------------------------ styling
    @staticmethod
    def _style_unlit(npath):
        npath.setLightOff(1)
        npath.setShaderOff(1)
        npath.setTransparency(enum(TransparencyAttrib, "M_alpha", "MAlpha"))
        npath.setDepthTest(False)
        npath.setDepthWrite(False)
        npath.setBin("fixed", 60)

    def _style_sphere(self, npath):
        if self.studio:
            npath.clearBin()
            npath.setAttrib(ColorWriteAttrib.make(enum(ColorWriteAttrib, "C_all", "CAll")))
            npath.setMaterial(self._skin_mat, 1)
            npath.setColor(SKIN[0], SKIN[1], SKIN[2], 1.0)
        else:  # depth-only: invisible, but hides whatever is behind it. Drawn before the virtual objects.
            npath.setAttrib(ColorWriteAttrib.make(enum(ColorWriteAttrib, "C_off", "COff")))
            npath.setBin("background", 1)

    def set_studio(self, studio):
        self.studio = studio
        for pool in self.pools.values():
            for npath in pool["paths"]:
                self._style_sphere(npath)

    def _make_pool(self, label):
        paths = []
        for _ in COLLIDER_R:
            npath = self.sphere_model.copyTo(self.root)
            self._style_sphere(npath)
            paths.append(npath)
        pool = {"paths": paths, "visible": True}
        self.pools[label] = pool
        return pool

    def _make_cursor(self, label):
        core = self.sphere_model.copyTo(self.render)
        halo = self.sphere_model.copyTo(self.render)
        for n in (core, halo):
            self._style_unlit(n)
        self.cursors[label] = (core, halo)
        return core, halo

    @staticmethod
    def _set_visible(pool, flag):
        if pool["visible"] != flag:
            pool["visible"] = flag
            for npath in pool["paths"]:
                if flag:
                    npath.show()
                else:
                    npath.hide()

    # ------------------------------------------------------------------ per frame
    def update(self, hands, floor_h, up):
        seen = set()
        show_body = self.studio or self.occlusion
        occ_scale = 1.0 if self.studio else 0.9
        for h in hands:
            seen.add(h.label)
            pool = self.pools.get(h.label) or self._make_pool(h.label)
            pts = COLLIDER_W @ h.p3
            for npath, p, r in zip(pool["paths"], pts, COLLIDER_R):
                npath.setPos(float(p[0]), float(p[1]), float(p[2]))
                npath.setScale(float(r) * occ_scale)
            self._set_visible(pool, show_body)

            core, halo = self.cursors.get(h.label) or self._make_cursor(h.label)
            pp = h.pinch_pt
            core.show()
            core.setPos(float(pp[0]), float(pp[1]), float(pp[2]))
            halo.setPos(float(pp[0]), float(pp[1]), float(pp[2]))
            if h.pinching:
                core.setScale(0.011)
                core.setColor(0.35, 1.0, 0.45, 0.95)
                halo.hide()
            else:
                core.setScale(0.007)
                core.setColor(1.0, 1.0, 1.0, 0.85)
                halo.show()
                halo.setScale(self.cfg.grab_radius)
                halo.setColor(1.0, 1.0, 1.0, 0.06)
        for label, pool in self.pools.items():
            if label not in seen:
                self._set_visible(pool, False)
        for label, (core, halo) in self.cursors.items():
            if label not in seen:
                core.hide()
                halo.hide()
        self._draw_lines(hands, floor_h, up)

    def _draw_lines(self, hands, floor_h, up):
        for child in list(self.overlay.getChildren()):
            child.removeNode()
        if not hands:
            return
        ls = LineSegs("hand_lines")
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.cross(up, e1)
        e2 /= np.linalg.norm(e2)
        for h in hands:
            if self.skeleton:
                ls.setThickness(2.0)
                ls.setColor(1.0, 1.0, 1.0, 0.55)
                for a, b in HAND_CONNECTIONS:
                    pa, pb = h.p3[a], h.p3[b]
                    ls.moveTo(float(pa[0]), float(pa[1]), float(pa[2]))
                    ls.drawTo(float(pb[0]), float(pb[1]), float(pb[2]))
            # depth cue: a line from the pinch point straight down to the floor + a ring where it lands
            pp = h.pinch_pt
            dist = float(np.dot(up, pp)) + floor_h
            if dist > 0.0:
                foot = pp - up * dist
                col = (0.35, 1.0, 0.45, 0.55) if h.pinching else (1.0, 1.0, 1.0, 0.35)
                ls.setThickness(1.5)
                ls.setColor(*col)
                ls.moveTo(float(pp[0]), float(pp[1]), float(pp[2]))
                ls.drawTo(float(foot[0]), float(foot[1]), float(foot[2]))
                for k in range(25):
                    a = 2.0 * math.pi * k / 24.0
                    q = foot + 0.025 * (math.cos(a) * e1 + math.sin(a) * e2)
                    if k == 0:
                        ls.moveTo(float(q[0]), float(q[1]), float(q[2]))
                    else:
                        ls.drawTo(float(q[0]), float(q[1]), float(q[2]))
        self.overlay.attachNewNode(ls.create())
