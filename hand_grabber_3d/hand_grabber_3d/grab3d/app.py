"""The Panda3D application: webcam background, lights + shadows, physics, hands, HUD."""
import os
import time

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (AmbientLight, AntialiasAttrib, CardMaker, ClockObject, DirectionalLight,
                          NodePath, Point3, TextNode, TransparencyAttrib, Vec4)

from . import textures as TX
from .hand_view import HandView
from .interaction import GrabController
from .meshes import cylinder_mesh, quad_mesh, rounded_box_mesh, sphere_mesh
from .panda_assets import (enum, geom_node, make_material, make_texture, make_video_texture,
                           update_video_texture)
from .physics import HandColliders, PhysicsWorld
from .rig import HandRig

AMB_BASE = np.array([0.50, 0.50, 0.54])
KEY_BASE = np.array([0.85, 0.82, 0.78])


class ShadowBlobs:
    """Cheap fallback shadows: a soft dark disc on the floor under every object."""

    def __init__(self, render, tex, physics, tilt_deg):
        self.render, self.tex, self.physics, self.tilt = render, tex, physics, tilt_deg

    def _make(self):
        cm = CardMaker("blob")
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        n = self.render.attachNewNode(cm.generate())
        n.setP(-90.0 - self.tilt)                       # lie flat on the (possibly tilted) floor
        n.setTexture(self.tex, 1)
        n.setTransparency(enum(TransparencyAttrib, "M_alpha", "MAlpha"))
        n.setLightOff(1)
        n.setShaderOff(1)
        n.setDepthWrite(False)
        return n

    def update(self):
        up, fh = self.physics.up, self.physics.floor_h
        for it in self.physics.items:
            if it.blob is None:
                it.blob = self._make()
            p = it.pos()
            h = float(np.dot(up, p)) + fh                # height of the centre above the floor
            foot = p - up * h + up * 0.003
            hh = max(h - it.radius(), 0.0)
            it.blob.setPos(float(foot[0]), float(foot[1]), float(foot[2]))
            it.blob.setScale(it.radius() * 2.8 * (1.0 + 3.0 * hh))
            it.blob.setAlphaScale(float(np.clip(0.60 - 2.5 * hh, 0.0, 0.60)))


class HandGrabberApp(ShowBase):
    def __init__(self, cfg, intr, worker, first_frame):
        ShowBase.__init__(self)
        self.cfg, self.intr, self.worker = cfg, intr, worker
        self.clock = ClockObject.getGlobalClock()
        self.disableMouse()
        self.setFrameRateMeter(False)
        self.setBackgroundColor(0, 0, 0, 1)

        # ---- camera = the webcam: same field of view, at the origin, looking along +Y
        self.camLens.setFov(intr.hfov_deg, intr.vfov_deg)
        self.camLens.setNearFar(0.03, 100.0)
        self.camera.setPos(0, 0, 0)
        self.camera.setHpr(0, 0, 0)
        if cfg.msaa:
            try:
                self.render.setAntialias(enum(AntialiasAttrib, "M_multisample", "MMultisample"))
            except Exception:  # noqa: BLE001
                pass

        self._seq = -1
        self._hands = []
        self._pkt_time = 0.0
        self._light = 0.45
        self._light_applied = -1.0
        self._hud_t = 0.0
        self.studio = cfg.studio

        self._build_assets(first_frame)
        self._build_background()
        self._build_lights()
        self.physics = PhysicsWorld(self.render, cfg, intr, self.models, self.tex)
        self.colliders = HandColliders(self.render, self.physics.bullet)
        self.rig = HandRig()
        self.grab = GrabController(self.physics, cfg)
        self.hand_view = HandView(self.render, self.models["sphere_lo"], cfg)
        self._build_floor()
        self._build_hud()
        self.blobs = (ShadowBlobs(self.render, self.tex["blob"], self.physics, cfg.tilt_deg)
                      if self.shadow_mode == "blob" else None)
        self.set_mode(self.studio)

        for k in range(5):                               # start with a few objects already falling
            self.physics.spawn()
        self._bind_keys()
        self.taskMgr.add(self.update, "update")

    # ================================================================== building the scene
    def _build_assets(self, first_frame):
        h, w = first_frame.shape[:2]
        self.cam_tex = make_video_texture(w, h)
        update_video_texture(self.cam_tex, first_frame)
        self.tex = {
            "paint": make_texture(TX.paint_texture(), "paint", repeat=False),
            "stripe": make_texture(TX.stripe_texture(), "stripe"),
            "floor_ar": make_texture(TX.floor_texture(True), "floor_ar"),
            "floor_studio": make_texture(TX.floor_texture(False), "floor_studio"),
            "backdrop": make_texture(TX.backdrop_texture(), "backdrop", mipmap=False, repeat=False),
            "blob": make_texture(TX.blob_texture(), "blob", mipmap=False, repeat=False),
        }
        self.models = {
            "box": NodePath(geom_node(rounded_box_mesh((0.5, 0.5, 0.5), 0.12), "box")),
            "ball": NodePath(geom_node(sphere_mesh(1.0, 18, 32), "ball")),
            "cyl": NodePath(geom_node(cylinder_mesh(1.0, 1.0, 36), "cyl")),
            "sphere_lo": NodePath(geom_node(sphere_mesh(1.0, 8, 14), "sphere_lo")),
        }

    def _build_background(self):
        """A card far away that exactly fills the camera's view and shows the webcam picture."""
        d = 50.0
        cm = CardMaker("background")
        cm.setFrame(-d * self.intr.tan_h, d * self.intr.tan_h, -d * self.intr.tan_v, d * self.intr.tan_v)
        self.bg_card = self.render.attachNewNode(cm.generate())
        self.bg_card.setPos(0, d, 0)
        self.bg_card.setTexture(self.cam_tex, 1)
        self.bg_card.setLightOff(1)
        self.bg_card.setShaderOff(1)
        self.bg_card.setDepthTest(False)
        self.bg_card.setDepthWrite(False)
        self.bg_card.setBin("background", 0)

        # picture-in-picture of the webcam (only shown in studio mode)
        cm = CardMaker("pip")
        cm.setFrame(-0.66, -0.02, 0.02, 0.38)
        self.pip = self.a2dBottomRight.attachNewNode(cm.generate())
        self.pip.setTexture(self.cam_tex, 1)

    def _build_lights(self):
        cfg = self.cfg
        self.shadow_mode = cfg.shadows if cfg.shaders else ("off" if cfg.shadows == "off" else "blob")
        if self.shadow_mode == "real":
            try:
                if not self.win.getGsg().getSupportsShadowFilter():
                    print("[info] GPU has no shadow-map support, using blob shadows instead.")
                    self.shadow_mode = "blob"
            except AttributeError:
                pass

        self.amb = AmbientLight("ambient")
        self.amb_np = self.render.attachNewNode(self.amb)
        self.render.setLight(self.amb_np)

        self.key = DirectionalLight("key")
        center = Point3(0.0, 0.8, -0.05)
        if self.shadow_mode == "real":
            self.key.setScene(self.render)
            self.key.setShadowCaster(True, 2048, 2048)
            lens = self.key.getLens()
            lens.setFilmSize(2.2, 2.2)
            lens.setNearFar(0.1, 6.0)
        self.key_np = self.render.attachNewNode(self.key)
        self.key_np.setPos(center + Point3(-0.5, -0.4, 1.4))
        self.key_np.lookAt(center)
        self.render.setLight(self.key_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.20, 0.22, 0.27, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(35, -25, 0)
        self.render.setLight(fill_np)

        self._apply_light(1.0, force=True)
        if cfg.shaders:
            self.render.setShaderAuto()

    def _apply_light(self, factor, force=False):
        if not force and abs(factor - self._light_applied) < 0.01:
            return
        self._light_applied = factor
        a, k = AMB_BASE * factor, KEY_BASE * factor
        self.amb.setColor(Vec4(float(a[0]), float(a[1]), float(a[2]), 1))
        self.key.setColor(Vec4(float(k[0]), float(k[1]), float(k[2]), 1))

    def _build_floor(self):
        self.floor_np = self.render.attachNewNode(geom_node(quad_mesh(-2.5, 2.5, 0.0, 3.2, 0.0, (50, 32)), "floor"))
        self.floor_np.setMaterial(make_material((1, 1, 1), 0.0, 1.0), 1)
        self.floor_np.setTransparency(enum(TransparencyAttrib, "M_alpha", "MAlpha"))
        self.floor_np.setDepthWrite(False)
        self._place_floor()

    def _place_floor(self):
        p = -self.physics.floor_h * self.physics.up
        self.floor_np.setPos(float(p[0]), float(p[1]), float(p[2]))
        self.floor_np.setP(-self.cfg.tilt_deg)

    def _build_hud(self):
        kw = dict(fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.9), align=TextNode.ALeft, mayChange=True)
        self.hud = OnscreenText(text="", parent=self.a2dTopLeft, pos=(0.04, -0.07), scale=0.042, **kw)
        self.hud2 = OnscreenText(text="", parent=self.a2dTopLeft, pos=(0.04, -0.13), scale=0.040,
                                 **{**kw, "fg": (0.55, 1.0, 0.65, 1)})
        OnscreenText(text="pinch=grab  open hand=push  SPACE spawn  R reset  G gravity  B studio/AR  "
                          "O occlusion  K skeleton  [ ] floor  S screenshot  Q quit",
                     parent=self.a2dBottomLeft, pos=(0.04, 0.04), scale=0.034,
                     **{**kw, "fg": (0.9, 0.9, 0.9, 1)})
        self.msg = OnscreenText(text="Show your hand to the camera", parent=self.aspect2d, pos=(0, 0.0),
                                scale=0.07, fg=(0.4, 0.9, 1, 1), shadow=(0, 0, 0, 1), mayChange=True)

    # ================================================================== modes & keys
    def set_mode(self, studio):
        self.studio = studio
        self.bg_card.setTexture(self.tex["backdrop"] if studio else self.cam_tex, 1)
        self.floor_np.setTexture(self.tex["floor_studio"] if studio else self.tex["floor_ar"], 1)
        if studio:
            self.pip.show()
        else:
            self.pip.hide()
        self.hand_view.set_studio(studio)

    def _bind_keys(self):
        self.accept("escape", self.userExit)
        self.accept("q", self.userExit)
        self.accept("space", lambda: [self.physics.spawn() for _ in range(4)])
        self.accept("r", self._reset)
        self.accept("g", lambda: print("Gravity", "ON" if self.physics.toggle_gravity() else "OFF (zero-g)"))
        self.accept("b", lambda: self.set_mode(not self.studio))
        self.accept("o", self._toggle_occlusion)
        self.accept("k", self._toggle_skeleton)
        self.accept("[", lambda: self._move_floor(+0.02))
        self.accept("]", lambda: self._move_floor(-0.02))
        self.accept("s", self._screenshot)

    def _reset(self):
        self.grab.reset()
        self.physics.reset()

    def _toggle_occlusion(self):
        self.hand_view.occlusion = not self.hand_view.occlusion
        print("Hand occlusion", "ON" if self.hand_view.occlusion else "OFF")

    def _toggle_skeleton(self):
        self.hand_view.skeleton = not self.hand_view.skeleton

    def _move_floor(self, delta):
        self.physics.set_floor(float(np.clip(self.physics.floor_h + delta, 0.02, 0.6)))
        self._place_floor()

    def _screenshot(self):
        os.makedirs("screenshots", exist_ok=True)
        self.screenshot(namePrefix=os.path.join("screenshots", "shot"), defaultFilename=True)
        print("Saved a screenshot to ./screenshots")

    def userExit(self):  # noqa: N802
        self.worker.stop()
        ShowBase.userExit(self)

    # ================================================================== main loop
    def update(self, task):
        if self.worker.error is not None:
            print("Fatal:", self.worker.error)
            self.userExit()
            return task.done

        dt = min(max(self.clock.getDt(), 1e-3), 1.0 / 30.0)
        now = time.perf_counter()

        pkt = self.worker.latest()
        if pkt is not None and pkt.seq != self._seq:
            self._seq = pkt.seq
            update_video_texture(self.cam_tex, pkt.frame)
            self._hands = pkt.hands
            self._pkt_time = now
            self._light += 0.15 * (pkt.brightness - self._light)
            self.grab.on_hands(pkt.hands, pkt.t)
        elif self._hands and now - self._pkt_time > 0.3:          # camera stalled: let go of everything
            self._hands = []
            self.grab.on_hands([], now)

        hands = self.rig.update(self._hands, dt)
        self.grab.drive(hands, dt)
        self.colliders.update(hands, self.grab.hidden_labels(now), now)
        self.physics.update(dt, self.grab.held_items())

        self._apply_light(float(np.clip(self._light / 0.45, 0.6, 1.35)))
        self._update_visuals(hands)
        self._update_hud(hands, now)
        return task.cont

    def _update_visuals(self, hands):
        held = self.grab.held_items()
        cand = set(self.grab.candidates.values())
        for it in self.physics.items:
            it.highlight(1.3 if it in held else (1.18 if it in cand else 1.0))
        self.hand_view.update(hands, self.physics.floor_h, self.physics.up)
        if self.blobs is not None:
            self.blobs.update()

    def _update_hud(self, hands, now):
        if now - self._hud_t < 0.2:
            return
        self._hud_t = now
        mode = "STUDIO" if self.studio else "AR"
        self.hud.setText(f"FPS {self.clock.getAverageFrameRate():3.0f}   objects {len(self.physics.items)}   "
                         f"{mode}   shadows {self.shadow_mode}")
        parts = [f"{h.label} {h.depth:.2f} m{' PINCH' if h.pinching else ''}" for h in hands]
        parts.append(f"grabs {self.grab.grabs}  last throw {self.grab.last_throw:.1f} m/s  "
                     f"best {self.grab.best_throw:.1f} m/s")
        self.hud2.setText("   |   ".join(parts))
        if hands:
            self.msg.hide()
        else:
            self.msg.show()
