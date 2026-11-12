"""Glue between numpy meshes/textures and Panda3D objects."""
import numpy as np
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
                          GeomVertexWriter, Material, Texture, Vec4)


def enum(cls, *names):
    """Return the first enum attribute that exists (Panda3D exposes both C_style and CamelCase names)."""
    for n in names:
        if hasattr(cls, n):
            return getattr(cls, n)
    raise AttributeError(f"{cls.__name__} has none of {names}")


def geom_node(mesh, name="mesh"):
    vdata = GeomVertexData(name, GeomVertexFormat.getV3n3t2(), enum(Geom, "UH_static", "UHStatic"))
    vdata.setNumRows(len(mesh.v))
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    tw = GeomVertexWriter(vdata, "texcoord")
    for v, n, t in zip(mesh.v.tolist(), mesh.n.tolist(), mesh.uv.tolist()):
        vw.addData3(v[0], v[1], v[2])
        nw.addData3(n[0], n[1], n[2])
        tw.addData2(t[0], t[1])
    prim = GeomTriangles(enum(Geom, "UH_static", "UHStatic"))
    for a, b, c in mesh.tri.tolist():
        prim.addVertices(a, b, c)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_texture(arr, name="tex", mipmap=True, repeat=True):
    """arr: uint8 (H, W, 3|4), rows top-to-bottom."""
    h, w, c = arr.shape
    tex = Texture(name)
    tex.setup2dTexture(w, h, enum(Texture, "T_unsigned_byte", "TUnsignedByte"),
                       enum(Texture, "F_rgba", "FRgba") if c == 4 else enum(Texture, "F_rgb", "FRgb"))
    tex.setRamImageAs(np.ascontiguousarray(arr[::-1]).tobytes(), "RGBA" if c == 4 else "RGB")
    if mipmap:
        tex.setMinfilter(enum(Texture, "FT_linear_mipmap_linear", "FTLinearMipmapLinear"))
        tex.setAnisotropicDegree(8)
    else:
        tex.setMinfilter(enum(Texture, "FT_linear", "FTLinear"))
    tex.setMagfilter(enum(Texture, "FT_linear", "FTLinear"))
    wrap = enum(Texture, "WM_repeat", "WMRepeat") if repeat else enum(Texture, "WM_clamp", "WMClamp")
    tex.setWrapU(wrap)
    tex.setWrapV(wrap)
    return tex


def make_video_texture(w, h):
    tex = Texture("webcam")
    tex.setup2dTexture(w, h, enum(Texture, "T_unsigned_byte", "TUnsignedByte"), enum(Texture, "F_rgb", "FRgb"))
    tex.setMinfilter(enum(Texture, "FT_linear", "FTLinear"))
    tex.setMagfilter(enum(Texture, "FT_linear", "FTLinear"))
    clamp = enum(Texture, "WM_clamp", "WMClamp")
    tex.setWrapU(clamp)
    tex.setWrapV(clamp)
    return tex


def update_video_texture(tex, frame_bgr):
    # Panda stores images bottom-to-top, OpenCV top-to-bottom -> flip vertically
    tex.setRamImageAs(np.ascontiguousarray(frame_bgr[::-1]).tobytes(), "BGR")


def make_material(rgb, specular=0.35, shininess=32.0):
    m = Material()
    c = Vec4(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    m.setDiffuse(c)
    m.setAmbient(c)
    m.setSpecular(Vec4(specular, specular, specular, 1.0))
    m.setShininess(shininess)
    return m
