"""Procedural textures as uint8 numpy arrays, rows top-to-bottom (image convention)."""
import numpy as np


def _blur(a, n=2):
    for _ in range(n):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


def _gray_rgb(img01):
    g = np.clip(img01, 0, 1) * 255
    return np.repeat(g[..., None], 3, axis=2).astype(np.uint8)


def paint_texture(size=128, seed=1):
    """Matte painted-wood look: fine grain + a slightly darker rim on every face."""
    rng = np.random.default_rng(seed)
    n = _blur(rng.normal(0, 1, (size, size)), 3)
    n = n / (n.std() + 1e-9)
    g = 0.93 + 0.035 * n
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1.0)
    e = np.minimum.reduce([xx, 1 - xx, yy, 1 - yy])
    g = g * (0.78 + 0.22 * np.clip(e / 0.07, 0, 1))
    return _gray_rgb(g)


def stripe_texture(size=256, panels=8, seed=2):
    """Beach-ball panels, so you can SEE a ball spin."""
    rng = np.random.default_rng(seed)
    x = np.arange(size) / size
    k = np.floor(x * panels).astype(int) % 2
    g = np.where(k == 0, 1.0, 0.72)
    img = np.tile(g[None, :], (size, 1))
    img = img + 0.02 * _blur(rng.normal(0, 1, (size, size)), 2)
    return _gray_rgb(img)


def floor_texture(ar=True, size=256, line=2):
    """One 10 cm tile. AR: faint translucent grid (a 'shadow catcher'). Studio: opaque dark floor."""
    rng = np.random.default_rng(3)
    grain = 0.03 * _blur(rng.normal(0, 1, (size, size)), 2)
    if ar:
        base, lines = np.array([200, 208, 218, 60.0]), np.array([255, 255, 255, 150.0])
    else:
        base, lines = np.array([44, 48, 57, 255.0]), np.array([84, 92, 108, 255.0])
    img = np.tile(base, (size, size, 1))
    img[..., :3] *= (1 + grain)[..., None]
    m = np.zeros((size, size), bool)
    m[:line, :] = True
    m[:, :line] = True
    img[m] = lines
    return np.clip(img, 0, 255).astype(np.uint8)


def backdrop_texture(w=512, h=288):
    """Studio backdrop: soft vertical gradient with vignette."""
    yy, xx = np.mgrid[0:h, 0:w]
    t = yy / (h - 1.0)
    top, bot = np.array([18, 22, 32.0]), np.array([60, 70, 90.0])
    img = top[None, None, :] * (1 - t[..., None]) + bot[None, None, :] * t[..., None]
    r = np.hypot((xx / (w - 1.0) - 0.5) * 1.4, (yy / (h - 1.0) - 0.45))
    img *= np.clip(1.15 - 0.9 * r, 0.35, 1.0)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def blob_texture(size=64):
    """Soft round shadow (RGBA, black with radial alpha)."""
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot((xx + 0.5) / size - 0.5, (yy + 0.5) / size - 0.5) * 2.0
    a = np.clip(1.0 - r, 0, 1) ** 1.6
    img = np.zeros((size, size, 4), np.uint8)
    img[..., 3] = (a * 230).astype(np.uint8)
    return img
