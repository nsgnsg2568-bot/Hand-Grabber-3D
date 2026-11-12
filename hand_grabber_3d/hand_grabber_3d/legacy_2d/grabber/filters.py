"""One-Euro filter: removes jitter when the hand is still, adds almost no lag when it moves fast.
Reference: Casiez et al., "1 Euro Filter", CHI 2012."""
import math


class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.008, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = 0.0
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        dt = max(t - self._t, 1e-4)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class OneEuro2D:
    def __init__(self, **kw):
        self.fx = OneEuro(**kw)
        self.fy = OneEuro(**kw)

    def __call__(self, p, t):
        return (self.fx(p[0], t), self.fy(p[1], t))
