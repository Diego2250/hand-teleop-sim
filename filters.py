"""Filtros para el objetivo del brazo. Interfaz común: f.update(z, t) -> estimado.

z: medición, np.array de 3 (metros, espacio del brazo). t: tiempo de captura en segundos.
Cada filtro guarda su estado; se crea uno nuevo por corrida.
"""
import math

import numpy as np


class NoFilter:
    """Línea base: devuelve la medición tal cual."""

    def update(self, z, t):
        return np.array(z, dtype=float)


class EMA:
    """Media móvil exponencial con constante de tiempo tau_ms.

    x = x + a * (z - x), con a = 1 - exp(-dt / tau). Usar el dt real en vez de un alpha fijo hace que el
    suavizado no dependa de los fps, y tras un hueco largo (mano perdida) el filtro alcanza la medición
    en vez de arrastrar el valor viejo. Equivale a alpha = 1 - exp(-33 / tau_ms) a 30 fps.
    Para un movimiento lento el retraso agregado es aproximadamente tau_ms.
    """

    def __init__(self, tau_ms):
        self.tau = tau_ms / 1000
        self.x = None
        self.t = None

    def update(self, z, t):
        z = np.array(z, dtype=float)
        if self.x is None:
            self.x = z
        else:
            a = 1 - math.exp(-max(t - self.t, 0) / self.tau)
            self.x = self.x + a * (z - self.x)
        self.t = t
        return self.x.copy()


class OneEuro:
    """One Euro Filter (Casiez, Roussel y Vogel, CHI 2012), aplicado a cada eje por separado.

    Pasa-bajas cuyo corte sube con la velocidad: corte = min_cutoff + beta * |velocidad filtrada|.
    Con la mano quieta corta bajo y suaviza mucho; al moverse corta alto y agrega poco retraso.
    min_cutoff y d_cutoff en Hz; beta en Hz por m/s. Con beta = 0 es una EMA con tau = 1 / (2 pi min_cutoff).
    """

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def update(self, z, t):
        z = np.array(z, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = z, np.zeros_like(z), t
            return z.copy()
        dt = max(t - self.t, 1e-6)
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * ((z - self.x) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (z - self.x)
        self.t = t
        return self.x.copy()


FILTERS = {"none": NoFilter, "ema": EMA, "oneeuro": OneEuro}


def make_filter(name, **params):
    return FILTERS[name](**params)
