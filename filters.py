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


FILTERS = {"none": NoFilter, "ema": EMA}


def make_filter(name, **params):
    return FILTERS[name](**params)
