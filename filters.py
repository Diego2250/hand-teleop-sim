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


class Kalman:
    """Filtro de Kalman con modelo de velocidad constante, aplicado a cada eje por separado.

    Estado por eje: posición y velocidad. Entre mediciones predice que la mano sigue a la misma velocidad,
    con incertidumbre por aceleración aleatoria de desviación accel (m/s²: qué tan brusco puede ser el
    movimiento). Corrige con cada medición suponiendo un ruido de desviación noise_mm. En régimen estable
    solo importa la proporción entre las dos. A diferencia de la EMA, sigue una velocidad constante sin retraso.
    """

    def __init__(self, accel, noise_mm=1.0):
        self.q = accel ** 2
        self.r = (noise_mm / 1000) ** 2
        self.x = None  # (2, 3): fila 0 posición, fila 1 velocidad, una columna por eje
        self.P = None  # covarianza 2x2, igual para los tres ejes (mismo modelo y mismas mediciones)
        self.t = None

    def update(self, z, t):
        z = np.array(z, dtype=float)
        if self.x is None:
            self.x = np.vstack([z, np.zeros(3)])
            self.P = np.diag([self.r, 1.0])  # velocidad inicial desconocida
            self.t = t
            return z.copy()
        dt = max(t - self.t, 1e-6)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.x = F @ self.x                        # predicción
        self.P = F @ self.P @ F.T + Q
        k = self.P[:, 0] / (self.P[0, 0] + self.r)  # ganancia de Kalman: cuánto creerle a la medición
        self.x = self.x + np.outer(k, z - self.x[0])  # corrección
        self.P = self.P - np.outer(k, self.P[0])
        self.t = t
        return self.x[0].copy()


FILTERS = {"none": NoFilter, "ema": EMA, "oneeuro": OneEuro, "kalman": Kalman}


def make_filter(name, **params):
    return FILTERS[name](**params)


def add_arguments(parser):
    """Argumentos de línea de comandos para elegir filtro; por defecto, el punto "rápido" de FINAL_CONFIGS."""
    parser.add_argument("--filter", choices=FILTERS, default="none")
    parser.add_argument("--tau", type=float, default=50.0, help="EMA: constante de tiempo en ms")
    parser.add_argument("--min-cutoff", type=float, default=0.5, help="One Euro: corte con la mano quieta, Hz")
    parser.add_argument("--beta", type=float, default=30.0, help="One Euro: aumento del corte por m/s, Hz")
    parser.add_argument("--accel", type=float, default=0.05, help="Kalman: aceleración aleatoria de la mano, m/s²")
    parser.add_argument("--noise-mm", type=float, default=1.0, help="Kalman: ruido de la medición, mm")


def params_from_arguments(args):
    """Parámetros del filtro elegido con add_arguments."""
    return {"none": {}, "ema": {"tau_ms": args.tau},
            "oneeuro": {"min_cutoff": args.min_cutoff, "beta": args.beta},
            "kalman": {"accel": args.accel, "noise_mm": args.noise_mm}}[args.filter]
