"""Reproduce una sesión grabada por el mismo camino que el vivo, sin visor, y mide temblor, retraso y error.

Uso:
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv

Camino: landmarks -> hand_msg (handDetector) -> hand_to_target -> clamp_reach -> IK + MuJoCo (armSim).
Cada mensaje llega al simulador en su tiempo de captura + la latencia de inferencia grabada; entre
mensajes se mantiene el último objetivo, igual que en vivo. La simulación corre a 500 Hz.

Métricas por repetición:
    temblor (fase quieta): desviación estándar del objetivo tras quitar la deriva lenta (media móvil de
        DETREND_S), descartando los primeros SETTLE_S de la ventana. También en la punta del brazo.
    error de seguimiento (fases seguir y profundidad): distancia punta-objetivo crudo en cada paso.
    retraso captura->punta: desfase que mejor alinea la punta con el objetivo crudo en tiempo de captura.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import armSim  # noqa: E402
from handDetector import hand_msg  # noqa: E402

SETTLE_S = 3.0       # se descarta el inicio de la fase quieta: la mano todavía está llegando
DETREND_S = 1.0      # ventana de la media móvil que separa la deriva lenta del temblor
MAX_LAG_MS = 400
PHASE_NAMES = {1: "quieta", 2: "seguir", 3: "profundidad"}
AXES = ("prof", "lat", "vert")  # x, y, z del brazo


def load_session(path):
    with open(path) as f:
        header = f.readline().strip().split(",")
    d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    c = {name: i for i, name in enumerate(header)}

    def points(r, first):
        return [SimpleNamespace(x=r[first + 3 * i], y=r[first + 3 * i + 1], z=r[first + 3 * i + 2]) for i in range(21)]

    msgs = [hand_msg(points(r, c["x0"]), points(r, c["wx0"]), int(r[c["t_ms"]]), r[c["w"]], r[c["h"]]) for r in d]
    return {
        "t": d[:, c["t_ms"]],
        "arrival": d[:, c["t_ms"]] + d[:, c["lat_ms"]],
        "rep": d[:, c["rep"]].astype(int),
        "phase": d[:, c["phase"]].astype(int),
        "rec": d[:, c["rec"]].astype(int),
        "t_phase": d[:, c["t_phase_ms"]],
        "msgs": msgs,
    }


def raw_targets(session):
    """Objetivo sin filtro por mensaje, igual que en armSim.main()."""
    return np.array([armSim.clamp_reach(armSim.hand_to_target(m)) for m in session["msgs"]])


def simulate(session, commands):
    """commands: (N, 3) objetivo enviado al brazo por cada mensaje. Devuelve t_sim (ms), punta (M, 3) e índice del mensaje vigente."""
    model, data, ik_data, site_id = armSim.make_sim()
    dt = model.opt.timestep * 1000
    t_sim = np.arange(session["arrival"][0], session["arrival"][-1] + 500, dt)
    current = np.searchsorted(session["arrival"], t_sim, side="right") - 1
    tip = np.empty((len(t_sim), 3))
    for k, i in enumerate(current):
        armSim.control_step(model, data, ik_data, site_id, armSim.HOME_TARGET if i < 0 else commands[i])
        tip[k] = data.site_xpos[site_id]
    if not np.all(np.isfinite(tip)):
        raise RuntimeError("La simulación se volvió inestable")
    return t_sim, tip, current


def detrended_std(x, fs):
    """Desviación estándar por eje después de restar una media móvil centrada de DETREND_S."""
    k = max(3, int(round(DETREND_S * fs)) | 1)
    trend = np.column_stack([np.convolve(x[:, i], np.ones(k) / k, mode="valid") for i in range(x.shape[1])])
    return (x[k // 2: k // 2 + len(trend)] - trend).std(axis=0)


def estimate_lag(ref, sig, k0, k1, dt, max_lag_ms=MAX_LAG_MS):
    """Retraso (ms) que minimiza el error cuadrático medio entre sig(t + lag) y ref(t) en los pasos [k0, k1)."""
    n = int(max_lag_ms / dt)
    k1 = min(k1, len(sig) - n)
    mse = [np.mean(np.sum((sig[k0 + s:k1 + s] - ref[k0:k1]) ** 2, axis=1)) for s in range(n + 1)]
    return int(np.argmin(mse)) * dt


def norm3(stds):
    return float(np.sqrt(np.sum(np.square(stds))))


def evaluate(session, raw, commands):
    """raw: objetivo sin filtro (referencia). commands: lo que se le manda al brazo (raw en la línea base)."""
    t_sim, tip, current = simulate(session, commands)
    dt = t_sim[1] - t_sim[0]
    ref_idx = np.maximum(np.searchsorted(session["t"], t_sim, side="right") - 1, 0)
    ref_capture = raw[ref_idx]  # objetivo crudo según el momento de captura
    in_msg = lambda mask: mask[np.maximum(current, 0)] & (current >= 0)  # noqa: E731

    results = []
    for rep in sorted(set(session["rep"])):
        for phase in PHASE_NAMES:
            win = (session["rep"] == rep) & (session["phase"] == phase) & (session["rec"] == 1)
            if not win.any():
                continue
            row = {"rep": rep, "phase": phase, "frames": int(win.sum())}
            if phase == 1:
                steady = win & (session["t_phase"] >= SETTLE_S * 1000)
                cmd = commands[steady] * 1000
                fs = (steady.sum() - 1) / (np.ptp(session["t"][steady]) / 1000)
                row["tremor"] = detrended_std(cmd, fs)
                row["jitter"] = np.std(np.diff(cmd, axis=0), axis=0) / np.sqrt(2)
                row["tremor_tip"] = detrended_std(tip[in_msg(steady)] * 1000, 1000 / dt)
            else:
                steps = in_msg(win)
                err = np.linalg.norm(tip[steps] - raw[current[steps]], axis=1) * 1000
                row["err_median"], row["err_p95"] = float(np.median(err)), float(np.percentile(err, 95))
                k = np.flatnonzero(win[ref_idx])
                row["lag_ms"] = estimate_lag(ref_capture, tip, k[0], k[-1] + 1, dt)
            results.append(row)
    return results


def print_table(results, title):
    print(f"\n{title}")
    still = [r for r in results if r["phase"] == 1]
    print(f"\nTemblor, fase quieta (mm; desde {SETTLE_S:.0f} s, quitando la deriva de {DETREND_S:.0f} s)")
    print(f"  {'':6s} objetivo: {'  '.join(f'{a:>5s}' for a in AXES)}     3D | punta 3D | jitter 3D")
    for r in still:
        print(f"  rep {r['rep']}            {'  '.join(f'{v:5.2f}' for v in r['tremor'])}  {norm3(r['tremor']):5.2f} |"
              f"    {norm3(r['tremor_tip']):5.2f} |     {norm3(r['jitter']):5.2f}")
    if still:
        mean = lambda key: np.mean([r[key] for r in still], axis=0)  # noqa: E731
        print(f"  media            {'  '.join(f'{v:5.2f}' for v in mean('tremor'))}  {norm3(mean('tremor')):5.2f} |"
              f"    {norm3(mean('tremor_tip')):5.2f} |     {norm3(mean('jitter')):5.2f}")

    print("\nSeguimiento (punta contra objetivo crudo)")
    print(f"  {'fase':12s} {'rep':>3s} | error mediana   p95 (mm) | retraso captura->punta (ms)")
    for phase in (2, 3):
        rows = [r for r in results if r["phase"] == phase]
        for r in rows:
            print(f"  {PHASE_NAMES[phase]:12s} {r['rep']:3d} | {r['err_median']:13.2f} {r['err_p95']:9.2f} | {r['lag_ms']:8.0f}")
        if rows:
            print(f"  {PHASE_NAMES[phase]:12s} {'med':>3s} | {np.mean([r['err_median'] for r in rows]):13.2f}"
                  f" {np.mean([r['err_p95'] for r in rows]):9.2f} | {np.mean([r['lag_ms'] for r in rows]):8.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="CSV grabado con tools/record_session.py")
    args = parser.parse_args()

    session = load_session(args.session)
    raw = raw_targets(session)
    print_table(evaluate(session, raw, raw), f"Línea base (sin filtro): {args.session.name}")


if __name__ == "__main__":
    main()
