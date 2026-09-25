"""Reproduce una sesión grabada por el mismo camino que el vivo, sin visor, y mide temblor, retraso y error.

Uso:
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv                      # línea base, tabla detallada
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv --filter ema --tau 100
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv --filter oneeuro --min-cutoff 1 --beta 30
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv --filter kalman --accel 0.01
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv --sweep              # barrido de filtros, tabla resumen
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv --final              # solo las configuraciones finales

Camino: landmarks -> hand_msg (handDetector) -> hand_to_target -> filtro -> clamp_reach -> IK + MuJoCo (armSim).
Cada mensaje llega al simulador en su tiempo de captura + la latencia de inferencia grabada; entre
mensajes se mantiene el último objetivo, igual que en vivo. La simulación corre a 500 Hz.

Métricas por repetición:
    temblor (fase quieta): desviación estándar del objetivo enviado al brazo tras quitar la deriva lenta
        (media móvil de DETREND_S), descartando los primeros SETTLE_S de la ventana. También en la punta.
    error de seguimiento (fases seguir y profundidad): distancia punta-objetivo crudo (sin filtro) en cada paso.
    retraso del filtro: desfase que mejor alinea el objetivo filtrado con el crudo. Solo es un buen resumen
        para filtros cuya salida es una copia atrasada de la entrada (EMA, One Euro); el Kalman se pasa y
        deforma, por eso los filtros se comparan contra el error de seguimiento.
    retraso captura->punta: desfase que mejor alinea la punta con el objetivo crudo en tiempo de captura.
    alcance y adelanto (fase saltos): tras cada salto del punto guía, cuánto tarda la punta en alcanzar a la mano
        (quedar a menos de TOL_MM de ella y no volver a separarse) y cuánto se le adelanta en la dirección del
        salto. Se compara contra la mano en cada instante porque la mano misma sigue corrigiendo y derivando.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import armSim  # noqa: E402
import filters  # noqa: E402
from handDetector import hand_msg  # noqa: E402

SETTLE_S = 3.0       # se descarta el inicio de la fase quieta: la mano todavía está llegando
DETREND_S = 1.0      # ventana de la media móvil que separa la deriva lenta del temblor
MAX_LAG_MS = 800
MAX_LEAD_MS = 150    # el Kalman puede adelantarse al predecir
EMA_TAUS = (10, 20, 33, 50, 75, 100, 150, 200, 300)
ONEEURO_CUTOFFS = (0.2, 0.3, 0.5, 1, 2, 4, 8)
ONEEURO_BETAS = (0, 3, 10, 30, 100, 300, 1000)
KALMAN_ACCELS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2)
COMPARE_ERR_MM = (3, 5, 10, 20)

# Configuraciones finales, elegidas con la sesión de afinación session_20260924_214146.csv ANTES de grabar la
# sesión de prueba: dos puntos de operación con error de seguimiento comparable en el círculo (~5 y ~9 mm).
# fc = 0.5 Hz y no 0.2 Hz: la ganancia en temblor era mínima y 0.2 Hz está en el borde de la cuadrícula probada.
FINAL_CONFIGS = [
    ("-", "none", {}),
    ("rápido", "ema", {"tau_ms": 50}),
    ("rápido", "oneeuro", {"min_cutoff": 0.5, "beta": 30}),
    ("rápido", "kalman", {"accel": 0.05}),
    ("suave", "ema", {"tau_ms": 75}),
    ("suave", "oneeuro", {"min_cutoff": 0.5, "beta": 10}),
    ("suave", "kalman", {"accel": 0.02}),
]
SWEEP = ([("none", {})]
         + [("ema", {"tau_ms": tau}) for tau in EMA_TAUS]
         + [("oneeuro", {"min_cutoff": fc, "beta": b}) for fc in ONEEURO_CUTOFFS for b in ONEEURO_BETAS]
         + [("kalman", {"accel": a}) for a in KALMAN_ACCELS])
TOL_MM = 5.0         # tolerancia del alcance
HOLD_S = 0.7         # la dirección de cada salto va de la posición previa a la mediana de este último tramo
PHASE_NAMES = {1: "quieta", 2: "seguir", 3: "profundidad", 4: "saltos"}
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
        "guide": d[:, [c["gu"], c["gv"]]],
        "msgs": msgs,
    }


def mapped_targets(session):
    """Objetivo mapeado por mensaje, antes de filtrar y recortar."""
    return np.array([armSim.hand_to_target(m) for m in session["msgs"]])


def commands_for(session, mapped, filt):
    """Lo que se le manda al brazo por mensaje: filtro sobre el objetivo mapeado y después el recorte de alcance."""
    return np.array([armSim.clamp_reach(filt.update(z, t / 1000)) for z, t in zip(mapped, session["t"])])


def raw_targets(session):
    """Objetivo sin filtro por mensaje, igual que en armSim.main()."""
    return commands_for(session, mapped_targets(session), filters.NoFilter())


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


def estimate_filter_lag(t, raw, cmd, mask, max_lag_ms=MAX_LAG_MS, max_lead_ms=MAX_LEAD_MS):
    """Retraso (ms, resolución 1 ms; negativo = adelanto) del objetivo filtrado contra el crudo, en los mensajes de mask.

    Compara cmd(t_i) contra raw(t_i - lag) interpolando raw linealmente entre frames: comparar las dos
    señales en escalera redondea el retraso a múltiplos de un frame (33 ms).
    """
    ti, ci = t[mask], cmd[mask]
    shifts = range(-max_lead_ms, max_lag_ms + 1)
    mse = [np.mean(np.sum((ci - np.column_stack([np.interp(ti - s, t, raw[:, a]) for a in range(3)])) ** 2, axis=1))
           for s in shifts]
    return shifts[int(np.argmin(mse))]


def jump_metrics(session, raw, t_sim, tip, win):
    """Medianas sobre los saltos del punto guía dentro de win: alcance (ms) y adelanto (mm) de la punta respecto a la mano."""
    t, guide = session["t"], session["guide"]
    idx = np.flatnonzero(win)
    jumps = [idx[k + 1] for k in range(len(idx) - 1) if np.any(guide[idx[k + 1]] != guide[idx[k]])]
    ends = [t[j] for j in jumps[1:]] + [t[idx[-1]]]
    catchup, lead = [], []
    for j, t1 in zip(jumps, ends):
        t0 = t[j]
        start = np.median(raw[(t >= t0 - 300) & (t < t0)], axis=0)
        final = np.median(raw[(t >= t1 - HOLD_S * 1000) & (t <= t1)], axis=0)
        direction = (final - start) / np.linalg.norm(final - start)
        arm = (t_sim >= t0) & (t_sim <= t1)
        times = t_sim[arm]
        hand = np.column_stack([np.interp(times, t, raw[:, a]) for a in range(3)])  # misma rejilla que la punta
        gap = tip[arm] - hand
        outside = np.flatnonzero(np.linalg.norm(gap, axis=1) * 1000 > TOL_MM)
        if not len(outside):
            catchup.append(0.0)
        elif outside[-1] + 1 < len(times):
            catchup.append(float(times[outside[-1] + 1] - t0))
        else:
            catchup.append(float(t1 - t0))  # no la alcanzó antes del siguiente salto
        lead.append(max(0.0, float(np.max(gap @ direction)) * 1000))
    return {"catchup_ms": float(np.median(catchup)), "lead_mm": float(np.median(lead)), "jumps": len(jumps)}


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
            elif phase == 4:
                row.update(jump_metrics(session, raw, t_sim, tip, win))
            else:
                steps = in_msg(win)
                err = np.linalg.norm(tip[steps] - raw[current[steps]], axis=1) * 1000
                row["err_median"], row["err_p95"] = float(np.median(err)), float(np.percentile(err, 95))
                k = np.flatnonzero(win[ref_idx])
                row["lag_ms"] = estimate_lag(ref_capture, tip, k[0], k[-1] + 1, dt)
                row["filter_lag_ms"] = estimate_filter_lag(session["t"], raw, commands, win)
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
    print(f"  {'fase':12s} {'rep':>3s} | error mediana   p95 (mm) | retraso filtro | retraso captura->punta (ms)")
    for phase in (2, 3):
        rows = [r for r in results if r["phase"] == phase]
        for r in rows:
            print(f"  {PHASE_NAMES[phase]:12s} {r['rep']:3d} | {r['err_median']:13.2f} {r['err_p95']:9.2f} |"
                  f" {r['filter_lag_ms']:11.0f}    | {r['lag_ms']:8.0f}")
        if rows:
            mean = lambda key: np.mean([r[key] for r in rows])  # noqa: E731
            print(f"  {PHASE_NAMES[phase]:12s} {'med':>3s} | {mean('err_median'):13.2f} {mean('err_p95'):9.2f} |"
                  f" {mean('filter_lag_ms'):11.0f}    | {mean('lag_ms'):8.0f}")

    jumps = [r for r in results if r["phase"] == 4]
    if jumps:
        print(f"\nSaltos (medianas por repetición; alcance = la punta queda a menos de {TOL_MM:g} mm de la mano)")
        print(f"  {'rep':>5s} | saltos | alcance (ms) | adelanto sobre la mano (mm)")
        for r in jumps:
            print(f"  {r['rep']:5d} | {r['jumps']:6d} | {r['catchup_ms']:12.0f} | {r['lead_mm']:10.1f}")


def summarize(results):
    """Una fila por configuración de filtro: medias sobre las repeticiones."""
    by_phase = lambda phase: [r for r in results if r["phase"] == phase]  # noqa: E731
    mean = lambda phase, key: float(np.mean([r[key] for r in by_phase(phase)], axis=0))  # noqa: E731
    still = by_phase(1)
    extra = {}
    if by_phase(4):
        extra = {"catchup": mean(4, "catchup_ms"), "lead": mean(4, "lead_mm")}
    return extra | {
        "tremor": norm3(np.mean([r["tremor"] for r in still], axis=0)),
        "tremor_prof": float(np.mean([r["tremor"][0] for r in still])),
        "jitter": norm3(np.mean([r["jitter"] for r in still], axis=0)),
        "filter_lag": mean(2, "filter_lag_ms"),
        "lag": mean(2, "lag_ms"),
        "err_follow": mean(2, "err_median"),
        "err_depth": mean(3, "err_median"),
    }


def per_rep(results):
    """Valores por repetición de las métricas principales, para mostrar cuánto varían."""
    by = lambda phase, key: [r[key] for r in results if r["phase"] == phase]  # noqa: E731
    out = {"tremor": [norm3(v) for v in by(1, "tremor")], "jitter": [norm3(v) for v in by(1, "jitter")],
           "err_follow": by(2, "err_median"), "err_depth": by(3, "err_median")}
    if any(r["phase"] == 4 for r in results):
        out |= {"catchup": by(4, "catchup_ms"), "lead": by(4, "lead_mm")}
    return out


def run_final(session, title):
    mapped = mapped_targets(session)
    raw = commands_for(session, mapped, filters.NoFilter())
    cols = [("tremor", "temblor mm", 2), ("jitter", "jitter mm", 2), ("err_follow", "error círculo mm", 1),
            ("err_depth", "error prof. mm", 1), ("catchup", "alcance ms", 0), ("lead", "se pasa mm", 1)]
    rows = []
    for point, name, params in FINAL_CONFIGS:
        reps = per_rep(evaluate(session, raw, commands_for(session, mapped, filters.make_filter(name, **params))))
        rows.append((point, label({"filter": name, **params}), reps))
    base = rows[0][2]  # FINAL_CONFIGS empieza sin filtro
    for _, _, reps in rows:  # comparación pareada: cada repetición contra sí misma sin filtro
        reps["tremor_change"] = [100 * (t / b - 1) for t, b in zip(reps["tremor"], base["tremor"])]
    cols = [c for c in cols if c[0] in base]
    cols.insert(1, ("tremor_change", "temblor vs sin filtro %", 0))
    print(f"\n{title}")
    print("media de las repeticiones (mínimo a máximo); 'temblor vs sin filtro' compara cada repetición consigo misma")
    print(f"  {'punto':7s} {'filtro':21s} | " + " | ".join(f"{head:>21s}" for _, head, _ in cols))
    for point, name, reps in rows:
        cells = [f"{np.mean(reps[k]):.{d}f} ({min(reps[k]):.{d}f} a {max(reps[k]):.{d}f})".rjust(21) for k, _, d in cols]
        print(f"  {point:7s} {name:21s} | " + " | ".join(cells))


def run_sweep(session):
    mapped = mapped_targets(session)
    raw = commands_for(session, mapped, filters.NoFilter())
    rows = []
    for i, (name, params) in enumerate(SWEEP, 1):
        print(f"  simulando {i}/{len(SWEEP)}", end="\r", file=sys.stderr)
        cmd = commands_for(session, mapped, filters.make_filter(name, **params))
        rows.append({"filter": name, **params, **summarize(evaluate(session, raw, cmd))})
    return rows


def label(row):
    if row["filter"] == "ema":
        return f"EMA tau={row['tau_ms']:g} ms"
    if row["filter"] == "oneeuro":
        return f"1€ fc={row['min_cutoff']:g} Hz β={row['beta']:g}"
    if row["filter"] == "kalman":
        return f"Kalman a={row['accel']:g} m/s²"
    return "sin filtro"


FILTER_NAMES = {"ema": "EMA", "kalman": "Kalman", "oneeuro": "One Euro"}


def frontier(rows, cost="err_follow"):
    """Configuraciones que nadie supera: para su costo (error de seguimiento o retraso), ninguna otra deja menos temblor."""
    best, out = float("inf"), []
    for r in sorted(rows, key=lambda r: (r[cost], r["tremor"])):
        if r["tremor"] < best:
            best = r["tremor"]
            out.append(r)
    return out


def print_rows(rows):
    jumps = "catchup" in rows[0]
    print(f"  {'filtro':21s} | temblor 3D (prof) | jitter 3D | retraso filtro | captura->punta | error círculo  prof."
          + (" | alcance adelanto" if jumps else ""))
    print(f"  {'':21s} | {'mm':>17s} | {'mm':>9s} | {'ms':>14s} | {'ms':>14s} | {'mm':>13s} {'mm':>6s}"
          + (f" | {'ms':>7s} {'mm':>8s}" if jumps else ""))
    for r in rows:
        print(f"  {label(r):21s} | {r['tremor']:10.2f} ({r['tremor_prof']:4.2f}) | {r['jitter']:9.2f} |"
              f" {r['filter_lag']:14.0f} | {r['lag']:14.0f} | {r['err_follow']:13.2f} {r['err_depth']:6.2f}"
              + (f" | {r['catchup']:7.0f} {r['lead']:8.1f}" if jumps else ""))


def print_sweep(rows, title):
    base = [r for r in rows if r["filter"] == "none"]
    groups = {name: [r for r in rows if r["filter"] == name] for name in FILTER_NAMES}
    print(f"\n{title}  (medias de las 3 repeticiones)")
    print("\nEMA")
    print_rows(base + groups["ema"])
    if groups["kalman"]:
        print("\nKalman, velocidad constante (ruido de medición 1 mm; su retraso no es un atraso puro, ver el error)")
        print_rows(base + groups["kalman"])
    if groups["oneeuro"]:
        print(f"\nOne Euro: frontera de {len(groups['oneeuro'])} combinaciones (fc en {ONEEURO_CUTOFFS} Hz, β en {ONEEURO_BETAS})")
        print_rows(frontier(base + groups["oneeuro"]))

    names = [n for n in FILTER_NAMES if groups[n]]
    curves = {n: frontier(base + groups[n]) for n in names}
    print("\nCon el mismo error de seguimiento en el círculo: temblor / jitter que deja cada filtro")
    print("(interpolado sobre la frontera de cada uno; '-' = fuera del rango probado)")
    print(f"  {'error':>7s} | " + " | ".join(f"{FILTER_NAMES[n]:>16s}" for n in names))
    for err in COMPARE_ERR_MM:
        cells = []
        for n in names:
            x = [r["err_follow"] for r in curves[n]]
            if not x[0] <= err <= x[-1]:
                cells.append(f"{'-':>16s}")
                continue
            at = lambda key: float(np.interp(err, x, [r[key] for r in curves[n]]))  # noqa: E731
            cells.append(f"{at('tremor'):.2f} / {at('jitter'):.2f} mm".rjust(16))
        print(f"  {err:4g} mm | " + " | ".join(cells))


PLOT_MAX_ERR_MM = 40
PLOT_LABELED = ((0.5, 30), (0.5, 10))  # combinaciones de One Euro con etiqueta en la gráfica


def plot_sweep(rows, path, title):
    """Temblor y jitter contra el error de seguimiento: EMA y Kalman (líneas) y One Euro (todas y su frontera)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"  # paleta de referencia de la skill dataviz, slots 1 a 3
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.edgecolor": grid})

    visible = lambda rs: [r for r in rs if r["err_follow"] <= PLOT_MAX_ERR_MM]  # noqa: E731
    by_err = lambda rs: sorted(rs, key=lambda r: r["err_follow"])  # noqa: E731
    base = [r for r in rows if r["filter"] == "none"]
    group = lambda name: [r for r in rows if r["filter"] == name]  # noqa: E731
    ema, kalman = visible(by_err(base + group("ema"))), visible(by_err(base + group("kalman")))
    euro, front = visible(group("oneeuro")), visible(frontier(base + group("oneeuro")))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, facecolor=surface)
    line = dict(linewidth=1.0, marker="o", markersize=4.5, markeredgecolor=surface, markeredgewidth=1.0)
    panels = (("tremor", "Temblor con la mano quieta (ventana de 1 s)"), ("jitter", "Jitter frame a frame con la mano quieta"))
    for ax, (key, heading) in zip(axes, panels):
        xs = lambda rs: [r["err_follow"] for r in rs]  # noqa: E731
        ys = lambda rs: [r[key] for r in rs]  # noqa: E731
        ax.scatter(xs(euro), ys(euro), s=14, color=orange, alpha=0.3, linewidths=0, label="One Euro, todas las combinaciones")
        ax.plot(xs(ema), ys(ema), color=blue, label="EMA", **line)
        ax.plot(xs(kalman), ys(kalman), color=aqua, label="Kalman (velocidad constante)", **line)
        ax.plot(xs(front), ys(front), color=orange, label="One Euro, frontera", **line)

        note = dict(textcoords="offset points", color=ink2, fontsize=8)
        name = dict(textcoords="offset points", color=ink, fontsize=9)
        ax.annotate("sin filtro", (base[0]["err_follow"], base[0][key]), xytext=(-8, 8), **note)
        for rs, text, offset in ((ema, "EMA", (6, 2)), (kalman, "Kalman", (6, -4)), (front, "One Euro", (6, -10))):
            ax.annotate(text, (rs[-1]["err_follow"], rs[-1][key]), xytext=offset, **name)
        for r in front if key == "tremor" else []:  # solo en el primer panel, para no encimar etiquetas
            if (r.get("min_cutoff"), r.get("beta")) in PLOT_LABELED:
                ax.annotate(f"fc={r['min_cutoff']:g} Hz, β={r['beta']:g}", (r["err_follow"], r[key]), xytext=(4, -12), **note)

        ax.set_title(heading, loc="left", fontsize=10, color=ink)
        ax.set_ylabel("mm (objetivo del brazo)")
        ax.set_xlabel("error de seguimiento al moverse en el círculo (mm, mediana)")
        ax.set_facecolor(surface)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, PLOT_MAX_ERR_MM * 1.12)
        ax.grid(True, color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8, labelcolor=ink2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=ink)
    fig.text(0.01, 0.01, f"No se muestran configuraciones con más de {PLOT_MAX_ERR_MM} mm de error. Más abajo y más a la izquierda es mejor.",
             fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    print(f"\nGráfica guardada en {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="CSV grabado con tools/record_session.py")
    parser.add_argument("--filter", choices=filters.FILTERS, default="none")
    parser.add_argument("--tau", type=float, default=100.0, help="EMA: constante de tiempo en ms")
    parser.add_argument("--min-cutoff", type=float, default=1.0, help="One Euro: corte con la mano quieta, Hz")
    parser.add_argument("--beta", type=float, default=30.0, help="One Euro: aumento del corte por m/s, Hz")
    parser.add_argument("--accel", type=float, default=0.01, help="Kalman: aceleración aleatoria de la mano, m/s²")
    parser.add_argument("--noise-mm", type=float, default=1.0, help="Kalman: ruido de la medición, mm")
    parser.add_argument("--sweep", action="store_true", help="prueba sin filtro, EMA, One Euro y Kalman con varios parámetros")
    parser.add_argument("--plot", type=Path, help="con --sweep, guarda la gráfica en este PNG")
    parser.add_argument("--final", action="store_true", help="evalúa solo las configuraciones finales (FINAL_CONFIGS)")
    args = parser.parse_args()

    session = load_session(args.session)
    if args.final:
        run_final(session, f"Configuraciones finales: {args.session.name}")
        return
    if args.sweep:
        rows = run_sweep(session)
        print_sweep(rows, f"Barrido de filtros: {args.session.name}")
        if args.plot:
            plot_sweep(rows, args.plot, f"EMA, Kalman y One Euro ({args.session.name}, medias de 3 repeticiones)")
        return

    mapped = mapped_targets(session)
    raw = commands_for(session, mapped, filters.NoFilter())
    params = {"none": {}, "ema": {"tau_ms": args.tau},
              "oneeuro": {"min_cutoff": args.min_cutoff, "beta": args.beta},
              "kalman": {"accel": args.accel, "noise_mm": args.noise_mm}}[args.filter]
    cmd = commands_for(session, mapped, filters.make_filter(args.filter, **params))
    title = "Línea base (sin filtro)" if args.filter == "none" else label({"filter": args.filter, **params})
    print_table(evaluate(session, raw, cmd), f"{title}: {args.session.name}")


if __name__ == "__main__":
    main()
