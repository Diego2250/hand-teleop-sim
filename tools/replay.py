"""Reproduce una sesión grabada por el mismo camino que el vivo, sin visor, y mide temblor, retraso y error.

Uso:
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv                      # línea base, tabla detallada
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv --filter ema --tau 100
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv --filter oneeuro --min-cutoff 1 --beta 30
    python tools/replay.py data/session_XXXXXXXX_XXXXXX.csv --sweep              # barrido de filtros, tabla resumen

Camino: landmarks -> hand_msg (handDetector) -> hand_to_target -> filtro -> clamp_reach -> IK + MuJoCo (armSim).
Cada mensaje llega al simulador en su tiempo de captura + la latencia de inferencia grabada; entre
mensajes se mantiene el último objetivo, igual que en vivo. La simulación corre a 500 Hz.

Métricas por repetición:
    temblor (fase quieta): desviación estándar del objetivo enviado al brazo tras quitar la deriva lenta
        (media móvil de DETREND_S), descartando los primeros SETTLE_S de la ventana. También en la punta.
    error de seguimiento (fases seguir y profundidad): distancia punta-objetivo crudo (sin filtro) en cada paso.
    retraso del filtro: desfase que mejor alinea el objetivo filtrado con el crudo.
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
import filters  # noqa: E402
from handDetector import hand_msg  # noqa: E402

SETTLE_S = 3.0       # se descarta el inicio de la fase quieta: la mano todavía está llegando
DETREND_S = 1.0      # ventana de la media móvil que separa la deriva lenta del temblor
MAX_LAG_MS = 800
EMA_TAUS = (10, 20, 33, 50, 75, 100, 150, 200, 300)
ONEEURO_CUTOFFS = (0.2, 0.3, 0.5, 1, 2, 4, 8)
ONEEURO_BETAS = (0, 3, 10, 30, 100, 300, 1000)
SWEEP = ([("none", {})]
         + [("ema", {"tau_ms": tau}) for tau in EMA_TAUS]
         + [("oneeuro", {"min_cutoff": fc, "beta": b}) for fc in ONEEURO_CUTOFFS for b in ONEEURO_BETAS])
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


def estimate_filter_lag(t, raw, cmd, mask, max_lag_ms=MAX_LAG_MS):
    """Retraso (ms, resolución 1 ms) del objetivo filtrado contra el crudo, en los mensajes de mask.

    Compara cmd(t_i) contra raw(t_i - lag) interpolando raw linealmente entre frames: comparar las dos
    señales en escalera redondea el retraso a múltiplos de un frame (33 ms).
    """
    ti, ci = t[mask], cmd[mask]
    mse = [np.mean(np.sum((ci - np.column_stack([np.interp(ti - s, t, raw[:, a]) for a in range(3)])) ** 2, axis=1))
           for s in range(max_lag_ms + 1)]
    return int(np.argmin(mse))


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


def summarize(results):
    """Una fila por configuración de filtro: medias sobre las repeticiones."""
    by_phase = lambda phase: [r for r in results if r["phase"] == phase]  # noqa: E731
    mean = lambda phase, key: float(np.mean([r[key] for r in by_phase(phase)], axis=0))  # noqa: E731
    still = by_phase(1)
    return {
        "tremor": norm3(np.mean([r["tremor"] for r in still], axis=0)),
        "tremor_prof": float(np.mean([r["tremor"][0] for r in still])),
        "jitter": norm3(np.mean([r["jitter"] for r in still], axis=0)),
        "filter_lag": mean(2, "filter_lag_ms"),
        "lag": mean(2, "lag_ms"),
        "err_follow": mean(2, "err_median"),
        "err_depth": mean(3, "err_median"),
    }


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
    return "sin filtro"


def frontier(rows):
    """Configuraciones que nadie supera: para su retraso, ninguna otra deja menos temblor."""
    best, out = float("inf"), []
    for r in sorted(rows, key=lambda r: (r["filter_lag"], r["tremor"])):
        if r["tremor"] < best:
            best = r["tremor"]
            out.append(r)
    return out


def print_rows(rows):
    print(f"  {'filtro':21s} | temblor 3D (prof) | jitter 3D | retraso filtro | captura->punta | error círculo  prof.")
    print(f"  {'':21s} | {'mm':>17s} | {'mm':>9s} | {'ms':>14s} | {'ms':>14s} | {'mm':>13s} {'mm':>6s}")
    for r in rows:
        print(f"  {label(r):21s} | {r['tremor']:10.2f} ({r['tremor_prof']:4.2f}) | {r['jitter']:9.2f} |"
              f" {r['filter_lag']:14.0f} | {r['lag']:14.0f} | {r['err_follow']:13.2f} {r['err_depth']:6.2f}")


def print_sweep(rows, title):
    base = [r for r in rows if r["filter"] == "none"]
    ema = base + [r for r in rows if r["filter"] == "ema"]
    euro = [r for r in rows if r["filter"] == "oneeuro"]
    print(f"\n{title}  (medias de las 3 repeticiones)")
    print("\nEMA")
    print_rows(ema)
    if not euro:
        return
    print(f"\nOne Euro: frontera de {len(euro)} combinaciones (fc en {ONEEURO_CUTOFFS} Hz, β en {ONEEURO_BETAS})")
    front = frontier(base + euro)
    print_rows(front)

    ema_sorted = sorted(ema, key=lambda r: r["filter_lag"])
    lags = [r["filter_lag"] for r in ema_sorted]
    at = lambda key, lag: float(np.interp(lag, lags, [r[key] for r in ema_sorted]))  # noqa: E731
    print("\nCon el mismo retraso del filtro, temblor que deja cada uno (EMA interpolada)")
    print(f"  {'retraso':>7s} | {'EMA':>7s} | {'One Euro':>8s} | cambio | jitter EMA -> One Euro")
    for r in front:
        if r["filter"] != "oneeuro" or r["filter_lag"] > lags[-1]:
            continue
        e = at("tremor", r["filter_lag"])
        print(f"  {r['filter_lag']:4.0f} ms | {e:4.2f} mm | {r['tremor']:5.2f} mm | {100 * (r['tremor'] - e) / e:+5.0f}% |"
              f" {at('jitter', r['filter_lag']):.2f} -> {r['jitter']:.2f} mm   ({label(r)})")


def plot_sweep(rows, path, title):
    """Dos paneles con el mismo eje x (retraso del filtro): lo que se gana (temblor) y lo que se paga (error)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"  # paleta de referencia de la skill dataviz, slots 1 a 3
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.edgecolor": grid})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, facecolor=surface)
    lag = [r["filter_lag"] for r in rows]
    line = dict(linewidth=1.0, marker="o", markersize=4.5, markeredgecolor=surface, markeredgewidth=1.0)

    ax1.plot(lag, [r["tremor"] for r in rows], color=blue, label="Temblor (ventana de 1 s)", **line)
    ax1.plot(lag, [r["jitter"] for r in rows], color=orange, label="Jitter (frame a frame)", **line)
    ax2.plot(lag, [r["err_follow"] for r in rows], color=aqua, **line)

    for r in rows:  # etiquetas selectivas: línea base y algunos tau
        if r["filter"] == "none" or r.get("tau_ms") in (33, 100, 300):
            name = "sin filtro" if r["filter"] == "none" else f"τ={r['tau_ms']:g} ms"
            ax1.annotate(name, (r["filter_lag"], r["tremor"]), xytext=(4, 6), textcoords="offset points", color=ink2, fontsize=8)
            ax2.annotate(name, (r["filter_lag"], r["err_follow"]), xytext=(4, -12), textcoords="offset points", color=ink2, fontsize=8)
    last = rows[-1]
    ax1.annotate("temblor", (last["filter_lag"], last["tremor"]), xytext=(6, -3), textcoords="offset points", color=ink, fontsize=9)
    ax1.annotate("jitter", (last["filter_lag"], last["jitter"]), xytext=(6, -3), textcoords="offset points", color=ink, fontsize=9)

    ax1.set_title("Lo que se gana: menos temblor con la mano quieta", loc="left", fontsize=10, color=ink)
    ax2.set_title("Lo que se paga: el brazo se queda atrás al moverte", loc="left", fontsize=10, color=ink)
    ax1.set_ylabel("mm (objetivo del brazo)")
    ax2.set_ylabel("error de seguimiento en el círculo, mm (mediana)")
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.set_xlabel("retraso agregado por el filtro (ms)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-10, max(lag) * 1.15)
        ax.grid(True, color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=ink2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout()
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
    parser.add_argument("--sweep", action="store_true", help="prueba sin filtro, EMA y One Euro con varios parámetros")
    parser.add_argument("--plot", type=Path, help="con --sweep, guarda la gráfica en este PNG")
    args = parser.parse_args()

    session = load_session(args.session)
    if args.sweep:
        rows = run_sweep(session)
        print_sweep(rows, f"Barrido de filtros: {args.session.name}")
        if args.plot:
            plot_sweep([r for r in rows if r["filter"] != "oneeuro"], args.plot,
                       f"EMA: temblor contra retraso ({args.session.name}, medias de 3 repeticiones)")
        return

    mapped = mapped_targets(session)
    raw = commands_for(session, mapped, filters.NoFilter())
    params = {"none": {}, "ema": {"tau_ms": args.tau},
              "oneeuro": {"min_cutoff": args.min_cutoff, "beta": args.beta}}[args.filter]
    cmd = commands_for(session, mapped, filters.make_filter(args.filter, **params))
    title = "Línea base (sin filtro)" if args.filter == "none" else label({"filter": args.filter, **params})
    print_table(evaluate(session, raw, cmd), f"{title}: {args.session.name}")


if __name__ == "__main__":
    main()
