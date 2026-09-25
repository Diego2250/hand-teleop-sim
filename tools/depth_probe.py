"""Diagnóstico de profundidad: graba landmarks crudos para comparar candidatos a señal de profundidad.

Uso (desde la raíz del repo o desde PyCharm):
    python tools/depth_probe.py

Teclas (pon la mano en posición y presiona con la otra mano):
    1, 2, 3  graba 5 s con la mano quieta, abierta y con la palma hacia la cámara, a ~30, ~45 y ~60 cm
    4        graba 10 s a ~45 cm inclinando y girando la mano, sin acercarla ni alejarla
    q        guarda el CSV en data/ y sale
"""
import csv
import math
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "hand_landmarker.task"
DATA_DIR = ROOT / "data"

PHASE_SECONDS = {1: 5, 2: 5, 3: 5, 4: 10}
PHASE_NAMES = {1: "~30 cm quieta", 2: "~45 cm quieta", 3: "~60 cm quieta", 4: "~45 cm girando"}
PALM_SEGMENTS = [(0, 9), (5, 17), (0, 5), (0, 17)]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

lock = threading.Lock()
state = {"phase": 0, "phase_end": 0, "frame": None, "feat": None}
rows = []
counts = {p: 0 for p in PHASE_SECONDS}


def now_ms():
    return time.monotonic_ns() // 1_000_000


def features(norm, world, w, h):
    """Candidatos a señal de profundidad. Todos crecen al acercar la mano.

    norm: (21, 3) landmarks normalizados (x por ancho, y por alto).
    world: (21, 3) world landmarks en metros.
    w, h: tamaño del frame en píxeles.
    """
    px = norm[:, :2] * [w, h]

    # El actual: mezcla unidades de ancho y alto.
    s = math.hypot(norm[9, 0] - norm[0, 0], norm[9, 1] - norm[0, 1])

    # Mismo segmento en píxeles, en unidades de alto del frame.
    s_px = np.linalg.norm(px[9] - px[0]) / h

    # Píxeles por metro de cada segmento de la palma; el escorzo solo acorta, así que se toma el mayor.
    palm_max = max(np.linalg.norm(px[a] - px[b]) / np.linalg.norm(world[a] - world[b])
                   for a, b in PALM_SEGMENTS) / h

    # Perspectiva débil: escala entre los 21 puntos en imagen y su proyección xy en metros.
    pc = px - px.mean(axis=0)
    wc = world[:, :2] - world[:, :2].mean(axis=0)
    wp = math.sqrt((pc ** 2).sum() / (wc ** 2).sum()) / h

    return {"s": s, "s_px": s_px, "palm_max": palm_max, "wp": wp}


def draw_hand(frame, norm):
    h, w = frame.shape[:2]
    for x, y in norm[:, :2] * [w, h]:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    for a, b in PALM_SEGMENTS:
        pa, pb = norm[a, :2] * [w, h], norm[b, :2] * [w, h]
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 0, 0), 2)


def on_result(result, output_image: mp.Image, timestamp_ms: int):
    frame = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)
    feat = None
    if result.hand_landmarks:
        norm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
        w, h = output_image.width, output_image.height
        feat = features(norm, world, w, h)
        draw_hand(frame, norm)

    with lock:
        phase = state["phase"]
        if feat is not None:
            rows.append([timestamp_ms, phase, w, h] + norm.ravel().tolist() + world.ravel().tolist())
            if phase:
                counts[phase] += 1
        state["frame"] = frame
        state["feat"] = feat


def save_csv():
    if not rows:
        print("No se grabó nada.")
        return
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"depth_probe_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    header = ["t_ms", "phase", "w", "h"]
    header += [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
    header += [f"w{c}{i}" for i in range(21) for c in ("x", "y", "z")]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Guardado {path} ({len(rows)} frames con mano). Frames por fase: {counts}")


def draw_overlay(frame, feat):
    with lock:
        phase, phase_end = state["phase"], state["phase_end"]
        done = dict(counts)

    if phase:
        left = (phase_end - now_ms()) / 1000
        cv2.putText(frame, f"GRABANDO {phase} ({PHASE_NAMES[phase]}): {left:.1f} s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "1-4: grabar fase | q: guardar y salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    status = "  ".join(f"{p}:{n}" for p, n in done.items())
    cv2.putText(frame, f"frames {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if feat is not None:
        for i, (name, val) in enumerate(feat.items()):
            cv2.putText(frame, f"{name}: {val:.3f}", (10, 95 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "sin mano", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=on_result,
        num_hands=1,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara.")
            return

        try:
            last_ts = -1
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer el frame de la cámara.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = max(now_ms(), last_ts + 1)
                last_ts = ts
                landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), ts)

                with lock:
                    if state["phase"] and now_ms() >= state["phase_end"]:
                        state["phase"] = 0
                    shown = state["frame"] if state["frame"] is not None else frame
                    feat = state["feat"]

                shown = shown.copy()
                draw_overlay(shown, feat)
                cv2.imshow("Depth probe", shown)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    p = key - ord("0")
                    with lock:
                        state["phase"], state["phase_end"] = p, now_ms() + PHASE_SECONDS[p] * 1000
        finally:
            cap.release()
            cv2.destroyAllWindows()
            save_csv()


if __name__ == "__main__":
    main()
