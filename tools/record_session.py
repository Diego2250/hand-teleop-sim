"""Grabador con protocolo guiado: graba landmarks crudos para evaluar filtros sin repetir tomas.

Uso (desde PyCharm, por el permiso de cámara):
    python tools/record_session.py

No usa teclas: arranca solo cuando ve tu mano 2 s seguidos y va mostrando cada fase.
Por repetición (3 en total, unos 2 minutos):
    1. quieta:      palma quieta en el punto blanco, a ~45 cm (temblor)
    2. seguir:      sigue con la palma el punto que da vueltas (retraso y error de seguimiento)
    3. profundidad: acerca y aleja la mano hasta que tu anillo verde iguale al blanco
Cada fase tiene 3 s de preparación (se graban con rec=0: sirven para que los filtros arranquen)
y 10 s de grabación (rec=1). q aborta y guarda lo grabado.
"""
import csv
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from armSim import DEPTH_K  # noqa: E402
from handDetector import palm_scale  # noqa: E402

MODEL_PATH = ROOT / "hand_landmarker.task"
DATA_DIR = ROOT / "data"

REPS = 3
PREP_MS, REC_MS = 3000, 10000
HAND_WAIT_MS = 2000
PERIOD_S = 5.0             # una vuelta del círculo o un ciclo de profundidad
CIRCLE_R = 0.2             # radio del círculo guía, en unidades de alto del frame
Z_MID, Z_AMP = 45.0, 15.0  # cm: la fase de profundidad va de 30 a 60 cm
RING_AT_MID = 0.12         # radio del anillo a 45 cm, en unidades de alto del frame
DISPLAY_W = 1280
PHASES = {
    1: ("quieta", "Manten la palma quieta en el punto, a ~45 cm"),
    2: ("seguir", "Sigue el punto con el centro de la palma"),
    3: ("profundidad", "Acerca y aleja la mano: iguala tu anillo verde al blanco"),
}
WHITE, GREEN, YELLOW, RED = (255, 255, 255), (0, 255, 0), (0, 255, 255), (0, 0, 255)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

lock = threading.Lock()
state = {"frame": None, "hand": None, "hand_since": None, "schedule": None}
rows = []
coverage = {(rep, phase): 0 for rep in range(1, REPS + 1) for phase in PHASES}


def now_ms():
    return time.monotonic_ns() // 1_000_000


def guide(phase, t, w, h):
    """Guía t segundos después de empezar a grabar: (u, v) normalizados en la imagen espejada y distancia en cm."""
    a = 2 * math.pi * t / PERIOD_S
    if phase == 2:
        return 0.5 + CIRCLE_R * h / w * math.cos(a), 0.5 + CIRCLE_R * math.sin(a), Z_MID
    if phase == 3:
        return 0.5, 0.5, Z_MID + Z_AMP * math.sin(a)
    return 0.5, 0.5, Z_MID


def build_schedule(t0):
    """Un bloque por fase: (inicio, fin, rep, fase, inicio_grabacion), todo en ms monotónicos."""
    blocks, t = [], t0
    for rep in range(1, REPS + 1):
        for phase in PHASES:
            blocks.append((t, t + PREP_MS + REC_MS, rep, phase, t + PREP_MS))
            t += PREP_MS + REC_MS
    return blocks


def block_at(schedule, t):
    for block in schedule or []:
        if block[0] <= t < block[1]:
            return block
    return None


def on_result(result, output_image: mp.Image, timestamp_ms: int):
    frame = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)
    lat_ms = now_ms() - timestamp_ms
    hand = None
    if result.hand_landmarks:
        lms, world = result.hand_landmarks[0], result.hand_world_landmarks[0]
        w, h = output_image.width, output_image.height
        hand = (lms[9].x, lms[9].y, DEPTH_K / palm_scale(lms, world, w, h))

    with lock:
        state["frame"], state["hand"] = frame, hand
        if hand is None:
            state["hand_since"] = None
        elif state["hand_since"] is None:
            state["hand_since"] = timestamp_ms

        block = block_at(state["schedule"], timestamp_ms)  # la fase se decide por el momento de captura
        if hand is None or block is None:
            return
        _, _, rep, phase, rec_start = block
        rec = int(timestamp_ms >= rec_start)
        t_phase = timestamp_ms - rec_start
        gu, gv, gz = guide(phase, max(t_phase, 0) / 1000, w, h)
        rows.append([timestamp_ms, rep, phase, rec, t_phase, gu, gv, gz, lat_ms, w, h]
                    + [c for p in lms for c in (p.x, p.y, p.z)]
                    + [c for p in world for c in (p.x, p.y, p.z)])
        if rec:
            coverage[(rep, phase)] += 1


def draw_target(img, u, v, z, color):
    h, w = img.shape[:2]
    center = (int(u * w), int(v * h))
    cv2.circle(img, center, int(RING_AT_MID * h * Z_MID / z), color, 2)
    cv2.circle(img, center, 8, color, -1)


def put(img, text, row, color=WHITE, scale=0.8):
    org = (15, 35 + 38 * row)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)  # borde oscuro para leer sobre cualquier fondo
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_ui(img, t, schedule, hand, hand_since):
    """Texto sin acentos: las fuentes de OpenCV no los dibujan."""
    h, w = img.shape[:2]
    block = block_at(schedule, t)
    if block is None:
        put(img, "Muestra tu mano a la camara para empezar", 0, YELLOW)
        if hand_since is not None:
            put(img, f"Empieza en {max(0, HAND_WAIT_MS - (t - hand_since)) / 1000:.1f} s", 1, YELLOW)
    else:
        _, end, rep, phase, rec_start = block
        draw_target(img, *guide(phase, max(0, t - rec_start) / 1000, w, h), WHITE)
        put(img, f"Repeticion {rep}/{REPS} | Fase {phase}/3: {PHASES[phase][0]}", 0)
        put(img, PHASES[phase][1], 1)
        if t < rec_start:
            put(img, f"Preparate: {(rec_start - t) / 1000:.1f}", 2, YELLOW, 1.2)
        else:
            put(img, f"GRABANDO {(end - t) / 1000:.1f} s", 2, RED, 1.2)

    if hand is not None:
        draw_target(img, *hand, GREEN)
        put(img, f"distancia ~{hand[2]:.0f} cm", 3, GREEN)
    else:
        put(img, "sin mano", 3, RED)


def run_protocol():
    """Devuelve True si se completaron todas las fases."""
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
            return False

        try:
            last_ts = -1
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer el frame de la cámara.")
                    return False

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = max(now_ms(), last_ts + 1)
                last_ts = ts
                landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), ts)

                t = now_ms()
                with lock:
                    if (state["schedule"] is None and state["hand_since"] is not None
                            and t - state["hand_since"] >= HAND_WAIT_MS):
                        state["schedule"] = build_schedule(t)
                    schedule, hand, since = state["schedule"], state["hand"], state["hand_since"]
                    shown = state["frame"] if state["frame"] is not None else frame

                if schedule is not None and t >= schedule[-1][1]:
                    return True

                shown = cv2.resize(shown, (DISPLAY_W, DISPLAY_W * shown.shape[0] // shown.shape[1]))
                draw_ui(shown, t, schedule, hand, since)
                cv2.imshow("Record session", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Abortado con q.")
                    return False
        finally:
            cap.release()
            cv2.destroyAllWindows()


def save_csv(finished):
    if not rows:
        print("No se grabó nada.")
        return
    DATA_DIR.mkdir(exist_ok=True)
    suffix = "" if finished else "_incompleta"
    path = DATA_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}{suffix}.csv"
    header = ["t_ms", "rep", "phase", "rec", "t_phase_ms", "gu", "gv", "gz", "lat_ms", "w", "h"]
    header += [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
    header += [f"w{c}{i}" for i in range(21) for c in ("x", "y", "z")]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Guardado {path} ({len(rows)} frames con mano).")
    print(f"Frames grabados con mano por fase (~{REC_MS // 1000 * 30} esperados a 30 fps):")
    for rep in range(1, REPS + 1):
        print(f"  rep {rep}: " + "  ".join(f"{PHASES[p][0]} {coverage[(rep, p)]}" for p in PHASES))


def main():
    finished = False
    try:
        finished = run_protocol()
    finally:
        save_csv(finished)  # fuera del landmarker: ya no llegan callbacks mientras se escribe


if __name__ == "__main__":
    main()
