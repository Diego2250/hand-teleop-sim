import cv2
import time
import mediapipe as mp
import threading
import json
import math
import socket

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # índice
    (9, 10), (10, 11), (11, 12),           # medio
    (13, 14), (14, 15), (15, 16),          # anular
    (0, 17), (17, 18), (18, 19), (19, 20), # meñique
    (5, 9), (9, 13), (13, 17),             # palma
]

lock = threading.Lock()
latest = {"frame": None, "landmarks": None, "world": None, "ts": None, "latency_ms": None}

UDP_ADDR = ("127.0.0.1", 5005)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

PALM_SEGMENTS = [(0, 9), (5, 17), (0, 5), (0, 17)]

def draw_landmarks_on_image(image_mp: mp.Image, hand_landmarks):
    annotated_image = cv2.cvtColor(image_mp.numpy_view(), cv2.COLOR_RGB2BGR)
    h, w = annotated_image.shape[:2]

    for single_hand_lm in hand_landmarks:
        points = [(int(lmk.x * w), int(lmk.y * h)) for lmk in single_hand_lm]

        for start, end in HAND_CONNECTIONS:
            cv2.line(annotated_image, points[start], points[end], (255, 0, 0), 2)

        for pt in points:
            cv2.circle(annotated_image, pt, 3, (0, 255, 0), -1)

    return annotated_image

def palm_scale(landmarks, world, w, h):
    """Señal de profundidad: crece al acercar la mano (proporcional a 1/distancia).

    Cada segmento de la palma en píxeles se divide entre su largo real en metros (world landmarks).
    El escorzo solo acorta, así que el segmento con mayor valor es el menos afectado por la inclinación.
    Ver tools/depth_probe.py (candidato palm_max).
    """
    best = 0.0
    for a, b in PALM_SEGMENTS:
        img = math.hypot((landmarks[a].x - landmarks[b].x) * w, (landmarks[a].y - landmarks[b].y) * h)
        real = math.dist((world[a].x, world[a].y, world[a].z), (world[b].x, world[b].y, world[b].z))
        best = max(best, img / real)
    return best / h

def send_hand(landmarks, world, ts, w, h):
    palm, wrist = landmarks[9], landmarks[0]
    scale = math.hypot(palm.x - wrist.x, palm.y - wrist.y)
    msg = {"u": palm.x, "v": palm.y, "s": scale, "d": palm_scale(landmarks, world, w, h), "ts": ts}
    sock.sendto(json.dumps(msg).encode(), UDP_ADDR)

def now_ms():
    return time.monotonic_ns() // 1_000_000

def on_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
        frame = draw_landmarks_on_image(output_image, result.hand_landmarks)
    else:
        frame = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)

    with lock:
        latest["frame"] = frame
        latest["landmarks"] = result.hand_landmarks[0] if result.hand_landmarks else None
        latest["world"] = result.hand_world_landmarks[0] if result.hand_world_landmarks else None
        latest["ts"] = timestamp_ms
        latest["latency_ms"] = now_ms() - timestamp_ms

    if result.hand_landmarks:
        send_hand(result.hand_landmarks[0], result.hand_world_landmarks[0], timestamp_ms,
                  output_image.width, output_image.height)

def main():
    model_path = "./hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=on_result,
        num_hands=1
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara.")
            return

        last_ts = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts = max(now_ms(), last_ts + 1)
            last_ts = ts
            landmarker.detect_async(mp_image, ts)

            with lock:
                shown = latest["frame"] if latest["frame"] is not None else frame
                lat = latest["latency_ms"]

            if lat is not None:
                cv2.putText(shown, f"lat: {lat} ms", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("HandLandmarker Live", shown)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
