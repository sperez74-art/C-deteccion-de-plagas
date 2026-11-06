import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from ultralytics import YOLO
from datetime import datetime

print("CARGANDO MODELO YOLO")
model = YOLO("best planta sana y enferma.pt")
print("MODELO LISTO")

drone_video_url = "rtsp://192.168.1.1:7070/webcam"

print(f"CONECTANDO AL DRON E88 EN {drone_video_url}")

cap = cv2.VideoCapture(drone_video_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("ERROR FATAL NO SE PUDO CONECTAR AL DRON")
    exit()

#  Reducir buffer
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW_NAME = "YOLO DRON - BAJA LATENCIA"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

log_file = "detecciones con dron 4.11.txt"
with open(log_file, "w", encoding="utf-8") as f:
    f.write("=== REGISTRO DE DETECCIONES ===\n")
    f.write(f"Inicio: {datetime.now()}\n")
    f.write("="*40 + "\n")

frame_count = 0

def guardar_detecciones(resultados, frame_num):
    if resultados.boxes is None:
        return

    objetos = []
    for box in resultados.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        objetos.append(f"{label} ({conf:.2%})")

    if objetos:
        with open(log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[Frame {frame_num}] {timestamp} -> {', '.join(objetos)}\n")

while True:
    for _ in range(3):
        cap.grab()

    success, frame = cap.read()
    if not success:
        print("ERROR FRAME")
        break

    results = model(frame, conf=0.6)

    annotated_frame = results[0].plot()

    frame_count += 1

    guardar_detecciones(results[0], frame_count)

    cv2.putText(annotated_frame, f"Frames: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open(log_file, "a", encoding="utf-8") as f:
    f.write("="*40 + "\n")
    f.write(f"Fin: {datetime.now()}\n")
    f.write(f"Total de frames: {frame_count}\n")

    f.write("\n--- RESUMEN FINAL ---\n")
    for objeto, cantidad in sorted(detecciones_totales.items(), key=lambda x: x[1], reverse=True):
        f.write(f"• {objeto}: {cantidad}\n")

cap.release()
cv2.destroyAllWindows()