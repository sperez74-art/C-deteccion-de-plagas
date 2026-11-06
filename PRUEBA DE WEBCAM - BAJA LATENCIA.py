import cv2
from ultralytics import YOLO
from datetime import datetime
from collections import Counter
from pathlib import Path

# ===== RUTAS ABSOLUTAS =====
CARPETA = Path(r"C:\Users\estudiante\Videos\PRUEBA DE WEBCAM Y DRON") # <-- AÑADRI DIRECCION DE DONDE ESTE GUARDADO
MODEL_PATH = CARPETA / "best planta sana y enferma.pt"
LOG_PATH   = CARPETA / "detecciones 1.txt"

if not MODEL_PATH.exists():
    print(f"ERROR: no se encontró el modelo en:\n{MODEL_PATH}")
    exit()

print("CARGANDO MODELO YOLO")
model = YOLO(str(MODEL_PATH))
print("MODELO LISTO")

print("CONECTANDO A LA WEBCAM EXTERNA")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR FATAL NO SE PUDO ABRIR LA WEBCAM")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW_NAME = "YOLO WEBCAM - BAJA LATENCIA"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("=== REGISTRO DE DETECCIONES ===\n")
    f.write(f"Inicio: {datetime.now()}\n")
    f.write("=" * 40 + "\n")

frame_count = 0
detecciones_totales = Counter()


def guardar_detecciones(resultados, frame_num):
    if resultados.boxes is None:
        return

    objetos = []
    for box in resultados.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        objetos.append(f"{label} ({conf:.2%})")
        detecciones_totales[label] += 1

    if objetos:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[Frame {frame_num}] {timestamp} -> {', '.join(objetos)}\n")


while True:
    success, frame = cap.read()
    if not success:
        print("ERROR FRAME")
        break

    results = model(frame, conf=0.7)
    annotated_frame = results[0].plot()

    frame_count += 1
    guardar_detecciones(results[0], frame_count)

    cv2.putText(annotated_frame, f"Frames: {frame_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow(WINDOW_NAME, annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

with open(LOG_PATH, "a", encoding="utf-8") as f:
    f.write("=" * 40 + "\n")
    f.write(f"Fin: {datetime.now()}\n")
    f.write(f"Total de frames: {frame_count}\n")

    f.write("\n--- RESUMEN FINAL ---\n")
    if detecciones_totales:
        for objeto, cantidad in sorted(detecciones_totales.items(),
                                       key=lambda x: x[1], reverse=True):
            f.write(f"• {objeto}: {cantidad}\n")
    else:
        f.write("(No se detectó ningún objeto)\n")

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Resultados guardados en:\n{LOG_PATH}")