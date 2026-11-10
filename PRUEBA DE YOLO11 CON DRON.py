import cv2
from ultralytics import YOLO
from datetime import datetime

print("CARGANDO MODELO YOLOV11")
model = YOLO("yolo11n.pt")
print("MODELO LISTO")

drone_video_url = "rtsp://192.168.1.1:7070/webcam"

print(f"CONECTANDO AL DRON E88 EN {drone_video_url}")
cap = cv2.VideoCapture(drone_video_url)

if not cap.isOpened():
    print("ERROR FATAL NO SE PUDO CONECTAR AL DRON")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Configurar ventana a PANTALLA COMPLETA
WINDOW_NAME = "DETECTANDO DESDE MI DRON E88 CON YOLO"
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Archivo de registro
log_file = "REGISTRO DE DETECCIONES.txt"
with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"=== REGISTRO DE DETECCIONES DEL DRON E88 ===\n")
    f.write(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 50 + "\n\n")

detecciones_totales = {}
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        print("ERROR NO SE PUDO LEER EL FRAME DEL DRON")
        break

    results = model(frame, stream=True, conf=0.5)

    for result in results:
        annotated_frame = result.plot()

        # Agregar información en pantalla (FPS, contador)
        frame_count += 1
        cv2.putText(annotated_frame, f"Frames: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Registrar detecciones
        if result.boxes is not None:
            objetos_frame = []
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                objetos_frame.append(f"{label} ({conf:.2%})")

                if label not in detecciones_totales:
                    detecciones_totales[label] = 0
                detecciones_totales[label] += 1

            if objetos_frame:
                with open(log_file, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    f.write(f"[Frame {frame_count}] {timestamp} -> {', '.join(objetos_frame)}\n")

        cv2.imshow(WINDOW_NAME, annotated_frame)

    # Presiona 'q' para salir, 'f' para salir de pantalla completa
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Alternar pantalla completa
        fullscreen = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              not fullscreen)

# Guardar resumen final
with open(log_file, "a", encoding="utf-8") as f:
    f.write("\n" + "=" * 50 + "\n")
    f.write(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total frames procesados: {frame_count}\n")
    f.write("\n--- RESUMEN FINAL ---\n")
    for objeto, cantidad in sorted(detecciones_totales.items(), key=lambda x: x[1], reverse=True):
        f.write(f"• {objeto}: {cantidad}\n")

cap.release()
cv2.destroyAllWindows()
print(f"\n REGISTRO GUARDADO EN: {log_file}")
print(f" TOTAL OBJETOS DETECTADOS: {sum(detecciones_totales.values())}")