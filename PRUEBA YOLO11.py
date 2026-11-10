import cv2
from ultralytics import YOLO

print("Cargando modelo YOLOv11...")
model = YOLO("yolo11n.pt")
print("Modelo listo!")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR FATAL: No se pudo abrir la cámara.")
    print("Revisa los permisos de PyCharm y que ningún otro programa la esté usando.")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        print("Error: No se pudo leer el frame de la cámara.")
        break

    results = model(frame, stream=True, conf=0.8)

    for result in results:
        annotated_frame = result.plot()

        cv2.imshow("Deteccion en Vivo con YOLOv11", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Aplicación cerrada correctamente.")