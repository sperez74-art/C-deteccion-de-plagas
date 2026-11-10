from ultralytics import YOLO
import cv2

try:
    # Cargamos el modelo YOLO
    model = YOLO("best planta sana y enferma.pt")
    print("MODELO CARGADO EXITOSAMENTE")
except Exception as e:
    print(f" ERROR AL CARGAR EL MODELO: {e}")
    exit()

# Cargamos la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: NO SE PUDO ABRIR LA CÁMARA")
    exit()

print("CÁMARA INICIADA CORRECTAMENTE")
print("PRESIONA ESC PARA SALIR")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("ADVERTENCIA - FRAME PERDIDO")
        continue

    results = model(frame, stream=True, conf=0.5)

    for result in results:
        annotated_frame = result.plot()
        cv2.imshow("PRESIONA ESC PARA SALIR", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("PROGRAMA FINALIZADO CORRECTAMENTE")