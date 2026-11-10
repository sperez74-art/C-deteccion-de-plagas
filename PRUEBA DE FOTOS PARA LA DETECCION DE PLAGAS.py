from ultralytics import YOLO
import cv2

model = YOLO("best planta sana y enferma.pt")

image_path = "PARA PRUEBA/planta_enferma (141).jpg"  # <--- MODIFICA AQUÍ
frame = cv2.imread(image_path)

if frame is None:
    print(f"ERROR: {image_path}")
    exit()

results = model(frame, stream=True, conf=0.60)

for result in results:
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Conf: {conf * 100:.1f}%",
                        (x1 + 10, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, label,
                        (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


cv2.imshow("Deteccion YOLOv11 - Imagen", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("planta_enferma detectada.jpg", frame)# <--- MODIFICAR AQUÍ TAMBIEN
print("Imagen guardada")