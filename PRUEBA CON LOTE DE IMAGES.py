from ultralytics import YOLO
import cv2
import os
import pandas as pd
from pathlib import Path

model = YOLO("best planta saludable y enferma.pt")

CARPETA_ENTRADA = Path("PARA PRUEBA")  # Carpeta con las imágenes originales
ARCHIVO_EXCEL = "resultados_deteccion_confi_60.xlsx"  # Nombre del Excel de salida

CONFIANZA = 0.60
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Lista para guardar los registros del Excel
registros = []

# Para obtener todas las imágenes de la carpeta
imagenes = [f for f in os.listdir(CARPETA_ENTRADA)
            if f.lower().endswith(EXTENSIONES_VALIDAS)]

if not imagenes:
    print(f"ERROR: No se encontraron imágenes en '{CARPETA_ENTRADA}'")
    exit()

print(f"Se encontraron {len(imagenes)} imágenes. Procesando...\n")

# Procesar cada imagen
for nombre_imagen in imagenes:
    ruta_imagen = os.path.join(CARPETA_ENTRADA, nombre_imagen)
    frame = cv2.imread(ruta_imagen)

    if frame is None:
        print(f"⚠️  ERROR al leer: {nombre_imagen}")
        continue

    # Ejecutar YOLO
    results = model(frame, stream=True, conf=CONFIANZA)

    # Listas temporales para esta imagen
    lista_objetos = []
    lista_confianzas = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                lista_objetos.append(label)
                lista_confianzas.append(f"{conf * 100:.1f}%")

    # Si no se detectó nada
    if not lista_objetos:
        registros.append({
            "imagen": nombre_imagen,
            "objeto": "Sin detección",
            "confianza": "-"
        })
        print(f"❌ {nombre_imagen}: sin detecciones")
    else:
        # Unir todos los objetos y confianzas de esta imagen en una sola cadena separados por comas
        objetos_concatenados = " ".join(lista_objetos)
        confianzas_concatenadas = ", ".join(lista_confianzas)

        registros.append({
            "imagen": nombre_imagen,
            "objeto": objetos_concatenados,
            "confianza": confianzas_concatenadas
        })
        print(f"✅ {nombre_imagen}: {objetos_concatenados} ({len(lista_objetos)} detecciones)")

# Crear DataFrame y exportar a Excel
df = pd.DataFrame(registros)
df.to_excel(ARCHIVO_EXCEL, index=False)

# ---- CONTEO FINAL GLOBAL ----
conteo_total = {}
for reg in registros:
    # Separamos por espacios o comas en caso de que haya varios objetos en el string unificado
    objs = reg["objeto"].replace(",", " ").split()
    for obj in objs:
        conteo_total[obj] = conteo_total.get(obj, 0) + 1

print("\n" + "=" * 50)
print("✔ Proceso terminado")
print(f"✔ Resultados guardados en: {ARCHIVO_EXCEL}")
print("\n--- RECONTEO FINAL DE DETECCIONES ---")
for etiqueta, cantidad in conteo_total.items():
    print(f"• {etiqueta}: {cantidad}")
print("=" * 50)