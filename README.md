# C-deteccion-de-plagas

 Sistema de Detección de Plagas, Fitopatologías y Monitoreo de Cultivos mediante YOLO

Este repositorio contiene un conjunto de scripts en Python diseñados para la automatización del monitoreo agrícola a través de visión artificial. Utilizando modelos YOLO (Ultralytics) y la librería OpenCV, las herramientas permiten identificar de manera sistemática anomalías, plagas y estados de salud (plantas sanas y enfermas) en cultivos.

El proyecto abarca soluciones para tres entornos operativos diferenciados: inspección en laboratorio mediante cámaras web, monitoreo en tiempo real con drones y auditorías masivas de imágenes mediante procesamiento por lotes.


 Estructura de Scripts y Archivos

 1. PRUEBA DE WEBCAM - BAJA LATENCIA.py
Optimizado para análisis estáticos en entornos controlados, laboratorios o invernaderos utilizando cámaras web locales.
 Reducción de Retraso: Configura un buffer mínimo de captura para asegurar la reproducción de video en tiempo real.
 Visualización a Pantalla Completa: Despliega una interfaz maximizada que incorpora superposición analítica con el contador dinámico de cuadros (frames).
 Registro de Datos: Escribe automáticamente las clases y porcentajes de confianza en un archivo log, consolidando al cierre un balance estadístico ordenado según la recurrencia de las detecciones.

 2. PRUEBA DE DRON - BAJA LATENCIA.py
Diseñado para la adquisición e inspección aérea mediante vehículos no tripulados (drones) acoplados a transmisiones de red.
 Control de Latencia de Red: Emplea mecanismos de descarte y vaciado de cuadros intermedios para amortiguar el desfase temporal típico de la señal Wi-Fi / RTSP del dron.
 Bitácora de Vuelo: Documenta cada objeto detectado de forma síncrona en un archivo de texto con marcas de tiempo de alta precisión.

 3. PRUEBA CON LOTE DE IMAGES.py
Script orientado a auditorías fitosanitarias masivas mediante el procesamiento secuencial de carpetas completas de imágenes (post-vuelo).
 Reportes Estructurados: Genera un archivo en formato Excel (.xlsx) estructurado fila por fila con el nombre de cada imagen, las etiquetas encontradas y sus respectivos niveles de confianza.
 Manejo de Errores y Excepciones: Discrimina y notifica de manera explícita archivos corruptos o imágenes que carecen de detecciones para agilizar la depuración del conjunto de datos.

 4. Archivo de Pesos (best planta sana y enferma.pt)
Corresponde a los pesos de la red neuronal convolucional entrenada específicamente para la clasificación del estado fitosanitario vegetal.


 Requisitos y Configuración del Entorno

Para ejecutar de manera satisfactoria cualquiera de las herramientas, el sistema requiere la siguiente base tecnológica:

 Lenguaje: Python 3.12.0
 Dependencias Principales:
   ultralytics
   opencv-python
   pandas
   openpyxl (requerido para la exportación y manipulación de archivos Excel)
