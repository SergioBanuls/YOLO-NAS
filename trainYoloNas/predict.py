import torch
from super_gradients.training import models
from PIL import Image

# --- CONFIGURACIÓN ---
# 1. Ruta a tu mejor modelo guardado
RUTA_CHECKPOINT = ''

# 2. Ruta a la imagen que quieres probar
RUTA_IMAGEN = ''

# 3. Parámetros que usaste en el entrenamiento
MODELO_ARQUITECTURA = 'yolo_nas_s'
NUM_CLASES = 2 # Solo tenías la clase "documento"

# 4. Umbral de confianza (ajusta este valor si es necesario)
# Solo mostrará detecciones con una confianza mayor a 0.5 (50%)
CONF_THRESHOLD = 0.6
# --- FIN DE LA CONFIGURACIÓN ---


print("Cargando modelo...")
model = models.get(
    MODELO_ARQUITECTURA,
    num_classes=NUM_CLASES,
    checkpoint_path=RUTA_CHECKPOINT
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Haciendo predicción en la imagen: {RUTA_IMAGEN}")
predictions = model.predict(RUTA_IMAGEN, conf=CONF_THRESHOLD)

print("Mostrando resultados... Cierra la ventana de la imagen para continuar.")
predictions.show()

# La línea de abajo daba error, la desactivamos por ahora con '#'
# predictions.save(output_folder=output_folder)

# Opcional: Imprimir las coordenadas y confianza de cada detección
print("\n--- Detalles de la Detección ---")
for image_prediction in predictions:
    if len(image_prediction.prediction.bboxes_xyxy) > 0:
        for i, bbox in enumerate(image_prediction.prediction.bboxes_xyxy):
            label_id = int(image_prediction.prediction.labels[i])
            clase_nombre = image_prediction.class_names[label_id]
            confianza = image_prediction.prediction.confidence[i]
            coordenadas = bbox

            print(f"✅ Objeto detectado: {clase_nombre}")
            print(f"   Confianza: {confianza:.2f}")
            print(f"   Coordenadas (x1, y1, x2, y2): {coordenadas}")
    else:
        print("❌ No se detectó ningún objeto con la confianza especificada.")

print("\nProceso finalizado.")