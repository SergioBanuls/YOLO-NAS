# YOLO-NAS: Dataset Preparation and Model Training

Este repositorio contiene dos aplicaciones principales para trabajar con el modelo YOLO-NAS, desde la preparación de datos hasta el entrenamiento y predicción del modelo.

## 📁 Estructura del Repositorio

```
YOLO-NAS/
├── imageScript/          # Preparación y conversión de datos
└── trainYoloNas/         # Entrenamiento y predicción con YOLO-NAS
```

## 🛠️ Aplicaciones

### 1. imageScript - Preparación de Dataset

**Propósito**: Convierte imágenes y sus anotaciones al formato correcto requerido por YOLO-NAS.

**Funcionalidades principales**:
- Conversión de metadatos de anotaciones desde formato JSON a formato YOLO
- Visualización de anotaciones sobre las imágenes para validar la conversión
- Reescalado automático de imágenes al tamaño requerido
- Generación de archivos de etiquetas compatibles con YOLO

**Casos de uso**:
- Preparar datasets personalizados para entrenamiento
- Validar la correcta conversión de anotaciones
- Normalizar imágenes y coordenadas de bounding boxes

📖 [Ver documentación completa de imageScript](./imageScript/README.md)

### 2. trainYoloNas - Entrenamiento y Predicción

**Propósito**: Contiene el código para entrenar modelos YOLO-NAS y realizar predicciones.

**Funcionalidades principales**:
- Entrenamiento de modelos YOLO-NAS personalizados
- Configuración de hiperparámetros de entrenamiento
- Evaluación de modelos entrenados
- Inferencia y predicción sobre nuevas imágenes
- Métricas de rendimiento y visualización de resultados

**Casos de uso**:
- Entrenar modelos de detección de objetos personalizados
- Evaluar el rendimiento del modelo
- Realizar predicciones sobre datos nuevos
- Fine-tuning de modelos pre-entrenados

📖 [Ver documentación completa de trainYoloNas](./trainYoloNas/README.md)

## 🚀 Flujo de Trabajo Recomendado

1. **Preparación de datos** (`imageScript/`):
   - Organizar imágenes en el directorio correspondiente
   - Convertir anotaciones al formato YOLO usando los scripts de conversión
   - Validar visualmente las anotaciones convertidas

2. **Entrenamiento del modelo** (`trainYoloNas/`):
   - Configurar los parámetros de entrenamiento
   - Entrenar el modelo YOLO-NAS con el dataset preparado
   - Evaluar el rendimiento del modelo

3. **Inferencia**:
   - Utilizar el modelo entrenado para realizar predicciones
   - Analizar resultados y métricas de rendimiento





## 📝 Notas

- Asegúrate de completar la preparación de datos en `imageScript/` antes de proceder con el entrenamiento en `trainYoloNas/`
- Cada aplicación tiene su propia documentación detallada con instrucciones específicas de uso
- Se recomienda usar un entorno virtual de Python para evitar conflictos de dependencias

---

