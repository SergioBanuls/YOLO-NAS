# Conversor de Anotaciones a Formato YOLO

Este proyecto convierte anotaciones de formato JSON a formato YOLO para entrenamiento de modelos de detección de objetos.

## Funcionalidad principal
- Convierte anotaciones de polígonos y rectángulos de formato JSON a formato YOLO.
- Soporta múltiples formas geométricas (rectángulos y polígonos).
- Genera automáticamente archivos .txt para cada imagen con las coordenadas normalizadas.
- Mapeo configurable de clases de objetos.

## Requisitos
- Python 3.6 o superior
- Pillow (PIL) - para procesamiento de imágenes (opcional)

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura de Directorios

```
.
├── images/              # Directorio con las imágenes de entrenamiento
├── annotations.json     # Archivo de anotaciones en formato JSON
├── labels/              # Directorio de salida para las etiquetas YOLO (se crea automáticamente)
├── new_conver_labels.py # Script principal de conversión
└── seeanotations.py     # Script auxiliar para visualización
```

## Uso

1. Coloca tus imágenes en el directorio `images/`
2. Asegúrate de tener el archivo `annotations.json` con las anotaciones
3. **Ejecuta el script principal de conversión:**

```bash
python new_conver_labels.py
```

Este script procesará todas las anotaciones del archivo JSON y generará los archivos .txt correspondientes en la carpeta `labels/` con el formato YOLO requerido.

### Script auxiliar para visualización (opcional)

Si deseas visualizar las anotaciones sobre las imágenes para verificar la conversión:

```bash
python seeanotations.py
```

## Formato de Anotaciones

El script `new_conver_labels.py` espera un archivo JSON con el siguiente formato:

```json
{
    "_via_img_metadata": {
        "image_name.jpg": {
            "filename": "image_name.jpg",
            "regions": [
                {
                    "shape_attributes": {
                        "name": "rect",
                        "x": 100,
                        "y": 100,
                        "width": 200,
                        "height": 200
                    },
                    "region_attributes": {
                        "field_name": "face"
                    }
                },
                {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [10, 20, 30, 40],
                        "all_points_y": [15, 25, 35, 45]
                    },
                    "region_attributes": {
                        "field_name": "doc_quad"
                    }
                }
            ]
        }
    }
}
```

### Tipos de formas soportadas:
- **Rectángulos**: Definidos con `x`, `y`, `width`, `height`
- **Polígonos**: Definidos con arrays `all_points_x` y `all_points_y`

## Mapeo de Clases

El script incluye un mapeo predefinido de clases que puedes modificar según tus necesidades:

- "face": 0
- "doc_quad": 1

Para agregar más clases, edita la sección correspondiente en `new_conver_labels.py`:

```python
# Asignar el class_id según el field_name
if field_name == "face":
    class_id = 0
elif field_name == "doc_quad":
    class_id = 1
elif field_name == "tu_nueva_clase":
    class_id = 2
# ... agregar más clases según sea necesario
```

## Configuración

### Resolución de imágenes
Por defecto, el script está configurado para imágenes de **2268x4032** píxeles. Si tus imágenes tienen una resolución diferente, modifica estas líneas en `new_conver_labels.py`:

```python
# Obtener el tamaño de la imagen
image_width, image_height = 2268, 4032  # Cambia estos valores
```

### Rutas de archivos
Puedes modificar las rutas por defecto editando estas variables al inicio del script:

```python
json_file = 'annotations.json'  # Ruta al archivo JSON
image_folder = 'images'         # Ruta a las imágenes
label_folder = 'labels'         # Ruta de salida para etiquetas
```

## Funcionamiento del algoritmo

1. **Lectura del JSON**: Carga las anotaciones desde `annotations.json`
2. **Procesamiento por imagen**: Para cada imagen en el JSON:
   - Extrae todas las regiones anotadas
   - Convierte coordenadas según el tipo de forma:
     - **Rectángulos**: Usa directamente x, y, width, height
     - **Polígonos**: Calcula el rectángulo envolvente (bounding box)
3. **Normalización**: Convierte coordenadas absolutas a coordenadas normalizadas (0-1)
4. **Formato YOLO**: Genera líneas en formato: `class_id x_center y_center width height`
5. **Guardado**: Crea archivos .txt en la carpeta `labels/` con el mismo nombre que la imagen

## Notas adicionales
- El directorio `labels/` se crea automáticamente si no existe
- Para polígonos, se calcula automáticamente el rectángulo envolvente
- Las coordenadas se normalizan automáticamente al rango [0, 1]
- Cada archivo .txt contiene una línea por objeto detectado en la imagen correspondiente
