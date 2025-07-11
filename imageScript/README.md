# Conversor de Anotaciones a Formato YOLO

Este script convierte anotaciones de formato JSON a formato YOLO para entrenamiento de modelos de detección de objetos.

## Funcionalidad principal
- Convierte metadatos de anotaciones al formato YOLO.
- Permite visualizar los metadatos sobre las imágenes para comprobar la conversión.
- Reescala las imágenes de origen a un tamaño más pequeño.

## Requisitos
- Python 3.6 o superior
- Pillow (PIL)
- matplotlib (para visualizar anotaciones)

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura de Directorios

```
.
├── images/           # Directorio con las imágenes
├── annotations.json  # Archivo de anotaciones en formato JSON
└── labels/           # Directorio de salida para las etiquetas YOLO
```

## Uso

1. Coloca tus imágenes en el directorio `images/`
2. Asegúrate de tener el archivo `annotations.json` con las anotaciones
3. Ejecuta el script principal para convertir y visualizar:

```bash
python seeanotations.py
```

Este script convertirá las anotaciones y mostrará la visualización de las mismas sobre las imágenes, permitiendo comprobar la correcta conversión y el reescalado.

## Formato de Anotaciones

El script espera un archivo JSON con el siguiente formato:

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
                }
            ]
        }
    }
}
```

## Mapeo de Clases

El script incluye un mapeo predefinido de clases:

- "face": 0
- "doc_quad": 1

## Notas adicionales
- El script reescala automáticamente las imágenes si es necesario para adaptarlas al tamaño requerido por YOLO.
- Puedes modificar el mapeo de clases o el formato de entrada según tus necesidades.
