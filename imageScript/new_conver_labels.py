import json
import os

def polygon_area(x_points, y_points):
    """Calcula el área de un polígono usando la fórmula del área de Gauss."""
    area = 0.0
    j = len(x_points) - 1
    for i in range(len(x_points)):
        area += (x_points[j] + x_points[i]) * (y_points[j] - y_points[i])
        j = i
    return abs(area) / 2.0

json_file = 'annotations.json'  # Cambia esto por la ruta correcta
image_folder = 'images'         # Ruta a las imágenes
label_folder = 'labels'         # Ruta a las etiquetas

# Crear el directorio de etiquetas si no existe
os.makedirs(label_folder, exist_ok=True)

# Función para convertir coordenadas a formato YOLO
def convert_to_yolo_format(image_width, image_height, region, shape_type):
    if shape_type == "rect":
        x = region["x"]
        y = region["y"]
        width = region["width"]
        height = region["height"]
        
        # Calcular las coordenadas normalizadas
        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        return x_center, y_center, width_norm, height_norm
    
    elif shape_type == "polygon":
        x_points = region["all_points_x"]
        y_points = region["all_points_y"]
        
        # Encontrar los puntos mínimos y máximos
        x_min = min(x_points)
        x_max = max(x_points)
        y_min = min(y_points)
        y_max = max(y_points)
        
        # Calcular el ancho y alto del rectángulo que envuelve
        width = x_max - x_min
        height = y_max - y_min
        
        # Calcular el centro del rectángulo
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        
        # Normalizar el ancho y alto
        width_norm = width / image_width
        height_norm = height / image_height
        
        return x_center, y_center, width_norm, height_norm

# Leer el archivo JSON
with open(json_file, 'r') as f:
    data = json.load(f)

# Recorrer cada imagen y extraer las regiones
for img_key, img_data in data["_via_img_metadata"].items():
    image_filename = img_data["filename"]
    image_path = os.path.join(image_folder, image_filename)
    
    # Obtener el tamaño de la imagen
    image_width, image_height = 2268, 4032  # Cambia esto si tus imágenes tienen una resolución diferente

    # Crear archivo de salida para las etiquetas
    label_file_path = os.path.join(label_folder, f"{image_filename.replace('.jpg', '.txt')}")
    
    # Crear la lista de líneas de las etiquetas
    label_lines = []
    
    for region in img_data["regions"]:
        shape_attributes = region["shape_attributes"]
        field_name = region["region_attributes"]["field_name"]
        
        # Asignar el class_id según el field_name
        if field_name == "face":
            class_id = 0  # Por ejemplo, "face" es la clase 0
        elif field_name == "doc_quad":
            class_id = 1  # Por ejemplo, "doc_quad" es la clase 1
        else:
            class_id = -1  # En caso de que haya un valor inesperado (maneja las excepciones si es necesario)
        
        # Convertir a formato YOLO
        x_center, y_center, width, height = convert_to_yolo_format(image_width, image_height, shape_attributes, shape_attributes["name"])
        
        # Crear la línea de la etiqueta
        label_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
        label_lines.append(label_line)
    
    # Guardar las etiquetas en el archivo .txt
    with open(label_file_path, 'w') as label_file:
        label_file.writelines(label_lines)

print("Las anotaciones han sido convertidas correctamente.")
