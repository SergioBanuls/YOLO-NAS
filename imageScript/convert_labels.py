import json
import os
from PIL import Image

# Mapea los field_name a IDs
LABEL_MAP = {
    "face": 0,
    "doc_quad": 1
}

with open("annotations.json") as f:
    data = json.load(f)
    # Accedemos a los datos dentro de _via_img_metadata
    data = data.get("_via_img_metadata", {})

output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

for item in data.values():
    filename = item["filename"]
    image_path = os.path.join("images", filename)
    img = Image.open(image_path)
    img_w, img_h = img.size

    label_lines = []

    for region in item["regions"]:
        shape = region["shape_attributes"]
        label_name = region["region_attributes"].get("field_name", "")
        class_id = LABEL_MAP.get(label_name)
        if class_id is None:
            continue  # ignorar etiquetas no reconocidas

        if shape["name"] == "rect":
            x = shape["x"]
            y = shape["y"]
            w = shape["width"]
            h = shape["height"]
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            label_lines.append(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}")

        # Aquí podrías añadir conversión de polígonos si YOLO lo soporta (ej. YOLOv8-seg)

    if label_lines:
        txt_path = os.path.join(output_dir, filename.replace(".jpg", ".txt"))
        with open(txt_path, "w") as out_f:
            out_f.write("\n".join(label_lines)) 