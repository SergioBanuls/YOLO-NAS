import cv2
import os

# Ruta de tus imágenes y anotaciones
image_folder = 'images/'
label_folder = 'labels/'

# Especifica la imagen que quieres comprobar (por ejemplo, '00.jpg')
image_to_check = '88.jpg'

# Asegúrate de que la imagen exista en la carpeta
image_path = os.path.join(image_folder, image_to_check)
label_path = os.path.join(label_folder, os.path.splitext(image_to_check)[0] + '.txt')

if os.path.exists(image_path) and os.path.exists(label_path):
    # Cargar la imagen
    img = cv2.imread(image_path)
    
    # Redimensionar la imagen a un tamaño más pequeño (por ejemplo, 800x600)
    img = cv2.resize(img, (800, 600))  # Puedes cambiar estas dimensiones

    # Cargar las anotaciones (.txt)
    with open(label_path) as f:
        annotations = f.readlines()

    # Para cada anotación, dibujar el rectángulo en la imagen
    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = int(parts[0])  # Clase (debe ser 0 en tu caso)
        x_center, y_center, width, height = map(float, parts[1:])

        # Convertir de coordenadas normalizadas a píxeles
        img_height, img_width, _ = img.shape
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # Dibujar el rectángulo
        color = (0, 255, 0)  # Color verde para la caja
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Mostrar la imagen con las anotaciones
    cv2.imshow(f'Image with annotations: {image_to_check}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("La imagen o las anotaciones no existen.")
