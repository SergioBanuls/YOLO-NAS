# YOLO-NAS para Detección de Documentos de Identidad

<div align="center">

![YOLO-NAS](https://img.shields.io/badge/YOLO--NAS-Object%20Detection-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Windows](https://img.shields.io/badge/Windows-Compatible-0078D4?style=for-the-badge&logo=windows)

</div>

## 📖 Descripción

Este repositorio contiene los recursos para entrenar un modelo de detección de objetos **YOLO-NAS** en un entorno local de Windows. El objetivo es crear un modelo capaz de localizar documentos de identidad en imágenes con alta precisión.

> 🎯 **Objetivo**: Detección automática de documentos de identidad en imágenes utilizando técnicas de Deep Learning

### ✨ Características

- ✅ Configuración optimizada para Windows
- ✅ Evita problemas de compilación comunes
- ✅ Utiliza Anaconda/Miniconda para gestión de dependencias
- ✅ Transfer Learning con pesos pre-entrenados COCO
- ✅ Interfaz de inferencia simple

---

## 📋 Tabla de Contenidos

- [⚙️ Instalación y Configuración del Entorno](#️-instalación-y-configuración-del-entorno)
- [🗂️ Preparación del Dataset](#️-preparación-del-dataset)
- [🚀 Fase 1: Entrenamiento del Modelo](#-fase-1-entrenamiento-del-modelo)
- [🔬 Fase 2: Comprobación y Predicción](#-fase-2-comprobación-y-predicción-inferencia)
- [💡 Resumen del Flujo de Trabajo](#-resumen-del-flujo-de-trabajo)

---

## ⚙️ Instalación y Configuración del Entorno

> ⚠️ **Importante**: Para una instalación limpia y sin errores, es crucial seguir estos pasos utilizando la terminal específica de Anaconda.

### 📋 Requisitos Previos

| Componente | Versión Recomendada | Link de Descarga |
|------------|-------------------|------------------|
| Miniconda | Última versión | [Descargar](https://docs.conda.io/en/latest/miniconda.html) |
| Python | 3.10 | Incluido con Miniconda |
| Sistema Operativo | Windows 10/11 | - |

### 📦 Pasos de Instalación

#### 1️⃣ Instalar Miniconda
```bash
# Descargar e instalar Miniconda para Windows desde su página oficial
# Asegúrate de marcar "Add to PATH" durante la instalación
```

#### 2️⃣ Abrir Anaconda Prompt
- Busca en el **Menú de Inicio**: `Anaconda Prompt (miniconda3)`
- 📝 **Nota**: Todos los comandos deben ejecutarse desde esta terminal

#### 3️⃣ Crear y Activar Entorno Virtual
```bash
# Crear nuevo entorno
conda create -n yolo-nas python=3.10 -y

# Activar entorno
conda activate yolo-nas
```

#### 4️⃣ Instalar Dependencias
```bash
# Instalar PyTorch y dependencias principales
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Instalar herramientas adicionales
conda install pycocotools onnx -c conda-forge

# Instalar super-gradients
pip install super-gradients
```

### ✅ Verificación de Instalación
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import super_gradients; print('Super Gradients instalado correctamente')"
```

---

## 🗂️ Preparación del Dataset

El modelo **YOLO-NAS** necesita que los datos estén organizados en una estructura específica para funcionar correctamente.

### 📁 Estructura de Carpetas Requerida

```
dataset/
├── images/
│   ├── train/          # Imágenes de entrenamiento
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/            # Imágenes de validación
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── labels/
    ├── train/          # Etiquetas de entrenamiento
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── val/            # Etiquetas de validación
        ├── img1.txt
        ├── img2.txt
        └── ...
```

### 🏷️ Formato de Anotaciones YOLO

Cada imagen debe tener un archivo `.txt` correspondiente con el formato:

```
<class_id> <x_center> <y_center> <width> <height>
```

| Parámetro | Descripción | Rango |
|-----------|-------------|-------|
| `class_id` | Índice de la clase del objeto | 0, 1, 2, ... |
| `x_center` | Coordenada X del centro (normalizada) | 0.0 - 1.0 |
| `y_center` | Coordenada Y del centro (normalizada) | 0.0 - 1.0 |
| `width` | Ancho del bounding box (normalizado) | 0.0 - 1.0 |
| `height` | Alto del bounding box (normalizado) | 0.0 - 1.0 |

#### 📝 Ejemplo de Archivo de Etiquetas
```
0 0.5 0.3 0.4 0.6
1 0.2 0.7 0.3 0.25
```

### ⚙️ Archivo de Configuración `data.yaml`

Este archivo es **fundamental** para el entrenamiento. Define la estructura del dataset:

```yaml
# data.yaml
path: /path/to/dataset  # Ruta raíz del dataset
train: images/train     # Ruta relativa a imágenes de entrenamiento
val: images/val         # Ruta relativa a imágenes de validación

# Número de clases
nc: 2

# Nombres de las clases
names:
  0: documento_identidad
  1: otro_documento
```

> ⚠️ **Crítico**: El valor `nc` debe coincidir exactamente con el número de clases en tu dataset.

---

## 🚀 Fase 1: Entrenamiento del Modelo

El entrenamiento se gestiona a través del script `train.py`, que está optimizado para **YOLO-NAS**.

### 🔧 Configuración del Entrenamiento

El script `train.py` está configurado para:

| Funcionalidad | Descripción |
|---------------|-------------|
| 📂 **Carga de Dataset** | Utiliza la información del archivo `data.yaml` |
| 🏗️ **Arquitectura del Modelo** | Construye YOLO-NAS con el número correcto de clases |
| 🎯 **Transfer Learning** | Carga pesos pre-entrenados del dataset COCO |
| ⚙️ **Hiperparámetros** | Define épocas, learning rate y función de pérdida |
| 💾 **Guardado** | Guarda checkpoints y resultados automáticamente |

### ▶️ Ejecutar Entrenamiento

```bash
# Activar el entorno
conda activate yolo-nas

# Navegar al directorio del proyecto
cd /path/to/trainYoloNas

# Iniciar entrenamiento
python train.py
```

### 📊 Parámetros de Entrenamiento

```python
# Ejemplo de configuración en train.py
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # ⚠️ Debe coincidir con data.yaml
```

### 📈 Monitoreo del Progreso

Durante el entrenamiento verás:
- 📉 **Loss curves**: Pérdida de entrenamiento y validación
- 📊 **Métricas**: mAP, Precision, Recall
- 💾 **Checkpoints**: Modelos guardados automáticamente
- 📁 **Logs**: Registros detallados del progreso

---

## 🔬 Fase 2: Comprobación y Predicción (Inferencia)

Una vez entrenado el modelo, utiliza el script `predict.py` para realizar detecciones en imágenes nuevas.

### ⚙️ Configuración de Parámetros

Antes de ejecutar la inferencia, **debes modificar manualmente** las siguientes variables en el archivo `predict.py`:

#### 🔧 Parámetros Obligatorios a Configurar

> ⚠️ **IMPORTANTE**: Estas rutas deben ser configuradas antes de ejecutar el script.

##### 📁 1. RUTA_CHECKPOINT
```python
# En predict.py - línea 8
RUTA_CHECKPOINT = '/ruta/completa/al/mejor/modelo/checkpoint.pth'

# Ejemplo:
RUTA_CHECKPOINT = '/Users/user/Documents/YOLO-NAS/trainYoloNas/runs/train/exp/weights/best.pth'
```

##### 🖼️ 2. RUTA_IMAGEN
```python
# En predict.py - línea 11
RUTA_IMAGEN = '/ruta/completa/a/imagen/de/prueba.jpg'

# Ejemplo:
RUTA_IMAGEN = '/Users/user/Documents/YOLO-NAS/imageScript/images/00.jpg'
```

#### 📋 Parámetros Adicionales de Configuración

##### 🏷️ 3. Número de Clases
```python
# ⚠️ DEBE coincidir con el entrenamiento
NUM_CLASES = 2
```

##### 🎯 4. Umbral de Confianza
```python
# Valor entre 0.0 y 1.0
CONF_THRESHOLD = 0.6  # Ajustar según necesidades
```

### 📋 Tabla de Configuración

| Parámetro | Descripción | Valor Recomendado | Notas |
|-----------|-------------|------------------|-------|
| `MODEL_PATH` | Ruta absoluta al archivo `.pth` | `/full/path/to/model.pth` | Usar ruta completa |
| `NUM_CLASSES` | Número de clases del modelo | `2` | Debe coincidir con entrenamiento |
| `CONFIDENCE_THRESHOLD` | Umbral mínimo de confianza | `0.2` - `0.5` | Empezar bajo para validar |

### ▶️ Ejecutar Predicción

> 🚨 **ANTES DE EJECUTAR**: Asegúrate de haber configurado las rutas en `predict.py`

```bash
# 1. Activar el entorno
conda activate yolo-nas

# 2. Navegar al directorio
cd /path/to/trainYoloNas

# 3. OBLIGATORIO: Editar predict.py y configurar:
#    - RUTA_CHECKPOINT (ruta al modelo entrenado)
#    - RUTA_IMAGEN (ruta a la imagen de prueba)

# 4. Ejecutar predicción
python predict.py
```

#### ✅ Checklist Previo a la Ejecución

- [ ] ✅ `RUTA_CHECKPOINT` configurada con la ruta completa al modelo
- [ ] ✅ `RUTA_IMAGEN` configurada con la ruta completa a la imagen de prueba
- [ ] ✅ `NUM_CLASES` coincide con el entrenamiento (2 en este caso)
- [ ] ✅ `CONF_THRESHOLD` ajustado según necesidades

### 🖼️ Resultado Esperado

El script mostrará:
- 🖼️ **Imagen original** con bounding boxes
- 🏷️ **Etiquetas** de las clases detectadas
- 📊 **Scores de confianza** para cada detección
- 📏 **Coordenadas** de los bounding boxes

---

## 💡 Resumen del Flujo de Trabajo

### 🔄 Workflow Completo

```mermaid
graph TD
    A[🔧 Instalar Miniconda] --> B[📂 Abrir Anaconda Prompt]
    B --> C[🐍 Crear entorno conda]
    C --> D[📦 Instalar dependencias]
    D --> E[🗂️ Preparar dataset]
    E --> F[⚙️ Configurar data.yaml]
    F --> G[🚀 Ejecutar train.py]
    G --> H[📊 Monitorear entrenamiento]
    H --> I[🔬 Configurar predict.py]
    I --> J[🎯 Realizar inferencia]
```

### 📋 Checklist de Comandos

#### ✅ Configuración Inicial
```bash
# 1. Activar entorno
conda activate yolo-nas

# 2. Navegar al proyecto
cd /Users/sergiobanuls/Documents/YOLO-NAS/trainYoloNas
```

#### ✅ Entrenamiento
```bash
# 3. Entrenar modelo
python train.py
```

#### ✅ Inferencia
```bash
# 4. Hacer predicciones (después de configurar predict.py)
python predict.py
```

### 🛠️ Solución de Problemas Comunes

| Problema | Solución |
|----------|----------|
| ❌ **Error de compilación** | Usar Anaconda Prompt exclusivamente |
| ❌ **Error de clases** | Verificar `nc` en `data.yaml` |
| ❌ **Error de checkpoint** | Usar ruta absoluta en `predict.py` |
| ❌ **Dependencias faltantes** | Reinstalar con conda/pip según instrucciones |
| ❌ **Error URLError: getaddrinfo failed** | Ver solución detallada abajo ⬇️ |

---

## 🔧 Solución al Error de Carga de Pesos Pre-entrenados

### ❌ Error: `URLError: <urlopen error [Errno 11001] getaddrinfo failed>`

Si encuentras este error al intentar cargar los pesos pre-entrenados (`pretrained_weights="coco"`) para YOLO-NAS, es debido a un cambio en la URL donde se alojan estos pesos. La versión actual de SuperGradients puede estar apuntando a una ubicación antigua.

#### 🔍 Pasos para la Solución

##### 1️⃣ Localizar la Instalación de SuperGradients

Encuentra la carpeta donde está instalada la librería `super_gradients`. Una ruta común es:

```bash
# Windows
C:\Users\tu_usuario\miniconda3\envs\yolo-nas\lib\site-packages\super_gradients\

# macOS/Linux
/Users/tu_usuario/miniconda3/envs/yolo-nas/lib/python3.10/site-packages/super_gradients/
```

> 💡 **Tip**: Puedes encontrar la ruta exacta ejecutando:
> ```python
> import super_gradients
> print(super_gradients.__file__)
> ```

##### 2️⃣ Modificar `pretrained_models.py`

**Archivo**: `.../super_gradients/training/pretrained_models.py`

1. Abre el archivo con un editor de texto
2. Busca todas las ocurrencias de: `sghub.deci.ai`
3. Reemplázalas por: `sg-hub-nv.s3.amazonaws.com`

```python
# ❌ URL antigua (buscar)
"https://sghub.deci.ai/models/..."

# ✅ URL nueva (reemplazar)
"https://sg-hub-nv.s3.amazonaws.com/models/..."
```

##### 3️⃣ Modificar `checkpoint_utils.py`

**Archivo**: `.../super_gradients/training/utils/checkpoint_utils.py`

1. Abre el archivo con un editor de texto
2. Busca la línea con `unique_filename`
3. Corrige la URL en esa línea

```python
# ❌ Línea antigua (buscar)
unique_filename = url.split("https://sghub.deci.ai/models/")[1].replace("/", "_").replace(" ", "_")

# ✅ Línea nueva (reemplazar)
unique_filename = url.split("https://sg-hub-nv.s3.amazonaws.com/models/")[1].replace("/", "_").replace(" ", "_")
```

#### ⚠️ Consideraciones Importantes

> 🔒 **Permisos**: Necesitarás permisos de administrador para guardar los cambios.

> 💾 **Backup**: Haz una copia de seguridad de los archivos antes de modificarlos:
> ```bash
> cp pretrained_models.py pretrained_models.py.backup
> cp checkpoint_utils.py checkpoint_utils.py.backup
> ```

> 🔄 **Actualizaciones**: Este problema es específico de ciertas versiones. En futuras actualizaciones de SuperGradients, el arreglo puede estar incluido.

#### ✅ Verificación

Después de realizar los cambios:

1. Guarda ambos archivos
2. Reinicia tu entorno Python
3. Vuelve a ejecutar tu script de entrenamiento
4. El modelo debería descargar los pesos pre-entrenados correctamente

---
