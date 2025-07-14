import torch
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

# --- Configuración del Entrenamiento ---
CHECKPOINT_DIR = 'checkpoints'
MODELO_A_ENTRENAR = 'yolo_nas_s' # Puedes elegir 'yolo_nas_m' o 'yolo_nas_l' para modelos más grandes
DATASET_PARAMS = {
    'data_dir': 'id_dataset',
    'train_images_dir': 'images/train',
    'train_labels_dir': 'labels/train',
    'val_images_dir': 'images/val',
    'val_labels_dir': 'labels/val',
    'class_names': ['face', 'document'],
    'num_classes': 2
}
TRAIN_PARAMS = {
    'max_epochs': 50, # Número de épocas de entrenamiento
    'lr_mode': 'cosine',
    'initial_lr': 0.001,
    'loss': PPYoloELoss(use_static_assigner=False, num_classes=DATASET_PARAMS['num_classes'], reg_max=16),
    'valid_metrics_list': [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=DATASET_PARAMS['num_classes'],
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    'metric_to_watch': 'mAP@0.50'
}

def main():
    # --- 1. Inicializar el Entrenador (Trainer) ---
    trainer = Trainer(experiment_name=MODELO_A_ENTRENAR + "_documentos", ckpt_root_dir=CHECKPOINT_DIR)

    # --- 2. Cargar los Datos ---
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': DATASET_PARAMS['data_dir'],
            'images_dir': DATASET_PARAMS['train_images_dir'],
            'labels_dir': DATASET_PARAMS['train_labels_dir'],
            'classes': DATASET_PARAMS['class_names']
        },
        dataloader_params={
            'batch_size': 8,  # Reduce si te quedas sin memoria en la GPU
            'num_workers': 2 # En Windows, es recomendable usar un valor bajo o incluso 0
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': DATASET_PARAMS['data_dir'],
            'images_dir': DATASET_PARAMS['val_images_dir'],
            'labels_dir': DATASET_PARAMS['val_labels_dir'],
            'classes': DATASET_PARAMS['class_names']
        },
        dataloader_params={
            'batch_size': 16, # Puede ser mayor para validación
            'num_workers': 2
        }
    )

    # --- 3. Cargar el Modelo ---
    # Usamos pesos pre-entrenados en COCO para un mejor rendimiento (transfer learning)
    model = models.get(MODELO_A_ENTRENAR, num_classes=DATASET_PARAMS['num_classes'], pretrained_weights="coco")
    #model = models.get(MODELO_A_ENTRENAR, num_classes=DATASET_PARAMS['num_classes'])

    # --- 4. Iniciar el Entrenamiento ---
    trainer.train(
        model=model,
        training_params=TRAIN_PARAMS,
        train_loader=train_data,
        valid_loader=val_data
    )

if __name__ == '__main__':
    # Verificar si hay una GPU disponible
    if torch.cuda.is_available():
        print("GPU detectada. Usando CUDA.")
    else:
        print("ADVERTENCIA: No se detectó GPU. El entrenamiento será muy lento.")
    main()