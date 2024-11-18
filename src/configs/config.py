from dataclasses import dataclass, field

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2


@dataclass
class DataConfig:
    IMAGES_PATH: str = "../data/images"
    ANNOTATIONS_PATH: str = "../data/annotations/instances_default.json"
    IMAGE_WIDTH: int = 1280
    IMAGE_HEIGHT: int = 960
    IMAGE_RESIZED: int = 520

    AUG_DATASET_PATH: str = "../aug_data"
    AUG_DATASET_INFO_PATH: str = "../aug_data/info.json"
    AUG_TRAIN_SET_PATH: str = "../aug_data/train"
    AUG_VAL_SET_PATH: str = "../aug_data/val"
    AUG_TEST_SET_PATH: str = "../aug_data/test"

    TRAIN_SIZE: float = 0.7
    VAL_SIZE: float = 0.15

    AUGMENTATIONS: list = field(default_factory=lambda: [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomToneCurve(scale=0.3, p=1.0),  # RandomToneCurve change only brightness and contrast
        A.GaussNoise(p=1.0),
        A.PixelDropout(mask_drop_value=0.0, p=1.0),
        A.Sharpen(p=1.0),
        A.GridElasticDeform(num_grid_xy=(10, 10), magnitude=30, p=1.0),
        A.ShiftScaleRotate(p=1.0)  # use it instead of Rotate because it can make rotate, translate or scale
    ])

    # for DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    IMAGE_TRAIN_TRANSFORMATION: A.Compose = A.Compose([
        A.Normalize(),
        A.Resize(height=520, width=520),
        ToTensorV2()
    ])
    MASK_TRAIN_TRANSFORMATION: A.Compose = A.Compose([
        A.Resize(height=520, width=520)
    ])


@dataclass
class ModelConfig:
    MODELS_DIR_PATH: str = "../models"
    DEVICE: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE: int = 10
    EPOCH_COUNT: int = 1  # TODO increase to ~100

    # Focal Loss configs
    GAMMA: float = 2.0  # by official paper "we found γ = 2 to work best in our experiments"


@dataclass
class Configs:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


config = Configs()
