from dataclasses import dataclass, field

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2


@dataclass
class DataConfig:
    IMAGES_PATH: str = "../data/images"
    ANNOTATIONS_PATH: str = "../data/annotations/instances_default.json"
    IMAGE_WIDTH: int = 1280
    IMAGE_HEIGHT: int = 960
    IMAGE_RESIZED: int = 520
    MAX_PIXEL_VALUE: int = 255

    RELIEF_IDX = [4, 5, 12]
    SHAPE_IDX = [6, 11, 14, 17]

    DATASET_PATH: str = "../aug_data"
    DATASET_INFO_PATH: str = "../aug_data/info.json"
    TRAIN_SET_PATH: str = "../aug_data/train"
    VAL_SET_PATH: str = "../aug_data/val"
    TEST_SET_PATH: str = "../aug_data/test"

    TRAIN_SIZE: float = 0.8
    VAL_SIZE: float = 0.10

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

    NORMALIZATION_MEAN: float = 68.451
    NORMALIZATION_STD: float = 32.061
    IMAGE_TRAIN_TRANSFORMATION: A.Compose = A.Compose([
        # A.CLAHE()
        A.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        # A.Resize(height=520, width=520),
        ToTensorV2()
    ])
    MASK_TRAIN_TRANSFORMATION: A.Compose = A.Compose([
        # A.Resize(height=520, width=520)
    ])
    # ===============================================================================================================
    IMAGE_PREDICTION_TRANSFORMATION: A.Compose = A.Compose([
        # A.CLAHE()
        # A.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        # A.Resize(height=520, width=520),
        ToTensorV2()
    ])

    PREDICTIONS_FOLDER_PATH: str = "../predictions"


@dataclass
class ModelConfig:
    MODELS_DIR_PATH: str = "../models"
    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE: int = 5
    LEARNING_RATE: float = 1e-6
    EPOCH_COUNT: int = 300
    THRESHOLD: float = 0.85

    # Focal Loss configs
    F_GAMMA: float = 2.0  # by official paper "we found γ = 2 to work best in our experiments"

    # Focal Tversky Loss configs. All values were best in original paper.
    FT_GAMMA: float = 0.75  # 1 / (4/3) => 3/4 => 0.75
    FT_ALPHA: float = 0.7
    FT_BETA: float = 0.3

    # Early stopping config
    PATIENCE: int = 70
    RESULTS_DIR: str = "../results"


@dataclass
class Configs:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


config = Configs()
