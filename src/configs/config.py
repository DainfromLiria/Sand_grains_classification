from dataclasses import dataclass, field

import albumentations as A
import torch
from pathlib import Path
from albumentations.pytorch import ToTensorV2


@dataclass
class Paths:
    # main
    ROOT: Path = Path(__file__).resolve().parent.parent.parent
    DATA: Path = ROOT / "data"

    # train dataset path
    TRAIN_DATA: Path = DATA / "train"
    TRAIN_IMAGES_FOLDER: Path = TRAIN_DATA / "images"
    TRAIN_ANNOTATION_FOLDER: Path = TRAIN_DATA / "annotations"
    TRAIN_ANNOTATIONS: Path = TRAIN_ANNOTATION_FOLDER / "instances_default.json"
    TRAIN_DATASET_INFO: Path = TRAIN_ANNOTATION_FOLDER / "dataset_info.json"

    # eval dataset path
    EVAL_DATA: Path = DATA / "eval"
    EVAL_IMAGES_FOLDER: Path = EVAL_DATA / "images"
    EVAL_ANNOTATION_FOLDER: Path = EVAL_DATA / "annotations"
    EVAL_ANNOTATIONS: Path = EVAL_ANNOTATION_FOLDER / "instances_default.json"


@dataclass
class DataConfig:

    # image properties # TODO find usage
    IMAGE_WIDTH: int = 1280
    IMAGE_HEIGHT: int = 960
    MAX_PIXEL_VALUE: int = 255

    # =====================================================================================
    # TODO update for new dataset !!!!
    RELIEF_IDX = [4, 5, 12]
    SHAPE_IDX = [6, 11, 14, 17]
    # =====================================================================================

    # train validation split
    TRAIN_SIZE: float = 0.9
    RANDOM_SEED: int = 42

    # =====================================================================================
    # augmentations TODO explore and change
    AUGMENTATIONS: A.Compose = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomToneCurve(scale=0.3, p=1.0),  # RandomToneCurve change only brightness and contrast
        # A.GaussNoise(p=1.0),
        # A.PixelDropout(mask_drop_value=0.0, p=1.0),
        # A.Sharpen(p=1.0),
        A.GridElasticDeform(num_grid_xy=(10, 10), magnitude=30, p=1.0),
        A.ShiftScaleRotate(p=1.0)  # use it instead of Rotate because it can make rotate, translate or scale
    ])
    # =====================================================================================

    # TTA_AUGMENTATIONS: list = field(default_factory=lambda: [
    #     A.HorizontalFlip(p=1.0),
    #     A.Rotate(limit=(-45, 45), p=1.0)
    # ])

    NORMALIZATION_MEAN: float = 68.451
    NORMALIZATION_STD: float = 32.061
    IMAGE_TRAIN_TRANSFORMATIONS: A.Compose = A.Compose([
        # A.CLAHE(),
        # A.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
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
    THRESHOLD: float = 0.9

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
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


config = Configs()
