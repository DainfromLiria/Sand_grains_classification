from dataclasses import dataclass, field
from pathlib import Path

import albumentations as A
import cv2
import torch


@dataclass
class Paths:
    # main
    ROOT: Path = Path(__file__).resolve().parent.parent.parent
    DATA: Path = ROOT / "data"

    PREDICTIONS_FOLDER: Path = ROOT / "predictions"
    PREDICTIONS_FOLDER.mkdir(parents=False, exist_ok=True)
    RESULTS_FOLDER: Path = ROOT / "results"
    RESULTS_FOLDER.mkdir(parents=False, exist_ok=True)
    MODELS_FOLDER: Path = ROOT / "models"
    MODELS_FOLDER.mkdir(parents=False, exist_ok=True)

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
    EVAL_DATASET_INFO: Path = EVAL_ANNOTATION_FOLDER / "dataset_info.json"



@dataclass
class Transformations:
    # augmentations
    USE_AUGMENTATIONS: bool = False
    AUGMENTATIONS: A.Compose = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-45, 45), p=0.5, border_mode=cv2.BORDER_REFLECT), # mirror reflection of the border elements
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(std_range=(0.1, 0.2), mean_range=(0.0, 0.0), p=0.2),
        A.ElasticTransform(alpha=80, sigma=50, p=0.3),
    ])

    # test time augmentations
    USE_TTA: bool = False
    TTA_AUGMENTATIONS: list = field(default_factory=lambda: [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=(-45, 45), p=1.0)
    ])

    # preprocessing transformations
    USE_PREPROCESSING: bool = False
    IMAGE_TRANSFORMATIONS: A.Compose = A.Compose([
        A.CLAHE(p=1.0),
        A.Sharpen(p=1.0),
        A.Normalize(p=1.0), # default mean and std is for ImageNet
        # A.Resize(256, 256),
    ])
    MASK_TRANSFORMATION: A.Compose = A.Compose([
        # A.Resize(256, 256),
    ])

@dataclass
class DataConfig:

    # image properties
    IMAGE_WIDTH: int = 1280
    IMAGE_HEIGHT: int = 960
    MAX_PIXEL_VALUE: int = 255
    CLASSES_COUNT: int = 5 # 5 - mechanical(3) and chemical(2); 7 - relief(3) and shape(4)

    # train validation split
    TRAIN_SIZE: float = 0.8
    RANDOM_SEED: int = 42


@dataclass
class Model:
    # Main
    DEVICE: str = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    AVAILABLE_MODELS: list = field(default_factory=lambda:['Unet', 'DeepLabV3Plus', 'Segformer'])
    MODEL: str = 'Segformer'
    ENCODER: str = 'mit_b0' # resnet50 for U-Net and DeepLabV3Plus, mit_b0 (or other b) for Segformer
    ENCODER_WEIGHTS: str = "imagenet" # for mit_b[1-5] available only imagenet, for other image-micronet
    BATCH_SIZE: int = 8 # 16, 8
    LEARNING_RATE: float = 0.0001 # 0.001, 0.0001

    # Metrics and activation function
    THRESHOLD: float = 0.5
    METRICS_COUNT: int = 3

    # Focal Loss configs
    F_GAMMA: float = 2.0  # by official paper "we found γ = 2 to work best in our experiments"

    # Focal Tversky Loss configs. All values were best in original paper.
    FT_GAMMA: float = 0.75  # 1 / (4/3) => 3/4 => 0.75
    FT_ALPHA: float = 0.7
    FT_BETA: float = 0.3

    # Epochs
    EPOCH_COUNT: int = 300
    PATIENCE: int = 100 # early stopping

    # Cosine Annealing with Warm Restarts
    # more about params:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
    USE_CA: bool = True
    CA_T0: int = 30 # or 10
    CA_TMULT: int = 1 # or 2

    # Overlapping patches
    USE_PATCHES: bool = True
    PATCH_SIZE: int = 224 # imagenet pretrained encoder's input size. But try 512?
    OVERLAP_RATE: float = 0.5 # half of PATCH_SIZE # try 0.6 or 0.7
    PATCH_STRIDE: int = int(PATCH_SIZE * (1 - OVERLAP_RATE))


@dataclass
class Configs:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: Model = field(default_factory=Model)
    transform: Transformations = field(default_factory=Transformations)


config = Configs()
