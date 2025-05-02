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

    INPUT_IMAGES_FOLDER: Path = ROOT / "input_images"
    INPUT_IMAGES_FOLDER.mkdir(parents=False, exist_ok=True)
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

    # test time augmentations (only flips in order neither, vertical, horizontal)
    USE_TTA: bool = True
    TTA_AUGMENTATIONS: list = field(default_factory=lambda:[None, [2], [3]])

    # preprocessing transformations
    USE_PREPROCESSING: bool = False
    IMAGE_TRANSFORMATIONS: A.Compose = A.Compose([
        A.CLAHE(clip_limit=4.0, p=1.0),
        A.Sharpen(alpha=(0.4, 0.4), lightness=(0.7, 0.7), p=1.0),
        A.Normalize(p=1.0), # default mean and std is for ImageNet
    ])
    USE_RESIZE: bool = False
    RESIZE_TRANSFORMATION: A.Compose = A.Compose([
        A.Resize(512, 512),
    ])

@dataclass
class DataConfig:

    # image properties
    IMAGE_WIDTH: int = 1280
    IMAGE_HEIGHT: int = 960
    CLASSES_COUNT: int = 5 # 5 - mechanical(3) and chemical(2)
    CLASSES_NAMES: dict[int, str] = field(default_factory=lambda: {
        0: "pitting",
        1: "edge_abrasion",
        2: "precipitation",
        3: "adhering_particles",
        4: "conchoidal_fracture"
    })

    # train validation split
    TRAIN_SIZE: float = 0.8
    RANDOM_SEED: int = 42


@dataclass
class Model:
    # Main
    DEVICE: str = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    AVAILABLE_MODELS: list = field(default_factory=lambda:['Unet', 'DeepLabV3Plus', 'Segformer'])
    MODEL: str = 'DeepLabV3Plus'
    ENCODER: str = 'resnet50' # resnet50 for U-Net and DeepLabV3Plus, mit_b0 (or other b) for Segformer
    ENCODER_WEIGHTS: str = "imagenet" # for mit_b[1-5] available only imagenet, for other image-micronet
    BATCH_SIZE: int = 8 # 16, 8
    LEARNING_RATE: float = 0.0001 # 0.0001, 0.00001
    LOSS_FUNCTION: str = "FocalTverskyLoss" # FocalLoss, BCEWithLogitsLoss
    USE_CLIPPING: bool = False

    # Metrics and activation function
    THRESHOLD: float = 0.5
    CONF_MAT_SIZE: int = 3
    EPS: float = 1e-6

    # Focal Loss configs
    F_ALPHA: float = 0.25
    F_GAMMA: float = 2.0  # by official paper "we found γ = 2 to work best in our experiments"

    # Focal Tversky Loss configs. All values were best in original paper.
    FT_GAMMA: float = 0.75  # 1 / (4/3) => 3/4 => 0.75
    FT_ALPHA: float = 0.7
    FT_BETA: float = 0.3

    # Epochs
    EPOCH_COUNT: int = 300
    PATIENCE: int = 100 # early stopping

    # Scheduler (Cosine Annealing and Cosine Annealing with Warm Restarts)
    USE_CA: bool = False
    SCHEDULER: str = "CosineAnnealingLR" # CosineAnnealingWarmRestarts
    CA_T0: int = 10
    CA_TMULT: int = 2
    CA_TMAX: int = EPOCH_COUNT

    # Overlapping patches
    USE_PATCHES: bool = False
    PATCH_SIZE: int = 512 # imagenet pretrained encoder's input size. But try 512?
    OVERLAP_RATE: float = 0.5 # half of PATCH_SIZE # try 0.6 or 0.7
    PATCH_STRIDE: int = int(PATCH_SIZE * (1 - OVERLAP_RATE))


@dataclass
class Configs:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: Model = field(default_factory=Model)
    transform: Transformations = field(default_factory=Transformations)


config = Configs()
