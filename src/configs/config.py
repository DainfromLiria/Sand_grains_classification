from dataclasses import dataclass, field

import albumentations as A


@dataclass
class DataConfig:
    IMAGES_PATH: str = "../data/images"
    ANNOTATIONS_PATH: str = "../data/annotations/instances_default.json"
    IMAGE_WIDTH: int = 1280
    IMAGE_HEIGHT: int = 960
    AUG_DATASET_PATH: str = "../aug_data"
    AUG_DATASET_INFO_PATH: str = "../aug_data/info.json"

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


config = DataConfig()
