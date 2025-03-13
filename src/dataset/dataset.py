import json
import logging
import random
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pathlib import Path

from configs import config

logger = logging.getLogger(__name__)


class SandGrainsDataset(Dataset):
    """
    This class read COCO dataset with original data, split it on train and validation if dataset_info.json
    not found and augment train dataset. Also, this class provide __getitem__ method that return images by
    index.
    """

    def __init__(self, mode: str) -> None:
        """
        Initialize the dataset creator.

        :param mode: can be "train", "val" or "eval"
        """
        self.mode = mode
        self.annot_path: Path = config.paths.TRAIN_ANNOTATIONS if self.mode in {"train", "val"} else config.paths.EVAL_ANNOTATIONS
        self.img_path: Path = config.paths.TRAIN_IMAGES_FOLDER if self.mode in {"train", "val"} else config.paths.EVAL_IMAGES_FOLDER
        self._coco: COCO = COCO(self.annot_path)
        self.dataset_info: Dict[str, Any] = {}
        if self.mode in {"train", "val"}:
            self._load_dataset_info()

    def __len__(self) -> int:
        if self.mode not in {"train", "val"}:
            return len(self._coco.getImgIds())
        return self.dataset_info["train_size"] if self.mode == "train" else self.dataset_info["val_size"]

    def _load_dataset_info(self) -> None:
        """Load information about the training dataset."""
        if not config.paths.TRAIN_DATASET_INFO.exists():
            logger.warning("Original train dataset isn't split on train and validation. Splitting it now.")
            self._train_val_split()
            with open(config.paths.TRAIN_DATASET_INFO, "w") as f:
                json.dump(self.dataset_info, f, indent=4)

        with open(config.paths.TRAIN_DATASET_INFO, "r", encoding="utf-8") as f:
            self.dataset_info = json.load(f)

    def _train_val_split(self) -> None:
        """Split data on train and validation set"""
        img_ids = self._coco.getImgIds()
        self.dataset_info["classes"] = {c['name']: c['id']-1 for c in self._coco.loadCats(self._coco.getCatIds())}
        self.dataset_info["num_classes"] = len(self.dataset_info["classes"])
        random.shuffle(img_ids)
        # calculate sizes
        self.dataset_info["img_count"] = len(img_ids)
        train_end = int(self.dataset_info["img_count"] * config.data.TRAIN_SIZE)
        # split indexes
        self.dataset_info["train_idx"] = img_ids[:train_end]
        self.dataset_info["train_size"] = len(self.dataset_info["train_idx"])
        self.dataset_info["val_idx"] = img_ids[train_end:]
        self.dataset_info["val_size"] = len(self.dataset_info["val_idx"])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one image with masks from memory by index.

        :param idx: index in array of coco indexes (from 0 to array_size).

        :return: two tensors, where first is image and second is mask.
        """
        coco_id: int = idx + 1 # coco indexes starts from 1
        if self.mode in {"train", "val"}:
            coco_id = self.dataset_info["train_idx" if self.mode == "train" else "val_idx"][idx]
        # load image
        image_path: Path = self.img_path / Path(self._coco.loadImgs(coco_id)[0]['file_name'])
        image: np.ndarray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # load annotations
        category_ids: List[int] = []
        masks: List[np.ndarray] = []
        ann_ids = self._coco.getAnnIds(imgIds=coco_id)
        annotations = self._coco.loadAnns(ann_ids)
        for ann in annotations:
            category_ids.append(ann['category_id'] - 1)
            mask = self._coco.annToMask(ann)
            masks.append(mask)

        # apply image preprocessing transformations
        image: torch.Tensor = config.transform.IMAGE_TRANSFORMATIONS(image=image)["image"]

        # apply augmentations
        if self.mode == "train":
            aug = config.transform.AUGMENTATIONS(image=image, masks=masks)
            image, masks = aug["image"], aug["masks"]

        # convert and join masks into tensor
        size = (self.dataset_info["num_classes"], masks[0].shape[0], masks[0].shape[1])
        tensor_masks: torch.Tensor = torch.zeros(size=size, dtype=torch.uint8)
        for i, c_idx in enumerate(category_ids):
            tensor_masks[c_idx, :, :] = tensor_masks[c_idx, :, :] | torch.tensor(masks[i], dtype=torch.uint8)

        # convert image to torch.Tensor
        if image.ndim < 3:
            image: np.ndarray = np.expand_dims(image, 2) # for grayscale (mono channel image)
        image: torch.Tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return image, tensor_masks
