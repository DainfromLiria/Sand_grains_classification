import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from configs import config

logger = logging.getLogger(__name__)


class DatasetCreator:
    """
    This class read COCO dataset with original data, split it on train and validation if dataset_info.json
    not found and augment train dataset. Also, this class provide load() method that return images by
    index for __getitem__ method of pytorch Dataset class.
    """

    def __init__(self, mode: str) -> None:
        """
        Initialize the dataset creator.

        :param mode: can be "train", "val" or "eval"
        """
        self.mode = mode
        self.annot_path: str = config.data.TRAIN_ANNOTATIONS_PATH if self.mode in {"train", "val"} else config.data.EVAL_ANNOTATIONS_PATH
        self.img_path: str = config.data.TRAIN_IMAGES_PATH if self.mode in {"train", "val"} else config.data.EVAL_IMAGES_PATH
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
        if not os.path.exists(path=config.data.TRAIN_DATASET_INFO_PATH):
            logger.warning("Original train dataset isn't split on train and validation. Splitting it now.")
            self._train_val_split()
            with open(config.data.TRAIN_DATASET_INFO_PATH, "w") as f:
                json.dump(self.dataset_info, f, indent=4)

        with open(config.data.TRAIN_DATASET_INFO_PATH, "r", encoding="utf-8") as f:
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

    def load(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one image with masks from memory by index.

        :param idx: index in array of coco indexes (from 0 to array_size).

        :return: two tensors, where first is image and second is mask.
        """
        coco_id: int = idx + 1 # coco indexes starts from 1
        if self.mode in {"train", "val"}:
            coco_id = self.dataset_info["train_idx" if self.mode == "train" else "val_idx"][idx]
        # load image
        image_path = os.path.join(self.img_path, self._coco.loadImgs(coco_id)[0]['file_name'])
        image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # load annotations
        category_ids: List[int] = []
        masks: List[np.ndarray] = []
        ann_ids = self._coco.getAnnIds(imgIds=coco_id)
        annotations = self._coco.loadAnns(ann_ids)
        for ann in annotations:
            category_ids.append(ann['category_id'] - 1)
            mask = self._coco.annToMask(ann)
            masks.append(mask)

        # apply augmentations
        if self.mode == "train":
            aug = config.data.AUGMENTATIONS(image=image, masks=masks)
            image, masks = aug["image"], aug["masks"]

        # convert and join masks into tensor
        size = (self.dataset_info["num_classes"], masks[0].shape[0], masks[0].shape[1])
        tensor_masks: torch.Tensor = torch.zeros(size=size, dtype=torch.uint8)
        for i, c_idx in enumerate(category_ids):
            tensor_masks[c_idx, :, :] = tensor_masks[c_idx, :, :] | torch.tensor(masks[i], dtype=torch.uint8)

        # apply final image transformations
        image: torch.Tensor = config.data.IMAGE_TRAIN_TRANSFORMATIONS(image=image)["image"]
        return image, tensor_masks
