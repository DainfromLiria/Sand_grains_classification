import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from configs import config
from utils import calculate_patch_positions, get_patch_by_position

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

        :param mode: can be "train", "val" or "eval".
        """
        if mode not in ["train", "val", "eval"]:
            raise ValueError(f"mode {mode} is not supported.")
        self.mode = mode
        self.annot_path: Path = config.paths.TRAIN_ANNOTATIONS if self.mode != "eval" else config.paths.EVAL_ANNOTATIONS
        self.img_path: Path = config.paths.TRAIN_IMAGES_FOLDER if self.mode != "eval" else config.paths.EVAL_IMAGES_FOLDER
        self.dataset_info_path: Path = config.paths.TRAIN_DATASET_INFO if self.mode != "eval" else config.paths.EVAL_DATASET_INFO
        self._coco: COCO = COCO(self.annot_path)
        self.dataset_info: Dict[str, Any] = {}
        self._load_dataset_info()

        if self.mode in ("train", "val"):
            self.calculate_counts_per_class()

        # patches
        if config.model.USE_PATCHES:
            self.positions: list = calculate_patch_positions()
            self.patches_count: int = len(self.positions)

    def __len__(self) -> int:
        if self.mode == "eval":
            ln: int = len(self._coco.getImgIds())
        else:
            ln: int = self.dataset_info["train_size"] if self.mode == "train" else self.dataset_info["val_size"]
        return ln * self.patches_count if config.model.USE_PATCHES else ln

    def _load_dataset_info(self) -> None:
        """Load information about the training dataset."""
        if not self.dataset_info_path.exists():
            logger.warning("Dataset doesn't have dataset_info.json file. Generating a new one.")
            self.get_main_dataset_info()
            if self.mode != "eval":
                logger.warning("Original train dataset isn't split on train and validation. Splitting it now.")
                self._train_val_split()
            with open(self.dataset_info_path, "w") as f:
                json.dump(self.dataset_info, f, indent=4)

        with open(self.dataset_info_path, "r", encoding="utf-8") as f:
            self.dataset_info = json.load(f)

    def _train_val_split(self) -> None:
        """Split data on train and validation set"""
        img_ids = self._coco.getImgIds()
        random.shuffle(img_ids)
        # calculate sizes
        train_end = int(self.dataset_info["img_count"] * config.data.TRAIN_SIZE)
        # split indexes
        self.dataset_info["train_idx"] = img_ids[:train_end]
        self.dataset_info["train_size"] = len(self.dataset_info["train_idx"])
        self.dataset_info["val_idx"] = img_ids[train_end:]
        self.dataset_info["val_size"] = len(self.dataset_info["val_idx"])

    def get_main_dataset_info(self) -> None:
        """Calculate main dataset info."""
        self.dataset_info["classes"] = {c['name']: c['id'] - 1 for c in self._coco.loadCats(self._coco.getCatIds())}
        self.dataset_info["num_classes"] = len(self.dataset_info["classes"])
        self.dataset_info["img_count"] = len(self._coco.getImgIds())

    def calculate_counts_per_class(self) -> None:
        """Calculate count of microtextures in train/val dataset per class"""
        category_ids: dict = defaultdict(int)
        ln: int = self.dataset_info["train_size"] if self.mode == "train" else self.dataset_info["val_size"]
        for i in range(0, ln):
            coco_id = self.dataset_info["train_idx" if self.mode == "train" else "val_idx"][i]
            ann_ids = self._coco.getAnnIds(imgIds=coco_id)
            annotations = self._coco.loadAnns(ann_ids)
            for ann in annotations:
                category_ids[ann['category_id'] - 1] += 1
        logger.warning(f"Stats in mode {self.mode}: {category_ids}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one image with masks from memory by index.

        :param idx: index in array of coco indexes (from 0 to array_size).

        :return: two tensors, where first is image and second is mask.
        """
        img_idx = idx
        if config.model.USE_PATCHES:
            img_idx = idx // self.patches_count
            idx = idx % self.patches_count
        coco_id: int = img_idx + 1 # coco indexes starts from 1
        if self.mode != "eval":
            coco_id = self.dataset_info["train_idx" if self.mode == "train" else "val_idx"][img_idx]
        # load image
        image_path: Path = self.img_path / Path(self._coco.loadImgs(coco_id)[0]['file_name'])
        image: np.ndarray = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if config.model.USE_PATCHES:
            image = get_patch_by_position(img=image, pos=self.positions[idx])

        # load annotations
        category_ids: List[int] = []
        masks: List[np.ndarray] = []
        ann_ids = self._coco.getAnnIds(imgIds=coco_id)
        annotations = self._coco.loadAnns(ann_ids)
        for ann in annotations:
            category_ids.append(ann['category_id'] - 1)
            mask = self._coco.annToMask(ann)
            if config.transform.USE_PREPROCESSING:
                mask: np.ndarray = config.transform.MASK_TRANSFORMATION(image=mask)["image"]
            if config.model.USE_PATCHES:
                mask = get_patch_by_position(img=mask, pos=self.positions[idx])
            masks.append(mask)

        # apply image preprocessing transformations
        if config.transform.USE_PREPROCESSING:
            image: torch.Tensor = config.transform.IMAGE_TRANSFORMATIONS(image=image)["image"]

        # apply augmentations
        if self.mode == "train" and config.transform.USE_AUGMENTATIONS:
            aug = config.transform.AUGMENTATIONS(image=image, masks=masks)
            image, masks = aug["image"], aug["masks"]

        # convert and join masks into tensor
        size = (self.dataset_info["num_classes"], masks[0].shape[0], masks[0].shape[1])
        tensor_masks: torch.Tensor = torch.zeros(size=size, dtype=torch.uint8)
        for i, c_idx in enumerate(category_ids):
            tensor_masks[c_idx, :, :] = tensor_masks[c_idx, :, :] | torch.tensor(masks[i], dtype=torch.uint8)

        # convert image to torch.Tensor
        image: torch.Tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return image, tensor_masks
