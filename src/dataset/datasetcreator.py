import json
import logging
import os
import random
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from configs import config

logger = logging.getLogger(__name__)


class DatasetCreator:
    """
    This class read COCO dataset with original data, split it on train, validation and test, augment
    train dataset and save it dataset in new format for training.
    """

    def __init__(self) -> None:
        self._coco: COCO = COCO(config.data.ANNOTATIONS_PATH)
        self._classes = {c['name']: c['id']-1 for c in self._coco.loadCats(self._coco.getCatIds())}
        self._categories: Dict[int, List[int]] = {}
        self._data: Dict[int, Tuple[np.ndarray, List[np.ndarray], List[int]]] = {}
        self._idx = 0

        self.train: Dict[int, Tuple[np.ndarray, List[np.ndarray], List[int]]] = {}
        self.val: Dict[int, Tuple[np.ndarray, List[np.ndarray], List[int]]] = {}
        self.test: Dict[int, Tuple[np.ndarray, List[np.ndarray], List[int]]] = {}

    def assemble(self) -> None:
        """Main public function of this class."""
        img_ids = self._coco.getImgIds()
        t_begin = time.monotonic()
        for i in tqdm(img_ids, desc="Read original data"):
            self._data[self._idx] = self._load(i)
            self._categories[self._idx] = self._data[self._idx][2]
            self._idx += 1
        self.train_test_split()
        self._augment_train_data()
        self._save_dataset()
        self._create_dataset_info()
        t_end = time.monotonic()
        logger.info(f"Train size: {len(self.train)}, Val size: {len(self.val)}, Test size: {len(self.test)}")
        logger.info(f"Time taken for assembling dataset: {round(t_end - t_begin, 2)}s")

    def _load(self, img_id: int) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
        """
        Load one image with annotation from memory.

        :param img_id: image id from coco annotation file.

        :return: image, binary masks of annotations and category ids.
        """
        # load image
        image_path = os.path.join(config.data.IMAGES_PATH, self._coco.loadImgs(img_id)[0]['file_name'])
        image = cv2.imread(image_path)

        # load annotations
        ann_ids = self._coco.getAnnIds(imgIds=img_id)
        annotations = self._coco.loadAnns(ann_ids)

        category_ids = []
        masks = []

        for ann in annotations:
            category_ids.append(ann['category_id'] - 1)
            mask = self._coco.annToMask(ann)
            masks.append(mask)

        return image, masks, category_ids

    def train_test_split(self) -> None:
        """Split the dataset into train, validation and test."""
        random.seed(config.data.RANDOM_SEED)
        keys = list(self._data.keys())
        random.shuffle(keys)
        # calculate sizes
        dataset_size = len(keys)
        train_end = int(dataset_size * config.data.TRAIN_SIZE)
        val_end = train_end + int(dataset_size * config.data.VAL_SIZE)
        # split keys
        train_keys = keys[:train_end]
        val_keys = keys[train_end:val_end]
        test_keys = keys[val_end:]
        # split data
        self.train = {k: self._data[k] for k in train_keys}
        self.val = {k: self._data[k] for k in val_keys}
        self.test = {k: self._data[k] for k in test_keys}

    def _augment_train_data(self):
        """Apply augmentations from configs on train dataset."""
        train_keys = list(self.train.keys())
        for i in tqdm(train_keys, desc="Augment train data"):
            image, masks, category_ids = self.train[i]
            augmentations = config.data.AUGMENTATIONS
            for a in augmentations:
                transform = a(image=image, masks=masks)
                t_image = transform["image"]
                t_masks = transform["masks"]
                self.train[self._idx] = (t_image, t_masks, category_ids)
                self._categories[self._idx] = category_ids
                self._idx += 1

    def _save_dataset(self) -> None:
        """Save images, masks and categories into folders."""
        if not os.path.exists(path=config.data.DATASET_PATH):
            os.mkdir(path=config.data.DATASET_PATH)
        self._save(self.train, path=config.data.TRAIN_SET_PATH)
        self._save(self.val, path=config.data.VAL_SET_PATH)
        self._save(self.test, path=config.data.TEST_SET_PATH)

    @staticmethod
    def _save(data: Dict, path: str) -> None:
        """Save part of dataset (train, validation or test)."""
        if not os.path.exists(path=path):
            os.mkdir(path=path)
        for idx, values in data.items():
            image, masks, category_ids = values
            # create dir for image, masks and categories
            img_folder_path = os.path.join(path, str(idx))
            if os.path.exists(img_folder_path):
                raise Exception(f"Folder for this image id already exists: {img_folder_path}")
            # save image
            os.mkdir(path=img_folder_path)
            img_path = os.path.join(img_folder_path, f"{idx}.tif")
            cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            # save masks
            masks_path = os.path.join(img_folder_path, "masks")
            os.mkdir(path=masks_path)
            for i, mask in enumerate(masks):
                mask_path = os.path.join(masks_path, f"{i}.png")
                cv2.imwrite(mask_path, mask)

    def _create_dataset_info(self) -> None:
        """Create json file with main information about dataset."""
        info_data = {
            "img_count": self._idx,
            "classes_count": len(self._classes),
            "classes": self._classes,
            "categories": self._categories
        }
        with open(config.data.DATASET_INFO_PATH, "w") as file:
            json.dump(info_data, file, indent=4)
