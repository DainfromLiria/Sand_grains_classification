import json
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from configs import config


class Augmentation:
    """
    This class provide all instruments for augmentation of coco format dataset of sand grains.
    """

    def __init__(self) -> None:
        self._coco: COCO = COCO(config.ANNOTATIONS_PATH)
        self._categories = {c['name']: c['id'] for c in self._coco.loadCats(self._coco.getCatIds())}
        self._id: int = 0

    def augment(self) -> None:
        """Main public function of this class."""
        img_ids = self._coco.getImgIds()
        t_begin = time.monotonic()
        for img_id in tqdm(img_ids, desc="Augmentation"):
            image, masks, category_ids = self._load(img_id)
            transformed_data = self._apply_augmentations(image=image, masks=masks)
            self._save(transformed_data=transformed_data, category_ids=category_ids)
            break  # TODO remove it
        t_end = time.monotonic()
        self._create_dataset_info()
        print(f"Time taken for generating new dataset: {round(t_end - t_begin, 2)}s")

    def _load(self, img_id: int) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
        """
        Load one image with annotation from memory.

        :param img_id: image id from coco annotation file.

        :return: image, binary masks of annotations and category ids.
        """
        # load image
        image_path = os.path.join(config.IMAGES_PATH, self._coco.loadImgs(img_id)[0]['file_name'])
        image = cv2.imread(image_path)

        # load annotations
        ann_ids = self._coco.getAnnIds(imgIds=img_id)
        annotations = self._coco.loadAnns(ann_ids)

        category_ids = []
        masks = []

        for ann in annotations:
            category_ids.append(ann['category_id'])
            mask = self._coco.annToMask(ann)
            masks.append(mask)

        return image, masks, category_ids

    @staticmethod
    def _apply_augmentations(image: np.ndarray, masks: List[np.ndarray]) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Apply augmentations from configs and return list with new transformed images and masks.

        :param image: original image.
        :param masks: original masks.

        :return: transformed image and masks.
        """
        augmentations = config.AUGMENTATIONS
        transformed_data = [(image, masks)]
        for a in augmentations:
            transform = a(image=image, masks=masks)
            t_image = transform["image"]
            t_masks = transform["masks"]
            transformed_data.append((t_image, t_masks))
        return transformed_data

    def _save(
            self,
            transformed_data: List[Tuple[np.ndarray, List[np.ndarray]]],
            category_ids: List[int]
    ) -> None:
        """
        Save transformed and original images, masks and categories into folders.

        :param transformed_data: transformed images and masks (+original).
        :param category_ids: list of categories id's from original image.
        """
        if os.path.exists(path=config.AUG_DATASET_PATH):
            raise Exception("Augmentations data folder already exists.")
        os.mkdir(path=config.AUG_DATASET_PATH)

        for image, masks in transformed_data:
            # create dir for image, masks and categories
            img_folder_path = os.path.join(config.AUG_DATASET_PATH, str(self._id))
            if os.path.exists(img_folder_path):
                raise Exception(f"Folder for this image id already exists: {img_folder_path}")
            # save image
            os.mkdir(path=img_folder_path)
            img_path = os.path.join(img_folder_path, f"{self._id}.tif")
            cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            # save masks
            masks_path = os.path.join(img_folder_path, "masks")
            os.mkdir(path=masks_path)
            for i, mask in enumerate(masks):
                mask_path = os.path.join(masks_path, f"{i}.png")
                cv2.imwrite(mask_path, mask)
            # save categories
            categories_path = os.path.join(img_folder_path, "categories.txt")
            with open(categories_path, "w") as file:
                file.write(str(category_ids))
            self._id += 1

    def _create_dataset_info(self) -> None:
        """Create json file with main information about dataset."""
        info_data = {
            "img_count": self._id,
            "classes": self._categories
        }
        with open(config.AUG_DATASET_INFO_PATH, "w") as file:
            json.dump(info_data, file, indent=4)
