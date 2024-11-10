import json
import os.path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from configs import config


class SandGrainsDataset(Dataset):

    def __init__(self) -> None:
        self._read_info()

    def __len__(self) -> int:
        return self.info["img_count"]

    def __getitem__(self, idx: int):
        data_path = os.path.join(config.AUG_DATASET_PATH, str(idx))
        if not os.path.exists(data_path):
            raise Exception(f"Folder with index {idx} does not exist")

        # read image
        image_path = os.path.join(data_path, f"{idx}.tif")
        if not os.path.exists(image_path):
            raise Exception(f"File with name {idx}.tif does not exist")
        image = cv2.imread(image_path)

        print(f"Categories: {self.info['categories'][str(idx)]}")
        # read masks
        masks_dir = os.path.join(data_path, "masks")
        if not os.path.exists(masks_dir):
            raise Exception(f"Folder with masks for index {idx} does not exist")
        masks = []
        for file in range(len(os.listdir(masks_dir))):
            mask_path = os.path.join(masks_dir, f"{file}.png")
            if not os.path.exists(mask_path):
                raise Exception(f"Mask with index {file} does not exist")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
        # TODO convert image into tensor
        return image

    def _read_info(self):
        """Read file with main information about dataset."""
        with open(config.AUG_DATASET_INFO_PATH, "r") as file:
            self.info = json.load(file)
