import json
import logging
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import config


class SandGrainsDataset(Dataset):

    def __init__(self, path: str) -> None:
        self.path = path
        self._folder_number = [f for f in os.listdir(path)]
        self._read_info()

    def __len__(self) -> int:
        return len(self._folder_number)

    def __getitem__(self, idx: int):
        real_idx = self._folder_number[idx]
        data_path = os.path.join(self.path, real_idx)
        if not os.path.exists(data_path):
            raise Exception(f"Folder with index {real_idx} does not exist")

        # read image
        image_path = os.path.join(data_path, f"{real_idx}.tif")
        if not os.path.exists(image_path):
            raise Exception(f"File with name {real_idx}.tif does not exist")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = config.data.IMAGE_TRAIN_TRANSFORMATION(image=image)["image"]

        # read masks
        masks_dir = os.path.join(data_path, "masks")
        if not os.path.exists(masks_dir):
            raise Exception(f"Folder with masks for index {real_idx} does not exist")
        masks = []
        for file in range(len(os.listdir(masks_dir))):
            mask_path = os.path.join(masks_dir, f"{file}.png")
            if not os.path.exists(mask_path):
                raise Exception(f"Mask with index {file} does not exist")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = config.data.MASK_TRAIN_TRANSFORMATION(image=mask)["image"]  # resize
            masks.append(mask)
        # convert masks into tensor
        tensor_masks = torch.zeros(
            size=(self.info["classes_count"], config.data.IMAGE_RESIZED, config.data.IMAGE_RESIZED),
            dtype=torch.uint8
        )
        for i, c_idx in enumerate(self.info["categories"][str(real_idx)]):
            tensor_masks[c_idx, :, :] = tensor_masks[c_idx, :, :] | torch.tensor(masks[i], dtype=torch.uint8)
        return image, tensor_masks

    def _read_info(self):
        """Read file with main information about dataset."""
        with open(config.data.AUG_DATASET_INFO_PATH, "r") as file:
            self.info = json.load(file)
