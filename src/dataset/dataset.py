import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from configs import config


class SandGrainsDataset(Dataset):

    def __init__(self) -> None:
        self._read_info()

    def __len__(self) -> int:
        return self.info["img_count"]

    def __getitem__(self, idx: int):
        pass

    def _read_info(self):
        """Read file with main information about dataset."""
        with open(config.AUG_DATASET_INFO_PATH, "r") as file:
            self.info = json.load(file)
