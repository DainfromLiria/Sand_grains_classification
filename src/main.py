from configs.config import config
from dataset.augmentation import Augmentation
from dataset.dataset import SandGrainsDataset
from detector.detector import MicroTextureDetector
from utils.logging import setup_logging
import logging

setup_logging()

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # aug = Augmentation()
    # aug.augment()

    dt = SandGrainsDataset(path=config.data.AUG_TRAIN_SET_PATH)
    img, masks = dt[0]
    logger.info(f"Image shape: {img.shape} - Mask shape: {masks.shape}")

    # net = MicroTextureDetector()
    # net.train()

