import torch

from configs import config
# from dataset import SandGrainsDataset
from detector import MicroTextureDetector
from utils.logging import setup_logging

# from torch.utils.data import DataLoader

if __name__ == '__main__':
    setup_logging()
    torch.hub.set_dir(config.paths.MODELS_FOLDER)  # TODO move to some setup file

    # dataset = SandGrainsDataset(mode="val")
    # img = dataset[0]
    #
    det = MicroTextureDetector(mode="train")
    det.train()
