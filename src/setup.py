import torch

from configs import config
from utils.logging import setup_logging


def setup() -> None:
    setup_logging()
    torch.hub.set_dir(config.paths.MODELS_FOLDER)
