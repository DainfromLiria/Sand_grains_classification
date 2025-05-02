import logging

import torch

from configs import config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setup() -> None:
    setup_logging()
    torch.hub.set_dir(config.paths.MODELS_FOLDER)
