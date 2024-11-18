import torch
from torch import nn
import torch.nn.functional as F
import logging
from configs.config import config

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    @staticmethod
    def forward(inputs: torch.Tensor, targets: torch.Tensor):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.flatten()
        logger.info(f"inputs: {inputs}")
        targets = targets.flatten()
        logger.info(f"targets: {targets}")
        bce_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt)**2 * bce_loss
        logger.info(f"focal_loss: {focal_loss}")
