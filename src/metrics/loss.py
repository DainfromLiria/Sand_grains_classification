import torch
from torch import nn
import torch.nn.functional as F
import logging
from configs.config import config

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)

        inputs = inputs.flatten()
        logger.info(f"inputs: {inputs}")
        targets = targets.flatten()
        logger.info(f"targets: {targets}")
        alpha = self._get_alpha(targets)
        bce_loss = F.binary_cross_entropy(weight=alpha, input=inputs, target=targets, reduction='none')  # -log(pt)
        logger.info(f"bce_loss: {bce_loss}")
        pt = torch.exp(-bce_loss)
        logger.info(f"pt: {pt}")
        focal_loss = (torch.tensor(1) - pt)**config.model.GAMMA * bce_loss
        logger.info(f"focal_loss: {focal_loss}")
        return torch.mean(focal_loss)

    @staticmethod
    def _get_alpha(targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Inverse-Frequency Class Weights, that will be used as alphas (weights) in cross entropy
        loss calculation.

        :param targets: input 1-dim tensor of targets with values 0 or 1.

        :return: Inverse-Frequency Class Weights (alphas)
        """
        assert targets.dim() == 1  # check if value is vector
        try:
            N = len(targets)
            p1 = sum(targets)
            p0 = N - p1
            w0 = N / (2 * p0)
            w1 = N / (2 * p1)
            alpha = torch.tensor([w0 if t == 0 else w1 for t in targets])
            return alpha
        except ZeroDivisionError as e:
            logger.error(f"{e}")  # in case if all mask contains only 0 or 1
            raise ZeroDivisionError
