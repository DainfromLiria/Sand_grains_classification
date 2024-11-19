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
        targets = targets.flatten()
        alpha = self._get_alpha(targets)
        bce_loss = F.binary_cross_entropy(weight=alpha, input=inputs, target=targets, reduction='none')  # -log(pt)
        pt = torch.exp(-bce_loss)
        focal_loss_per_pixel = (torch.tensor(1) - pt)**config.model.GAMMA * bce_loss
        return torch.mean(focal_loss_per_pixel)

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
            _, p = torch.unique(targets, return_counts=True)
            w0 = N / (2 * p[0])
            w1 = N / (2 * p[1])
            alpha = torch.where(targets == 0, w0, w1)
            return alpha
        except Exception as e:
            logger.error(f"{e}")  # in case if all mask contains only 0 or 1
            raise e
