import logging

import torch
import torch.nn.functional as F
from torch import nn

from configs.config import config

logger = logging.getLogger(__name__)


# ====================================================================================================================
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate alpha weighted focal loss for all input batch. Use mean for reduction.

        :param outputs: output of last nn layer.
        :param targets: masks.

        :return: mean alpha weighted focal loss for batch.
        """
        outputs = torch.sigmoid(outputs)

        outputs = outputs.flatten()
        targets = targets.flatten()
        alpha = self._get_alpha(targets)
        bce_loss = F.binary_cross_entropy(weight=alpha, input=outputs, target=targets, reduction='none')  # -log(pt)
        pt = torch.exp(-bce_loss)
        focal_loss_per_pixel = (torch.tensor(1) - pt)**config.model.F_GAMMA * bce_loss
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
# ====================================================================================================================


# ====================================================================================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = 1e-6  # for situations when mask and target contains only zeros
        outputs = torch.sigmoid(outputs)

        outputs = outputs.flatten()
        targets = targets.flatten()

        # ==========================================================================================================
        # https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/losses/tversky.py
        tp = torch.sum(outputs * targets, dim=0)
        fn = torch.sum((1 - outputs) * targets, dim=0)
        fp = torch.sum(outputs * (1 - targets), dim=0)
        # ==========================================================================================================

        ti = (tp + eps) / (tp + (config.model.FT_ALPHA * fn) + (config.model.FT_BETA * fp) + eps)
        tfl = (1 - ti)**config.model.FT_GAMMA
        return torch.mean(tfl.squeeze())
# ====================================================================================================================
