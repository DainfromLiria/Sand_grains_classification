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
        outputs = outputs.reshape(outputs.size(1), -1)
        targets = targets.reshape(targets.size(1), -1)

        alpha = torch.tensor(config.model.F_ALPHA)
        bce_loss = F.binary_cross_entropy_with_logits(input=outputs, target=targets, reduction='none')  # -log(pt)
        pt = torch.exp(-bce_loss)
        focal_loss_per_pixel = alpha * (torch.tensor(1) - pt)**config.model.F_GAMMA * bce_loss
        return torch.mean(focal_loss_per_pixel)
# ====================================================================================================================


# ====================================================================================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()
        self.eps = 1e-6  # for situations when mask and target contains only zeros

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = torch.sigmoid(outputs)

        # ==========================================================================================================
        # https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/losses/tversky.py
        tp = torch.sum(outputs * targets, dim=(0, 2, 3))
        fn = torch.sum((1 - outputs) * targets, dim=(0, 2, 3))
        fp = torch.sum(outputs * (1 - targets), dim=(0, 2, 3))
        # ==========================================================================================================

        ti = (tp + self.eps) / (tp + (config.model.FT_ALPHA * fn) + (config.model.FT_BETA * fp) + self.eps)
        tfl = (1 - ti)**config.model.FT_GAMMA

        return torch.mean(tfl)
# ====================================================================================================================
