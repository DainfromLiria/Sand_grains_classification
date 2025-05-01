import logging

import torch

from configs import config

logger = logging.getLogger(__name__)


def calculate_confusion_matrix(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate confusion matrix per batch.

    :param outputs: inputs with predictions (values 0 or 1).
    :param targets: binary masks.

    :return: confusion matrix in Tensor format with values in order tp, fp, fn. All values are torch.Tensor.
    """
    # convert probabilities into labels (only TTA returns binary mask)
    if not config.transform.USE_TTA:
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > config.model.THRESHOLD).type(torch.uint8)

    # calculate TP
    tp_mat = torch.logical_and(outputs, targets)
    tp = torch.sum(tp_mat, dim=(0, 2, 3))
    # calculate FP and FN
    inp_xor_target = torch.logical_xor(outputs, targets)
    # FP
    fp_mat = torch.logical_and(inp_xor_target, outputs)
    fp = torch.sum(fp_mat, dim=(0, 2, 3))
    # FN
    fn_mat = torch.logical_and(inp_xor_target, targets)
    fn = torch.sum(fn_mat, dim=(0, 2, 3))

    return torch.stack((tp, fp, fn))
