import logging

import torch

from configs import config
from utils import predict_morphological_feature

logger = logging.getLogger(__name__)


def confusion_matrix(outputs: torch.Tensor, targets: torch.Tensor):
    """
    Calculate confusion matrix per batch.

    :param outputs: inputs with predictions (values 0 or 1).
    :param targets: binary masks.

    :return: confusion matrix values in order tp, fp, fn, tn. All values are torch.Tensor.
    """
    # calculate TP
    tp_mat = torch.logical_and(outputs, targets)
    tp = torch.sum(tp_mat, dim=(0, 2, 3), keepdim=True)
    # calculate FP and FN
    inp_xor_target = torch.logical_xor(outputs, targets)
    fp_mat = torch.logical_and(inp_xor_target, outputs)
    fp = torch.sum(fp_mat, dim=(0, 2, 3), keepdim=True)
    fn_mat = torch.logical_and(inp_xor_target, targets)
    fn = torch.sum(fn_mat, dim=(0, 2, 3), keepdim=True)
    # calculate TN
    tn_mat = torch.logical_not(torch.logical_or(outputs, targets))
    tn = torch.sum(tn_mat, dim=(0, 2, 3), keepdim=True)
    return tp, fp, fn, tn


def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-6  # for situations when mask and target contains only zeros
    # convert probabilities into labels
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > config.model.THRESHOLD).type(torch.uint8)
    outputs = predict_morphological_feature(outputs)

    tp, fp, fn, tn = confusion_matrix(outputs=outputs, targets=targets)

    precision_per_class = torch.squeeze((tp + eps) / (tp + fp + eps))
    recall_per_class = torch.squeeze((tp + eps) / (tp + fn + eps))
    iou_per_class = torch.squeeze((tp + eps) / (tp + fp + fn + eps))

    return torch.stack((iou_per_class, recall_per_class, precision_per_class))
