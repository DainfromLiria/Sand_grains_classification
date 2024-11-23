import torch
import logging

logger = logging.getLogger(__name__)


def confusion_matrix(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Calculate confusion matrix.

    :param inputs: inputs with predictions (values 0 or 1).
    :param targets: binary masks.

    :return: confusion matrix values in order tp, fp, fn, tn. All values are torch.Tensor.
    """
    # calculate TP
    tp_mat = torch.logical_and(inputs, targets)
    tp = torch.sum(tp_mat, dim=(0, 2, 3), keepdim=True)
    # calculate FP and FN
    inp_xor_target = torch.logical_xor(inputs, targets)
    fp_mat = torch.logical_and(inp_xor_target, inputs)
    fp = torch.sum(fp_mat, dim=(0, 2, 3), keepdim=True)
    fn_mat = torch.logical_and(inp_xor_target, targets)
    fn = torch.sum(fn_mat, dim=(0, 2, 3), keepdim=True)
    # calculate TN
    tn_mat = torch.logical_not(torch.logical_or(inputs, targets))
    tn = torch.sum(tn_mat, dim=(0, 2, 3), keepdim=True)
    return tp, fp, fn, tn


def calculate_metrics(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-6  # for situations when mask and target contains only zeros
    inputs = torch.sigmoid(inputs)
    inputs = (inputs >= 0.5).int()  # convert probabilities into labels

    tp, fp, fn, tn = confusion_matrix(inputs=inputs, targets=targets)

    precision_per_class = torch.squeeze(tp / (tp + fp + eps))
    recall_per_class = torch.squeeze(tp / (tp + fn + eps))
    iou_per_class = torch.squeeze(tp / (tp + fp + fn + eps))

    return torch.stack((iou_per_class, recall_per_class, precision_per_class))
