import logging

import torch

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
    outputs = convert_nn_output(outputs=outputs, to_mask=True)  # convert probabilities into labels

    tp, fp, fn, tn = confusion_matrix(outputs=outputs, targets=targets)

    precision_per_class = torch.squeeze((tp + eps) / (tp + fp + eps))
    recall_per_class = torch.squeeze((tp + eps) / (tp + fn + eps))
    iou_per_class = torch.squeeze((tp + eps) / (tp + fp + fn + eps))

    return torch.stack((iou_per_class, recall_per_class, precision_per_class))


def convert_nn_output(outputs: torch.Tensor, to_mask: bool) -> torch.Tensor:
    relief_idx = [4, 5, 12]
    shape_idx = [6, 11, 14, 17]  # TODO hardcoded part, ideally must be dynamic.
    other_idx = [0, 1, 2, 3, 7, 8, 9, 10, 13, 15, 16, 18]

    outputs[:, other_idx] = torch.sigmoid(outputs[:, other_idx])
    outputs[:, relief_idx] = torch.softmax(outputs[:, relief_idx], dim=1)
    outputs[:, shape_idx] = torch.softmax(outputs[:, shape_idx], dim=1)
    if to_mask:
        # convert probabilities into labels
        # =========================================================================================================
        # https://discuss.pytorch.org/t/how-to-convert-argmax-result-to-an-one-hot-matrix/125508
        relief_max_vals = torch.argmax(outputs[:, relief_idx], dim=1)
        outputs[:, relief_idx] = torch.zeros_like(outputs[:, relief_idx]).scatter_(1, relief_max_vals.unsqueeze(1), 1.)
        shape_max_vals = torch.argmax(outputs[:, shape_idx], dim=1)
        outputs[:, shape_idx] = torch.zeros_like(outputs[:, shape_idx]).scatter_(1, shape_max_vals.unsqueeze(1), 1.)
        # =========================================================================================================
        outputs[:, other_idx] = (outputs[:, other_idx] > 0.5).float()
    return outputs
