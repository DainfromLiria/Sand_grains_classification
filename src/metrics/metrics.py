import torch
import numpy as np
from collections import defaultdict
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def IOU(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Calculate Interception Over Union per class.

    :param inputs: predictions.
    :param targets: trues.

    :return: IOU per class.
    """
    interception = torch.logical_and(inputs, targets)
    union = torch.logical_or(inputs, targets)
    interception_sum = torch.sum(interception, dim=(0, 2, 3), keepdim=True)
    union_sum = torch.sum(union, dim=(0, 2, 3), keepdim=True)
    return interception_sum / union_sum


def calculate_metrics(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = torch.sigmoid(inputs)
    inputs = (inputs >= 0.5).long()  # convert probabilities into labels

    iou_per_class = IOU(inputs, targets)
    mean_iou = torch.mean(iou_per_class)
    logger.info(f"IOU per class: {iou_per_class.flatten().tolist()}")
    logger.info(f"IOU Mean: {mean_iou}")

