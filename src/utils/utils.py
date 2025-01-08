import torch
from configs import config


def predict_morphological_feature(outputs: torch.Tensor) -> torch.Tensor:
    """
    Select only one type of relief and one type of shape.
    Keep it and make all other predictions of relief and shape in outputs mask as zero.
    The best prediction is selected by max count of predicted pixels.

    :param outputs: tensor with predictions.

    :return: tensor with updated predictions where only one relief and one shape type has pixels with 1 value.
    """
    relief_idx = config.data.RELIEF_IDX
    shape_idx = config.data.SHAPE_IDX

    relief_counts = [outputs[:, idx, :, :].sum().item() for idx in relief_idx]
    shape_counts = [outputs[:, idx, :, :].sum().item() for idx in shape_idx]
    relief_max_idx = relief_idx[relief_counts.index(max(relief_counts))]
    shape_max_idx = shape_idx[shape_counts.index(max(shape_counts))]
    for idx in relief_idx + shape_idx:
        if idx != relief_max_idx and idx != shape_max_idx:
            outputs[:, idx, :, :] = 0
    return outputs
