import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs import config

logger = logging.getLogger(__name__)


def get_padding_height_and_width() -> tuple[int, int]:
    """
    Return the number of pixels by which the picture should be padded in height and width.

    :return: (count_for_height, count_for_width)
    """
    height = config.data.IMAGE_HEIGHT
    width = config.data.IMAGE_WIDTH
    h_mod = height % config.model.PATCH_STRIDE
    w_mod = width % config.model.PATCH_STRIDE
    p_h: int = 0
    p_w: int = 0
    if h_mod >= 0:
        p_h = ((height - h_mod) + config.model.PATCH_SIZE) - height
    if w_mod >= 0:
        p_w = ((width - w_mod) + config.model.PATCH_SIZE) - width
    return p_h, p_w


def pad_image(img: np.ndarray) -> np.ndarray:
    """Padded image from right downside using mirroring of the border."""
    pad_h, pad_w = get_padding_height_and_width()
    # (top, bottom), (left, right), (channel dim must be ignored)
    pad_width = ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w))
    return np.pad(img, pad_width, mode="reflect")


def calculate_patch_positions() -> list[tuple[int, int, int, int]]:
    """Return list with positions of the patches on the original image if format (xmin,ymin,xmax,ymax)."""
    positions: list = []
    for y in range(0, config.data.IMAGE_HEIGHT, config.model.PATCH_STRIDE):
        for x in range(0, config.data.IMAGE_WIDTH, config.model.PATCH_STRIDE):
            positions.append((y, y + config.model.PATCH_SIZE, x, x + config.model.PATCH_SIZE))
    return positions


def get_patch_by_position(img: np.ndarray, pos: tuple[int, int, int, int]) -> np.ndarray:
    """Return patch by the given position."""
    img: np.ndarray = pad_image(img)
    if img.ndim not in (2, 3):
        raise ValueError('Input image must be 2 or 3 dimensions.')
    y0, y1, x0, x1 = pos
    return img[y0:y1, x0:x1, :] if img.ndim == 3 else img[y0:y1, x0:x1]


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Denormalize image normalized by ImageNet std and mean."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    logger.warning("Image has been denormalized.")
    return (image * std) + mean


def join_patches(
        patches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Join image, mask and nn prediction patches.

    :param patches: tuple with patches in order (image, mask, prediction).

    :return: tuple with joined patches in order (image, mask, prediction).
    """
    pad_h, pad_w = get_padding_height_and_width()
    h, w = config.data.IMAGE_HEIGHT+pad_h, config.data.IMAGE_WIDTH+pad_w
    image: torch.Tensor = torch.zeros(3, h, w)
    mask: torch.Tensor = torch.zeros(config.data.CLASSES_COUNT, h, w)
    prediction: torch.Tensor = torch.zeros(config.data.CLASSES_COUNT, h, w)

    idx: int = 0
    for y in range(0, config.data.IMAGE_HEIGHT, config.model.PATCH_STRIDE):
        for x in range(0, config.data.IMAGE_WIDTH, config.model.PATCH_STRIDE):
            image[:, y:y + config.model.PATCH_SIZE, x:x + config.model.PATCH_SIZE] += patches[idx][0]
            mask[:, y:y + config.model.PATCH_SIZE, x:x + config.model.PATCH_SIZE] += patches[idx][1]
            prediction[:, y:y + config.model.PATCH_SIZE, x:x + config.model.PATCH_SIZE] += patches[idx][2]
            idx += 1
    return image, mask, prediction


def visualize_binary_mask(
        name: str,
        image: torch.Tensor,
        mask: torch.Tensor,
        color: tuple = (0, 255, 0)
) -> None:
    """
    Visualize one binary mask (ground truth or nn prediction) on input image and show it.

    :param name: name of the opencv window.
    :param image: input image.
    :param mask: binary mask.
    :param color: color in RGB format.
    """
    denorm_img = denormalize(image) if config.transform.USE_PREPROCESSING else image
    image_rgb = denorm_img.permute(1, 2, 0).numpy()
    image_rgb = image_rgb if config.transform.USE_PREPROCESSING else image_rgb
    mask_np = mask.numpy()
    if len(np.unique(mask_np)) > 1:
        overlay = apply_mask(image_rgb, mask_np, color)
        # show image
        cv2.imshow(name, overlay / 255)
        cv2.waitKey(2555904)
        cv2.destroyAllWindows()

def visualize_prediction(img: np.ndarray, pred: np.ndarray, color: tuple = (0, 255, 0)) -> None:
    """
    Visualize prediction on the image using input color (by default green).
    The Main use case is visualizations in Jupiter Notebook.
    """
    names: dict[int, str] = config.data.CLASSES_NAMES
    masks: dict[str, np.ndarray] = {names[i]: mask for i, mask in enumerate(pred) if len(np.unique(mask)) > 1}
    fig, axes = plt.subplots(len(masks.keys()), 1, figsize=(30, 50))
    for i, (name, mask) in enumerate(masks.items()):
        overlay = apply_mask(img, mask, color)
        axes[i].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
        axes[i].axis('off')
        axes[i].set_title(name)
    plt.tight_layout()
    plt.show()

def apply_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Overlay mask on the image using user defines color (by default green)."""
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask >= 1] = color
    # alpha blending
    alpha = 0.5
    return image + alpha * mask_rgb