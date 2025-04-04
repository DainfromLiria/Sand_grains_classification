from typing import List, Tuple

import numpy as np

from configs import config


def get_padding_height_and_width() -> Tuple[int, int]:
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
    pad_h, pad_w = get_padding_height_and_width()
    # (top, bottom), (left, right), (channel dim must be ignored)
    pad_width = ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w))
    return np.pad(img, pad_width, mode="reflect")


def calculate_patches_count() -> int:
    height = config.data.IMAGE_HEIGHT
    width = config.data.IMAGE_WIDTH
    pad_h, pad_w = get_padding_height_and_width()
    print(f"Padding height: {pad_h} Padding width: {pad_w}")
    print(f"Height: {(height + pad_h)}, Width: {(width + pad_w)}")
    h = ((height + pad_h) // config.model.PATCH_STRIDE)
    w = ((width + pad_w) // config.model.PATCH_STRIDE)
    return h * w


def calculate_patch_positions() -> List:
    positions: list = []
    for y in range(0, config.data.IMAGE_HEIGHT, config.model.PATCH_STRIDE):
        for x in range(0, config.data.IMAGE_WIDTH, config.model.PATCH_STRIDE):
            positions.append((y, y + config.model.PATCH_SIZE, x, x + config.model.PATCH_SIZE))
    return positions


def get_patch_by_position(img: np.ndarray, pos: Tuple) -> np.ndarray:
    img: np.ndarray = pad_image(img)
    if img.ndim not in (2, 3):
        raise ValueError('Input image must be 2 or 3 dimensions.')
    y0, y1, x0, x1 = pos
    return img[y0:y1, x0:x1, :] if img.ndim == 3 else img[y0:y1, x0:x1]


def join_patches():
    pass
