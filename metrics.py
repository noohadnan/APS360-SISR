# metrics.py

import math
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Assumes the images are normalized in the [0, 1] range.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: PSNR value in decibels.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1 / mse)

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    Assumes the images are normalized in the [0, 1] range.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: SSIM value.
    """
    ssim_val, _ = ssim(img1, img2, full=True, data_range=img2.max() - img2.min())
    return ssim_val
