import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

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

    Automatically adjusts the window size if the image is small and
    sets the channel_axis for multichannel images.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: SSIM value.
    """
    # Determine an appropriate win_size.
    # Default is 7, but if the image is smaller, pick the largest odd integer <= min(height, width)
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = 7 if min_dim >= 7 else (min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(win_size, 3)  # Ensure win_size is at least 3

    if img1.ndim == 3:
        # Assume the last axis is the channel axis.
        ssim_val, _ = ssim(
            img1,
            img2,
            full=True,
            data_range=img2.max() - img2.min(),
            win_size=win_size,
            channel_axis=-1
        )
    else:
        ssim_val, _ = ssim(
            img1,
            img2,
            full=True,
            data_range=img2.max() - img2.min(),
            win_size=win_size
        )
    return ssim_val


if __name__ == '__main__':
    # Example usage using input images
    img1_path = "/Users/noohadnan/Desktop/APS360/APS360-SISR/Model/outputs/epoch_39_data/original/origepoch39_image0.jpg"  
    pil_image1 = Image.open(img1_path).convert('L')  
    img1np = np.array(pil_image1)

    img1 = img1np.astype(np.float32) / 255.0
    
    img2_path = "output_images/upscaled_image.jpg"
    pil_image2 = Image.open(img2_path).convert('L')
    img2np = np.array(pil_image2)    

    img2 = img2np.astype(np.float32) / 255.0

    psnr_value = calculate_psnr(img1, img2)
    ssim_value = calculate_ssim(img2, img1)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")