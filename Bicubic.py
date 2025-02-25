import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as ssim

from metrics import calculate_psnr, calculate_ssim

def bicubic_interpolate_2d(image: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    """
    Upscale a 2D (grayscale) image using bicubic interpolation with PyTorch.
    
    Args:
        image (torch.Tensor): A 2D tensor of shape (H, W) representing a grayscale image.
        new_height (int): The target height.
        new_width (int): The target width.
    
    Returns:
        torch.Tensor: The upscaled image as a 2D tensor of shape (new_height, new_width).
    """
    # Add batch and channel dimensions: (1, 1, H, W)
    image = image.unsqueeze(0).unsqueeze(0)
    # Use PyTorch's built-in bicubic interpolation
    upscaled = F.interpolate(image, size=(new_height, new_width), mode='bicubic', align_corners=False)
    # Remove the added dimensions
    return upscaled.squeeze(0).squeeze(0)

if __name__ == '__main__':
    # --- Load the high-resolution image ---
    img_path = "SampleData/sample.jpg"  
    pil_image = Image.open(img_path).convert('L')  
    original_np = np.array(pil_image)
    
    # Normalize and convert to PyTorch tensor (values in [0,1])
    original_tensor = torch.from_numpy(original_np).float() / 255.0
    original_height, original_width = original_tensor.shape

    # --- Downscale to simulate a low-resolution image ---
    # For example, reduce resolution by a factor of 2.
    low_res_width = original_width // 2
    low_res_height = original_height // 2
    pil_low_res = pil_image.resize((low_res_width, low_res_height), Image.BICUBIC)
    low_res_np = np.array(pil_low_res).astype(np.float32) / 255.0
    low_res_tensor = torch.from_numpy(low_res_np)

    # --- Upscale the low-resolution image using bicubic interpolation ---
    upscaled_tensor = bicubic_interpolate_2d(low_res_tensor, original_height, original_width)
    upscaled_np = upscaled_tensor.numpy()

    # --- Compute PSNR and SSIM with the original high-resolution image ---
    # Ensure both images are in [0,1]
    original_norm = original_np.astype(np.float32) / 255.0
    psnr_value = calculate_psnr(original_norm, upscaled_np)
    ssim_value = calculate_ssim(original_norm, upscaled_np)

    print("PSNR between original and upscaled image: {:.2f} dB".format(psnr_value))
    print("SSIM between original and upscaled image: {:.4f}".format(ssim_value))

    # --- Display the images ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original High Resolution")
    plt.imshow(original_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Low Resolution Input")
    plt.imshow(low_res_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Upscaled with Bicubic")
    plt.imshow(upscaled_np, cmap='gray')
    plt.axis('off')

    plt.show()