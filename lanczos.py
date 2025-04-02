import cv2
import numpy as np
from PIL import Image
from noise import pnoise2
import metrics
import matplotlib.pyplot as plt

from image_processing import process_image

def lanczos_kernel(x, a):
    """
    Compute the Lanczos kernel value for a given x and window parameter a.
    Uses np.sinc, which computes sin(pi*x)/(pi*x).
    """
    x = np.array(x, dtype=np.float64)
    abs_x = np.abs(x)
    # Compute sinc(x)*sinc(x/a) for |x| < a, else 0.
    kernel = np.where(abs_x < a, np.sinc(x) * np.sinc(x / a), 0.0)
    return kernel

def lanczos_resize_vectorized(image, scale, a=3):
    """
    Resize an image using Lanczos interpolation with vectorized inner loops.
    
    Parameters:
      image : np.ndarray (grayscale or RGB)
      scale : float - scale factor (>1 for upscaling, <1 for downscaling)
      a : int - Lanczos window parameter.
      
    Returns:
      np.ndarray: The resized image.
    """
    # Ensure the image has a channel dimension
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    in_height, in_width, channels = image.shape
    out_height = int(in_height * scale)
    out_width = int(in_width * scale)
    output = np.zeros((out_height, out_width, channels), dtype=np.float64)
    
    for y_out in range(out_height):
        src_y = y_out / scale
        y_floor = int(np.floor(src_y))
        y_min = y_floor - a + 1
        y_max = y_floor + a
        y_indices = np.arange(y_min, y_max + 1)
        valid_y_mask = (y_indices >= 0) & (y_indices < in_height)
        y_indices_valid = y_indices[valid_y_mask]
        dy = src_y - y_indices_valid
        kernel_y = lanczos_kernel(dy, a)
        
        for x_out in range(out_width):
            src_x = x_out / scale
            x_floor = int(np.floor(src_x))
            x_min = x_floor - a + 1
            x_max = x_floor + a
            x_indices = np.arange(x_min, x_max + 1)
            valid_x_mask = (x_indices >= 0) & (x_indices < in_width)
            x_indices_valid = x_indices[valid_x_mask]
            dx = src_x - x_indices_valid
            kernel_x = lanczos_kernel(dx, a)
            
            # Combined 2D weight matrix
            weights = np.outer(kernel_y, kernel_x)
            sum_weights = np.sum(weights)
            if sum_weights != 0:
                for c in range(channels):
                    patch = image[np.ix_(y_indices_valid, x_indices_valid, [c])].squeeze(axis=2)
                    output[y_out, x_out, c] = np.sum(patch * weights) / sum_weights
            else:
                for c in range(channels):
                    output[y_out, x_out, c] = 0.0
    
    # Remove singleton channel for grayscale images
    if output.shape[2] == 1:
        output = output[:, :, 0]
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def depixelate_image(image, lanczos_scale=2.0, a=3):
    """
    Depixelates an image using the Lanczos filter.
    The process first upscales the image using Lanczos interpolation,
    then downscales it back to the original dimensions with the same method.
    
    Args:
      image : np.ndarray - the pixelated image (RGB or grayscale)
      lanczos_scale : float - the upscale factor used for depixelation.
      a : int - Lanczos window parameter.
      
    Returns:
      np.ndarray: The depixelated image.
    """
    # Upscale the pixelated image
    upscaled = lanczos_resize_vectorized(image, lanczos_scale, a)
    # Downscale back to the original size (scale factor <1)
    depixelated = lanczos_resize_vectorized(upscaled, 1 / lanczos_scale, a)
    return depixelated

##############################################
# Main Pipeline
##############################################

if __name__ == "__main__":
    input_image = "procepoch39_image0.jpg"  # Change to your desired image filename
    mode = "none"  # Options: "none", "gaussian", "perlin", "both"
    
    # --- Pixelation Stage ---
    # Process the image using your pixelation function (blurring + resolution loss)
    pixelated = process_image("data_images/" + input_image, mode, scale_factor=4)
    if pixelated is None:
        exit(1)
    pixelated.save("output_images/" + input_image[:-4] + "_blur_compress.jpg", quality=10)
    
    # Convert pixelated image (PIL) to NumPy array (RGB)
    pixelated_np = np.array(pixelated)
    
    # --- Depixelation Stage ---
    # Depixelate the image using the vectorized Lanczos filter (upscale then downscale)
    depixelated_np = depixelate_image(pixelated_np, lanczos_scale=2.0, a=3)
    depixelated = Image.fromarray(depixelated_np)
    depixelated.save("output_images/" + input_image[:-4] + "_depixelated.jpg")
    
    # --- Load Original Image for Comparison ---
    original_image = cv2.imread("data_images/" + input_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Normalize images to [0, 1] for PSNR/SSIM calculations
    original_norm = original_image.astype(np.float32) / 255.0
    pixelated_norm = pixelated_np.astype(np.float32) / 255.0
    depixelated_norm = depixelated_np.astype(np.float32) / 255.0
    
    # --- Calculate PSNR and SSIM for every step ---
    # Original vs. Pixelated
    psnr_orig_pix = metrics.calculate_psnr(original_norm, pixelated_norm)
    ssim_orig_pix = metrics.calculate_ssim(original_norm, pixelated_norm)
    
    # Pixelated vs. Depixelated
    psnr_pix_depix = metrics.calculate_psnr(pixelated_norm, depixelated_norm)
    ssim_pix_depix = metrics.calculate_ssim(pixelated_norm, depixelated_norm)
    
    # Original vs. Depixelated
    psnr_orig_depix = metrics.calculate_psnr(original_norm, depixelated_norm)
    ssim_orig_depix = metrics.calculate_ssim(original_norm, depixelated_norm)
    
    print("----- PSNR and SSIM Metrics -----")
    print(f"Original vs. Pixelated - PSNR: {psnr_orig_pix:.2f} dB, SSIM: {ssim_orig_pix:.4f}")
    print(f"Pixelated vs. Depixelated - PSNR: {psnr_pix_depix:.2f} dB, SSIM: {ssim_pix_depix:.4f}")
    print(f"Original vs. Depixelated - PSNR: {psnr_orig_depix:.2f} dB, SSIM: {ssim_orig_depix:.4f}")
    
    ##############################################
    # Plotting the Images
    ##############################################
    plt.figure(figsize=(18, 6))
    
    # Plot Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    
    # Plot Pixelated Image
    plt.subplot(1, 3, 2)
    plt.imshow(pixelated_np)
    plt.title("Pixelated (Blur & Compress)")
    plt.axis("off")
    
    # Plot Depixelated Image
    plt.subplot(1, 3, 3)
    plt.imshow(depixelated_np)
    plt.title("Depixelated (Lanczos)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()