import cv2
import numpy as np
from PIL import Image
from noise import pnoise2
import metrics

def apply_gaussian_blur(image, kernel_size=21):
    """Applies Gaussian blur to the entire image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def generate_perlin_noise(image_shape, scale=10):
    """Generates a Perlin noise mask for non-uniform blurring."""
    height, width = image_shape[:2]
    noise_map = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            noise_map[i, j] = pnoise2(i / scale, j / scale, octaves=6)

    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    return noise_map

def apply_perlin_blur(image, perlin_scale=5, max_kernel_size=21):
    """Applies Perlin noise-based variable blurring in patches instead of per pixel."""
    perlin_mask = generate_perlin_noise(image.shape, scale=perlin_scale)

    blurred_image = np.copy(image)

    patch_size = 10

    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            kernel_size = int(1 + perlin_mask[i, j] * (max_kernel_size - 1))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            blurred_image[i:i+patch_size, j:j+patch_size] = cv2.GaussianBlur(
                image[i:i+patch_size, j:j+patch_size], (kernel_size, kernel_size), 0
            )

    return blurred_image

def downscale_then_upscale(image, scale_factor=4):
    """Simulates resolution loss by downscaling and upscaling the image."""
    height, width = image.shape[:2]
    new_size = (width // scale_factor, height // scale_factor)

    # Downscale
    low_res = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Upscale back
    restored = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_CUBIC)
    
    return restored

def process_image(image_path, mode="none", scale_factor=32):
    """
    Process the image with the specified mode:
    - "none": No blur, just downscale-upscale
    - "gaussian": Gaussian blur + downscale-upscale
    - "perlin": Perlin noise blur + downscale-upscale
    - "both": Gaussian + Perlin blur + downscale-upscale
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mode == "gaussian":
        image = apply_gaussian_blur(image)
    elif mode == "perlin":
        image = apply_perlin_blur(image)
    elif mode == "both":
        image = apply_gaussian_blur(image)
        image = apply_perlin_blur(image)
    elif mode == "no":
        return Image.fromarray(image) # No blur
    image = downscale_then_upscale(image, scale_factor)

    return Image.fromarray(image)