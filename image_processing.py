import cv2
import numpy as np
from PIL import Image
from noise import pnoise2
import metrics

def apply_gaussian_blur(image, kernel_size=7):
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

def apply_perlin_blur(image, perlin_scale=10, max_kernel_size=9):
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

def downscale_then_upscale(image, scale_factor=16):
    """Simulates resolution loss by downscaling and upscaling the image."""
    height, width = image.shape[:2]
    new_size = (width // scale_factor, height // scale_factor)

    # Downscale
    low_res = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Upscale back
    restored = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_CUBIC)
    
    return restored

def process_image(image_path, mode="none", scale_factor=4):
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

    image = downscale_then_upscale(image, scale_factor)

    return Image.fromarray(image)

if __name__ == "__main__":
    input_image = "downtown_toronto.jpg" # change this to the image you want to process
    mode = "both"  # options to run: "none", "gaussian", "perlin", "both", but just run "both" for this project
    output = process_image("data_images/" + input_image, mode)

    if output:
        output.save("output_images/" + input_image[:-4] + "_blur_compress.jpg", quality=10)

    original_image = cv2.imread("data_images/" + input_image)
    processed_image = cv2.imread("output_images/" + input_image[:-4] + "_blur_compress.jpg")

    original_image = original_image.astype(np.float32) / 255.0
    processed_image = processed_image.astype(np.float32) / 255.0

    processed_image = cv2.resize(processed_image, (original_image.shape[1], original_image.shape[0]))

    psnr = metrics.calculate_psnr(original_image, processed_image)
    print(f"PSNR: {psnr:.2f} dB")

    gray_original = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_processed = cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    ssim_value = np.array(metrics.calculate_ssim(gray_original, gray_processed), dtype=np.float32)
    print(f"SSIM: {ssim_value:.4f}")
