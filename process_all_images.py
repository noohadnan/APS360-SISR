import os
import cv2
import numpy as np
import csv
import metrics
from image_processing import process_image
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_image(args):
    """
    Process a single image: save the original as jpg, process it, compute metrics.
    
    args is a tuple containing:
      image_counter, input_path, original_images_dir, processed_images_dir, mode, scale_factor
    """
    image_counter, input_path, original_images_dir, processed_images_dir, mode, scale_factor = args
    new_filename = f"{image_counter}.jpg"
    original_output_path = os.path.join(original_images_dir, new_filename)
    processed_output_path = os.path.join(processed_images_dir, new_filename)
    
    try:
        orig_img = cv2.imread(input_path)
        if orig_img is None:
            raise ValueError(f"Failed to load {input_path}")
        cv2.imwrite(original_output_path, orig_img)
        
        processed_image = process_image(input_path, mode, scale_factor)
        if processed_image:
            processed_image.save(processed_output_path, quality=40)
        else:
            raise ValueError(f"Processing failed for {input_path}")

        original = cv2.imread(original_output_path).astype(np.float32) / 255.0
        processed = cv2.imread(processed_output_path).astype(np.float32) / 255.0

        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        psnr = metrics.calculate_psnr(original, processed)
        
        gray_original = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_original = gray_original.astype(np.float32) / 255.0
        gray_processed = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_processed = gray_processed.astype(np.float32) / 255.0
        ssim_value = float(metrics.calculate_ssim(gray_original, gray_processed))
        
        return (new_filename, psnr, ssim_value)
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return (new_filename, None, None)

def process_all_images_parallel(root_dir, mode="both", scale_factor=4, start_at=1354):
    """
    Processes all images under root_dir in parallel starting from a given image number,
    saving originals, processed images, and metrics.
    """
    original_images_dir = r"action_camera\original_images"
    processed_images_dir = r"action_camera\processed_images"
    metrics_file = r"action_camera\image_metrics.csv"

    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(processed_images_dir, exist_ok=True)
    
    image_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(subdir, file))
    
    image_files.sort()
    
    tasks = []
    image_counter = 1
    for input_path in image_files:
        if image_counter < start_at:
            image_counter += 1
            continue
        tasks.append((image_counter, input_path, original_images_dir, processed_images_dir, mode, scale_factor))
        image_counter += 1

    print(f"Processing {len(tasks)} images in parallel...")
    
    results = []

    with ProcessPoolExecutor() as executor:
        future_to_image = {executor.submit(process_single_image, task): task[0] for task in tasks}
        for future in as_completed(future_to_image):
            res = future.result()
            results.append(res)
            print(f"Processed image: {res[0]}")
    
    with open(metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename".ljust(10), "PSNR (dB)".ljust(10), "SSIM".ljust(10)])
        writer.writerow(["-" * 10, "-" * 10, "-" * 10])
        for filename, psnr, ssim_value in sorted(results, key=lambda x: int(x[0].split('.')[0])):
            writer.writerow([
                filename.ljust(10),
                (f"{psnr:.2f}" if psnr is not None else "Error").ljust(10),
                (f"{ssim_value:.4f}" if ssim_value is not None else "Error").ljust(10)
            ])
    
    print(f"\nImage processing complete! Metrics saved to {metrics_file}")

if __name__ == "__main__":
    extracted_frames_dir = r"extracted_frames"
    if not os.path.exists(extracted_frames_dir):
        print(f"Error: Directory '{extracted_frames_dir}' not found.")
    else:
        process_all_images_parallel(extracted_frames_dir, mode="both", scale_factor=4, start_at=1354)
