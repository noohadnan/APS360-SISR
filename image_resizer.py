import os
from PIL import Image

input_dir =  "action_camera/processed_images"
output_dir = "action_camera/processed_images_resized"

os.makedirs(output_dir, exist_ok=True)

target_size = (640, 360) # 16:9 aspect ratio

for file_name in os.listdir(input_dir):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_img.save(output_path, "JPEG")
                
            print(f"Resized {file_name} to {target_size}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print("All images have been resized and saved to the 'processed_images_resized' directory.")
