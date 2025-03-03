import os
import random
from PIL import Image

# Set your USB drive as the source and output directory on I:
source_root = r"I:\\"  # Change to your USB drive path
output_dataset = r"I:\dataset"  # New dataset will be created here on the I: drive

split_ratios = {'train': 0.7, 'validation': 0.15, 'test': 0.15}

splits = ['train', 'validation', 'test']
subfolders = ['original_images', 'processed_images']
for split in splits:
    for sub in subfolders:
        os.makedirs(os.path.join(output_dataset, split, sub), exist_ok=True)

def recursive_list_images(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                image_files.append(rel_path)
    return image_files

image_pairs = []

for root, dirs, _ in os.walk(source_root):
    if "original_images" in dirs and "processed_images" in dirs:
        orig_dir = os.path.join(root, "original_images")
        proc_dir = os.path.join(root, "processed_images")

        orig_files = recursive_list_images(orig_dir)
        for rel_path in orig_files:
            orig_file_path = os.path.join(orig_dir, rel_path)
            proc_file_path = os.path.join(proc_dir, rel_path)

            if not os.path.exists(proc_file_path):
                base, ext = os.path.splitext(rel_path)
                alternative_ext = '.jpg' if ext.lower() == '.png' else '.png'
                alternative_rel = base + alternative_ext
                proc_file_path = os.path.join(proc_dir, alternative_rel)
                if not os.path.exists(proc_file_path):
                    continue  # skip if no matching processed file found
            image_pairs.append((orig_file_path, proc_file_path))

print(f"Found {len(image_pairs)} paired images.")

random.shuffle(image_pairs)
total = len(image_pairs)
train_end = int(total * split_ratios['train'])
val_end = train_end + int(total * split_ratios['validation'])

train_pairs = image_pairs[:train_end]
val_pairs = image_pairs[train_end:val_end]
test_pairs = image_pairs[val_end:]

print(f"Split into: {len(train_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test pairs.")

def process_and_save(pair_list, split_name):
    orig_dest_dir = os.path.join(output_dataset, split_name, "original_images")
    proc_dest_dir = os.path.join(output_dataset, split_name, "processed_images")
    count = 1
    for orig_path, proc_path in pair_list:
        try:
            with Image.open(orig_path) as img:
                img = img.convert("RGB")
                dest_orig = os.path.join(orig_dest_dir, f"{count}.jpg")
                img.save(dest_orig, format="JPEG", quality=95)
        except Exception as e:
            print(f"Error processing original {orig_path}: {e}")
            continue

        try:
            with Image.open(proc_path) as img:
                img = img.convert("RGB")
                dest_proc = os.path.join(proc_dest_dir, f"{count}.jpg")
                img.save(dest_proc, format="JPEG", quality=95)
        except Exception as e:
            print(f"Error processing processed {proc_path}: {e}")
            continue

        count += 1

process_and_save(train_pairs, "train")
process_and_save(val_pairs, "validation")
process_and_save(test_pairs, "test")

print("Dataset created successfully in the 'dataset' folder on your I: drive.")