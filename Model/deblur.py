import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F

# Define the SISR model exactly as in your training file
class SISR(nn.Module):
    def __init__(self):
        super(SISR, self).__init__()
        self.name = "SISR"
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Patch extraction and representation
        x = F.relu(self.conv2(x))  # Non-linear mapping
        x = self.conv3(x)          # Reconstruction
        return x

# Unnormalize image: inverse of Normalize(mean=0.5, std=0.5)
def unnormalize_image(img):
    return img * 0.5 + 0.5

def load_image(image_path):
    """Load and preprocess image with same transforms as training."""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def save_image(tensor, output_path):
    """Unnormalize, clamp, convert to PIL and save."""
    tensor = tensor.squeeze(0)
    tensor = unnormalize_image(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    img = transforms.ToPILImage()(tensor.cpu())
    img.save(output_path)
    print(f"Deblurred image saved to {output_path}")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate the model and load checkpoint
    model = SISR().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load and process the input image
    input_tensor = load_image(args.input).to(device)

    # Run inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Save the deblurred output image
    save_image(output_tensor, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deblur an image using the trained SISR model")
    parser.add_argument('--checkpoint', type=str,
                        default="/Users/jamesaliev/APS360-SISR/Model/model_savepoint/model_SISR_bs8_lr0.005_epoch39",
                        help="Path to the SISR model checkpoint")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the input .jpg image")
    parser.add_argument('--output', type=str, default="output.jpg",
                        help="Path to save the deblurred image")
    args = parser.parse_args()
    main(args)