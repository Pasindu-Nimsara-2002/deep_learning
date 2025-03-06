import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
from model import Autoencoder

# Define transformations
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Dataset class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, modified_dir, transform=None):
        self.modified_dir = modified_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(modified_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.modified_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, img_name

# Load dataset
modified_dir = "dataset/modified_images"
dataset = ImageDataset(modified_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
model.eval()

# Create output directory
output_dir = "dataset/output_images"
os.makedirs(output_dir, exist_ok=True)

# Run inference
for modified_img, img_name in dataloader:
    modified_img = modified_img.to(device)
    
    with torch.no_grad():
        restored_img = model(modified_img)

    # Convert tensor to numpy array
    restored_img = restored_img.cpu().squeeze(0).squeeze(0).numpy() * 255.0
    restored_img = restored_img.astype(np.uint8)

    # Save output
    output_path = os.path.join(output_dir, img_name[0])
    cv2.imwrite(output_path, restored_img)

    print(f"Saved: {img_name[0]}")

print("Inference complete! All images saved.")
