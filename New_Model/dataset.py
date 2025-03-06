# dataset.py
import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, modified_dir, original_dir, transform=None, target_size=(128, 128)):
        self.modified_dir = modified_dir
        self.original_dir = original_dir
        self.transform = transform
        self.target_size = target_size
        self.modified_images = sorted(os.listdir(modified_dir))
        self.original_images = sorted(os.listdir(original_dir))

    def __len__(self):
        return len(self.modified_images)

    def __getitem__(self, idx):
        # Load modified image (input to the model)
        modified_img_path = os.path.join(self.modified_dir, self.modified_images[idx])
        modified_img = cv2.imread(modified_img_path, cv2.IMREAD_GRAYSCALE)
        modified_img = Image.fromarray(modified_img)

        # Load the corresponding original image (target output)
        original_img_path = os.path.join(self.original_dir, self.original_images[idx])
        original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
        original_img = Image.fromarray(original_img)

        # Resize the target image to match the model output size
        original_img = original_img.resize(self.target_size, Image.Resampling.LANCZOS)

        if self.transform:
            modified_img = self.transform(modified_img)
            original_img = self.transform(original_img)

        return modified_img, original_img
