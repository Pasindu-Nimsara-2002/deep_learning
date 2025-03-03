import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class KAISTThermalDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        """
        root_dir: Path to KAIST dataset (e.g., "KAIST_Dataset/")
        mode: "train", "test", or "val"
        transform: Data transformation pipeline
        """
        self.input_dir = os.path.join(root_dir, mode, "input")
        self.target_dir = os.path.join(root_dir, mode, "target")
        self.transform = transform
        self.image_filenames = sorted(os.listdir(self.input_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.image_filenames[idx])
        target_path = os.path.join(self.target_dir, self.image_filenames[idx])

        # Load images
        input_image = Image.open(input_path).convert("L")  # Grayscale
        target_image = Image.open(target_path).convert("L")

        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image
