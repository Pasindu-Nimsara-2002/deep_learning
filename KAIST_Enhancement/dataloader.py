import torch
from torch.utils.data import DataLoader
from dataset_loader import KAISTThermalDataset
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the model input
    transforms.ToTensor()
])

# Dataset paths
dataset_root = "KAIST_Dataset"

# Load datasets
train_dataset = KAISTThermalDataset(dataset_root, mode="train", transform=transform)
test_dataset = KAISTThermalDataset(dataset_root, mode="test", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Example: Checking a batch
sample_batch = next(iter(train_loader))
input_images, target_images = sample_batch
print("Batch size:", input_images.shape)  # torch.Size([16, 1, 256, 256])
