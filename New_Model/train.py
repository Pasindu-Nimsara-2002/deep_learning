import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

# Define dataset class
class ImageDataset(Dataset):
    def __init__(self, original_dir, modified_dir, transform=None):
        self.original_dir = original_dir
        self.modified_dir = modified_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(original_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        original_path = os.path.join(self.original_dir, img_name)
        modified_path = os.path.join(self.modified_dir, img_name)

        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        modified_img = cv2.imread(modified_path, cv2.IMREAD_GRAYSCALE)

        original_img = Image.fromarray(original_img)
        modified_img = Image.fromarray(modified_img)

        if self.transform:
            original_img = self.transform(original_img)
            modified_img = self.transform(modified_img)

        return modified_img, original_img  # (input, target)


# Define transformation with resizing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.ToTensor()
])

# Load dataset
dataset = ImageDataset("dataset/original_images", "dataset/modified_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 10
for epoch in range(num_epochs):
    for modified_img, original_img in dataloader:
        modified_img, original_img = modified_img.to(device), original_img.to(device)

        optimizer.zero_grad()
        outputs = model(modified_img)
        loss = criterion(outputs, original_img)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "autoencoder.pth")
print("Training complete! Model saved.")
