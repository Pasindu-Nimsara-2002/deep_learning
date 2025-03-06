import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Define U-Net model (autoencoder architecture)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pooling to downsample
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pooling to downsample
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pooling to downsample
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),  # Upsample to original size
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Custom Dataset class to load images from folders
class ImagePairDataset(Dataset):
    def __init__(self, modified_folder, original_folder, transform=None):
        self.modified_images = sorted(os.listdir(modified_folder))
        self.original_images = sorted(os.listdir(original_folder))
        self.modified_folder = modified_folder
        self.original_folder = original_folder
        self.transform = transform

    def __len__(self):
        return len(self.modified_images)

    def __getitem__(self, idx):
        modified_image = Image.open(os.path.join(self.modified_folder, self.modified_images[idx])).convert('RGB')
        original_image = Image.open(os.path.join(self.original_folder, self.original_images[idx])).convert('RGB')

        if self.transform:
            modified_image = self.transform(modified_image)
            original_image = self.transform(original_image)

        return modified_image, original_image


# Define image transformations (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize all images to 64x64
    transforms.ToTensor(),  # Convert to tensor
])

# Define file paths
modified_folder = 'dataset/modified_images'
original_folder = 'dataset/original_images'

# Load dataset
train_dataset = ImagePairDataset(modified_folder, original_folder, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = UNet().cuda()  # Move model to GPU if available
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU if available

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'unet_model.pth')
print("Model saved!")

# Load the model for inference
model.load_state_dict(torch.load('unet_model.pth'))
model.eval()
