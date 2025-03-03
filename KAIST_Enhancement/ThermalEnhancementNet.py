import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalEnhancementNet(nn.Module):
    def __init__(self):
        super(ThermalEnhancementNet, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Output Reconstruction Layer
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Convolution + BatchNorm + ReLU
        x = F.relu(self.bn2(self.conv2(x)))  
        x = F.relu(self.bn3(self.conv3(x)))  
        x = self.conv_out(x)  # Final layer without activation (output remains in raw form)
        return x
