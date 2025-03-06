import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Define your transform (same as used during training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjust based on input size during training
    transforms.ToTensor(),
])

# Load the trained model
model = UNet(in_channels=3, out_channels=3).cuda()  # Replace with the correct model if different
model.load_state_dict(torch.load('unet_model.pth', weights_only=True))  # Load the model weights
model.eval()  # Set model to evaluation mode

# Load the input image
sample_input = Image.open('sample_image.jpg').convert('RGB')

# Preprocess the image
sample_input = transform(sample_input).unsqueeze(0).cuda()  # Add batch dimension and move to GPU if available

# Perform inference
with torch.no_grad():  # Disable gradient calculations during inference
    output = model(sample_input)

# Convert the output tensor back to an image (optional)
output_image = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Remove batch dimension and convert to numpy
output_image = (output_image * 255).astype(np.uint8)  # Scale back to 0-255 range

# Convert the NumPy array to a PIL image
output_pil = Image.fromarray(output_image)

# Show the result
output_pil.show()  # Display the image
