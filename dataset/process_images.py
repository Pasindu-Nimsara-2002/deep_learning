import os
import cv2 as cv
import numpy as np

# Define paths
base_dir = "dataset"
original_dir = os.path.join(base_dir, "original_images")
modified_dir = os.path.join(base_dir, "modified_images")

# Create modified directory if it doesn't exist
os.makedirs(modified_dir, exist_ok=True)

# Process images
for filename in os.listdir(original_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(original_dir, filename)
        img_orig = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        if img_orig is None:
            print(f"Skipping {filename}, cannot read image.")
            continue

        # Apply transformation
        h, w = img_orig.shape
        m = np.full((h, w), 30, dtype=np.uint8)
        new_image = cv.subtract(img_orig, m)
        new_image = cv.GaussianBlur(new_image, (9, 9), 0)

        # Save modified image
        modified_path = os.path.join(modified_dir, filename)
        cv.imwrite(modified_path, new_image)
        print(f"Processed and saved: {filename}")

print("All images processed successfully!")
