from datasets import load_dataset
from PIL import Image
import os
import numpy as np

# Load the dataset
dataset = load_dataset("samp3209/logo-dataset")

# Directory to save the transparent logos
save_dir = "/dataset/datasets/watermark"
os.makedirs(save_dir, exist_ok=True)


# Function to convert image to transparent PNG
def save_transparent_image(image, save_path):
    # Convert the image to RGBA format with transparency
    image = image.convert("RGBA")

    # Make white pixels transparent
    data = np.array(image)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    white_mask = (r > 200) & (g > 200) & (b > 200)
    data[..., 3][white_mask] = 0  # Set alpha channel to 0 for white pixels

    # Save the image as PNG with transparency
    transparent_image = Image.fromarray(data)
    transparent_image.save(save_path, format="PNG")


# Process each logo in the dataset and save it
for i, item in enumerate(dataset['train']):
    image = item['image']  # Directly use the image object
    save_path = os.path.join(save_dir, f"logo_{i}.png")
    save_transparent_image(image, save_path)

print("Logos have been saved with transparent backgrounds.")
