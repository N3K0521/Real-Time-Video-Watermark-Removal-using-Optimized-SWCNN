from PIL import Image
import os

# Define your directory path containing the logo images
image_dir = source_directory = 'D:\Thesis\Thesis_Watermark\dataset\watermark'

# List all image files in the directory
images = [img for img in os.listdir(image_dir) if img.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

# Rename and resize images
for index, image_name in enumerate(images):
    image_path = os.path.join(image_dir, image_name)
    img = Image.open(image_path)

    # Calculate the resizing ratio to fit within 100x100 while maintaining aspect ratio
    img.thumbnail((100, 100), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS

    # New name for the image
    new_name = f'logo{index + 1}.png'
    new_path = os.path.join(image_dir, new_name)

    # Save the resized image
    img.save(new_path)

    # Optionally, delete the original image if needed
    # os.remove(image_path)

print("Renaming and resizing completed.")
