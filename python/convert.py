import os
import shutil

# Define the source and destination directories
source_dir = r'C:\Users\USER\Desktop\hackathon\hackathon\python\lfw-deepfunneled'
destination_dir = r'C:\Users\USER\Desktop\hackathon\hackathon\python\tfwimages'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Check if the file is an image (you can add more extensions if needed)
        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Construct full file path
            file_path = os.path.join(root, file)
            # Move the file to the destination directory
            shutil.move(file_path, os.path.join(destination_dir, file))

print("All images have been moved to the 'tfwimages' folder.")
