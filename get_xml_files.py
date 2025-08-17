import cv2
import os
import shutil

# Get the path to OpenCV's haarcascades directory
haarcascades_path = cv2.data.haarcascades
print(f'OpenCV haarcascades path: {haarcascades_path}')

# Ensure data directory exists
data_dir = '/home/ubuntu/face_detection_project/data'
os.makedirs(data_dir, exist_ok=True)

# Copy all XML files to the project data directory
for file in os.listdir(haarcascades_path):
    if file.endswith('.xml'):
        print(file)
        src = os.path.join(haarcascades_path, file)
        dst = os.path.join(data_dir, file)
        shutil.copy(src, dst)
        print(f'Copied {file} to project data directory')
