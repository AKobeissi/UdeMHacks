import os
import cv2
import numpy as np

def process_images(dataset_folder, target_size):
    image_set = []
    labels = []
    
    # For each class/subfolder in the dataset
    for class_folder in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_folder)
        
        # Skip if not a directory or if it's a hidden file
        if not os.path.isdir(class_path) or class_folder.startswith('.'):
            continue
        
        label_name = class_folder.split('_')[0]

        # Process each image in the class folder
        for img_file in os.listdir(class_path):
            # Skip hidden files
            if img_file.startswith('.'):
                continue
                
            img_path = os.path.join(class_path, img_file)
            
            # Make sure it's a file and has an image extension
            if not os.path.isfile(img_path):
                continue
                
            # Common image extensions
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')):
                continue
            
            # Load the image properly with OpenCV
            img = cv2.imread(img_path)
            
            # Skip if image couldn't be loaded
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
                
            # Resize the image to target size
            try:
                processed_image = cv2.resize(img, target_size)
                image_set.append(processed_image)
                labels.append(label_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays if we have images
    if image_set:
        image_set = np.array(image_set)
        labels = np.array(labels)
        
    return image_set, labels