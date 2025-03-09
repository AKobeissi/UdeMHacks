import os
import cv2
import numpy as np

def preprocess_image(image, target_size):
    original_height, original_width, _ = image.shape
    target_width, target_height = target_size

    # Determine scaling factors
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)

    # Only use resize if the image is larger than target resolution
    if scale < 1:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image.copy()

    # If one dimension is still smaller, pad minimally to reach target size
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    final_height, final_width, _ = resized_image.shape

    if final_width < target_width:
        pad_left = (target_width - final_width) // 2
        pad_right = target_width - final_width - pad_left
    if final_height < target_height:
        pad_top = (target_height - final_height) // 2
        pad_bottom = target_height - final_height - pad_top

    # Apply padding only if necessary by adding black pixels
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        processed_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        processed_image = resized_image

    return processed_image

def load_images_from_folder(dir_path, label):
    images, labels = [], []
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels

# Process all images
def process_images(dataset_folder, target_size):
    image_set, label_set = [], []
    for dir in os.listdir(dataset_folder):
        dir_path = os.path.join(dataset_folder, dir)
        label = dir.split("_")[0]
        images, labels = load_images_from_folder(dir_path, label)
        image_set.append(images)
        label_set.append(labels)
    
    processed_images = []
    for img in image_set:
        processed_image = preprocess_image(img, target_size)
        processed_images.append(processed_image / 255)

    # Convert processed images and label set array to a numpy array
    processed_images = np.array(processed_images)
    image_labels = np.array(label_set)

    # Reshape image data set to be a 2D array instead of a 4D array
    # From (num_samples, height, width, channels) to (num_samples, height*width*channels)
    num_images = processed_images.shape[0]
    processed_images = processed_images.reshape(num_images, -1)
    
    return processed_images, image_labels

if __name__ == "__main__":
    input_folder = "/UdeMHacks/Parasite Data Set"
    target_size = (256, 256)

    process_images(input_folder, target_size)
