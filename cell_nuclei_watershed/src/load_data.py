import os
import cv2

def load_image(sample_dir):
    image_dir = os.path.join(sample_dir, "images")
    image_name = os.listdir(image_dir)[0]
    image_path = os.path.join(image_dir, image_name)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image