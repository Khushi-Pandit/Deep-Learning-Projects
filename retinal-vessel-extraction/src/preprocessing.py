import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    green_channel = img[:, :, 1]  # best for vessels

    # Normalize
    norm = cv2.normalize(green_channel, None, 0, 255, cv2.NORM_MINMAX)

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(norm)

    return enhanced