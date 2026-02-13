import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def normalize(image):
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    return (image * 255).astype(np.uint8)