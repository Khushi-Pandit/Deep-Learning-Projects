import cv2
import numpy as np
from skimage.filters import threshold_sauvola

def otsu_segmentation(image):
    _, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh

def sauvola_segmentation(image, window_size=25):
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    binary = image > thresh_sauvola
    return (binary * 255).astype(np.uint8)