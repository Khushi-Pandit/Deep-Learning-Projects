import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola

def apply_niblack(image, window_size=25, k=0.2):
    thresh = threshold_niblack(image, window_size=window_size, k=k)
    binary = image > thresh
    return binary.astype(np.uint8) * 255

def apply_sauvola(image, window_size=25, k=0.2):
    thresh = threshold_sauvola(image, window_size=window_size, k=k)
    binary = image > thresh
    return binary.astype(np.uint8) * 255