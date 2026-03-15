# src/watershed_marker.py
import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def watershed_with_markers(image):
    # Step 1: Binary threshold
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 2: Distance transform
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Step 3: Find local maxima (coordinates)
    coords = peak_local_max(
        distance,
        min_distance=15,
        labels=binary
    )

    # Step 4: Create marker image SAME SIZE as input
    markers = np.zeros(distance.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # Step 5: Watershed requires 3-channel image
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    labels = cv2.watershed(image_bgr, markers)
    return labels