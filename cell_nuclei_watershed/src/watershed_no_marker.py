import cv2
import numpy as np

def watershed_without_markers(image):
    _, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.3 * dist_transform.max(), 255, 0
    )

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers
    )
    return markers