import cv2
import os

def load_ground_truth(path):
    gt = cv2.imread(path, 0)
    _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    return gt

def save_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)