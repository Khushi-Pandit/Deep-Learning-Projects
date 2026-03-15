import cv2

def preprocess(image):
    blur = cv2.GaussianBlur(image, (5,5), 0)
    return blur