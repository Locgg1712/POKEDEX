import cv2
import numpy as np

def low_pass_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def high_pass_filter(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.subtract(image, blur)

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))

    # áp dụng DSP
    img = low_pass_filter(img)

    return img