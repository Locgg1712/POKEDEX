import cv2
import numpy as np
from PIL import Image

def preprocess_image(pil_img):
    """
    DSP preprocessing:
    - Giữ màu (RGB)
    - Lọc Gaussian để giảm nhiễu
    """

    # convert PIL → numpy
    img = np.array(pil_img)

    # đảm bảo ảnh là RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Gaussian Blur (DSP)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # convert lại về PIL
    return Image.fromarray(blur)