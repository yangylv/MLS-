import numpy as np
import cv2

def get_ratio(image_file):
    image = cv2.imread(image_file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    total_pixels = gray_image.size
    black_pixels = (gray_image < 50).sum()
    balck_ratio = black_pixels/total_pixels
    return balck_ratio

if __name__ == "__main__":
    print("Hello")