import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data = [
    [[0, 1], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 1], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 1], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[1, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[0, 1], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 1], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[1, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[1, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[1, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[0, 1], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
]


image_path = "mls.png"
image = Image.open(image_path)
x = 220; y = 160
d = 160
box = (x, y, x+d, y+d)
cropped_image = image.crop(box)
cropped_image.save("test3.png")
