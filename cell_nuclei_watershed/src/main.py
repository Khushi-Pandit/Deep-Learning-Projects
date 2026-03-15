import os
import matplotlib.pyplot as plt

from load_data import load_image
from preprocess import preprocess
from watershed_no_marker import watershed_without_markers
from watershed_marker import watershed_with_markers

DATA_DIR = "data/stage1_train"

sample_id = os.listdir(DATA_DIR)[0]
sample_path = os.path.join(DATA_DIR, sample_id)

image = load_image(sample_path)
image = preprocess(image)

wm = watershed_with_markers(image)
wnm = watershed_without_markers(image)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1,3,2)
plt.title("Without Markers")
plt.imshow(wnm)

plt.subplot(1,3,3)
plt.title("With Markers")
plt.imshow(wm)

plt.tight_layout()
plt.show()