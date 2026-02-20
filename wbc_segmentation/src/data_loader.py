"""
Data loader for Kaggle Blood Cell Images dataset.
Dataset: https://www.kaggle.com/datasets/paultimothymooney/blood-cells

Expected structure after download:
  data/
    dataset-master/
      JPEGImages/
        BloodImage_00001.jpg
        BloodImage_00002.jpg
        ...
      Annotations/         (XML bounding-box annotations, optional)
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_images(root_dir: str) -> list[str]:
    """Recursively find all supported image files under root_dir."""
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(paths))


def load_dataset(data_dir: str = "data", max_images: int | None = None) -> list[np.ndarray]:
    """
    Load images from the Kaggle Blood Cell dataset directory.

    Args:
        data_dir:   Root directory containing the dataset.
        max_images: Optional cap on number of images to load.

    Returns:
        List of RGB numpy arrays.
    """
    images_paths = find_images(data_dir)

    if not images_paths:
        print(f"[DataLoader] No images found in '{data_dir}'. "
              "Using synthetic data for demo.")
        from segmentation import create_synthetic_wbc
        return [create_synthetic_wbc(256)]

    if max_images:
        images_paths = images_paths[:max_images]

    images = []
    for p in images_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    print(f"[DataLoader] Loaded {len(images)} images from '{data_dir}'.")
    return images


def get_single_image(data_dir: str = "data") -> np.ndarray | None:
    """Return the first available image or None."""
    paths = find_images(data_dir)
    if not paths:
        return None
    img = cv2.imread(paths[0])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def download_instructions() -> str:
    return """
========================================================
  HOW TO DOWNLOAD THE KAGGLE BLOOD CELL DATASET
========================================================
1. Install Kaggle CLI:
     pip install kaggle

2. Place your API key at ~/.kaggle/kaggle.json
   (Get it from https://www.kaggle.com/account)

3. Run:
     kaggle datasets download -d paultimothymooney/blood-cells
     unzip blood-cells.zip -d data/

4. Your data/ folder should then contain:
     data/dataset-master/JPEGImages/*.jpg

5. Re-run the pipeline:
     python src/segmentation.py data/dataset-master/JPEGImages/BloodImage_00001.jpg
========================================================
"""


if __name__ == "__main__":
    print(download_instructions())
    imgs = load_dataset("data", max_images=5)
    print(f"Loaded {len(imgs)} image(s), first shape: {imgs[0].shape}")