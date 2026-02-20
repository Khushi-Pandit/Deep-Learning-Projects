import os
import glob
import cv2
import numpy as np
from pathlib import Path


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_images(root_dir: str) -> list[str]:
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(paths))


def load_dataset(data_dir: str = "data", max_images: int | None = None) -> list[np.ndarray]:
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
    return 


if __name__ == "__main__":
    print(download_instructions())
    imgs = load_dataset("data", max_images=5)
    print(f"Loaded {len(imgs)} image(s), first shape: {imgs[0].shape}")