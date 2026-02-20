import os
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import wraps


def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  [{fn.__name__}] completed in {elapsed:.3f}s")
        return result
    return wrapper


def apply_colormap(label_image: np.ndarray, n_colors: int = 3) -> np.ndarray:
    """Map integer label image to an RGB image using a fixed palette."""
    palette = np.array([
        [255,  50,  50],
        [ 50, 200,  50],
        [ 30,  30, 200],
        [200, 200,  50],
        [200,  50, 200],
    ], dtype=np.uint8)
    h, w = label_image.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n_colors):
        rgb[label_image == i] = palette[i % len(palette)]
    return rgb


def resize_image(img: np.ndarray, max_dim: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def overlay_boundary(img: np.ndarray, mask: np.ndarray,
                      color=(0, 255, 0), thickness=2) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_image_summary(img, km_regions, fcm_regions,
                        title="WBC Segmentation Summary", save_path=None):
    km_nuc  = overlay_boundary(img, km_regions["nucleus"],  color=(255, 50, 50))
    km_cyt  = overlay_boundary(km_nuc, km_regions["cytoplasm"], color=(50, 200, 50))
    fcm_nuc = overlay_boundary(img, fcm_regions["nucleus"], color=(255, 50, 50))
    fcm_cyt = overlay_boundary(fcm_nuc, fcm_regions["cytoplasm"], color=(50, 200, 50))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes[0].imshow(img);    axes[0].set_title("Original")
    axes[1].imshow(km_cyt); axes[1].set_title("K-Means Boundaries")
    axes[2].imshow(fcm_cyt);axes[2].set_title("FCM Boundaries")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("utils.py – utility module, import to use.")