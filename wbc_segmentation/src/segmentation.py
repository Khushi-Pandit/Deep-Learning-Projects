"""
White Blood Cell Segmentation using K-Means and Fuzzy C-Means Clustering
Segments WBC nucleus and cytoplasm regions and compares boundary accuracy.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

try:
    import skfuzzy as fuzz
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False



class FuzzyCMeans:
    """Minimal Fuzzy C-Means implementation."""

    def __init__(self, n_clusters=3, m=2.0, max_iter=150, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        c = self.n_clusters
        m = self.m

        u = rng.dirichlet(np.ones(c), size=n).T   # (c, n)

        for _ in range(self.max_iter):
            um = u ** m
            centers = (um @ X) / um.sum(axis=1, keepdims=True)  # (c, features)
            dist = np.linalg.norm(
                X[np.newaxis, :, :] - centers[:, np.newaxis, :], axis=2
            )  # (c, n)
            dist = np.fmax(dist, 1e-10)
            exp = 2.0 / (m - 1)
            inv = (1.0 / dist) ** exp
            u_new = inv / inv.sum(axis=0, keepdims=True)
            if np.linalg.norm(u_new - u) < self.tol:
                u = u_new
                break
            u = u_new

        self.u_ = u
        self.centers_ = centers
        self.labels_ = np.argmax(u, axis=0)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    r = blurred[:, :, 0].flatten().astype(np.float32) / 255.0
    g = blurred[:, :, 1].flatten().astype(np.float32) / 255.0
    b = blurred[:, :, 2].flatten().astype(np.float32) / 255.0
    h = hsv[:, :, 0].flatten().astype(np.float32) / 179.0
    s = hsv[:, :, 1].flatten().astype(np.float32) / 255.0

    features = np.stack([r, g, b, h, s], axis=1)
    return blurred, features


def features_to_label_image(labels, shape):
    return labels.reshape(shape[:2])


def kmeans_segmentation(features, n_clusters=3, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(features)
    return labels, km


def fcm_segmentation(features, n_clusters=3, m=2.0):
    if SKFUZZY_AVAILABLE:
        data = features.T
        cntr, u, *_ = fuzz.cluster.cmeans(data, n_clusters, m, error=1e-4, maxiter=150, init=None)
        labels = np.argmax(u, axis=0)
        return labels, (cntr, u)
    else:
        fcm = FuzzyCMeans(n_clusters=n_clusters, m=m)
        labels = fcm.fit_predict(features)
        return labels, fcm


def identify_wbc_regions(label_image, img):
    n_clusters = len(np.unique(label_image))
    cluster_means = {}
    for c in range(n_clusters):
        mask = label_image == c
        cluster_means[c] = img[mask].mean(axis=0) if mask.any() else np.zeros(3)

    nucleus_label = min(cluster_means, key=lambda c: cluster_means[c][1])
    brightness = {c: cluster_means[c].mean() for c in cluster_means}
    background_label = max(brightness, key=brightness.get)
    remaining = [c for c in range(n_clusters) if c != nucleus_label and c != background_label]
    cytoplasm_label = remaining[0] if remaining else (1 if nucleus_label != 1 else 0)

    return {
        "nucleus":    (label_image == nucleus_label).astype(np.uint8),
        "cytoplasm":  (label_image == cytoplasm_label).astype(np.uint8),
        "background": (label_image == background_label).astype(np.uint8),
    }


def compute_boundary_accuracy(mask, gt_mask=None):
    metrics = {}
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        peri = cv2.arcLength(largest, True)
        metrics["circularity"] = (4 * np.pi * area / (peri ** 2 + 1e-6))
        metrics["contour_smoothness"] = peri ** 2 / (area + 1e-6)
        metrics["boundary_pixels"] = int(peri)
        metrics["area_pixels"] = int(area)
    else:
        metrics.update(circularity=0.0, contour_smoothness=999.0, boundary_pixels=0, area_pixels=0)

    if gt_mask is not None:
        intersection = np.logical_and(mask, gt_mask).sum()
        union = np.logical_or(mask, gt_mask).sum()
        metrics["iou"] = float(intersection / (union + 1e-6))
        metrics["dice"] = float(2 * intersection / (mask.sum() + gt_mask.sum() + 1e-6))
    return metrics


def compare_methods(km_regions, fcm_regions, gt_masks=None):
    comparison = {}
    for region in ("nucleus", "cytoplasm"):
        km_m = km_regions.get(region, np.zeros((1, 1), dtype=np.uint8))
        fc_m = fcm_regions.get(region, np.zeros((1, 1), dtype=np.uint8))
        gt_m = gt_masks.get(region) if gt_masks else None
        comparison[region] = {
            "kmeans": compute_boundary_accuracy(km_m, gt_m),
            "fcm":    compute_boundary_accuracy(fc_m, gt_m),
        }
    return comparison


_PALETTE = {"nucleus": [255, 50, 50], "cytoplasm": [50, 200, 50], "background": [30, 30, 200]}


def create_colored_segmentation(label_image, regions):
    vis = np.zeros((*label_image.shape, 3), dtype=np.uint8)
    for name, mask in regions.items():
        color = _PALETTE.get(name, [128, 128, 128])
        vis[mask.astype(bool)] = color
    return vis


def visualize_results(original, km_regions, fcm_regions, km_labels, fcm_labels, comparison, output_path="results/comparison.png"):
    km_vis  = create_colored_segmentation(km_labels, km_regions)
    fcm_vis = create_colored_segmentation(fcm_labels, fcm_regions)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("WBC Segmentation: K-Means vs Fuzzy C-Means", fontsize=15, fontweight="bold")

    axes[0, 0].imshow(original);         axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(km_vis);           axes[0, 1].set_title("K-Means Segmentation")
    axes[0, 2].imshow(km_regions["nucleus"], cmap="Reds");   axes[0, 2].set_title("K-Means Nucleus")
    axes[0, 3].imshow(km_regions["cytoplasm"], cmap="Greens");axes[0, 3].set_title("K-Means Cytoplasm")

    axes[1, 0].imshow(original);         axes[1, 0].set_title("Original Image")
    axes[1, 1].imshow(fcm_vis);          axes[1, 1].set_title("FCM Segmentation")
    axes[1, 2].imshow(fcm_regions["nucleus"], cmap="Reds");  axes[1, 2].set_title("FCM Nucleus")
    axes[1, 3].imshow(fcm_regions["cytoplasm"], cmap="Greens");axes[1, 3].set_title("FCM Cytoplasm")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")


def visualize_comparison_metrics(comparison, output_path="results/metrics.png"):
    regions = list(comparison.keys())
    metrics = ["circularity", "contour_smoothness"]

    fig, axes = plt.subplots(len(metrics), len(regions), figsize=(12, 8))
    fig.suptitle("Boundary Accuracy Comparison: K-Means vs FCM", fontsize=13, fontweight="bold")

    for row, metric in enumerate(metrics):
        for col, region in enumerate(regions):
            ax = axes[row, col]
            data = comparison[region]
            km_val  = data["kmeans"].get(metric, 0)
            fcm_val = data["fcm"].get(metric, 0)
            bars = ax.bar(["K-Means", "FCM"], [km_val, fcm_val], color=["#2196F3", "#FF9800"], edgecolor="black")
            ax.set_title(f"{region.capitalize()} – {metric.replace('_', ' ').title()}")
            ax.set_ylabel(metric.replace("_", " ").title())
            for bar, val in zip(bars, [km_val, fcm_val]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(km_val, fcm_val) * 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")


def create_synthetic_wbc(size=256):
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    cx, cy = size // 2, size // 2
    cv2.circle(img, (cx, cy), size // 3, (230, 180, 200), -1)
    cv2.circle(img, (cx - 10, cy + 10), size // 6, (80, 40, 120), -1)
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def run_segmentation_pipeline(image_path=None, n_clusters=3, output_dir="results"):
    print("=" * 60)
    print("  WBC Segmentation Pipeline")
    print("=" * 60)

    if image_path and os.path.exists(image_path):
        print(f"[1] Loading image: {image_path}")
        img = load_image(image_path)
    else:
        print("[1] Generating synthetic WBC image …")
        img = create_synthetic_wbc(256)

    print(f"    Shape: {img.shape}")
    print("[2] Preprocessing …")
    img_proc, features = preprocess_image(img)

    print("[3] Running K-Means …")
    km_labels_flat, km_model = kmeans_segmentation(features, n_clusters)
    km_label_img = features_to_label_image(km_labels_flat, img.shape)

    print("[4] Running Fuzzy C-Means …")
    fcm_labels_flat, fcm_model = fcm_segmentation(features, n_clusters)
    fcm_label_img = features_to_label_image(fcm_labels_flat, img.shape)

    print("[5] Identifying WBC regions …")
    km_regions  = identify_wbc_regions(km_label_img,  img)
    fcm_regions = identify_wbc_regions(fcm_label_img, img)

    print("[6] Computing metrics …")
    comparison = compare_methods(km_regions, fcm_regions)

    print("[7] Visualising …")
    os.makedirs(output_dir, exist_ok=True)
    visualize_results(img, km_regions, fcm_regions, km_label_img, fcm_label_img, comparison,
                      output_path=os.path.join(output_dir, "comparison.png"))
    visualize_comparison_metrics(comparison, output_path=os.path.join(output_dir, "metrics.png"))

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for region, data in comparison.items():
        print(f"\n  [{region.upper()}]")
        for method, mets in data.items():
            print(f"    {method.upper():8s} → circularity={mets.get('circularity',0):.4f}  "
                  f"smoothness={mets.get('contour_smoothness',0):.2f}  "
                  f"boundary_px={mets.get('boundary_pixels',0)}")

    print(f"\n[Done] Results saved to: {output_dir}/")
    return comparison


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_segmentation_pipeline(image_path, n_clusters=3, output_dir="results")