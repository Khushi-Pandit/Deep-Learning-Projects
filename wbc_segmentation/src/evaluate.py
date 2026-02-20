import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segmentation import (
    load_image, preprocess_image, features_to_label_image,
    kmeans_segmentation, fcm_segmentation,
    identify_wbc_regions, compute_boundary_accuracy,
    create_synthetic_wbc
)


def compute_extended_metrics(km_regions, fcm_regions, img):
    results = {}

    for region in ("nucleus", "cytoplasm"):
        km_mask  = km_regions[region]
        fcm_mask = fcm_regions[region]

        km_pts  = np.column_stack(np.where(km_mask  > 0)) if km_mask.any()  else np.array([[0, 0]])
        fcm_pts = np.column_stack(np.where(fcm_mask > 0)) if fcm_mask.any() else np.array([[0, 0]])

        def hausdorff(A, B):
            from scipy.spatial.distance import cdist
            D = cdist(A, B)
            return max(D.min(axis=1).max(), D.min(axis=0).max())

        try:
            h_dist = hausdorff(km_pts, fcm_pts)
        except Exception:
            h_dist = -1.0

        intersection = np.logical_and(km_mask, fcm_mask).sum()
        union        = np.logical_or(km_mask,  fcm_mask).sum()
        iou  = float(intersection / (union + 1e-6))
        dice = float(2 * intersection / (km_mask.sum() + fcm_mask.sum() + 1e-6))

        km_intensity  = float(img[km_mask.astype(bool)].mean())  if km_mask.any()  else 0.0
        fcm_intensity = float(img[fcm_mask.astype(bool)].mean()) if fcm_mask.any() else 0.0

        results[region] = {
            "hausdorff_distance":        round(h_dist, 3),
            "iou_kmeans_vs_fcm":         round(iou, 4),
            "dice_kmeans_vs_fcm":        round(dice, 4),
            "km_mean_intensity":         round(km_intensity, 2),
            "fcm_mean_intensity":        round(fcm_intensity, 2),
            "km_area_pixels":            int(km_mask.sum()),
            "fcm_area_pixels":           int(fcm_mask.sum()),
        }

    return results


def save_metrics_json(comparison, extended, output_path="results/metrics.json"):
    report = {
        "boundary_accuracy": comparison,
        "extended_metrics":  extended,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved metrics JSON → {output_path}")


def run_evaluation(image_path=None, output_dir="results"):
    print("\n=== Extended Evaluation ===")

    if image_path and os.path.exists(image_path):
        from segmentation import load_image
        img = load_image(image_path)
    else:
        img = create_synthetic_wbc(256)

    _, features = preprocess_image(img)

    km_labels_flat, _  = kmeans_segmentation(features, 3)
    fcm_labels_flat, _ = fcm_segmentation(features, 3)

    km_label_img  = features_to_label_image(km_labels_flat,  img.shape)
    fcm_label_img = features_to_label_image(fcm_labels_flat, img.shape)

    km_regions  = identify_wbc_regions(km_label_img,  img)
    fcm_regions = identify_wbc_regions(fcm_label_img, img)

    # Boundary accuracy
    comparison = {}
    for region in ("nucleus", "cytoplasm"):
        comparison[region] = {
            "kmeans": compute_boundary_accuracy(km_regions[region]),
            "fcm":    compute_boundary_accuracy(fcm_regions[region]),
        }

    extended = compute_extended_metrics(km_regions, fcm_regions, img)

    os.makedirs(output_dir, exist_ok=True)
    save_metrics_json(comparison, extended,
                      output_path=os.path.join(output_dir, "metrics.json"))

    print("\n--- Extended Metrics ---")
    for region, vals in extended.items():
        print(f"\n[{region.upper()}]")
        for k, v in vals.items():
            print(f"  {k}: {v}")

    return comparison, extended


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(image_path, output_dir="results")