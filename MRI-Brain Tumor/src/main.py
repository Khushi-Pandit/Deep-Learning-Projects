import os
import numpy as np
import cv2

from preprocess import load_image, normalize
from segmentation import otsu_segmentation, sauvola_segmentation
from metrics import dice_score, jaccard_index

DATA_DIR = "../data/lgg-mri-segmentation"

def binarize_mask(mask):
    return (mask > 0).astype(np.uint8)

def process_dataset():
    otsu_dice_scores = []
    sauvola_dice_scores = []

    otsu_jaccard_scores = []
    sauvola_jaccard_scores = []

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:

            if file.endswith(".tif") and not file.endswith("_mask.tif"):

                img_path = os.path.join(root, file)
                mask_path = os.path.join(root, file.replace(".tif", "_mask.tif"))

                if not os.path.exists(mask_path):
                    continue

                image = load_image(img_path)
                image = normalize(image)

                mask = load_image(mask_path)
                mask = binarize_mask(mask)

                # Otsu
                otsu_result = otsu_segmentation(image)
                otsu_bin = binarize_mask(otsu_result)

                # Sauvola
                sauvola_result = sauvola_segmentation(image)
                sauvola_bin = binarize_mask(sauvola_result)

                # Metrics
                otsu_dice_scores.append(dice_score(mask, otsu_bin))
                sauvola_dice_scores.append(dice_score(mask, sauvola_bin))

                otsu_jaccard_scores.append(jaccard_index(mask, otsu_bin))
                sauvola_jaccard_scores.append(jaccard_index(mask, sauvola_bin))

    print("\n===== FINAL RESULTS =====")
    print(f"Otsu Dice: {np.mean(otsu_dice_scores):.4f}")
    print(f"Sauvola Dice: {np.mean(sauvola_dice_scores):.4f}")
    print(f"Otsu Jaccard: {np.mean(otsu_jaccard_scores):.4f}")
    print(f"Sauvola Jaccard: {np.mean(sauvola_jaccard_scores):.4f}")

if __name__ == "__main__":
    process_dataset()