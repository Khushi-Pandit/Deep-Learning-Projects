import os
from tqdm import tqdm
from preprocessing import preprocess_image
from thresholding import apply_niblack, apply_sauvola
from evaluation import compute_metrics
from utils import load_ground_truth, save_image

IMAGE_DIR = "../data/images"
MASK_DIR = "../data/masks"

def main():
    niblack_scores = []
    sauvola_scores = []

    for file in tqdm(os.listdir(IMAGE_DIR)):
        img_path = os.path.join(IMAGE_DIR, file)
        mask_path = os.path.join(MASK_DIR, file)

        image = preprocess_image(img_path)
        gt = load_ground_truth(mask_path)

        niblack = apply_niblack(image)
        sauvola = apply_sauvola(image)

        save_image(f"../outputs/niblack/{file}", niblack)
        save_image(f"../outputs/sauvola/{file}", sauvola)

        n_score = compute_metrics(niblack, gt)
        s_score = compute_metrics(sauvola, gt)

        niblack_scores.append(n_score)
        sauvola_scores.append(s_score)

    print("\n📊 RESULTS:")
    print(f"Niblack Sensitivity: {sum(niblack_scores)/len(niblack_scores):.4f}")
    print(f"Sauvola Sensitivity: {sum(sauvola_scores)/len(sauvola_scores):.4f}")

if __name__ == "__main__":
    main()