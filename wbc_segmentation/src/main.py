import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from segmentation import run_segmentation_pipeline
from data_loader import load_dataset, download_instructions


def parse_args():
    p = argparse.ArgumentParser(description="WBC Segmentation: K-Means vs FCM")
    p.add_argument("--image",  type=str, default=None,
                   help="Path to a single WBC image")
    p.add_argument("--batch",  type=str, default=None,
                   help="Directory of WBC images for batch processing")
    p.add_argument("--clusters", type=int, default=3,
                   help="Number of clusters (default: 3)")
    p.add_argument("--output", type=str, default="results",
                   help="Output directory (default: results/)")
    p.add_argument("--eval",   action="store_true",
                   help="Run extended evaluation after segmentation")
    p.add_argument("--download-info", action="store_true",
                   help="Show Kaggle dataset download instructions")
    return p.parse_args()


def main():
    args = parse_args()

    if args.download_info:
        print(download_instructions())
        return

    if args.batch:
        images = load_dataset(args.batch, max_images=10)
        print(f"Batch mode: processing {len(images)} images …")
        for i, img in enumerate(images):
            out_dir = os.path.join(args.output, f"image_{i:03d}")
            # save temp image and run pipeline
            import cv2, tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            run_segmentation_pipeline(tmp.name, n_clusters=args.clusters, output_dir=out_dir)
            os.unlink(tmp.name)
    else:
        run_segmentation_pipeline(
            image_path=args.image,
            n_clusters=args.clusters,
            output_dir=args.output
        )

    if args.eval:
        from evaluate import run_evaluation
        run_evaluation(image_path=args.image, output_dir=args.output)


if __name__ == "__main__":
    main()