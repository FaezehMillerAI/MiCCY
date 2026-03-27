from __future__ import annotations

import argparse
from pathlib import Path

from deepeyenet.data.dataset import load_split
from deepeyenet.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DeepEyeNet metadata paths and summarize splits")
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--images-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {}
    for split, filename in [("train", "DeepEyeNet_train.json"), ("valid", "DeepEyeNet_valid.json"), ("test", "DeepEyeNet_test.json")]:
        records = load_split(args.metadata_dir, filename)
        missing = [record.image_path for record in records if not (Path(args.images_root) / record.image_path).exists()]
        summary[split] = {"num_samples": len(records), "missing_images": missing[:20], "missing_count": len(missing)}
    output_path = Path(args.metadata_dir) / "dataset_summary.json"
    save_json(summary, output_path)
    print(summary)


if __name__ == "__main__":
    main()
