from __future__ import annotations

import argparse
from pathlib import Path

from deepeyenet.evaluation.visualization import plot_history, plot_prediction_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create summary figures for a run directory")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    figure_dir = run_dir / "figures"
    plot_history(run_dir / "history.json", figure_dir)
    if (run_dir / "best_valid_predictions.json").exists():
        plot_prediction_examples(run_dir / "best_valid_predictions.json", figure_dir)
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
