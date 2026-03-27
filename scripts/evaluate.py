from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepeyenet.config import ExperimentConfig
from deepeyenet.data.dataset import DeepEyeNetDataset, collate_batch, load_split
from deepeyenet.data.vocab import Vocabulary
from deepeyenet.evaluation.visualization import plot_metric_radar, plot_prediction_examples
from deepeyenet.models.multimodal import DeepEyeReasoner
from deepeyenet.training.graph import build_keyword_adjacency
from deepeyenet.training.trainer import Trainer
from deepeyenet.utils.io import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepEyeReasoner")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["valid", "test", "train"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = ExperimentConfig(**load_json(run_dir / "config.json"))
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        config.device = "cpu"
    clinical_vocab = Vocabulary.from_dict(load_json(run_dir / "clinical_vocab.json"))
    report_vocab = Vocabulary.from_dict(load_json(run_dir / "report_vocab.json"))
    keyword_to_idx = load_json(run_dir / "keyword_index.json")["keyword_to_idx"]
    keyword_names = [keyword for keyword, _ in sorted(keyword_to_idx.items(), key=lambda item: item[1])]

    train_records = load_split(config.metadata_dir, config.train_json)
    split_filename = {
        "train": config.train_json,
        "valid": config.valid_json,
        "test": config.test_json,
    }[args.split]
    eval_records = load_split(config.metadata_dir, split_filename)
    adjacency = build_keyword_adjacency(train_records, keyword_to_idx)
    dataset = DeepEyeNetDataset(eval_records, config.images_root, clinical_vocab, report_vocab, keyword_to_idx, config.image_size, config.max_clinical_tokens, config.max_report_tokens, training=False)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=partial(collate_batch, report_pad_id=report_vocab.pad_id, clinical_pad_id=clinical_vocab.pad_id))

    model = DeepEyeReasoner(
        clinical_vocab_size=len(clinical_vocab.itos),
        report_vocab_size=len(report_vocab.itos),
        num_keywords=len(keyword_to_idx),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        clinical_pad_id=clinical_vocab.pad_id,
        report_pad_id=report_vocab.pad_id,
        graph_steps=config.graph_steps,
    ).to(torch.device(config.device))
    checkpoint = torch.load(run_dir / "best.pt", map_location=config.device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    trainer = Trainer(model, optimizer, config, report_vocab.pad_id, adjacency, report_vocab, keyword_names, torch.device(config.device))
    result, samples = trainer.evaluate_epoch(loader)
    eval_dir = run_dir / f"eval_{args.split}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(result.metrics, eval_dir / "metrics.json")
    save_json(samples, eval_dir / "predictions.json")
    plot_metric_radar(result.metrics, eval_dir / "metric_radar.png")
    plot_prediction_examples(eval_dir / "predictions.json", eval_dir)
    print(result.metrics)


if __name__ == "__main__":
    main()
