from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepeyenet.config import ExperimentConfig
from deepeyenet.data.dataset import DeepEyeNetDataset, build_keyword_index, collate_batch, load_split
from deepeyenet.data.vocab import Vocabulary
from deepeyenet.evaluation.visualization import plot_dataset_statistics, plot_history, plot_keyword_graph
from deepeyenet.models.multimodal import DeepEyeReasoner
from deepeyenet.training.graph import build_keyword_adjacency
from deepeyenet.training.trainer import Trainer
from deepeyenet.utils.io import ensure_dir, save_json
from deepeyenet.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepEyeReasoner on DeepEyeNet")
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-dir", default="outputs/run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-clinical-tokens", type=int, default=96)
    parser.add_argument("--max-report-tokens", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--graph-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-keyword-freq", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        metadata_dir=args.metadata_dir,
        images_root=args.images_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        max_clinical_tokens=args.max_clinical_tokens,
        max_report_tokens=args.max_report_tokens,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        graph_steps=args.graph_steps,
        num_workers=args.num_workers,
        device=args.device,
        min_keyword_freq=args.min_keyword_freq,
    )
    output_dir = ensure_dir(config.output_dir)
    set_seed(config.random_seed)
    train_records = load_split(config.metadata_dir, config.train_json)
    valid_records = load_split(config.metadata_dir, config.valid_json)
    test_records = load_split(config.metadata_dir, config.test_json)
    plot_dataset_statistics({"train": train_records, "valid": valid_records, "test": test_records}, output_dir / "figures")

    clinical_vocab = Vocabulary.build((record.clinical_description for record in train_records), min_freq=1)
    report_vocab = Vocabulary.build((record.report_text for record in train_records), min_freq=1)
    keyword_to_idx = build_keyword_index(train_records, config.min_keyword_freq)
    keyword_names = [keyword for keyword, _ in sorted(keyword_to_idx.items(), key=lambda item: item[1])]
    adjacency = build_keyword_adjacency(train_records, keyword_to_idx)
    plot_keyword_graph(adjacency, keyword_names, output_dir / "figures" / "keyword_graph.png")

    train_dataset = DeepEyeNetDataset(train_records, config.images_root, clinical_vocab, report_vocab, keyword_to_idx, config.image_size, config.max_clinical_tokens, config.max_report_tokens, training=True)
    valid_dataset = DeepEyeNetDataset(valid_records, config.images_root, clinical_vocab, report_vocab, keyword_to_idx, config.image_size, config.max_clinical_tokens, config.max_report_tokens, training=False)
    collate = partial(collate_batch, report_pad_id=report_vocab.pad_id, clinical_pad_id=clinical_vocab.pad_id)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    trainer = Trainer(model, optimizer, config, report_vocab.pad_id, adjacency, report_vocab, keyword_names, torch.device(config.device))
    fit_result = trainer.fit(train_loader, valid_loader, output_dir)

    save_json(clinical_vocab.to_dict(), output_dir / "clinical_vocab.json")
    save_json(report_vocab.to_dict(), output_dir / "report_vocab.json")
    save_json({"keyword_to_idx": keyword_to_idx}, output_dir / "keyword_index.json")
    config.dump(output_dir / "config.json")
    plot_history(output_dir / "history.json", output_dir / "figures")
    print(f"Training complete. Best checkpoint: {fit_result['best_checkpoint']}")


if __name__ == "__main__":
    main()
