from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from deepeyenet.data.dataset import SampleRecord
from deepeyenet.utils.io import ensure_dir, load_json

sns.set_theme(style="whitegrid")


def plot_dataset_statistics(records_by_split: dict[str, list[SampleRecord]], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    split_rows = []
    keyword_counter = Counter()
    report_lengths = []
    for split_name, records in records_by_split.items():
        split_rows.append({"split": split_name, "samples": len(records)})
        for record in records:
            keyword_counter.update(record.keywords)
            report_lengths.append({"split": split_name, "length": len(record.report_text.split())})
    pd.DataFrame(split_rows).plot.bar(x="split", y="samples", legend=False, figsize=(6, 4), color="#1f77b4")
    plt.title("Dataset Split Sizes")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "split_sizes.png", dpi=220)
    plt.close()

    top_keywords = pd.DataFrame(keyword_counter.most_common(15), columns=["keyword", "count"])
    plt.figure(figsize=(10, 6))
    sns.barplot(top_keywords, x="count", y="keyword", palette="viridis")
    plt.title("Top Clinical Keywords")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "top_keywords.png", dpi=220)
    plt.close()

    report_df = pd.DataFrame(report_lengths)
    plt.figure(figsize=(8, 5))
    sns.histplot(report_df, x="length", hue="split", bins=20, kde=True)
    plt.title("Report Length Distribution")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "report_lengths.png", dpi=220)
    plt.close()


def plot_history(history_path: str | Path, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    history = pd.DataFrame(load_json(history_path))
    if history.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["valid_loss"], label="Valid Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_curves.png", dpi=220)
    plt.close()

    metric_cols = [col for col in history.columns if col.startswith("valid_bleu_") or col in ["valid_rouge_l", "valid_keyword_micro_f1", "valid_keyword_map"]]
    if metric_cols:
        plt.figure(figsize=(10, 5))
        for col in metric_cols:
            plt.plot(history["epoch"], history[col], label=col.replace("valid_", ""))
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "validation_metrics.png", dpi=220)
        plt.close()


def plot_keyword_graph(adjacency, keyword_names: list[str], output_path: str | Path, top_k: int = 25) -> None:
    matrix = adjacency.cpu().numpy() if hasattr(adjacency, "cpu") else np.asarray(adjacency)
    strengths = matrix.sum(axis=1)
    chosen = np.argsort(strengths)[-top_k:]
    graph = nx.Graph()
    for idx in chosen:
        graph.add_node(keyword_names[idx])
    for i in chosen:
        for j in chosen:
            if j <= i:
                continue
            weight = float(matrix[i, j])
            if weight > matrix.mean():
                graph.add_edge(keyword_names[i], keyword_names[j], weight=weight)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=7)
    widths = [1 + 5 * graph[u][v]["weight"] for u, v in graph.edges()]
    nx.draw_networkx(graph, pos, node_color="#0ea5a4", edge_color="#94a3b8", width=widths, font_size=9)
    plt.title("Knowledge Graph of Retinal Keyword Co-occurrence")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_metric_radar(metrics: dict[str, float], output_path: str | Path) -> None:
    keys = [
        "bleu_4",
        "rouge_l",
        "keyword_micro_f1",
        "keyword_map",
        "report_token_f1",
    ]
    values = [float(metrics.get(key, 0.0)) for key in keys]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys)
    ax.set_title("Evaluation Radar")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_prediction_examples(predictions_path: str | Path, output_dir: str | Path, max_examples: int = 8) -> None:
    output_dir = ensure_dir(output_dir)
    records = load_json(predictions_path)[:max_examples]
    if not records:
        return
    scores = []
    for row in records:
        ref_len = max(len(row["reference_report"].split()), 1)
        pred_len = len(row["predicted_report"].split())
        scores.append({"image": Path(row["image_path"]).name, "length_ratio": pred_len / ref_len, "predicted_keywords": len(row["predicted_keywords"])})
    frame = pd.DataFrame(scores)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(frame, x="length_ratio", y="predicted_keywords", s=120)
    for _, row in frame.iterrows():
        plt.text(row["length_ratio"], row["predicted_keywords"], row["image"], fontsize=8)
    plt.title("Prediction Diagnostics")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "prediction_diagnostics.png", dpi=220)
    plt.close()
