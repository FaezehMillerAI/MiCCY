from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepeyenet.evaluation.metrics import compute_classification_metrics, compute_generation_metrics
from deepeyenet.utils.io import save_json


@dataclass
class EpochResult:
    loss: float
    generation_loss: float
    keyword_loss: float
    metrics: dict[str, float]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,
        report_pad_id: int,
        adjacency: torch.Tensor,
        report_vocab,
        keyword_names: list[str],
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.report_pad_id = report_pad_id
        self.adjacency = adjacency.to(device)
        self.report_vocab = report_vocab
        self.keyword_names = keyword_names
        self.device = device
        self.gen_criterion = nn.CrossEntropyLoss(ignore_index=report_pad_id)
        self.keyword_criterion = nn.BCEWithLogitsLoss()

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch["image"] = batch["image"].to(self.device)
        batch["clinical_ids"] = batch["clinical_ids"].to(self.device)
        batch["report_ids"] = batch["report_ids"].to(self.device)
        batch["keyword_target"] = batch["keyword_target"].to(self.device)
        return batch

    def train_epoch(self, loader: DataLoader) -> EpochResult:
        self.model.train()
        losses = {"total": 0.0, "generation": 0.0, "keyword": 0.0}
        count = 0
        for batch in tqdm(loader, desc="train", leave=False):
            batch = self._move_batch(batch)
            decoder_input_ids = batch["report_ids"][:, :-1]
            target_ids = batch["report_ids"][:, 1:]
            outputs = self.model(
                images=batch["image"],
                clinical_ids=batch["clinical_ids"],
                keyword_indices=batch["keyword_indices"],
                adjacency=self.adjacency,
                decoder_input_ids=decoder_input_ids,
            )
            generation_loss = self.gen_criterion(outputs["decoder_logits"].reshape(-1, outputs["decoder_logits"].size(-1)), target_ids.reshape(-1))
            keyword_loss = self.keyword_criterion(outputs["keyword_logits"], batch["keyword_target"])
            loss = self.config.generation_loss_weight * generation_loss + self.config.keyword_loss_weight * keyword_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            bs = batch["image"].size(0)
            losses["total"] += loss.item() * bs
            losses["generation"] += generation_loss.item() * bs
            losses["keyword"] += keyword_loss.item() * bs
            count += bs
        return EpochResult(
            loss=losses["total"] / max(count, 1),
            generation_loss=losses["generation"] / max(count, 1),
            keyword_loss=losses["keyword"] / max(count, 1),
            metrics={},
        )

    @torch.no_grad()
    def evaluate_epoch(self, loader: DataLoader) -> tuple[EpochResult, list[dict[str, Any]]]:
        self.model.eval()
        losses = {"total": 0.0, "generation": 0.0, "keyword": 0.0}
        count = 0
        all_keyword_targets = []
        all_keyword_probs = []
        references = []
        predictions = []
        samples = []
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = self._move_batch(batch)
            decoder_input_ids = batch["report_ids"][:, :-1]
            target_ids = batch["report_ids"][:, 1:]
            outputs = self.model(
                images=batch["image"],
                clinical_ids=batch["clinical_ids"],
                keyword_indices=batch["keyword_indices"],
                adjacency=self.adjacency,
                decoder_input_ids=decoder_input_ids,
            )
            generation_loss = self.gen_criterion(outputs["decoder_logits"].reshape(-1, outputs["decoder_logits"].size(-1)), target_ids.reshape(-1))
            keyword_loss = self.keyword_criterion(outputs["keyword_logits"], batch["keyword_target"])
            loss = self.config.generation_loss_weight * generation_loss + self.config.keyword_loss_weight * keyword_loss
            generated = self.model.generate(
                images=batch["image"],
                clinical_ids=batch["clinical_ids"],
                keyword_indices=batch["keyword_indices"],
                adjacency=self.adjacency,
                bos_id=self.report_vocab.bos_id,
                eos_id=self.report_vocab.eos_id,
                max_steps=self.config.max_report_tokens - 1,
            )
            keyword_probs = torch.sigmoid(generated["keyword_logits"]).cpu()
            keyword_pred_mask = (keyword_probs >= 0.5).int().numpy()
            keyword_target = batch["keyword_target"].cpu().numpy()
            decoded = [self.report_vocab.decode(row.tolist()) for row in generated["tokens"].cpu()]
            for idx, pred_text in enumerate(decoded):
                ref_text = batch["report_text"][idx]
                references.append(ref_text)
                predictions.append(pred_text)
                pred_keywords = [self.keyword_names[i] for i, flag in enumerate(keyword_pred_mask[idx]) if flag]
                samples.append({
                    "image_path": batch["image_path"][idx],
                    "clinical_text": batch["clinical_text"][idx],
                    "reference_report": ref_text,
                    "predicted_report": pred_text,
                    "reference_keywords": batch["raw_keywords"][idx],
                    "predicted_keywords": pred_keywords,
                    "attention_trace": generated["attention_maps"][idx].cpu().tolist(),
                })
            bs = batch["image"].size(0)
            losses["total"] += loss.item() * bs
            losses["generation"] += generation_loss.item() * bs
            losses["keyword"] += keyword_loss.item() * bs
            count += bs
            all_keyword_targets.append(keyword_target)
            all_keyword_probs.append(keyword_probs.numpy())
        classification_metrics = compute_classification_metrics(
            y_true=self._stack(all_keyword_targets),
            y_prob=self._stack(all_keyword_probs),
            keyword_names=self.keyword_names,
        )
        generation_metrics = compute_generation_metrics(references, predictions)
        metrics = {**classification_metrics, **generation_metrics}
        return (
            EpochResult(
                loss=losses["total"] / max(count, 1),
                generation_loss=losses["generation"] / max(count, 1),
                keyword_loss=losses["keyword"] / max(count, 1),
                metrics=metrics,
            ),
            samples,
        )

    def _stack(self, chunks):
        import numpy as np
        if not chunks:
            return np.empty((0, len(self.keyword_names)))
        return np.concatenate(chunks, axis=0)

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, output_dir: str | Path) -> dict[str, Any]:
        output_dir = Path(output_dir)
        history = []
        best_score = float("-inf")
        best_path = output_dir / "best.pt"
        for epoch in range(1, self.config.epochs + 1):
            train_result = self.train_epoch(train_loader)
            valid_result, valid_samples = self.evaluate_epoch(valid_loader)
            row = {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_generation_loss": train_result.generation_loss,
                "train_keyword_loss": train_result.keyword_loss,
                "valid_loss": valid_result.loss,
                "valid_generation_loss": valid_result.generation_loss,
                "valid_keyword_loss": valid_result.keyword_loss,
                **{f"valid_{k}": v for k, v in valid_result.metrics.items()},
            }
            history.append(row)
            score = valid_result.metrics.get("bleu_4", 0.0) + valid_result.metrics.get("keyword_micro_f1", 0.0)
            checkpoint = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": history,
            }
            if score > best_score:
                best_score = score
                torch.save(checkpoint, best_path)
                save_json(valid_samples, output_dir / "best_valid_predictions.json")
            if self.config.save_every_epoch:
                torch.save(checkpoint, output_dir / f"epoch_{epoch}.pt")
        save_json(history, output_dir / "history.json")
        return {"history": history, "best_checkpoint": str(best_path)}
