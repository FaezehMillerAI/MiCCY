from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
from typing import Any

import pandas as pd
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

from deepeyenet.data.vocab import Vocabulary
from deepeyenet.utils.io import load_json


@dataclass
class SampleRecord:
    image_path: str
    keywords: list[str]
    clinical_description: str
    report_text: str


def parse_keywords(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [part.strip().lower() for part in text.split(",") if part.strip()]


def load_split(metadata_dir: str | Path, filename: str) -> list[SampleRecord]:
    path = Path(metadata_dir) / filename
    if path.suffix.lower() == ".json":
        payload = load_json(path)
        records = []
        for image_path, item in payload.items():
            records.append(
                SampleRecord(
                    image_path=image_path,
                    keywords=parse_keywords(item.get("Keywords", [])),
                    clinical_description=str(item.get("clinical-description", "")),
                    report_text=str(item.get("report_text", "")),
                )
            )
        return records
    frame = pd.read_csv(path)
    return [
        SampleRecord(
            image_path=row["image_path"],
            keywords=parse_keywords(row.get("Keywords", [])),
            clinical_description=str(row.get("clinical-description", "")),
            report_text=str(row.get("report_text", "")),
        )
        for _, row in frame.iterrows()
    ]


class DeepEyeNetDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        images_root: str | Path,
        clinical_vocab: Vocabulary,
        report_vocab: Vocabulary,
        keyword_to_idx: dict[str, int],
        image_size: int = 224,
        max_clinical_tokens: int = 96,
        max_report_tokens: int = 128,
        training: bool = True,
    ) -> None:
        self.records = records
        self.images_root = Path(images_root)
        self.clinical_vocab = clinical_vocab
        self.report_vocab = report_vocab
        self.keyword_to_idx = keyword_to_idx
        self.max_clinical_tokens = max_clinical_tokens
        self.max_report_tokens = max_report_tokens
        augmentations = []
        if training:
            augmentations.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
            ])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            *augmentations,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(self.images_root / record.image_path).convert("RGB")
        image_tensor = self.transform(image)
        clinical_ids = torch.tensor(
            self.clinical_vocab.encode(record.clinical_description, self.max_clinical_tokens),
            dtype=torch.long,
        )
        report_ids = torch.tensor(
            self.report_vocab.encode(record.report_text, self.max_report_tokens),
            dtype=torch.long,
        )
        keyword_target = torch.zeros(len(self.keyword_to_idx), dtype=torch.float32)
        keyword_indices = []
        for keyword in record.keywords:
            if keyword in self.keyword_to_idx:
                idx = self.keyword_to_idx[keyword]
                keyword_target[idx] = 1.0
                keyword_indices.append(idx)
        return {
            "image": image_tensor,
            "clinical_ids": clinical_ids,
            "report_ids": report_ids,
            "keyword_target": keyword_target,
            "keyword_indices": torch.tensor(keyword_indices, dtype=torch.long),
            "raw_keywords": record.keywords,
            "clinical_text": record.clinical_description,
            "report_text": record.report_text,
            "image_path": record.image_path,
        }


def build_keyword_index(records: list[SampleRecord], min_freq: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for keyword in record.keywords:
            counts[keyword] = counts.get(keyword, 0) + 1
    keywords = sorted([keyword for keyword, freq in counts.items() if freq >= min_freq])
    return {keyword: idx for idx, keyword in enumerate(keywords)}


def collate_batch(batch: list[dict[str, Any]], report_pad_id: int, clinical_pad_id: int) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "clinical_ids": pad_sequence([item["clinical_ids"] for item in batch], batch_first=True, padding_value=clinical_pad_id),
        "report_ids": pad_sequence([item["report_ids"] for item in batch], batch_first=True, padding_value=report_pad_id),
        "keyword_target": torch.stack([item["keyword_target"] for item in batch]),
        "keyword_indices": [item["keyword_indices"] for item in batch],
        "raw_keywords": [item["raw_keywords"] for item in batch],
        "clinical_text": [item["clinical_text"] for item in batch],
        "report_text": [item["report_text"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }
