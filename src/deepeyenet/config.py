from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class ExperimentConfig:
    metadata_dir: str
    images_root: str
    output_dir: str = "outputs/run"
    train_json: str = "DeepEyeNet_train.json"
    valid_json: str = "DeepEyeNet_valid.json"
    test_json: str = "DeepEyeNet_test.json"
    image_size: int = 224
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_clinical_tokens: int = 96
    max_report_tokens: int = 128
    embed_dim: int = 128
    hidden_dim: int = 256
    graph_steps: int = 2
    teacher_forcing: float = 0.7
    generation_loss_weight: float = 1.0
    keyword_loss_weight: float = 0.5
    random_seed: int = 42
    device: str = "cuda"
    min_keyword_freq: int = 1
    save_every_epoch: bool = False
    sample_visualizations: int = 8
    notes: dict = field(default_factory=dict)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
