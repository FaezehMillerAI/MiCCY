from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9/+-]+|[.,;:()]", text.lower())


@dataclass
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "Vocabulary":
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))
        specials = [PAD, BOS, EOS, UNK]
        itos = specials[:]
        for token, freq in counter.items():
            if freq >= min_freq and token not in specials:
                itos.append(token)
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, max_length: int, add_special_tokens: bool = True) -> list[int]:
        tokens = simple_tokenize(text)
        if add_special_tokens:
            tokens = [BOS] + tokens[: max_length - 2] + [EOS]
        else:
            tokens = tokens[:max_length]
        return [self.stoi.get(token, self.stoi[UNK]) for token in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        specials = {PAD, BOS, EOS} if skip_special_tokens else set()
        tokens = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if token == EOS:
                break
            if token in specials:
                continue
            tokens.append(token)
        text = " ".join(tokens)
        return re.sub(r"\s+([.,;:()])", r"", text).strip()

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        itos = list(data["itos"])
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
