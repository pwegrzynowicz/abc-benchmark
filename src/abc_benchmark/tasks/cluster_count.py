from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClusterCountExample:
    prompt: str
    image_path: str
    gold_label: int


def normalize_model_output(text: str) -> int:
    text = text.strip()
    if not text.isdigit():
        raise ValueError(f"Invalid model output: {text!r}")
    return int(text)


def score_prediction(prediction: int, gold_label: int) -> int:
    return int(prediction == gold_label)


# Kaggle Benchmarks integration belongs here later.
