from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.datasets.build_dataset import build_dataset
from abc_benchmark.generation.feature_generator import make_generator


def _feature_row(scene, difficulty: str, seed: int) -> dict:
    return {
        "seed": scene.seed,
        "difficulty": scene.difficulty,
        "prompt": scene.prompt,
        "image_path": str(Path("images") / difficulty / f"{difficulty}_{seed}.png"),
        "gold_label": scene.gold_label,
        "target_shape": scene.target_shape,
        "target_color": scene.target_color,
        "num_items": len(scene.items),
    }


def build_feature_dataset(
    output_dir: str | Path,
    difficulty: str,
    count: int,
    start_seed: int = 0,
) -> pd.DataFrame:
    return build_dataset(
        output_dir=output_dir,
        difficulty=difficulty,
        count=count,
        make_generator=make_generator,
        row_builder=_feature_row,
        start_seed=start_seed,
    )