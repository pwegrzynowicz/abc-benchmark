from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.datasets.build_dataset import build_dataset
from abc_benchmark.generation.cluster_generator import make_generator


def _cluster_row(scene, difficulty: str, seed: int) -> dict:
    return {
        "seed": scene.seed,
        "difficulty": scene.difficulty,
        "prompt": scene.prompt,
        "image_path": str(Path("images") / difficulty / f"{difficulty}_{seed}.png"),
        "gold_label": scene.gold_label,
        "target_shape": scene.target_shape,
        "target_color": scene.target_color,
        "target_cluster_id": scene.target_cluster_id,
        "num_items": len(scene.items),
    }


def build_cluster_dataset(
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
        row_builder=_cluster_row,
        start_seed=start_seed,
    )