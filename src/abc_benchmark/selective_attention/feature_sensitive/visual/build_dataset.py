from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.selective_attention.feature_sensitive.visual.generator import (
    FeatureSensitiveVisualGenerator,
    scene_to_dataset_row,
)


def build_feature_sensitive_visual_dataset(
    output_dir: str | Path,
    *,
    dimension: str,
    variant: str,
    count: int,
    start_seed: int = 0,
    csv_name: str | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images" / dimension / variant
    image_dir.mkdir(parents=True, exist_ok=True)

    generator = FeatureSensitiveVisualGenerator()
    rows: list[dict[str, object]] = []

    for i in range(count):
        seed = start_seed + i
        scene = generator.generate(seed=seed, dimension=dimension, variant=variant)
        image_path = image_dir / f"{dimension}__{variant}__{seed}.png"
        generator.render(scene, image_path)

        row = scene_to_dataset_row(scene)
        row["image_path"] = str(Path("images") / dimension / variant / image_path.name)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_name = csv_name or f"{dimension}__{variant}.csv"
    df.to_csv(output_dir / out_name, index=False)
    return df
