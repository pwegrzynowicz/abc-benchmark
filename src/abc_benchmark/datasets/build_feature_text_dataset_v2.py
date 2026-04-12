from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.generation.feature_text_generator_v2 import (
    FeatureTextSelectiveAttentionGeneratorV2,
    scene_to_dataset_row,
)


def build_feature_text_dataset_v2(
    output_dir: str | Path,
    *,
    regime: str,
    regime_level: str,
    count: int,
    start_seed: int = 0,
) -> pd.DataFrame:
    """Build a wide feature-text v2 dataset.

    Each generated scene supports both tasks:
    - counting via count_prompt + gold_count
    - filtering via filter_prompt + gold_lines

    So this builder intentionally generates each scene only once and does not
    split rows by a fake response_mode.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = FeatureTextSelectiveAttentionGeneratorV2()
    rows: list[dict] = []

    for i in range(count):
        seed = start_seed + i
        scene = gen.generate(
            seed=seed,
            regime=regime,  # type: ignore[arg-type]
            regime_level=regime_level,
        )
        rows.append(scene_to_dataset_row(scene))

    df = pd.DataFrame(rows)
    filename = f"{regime}_{regime_level}.csv"
    df.to_csv(output_dir / filename, index=False)
    return df
