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
    response_mode: str = "count",
    start_seed: int = 0,
) -> pd.DataFrame:
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
            response_mode=response_mode,  # type: ignore[arg-type]
        )
        rows.append(scene_to_dataset_row(scene))

    df = pd.DataFrame(rows)
    filename = f"{regime}_{regime_level}_{response_mode}.csv"
    df.to_csv(output_dir / filename, index=False)
    return df
