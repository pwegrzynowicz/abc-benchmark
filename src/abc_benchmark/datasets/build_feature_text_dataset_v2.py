
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
    position_mode: str | None = None,
    target_count_override: int | None = None,
) -> pd.DataFrame:
    """Build a wide feature-text dataset.

    Each generated scene is exported once and supports both:
    - counting via count_prompt + gold_count
    - filtering via filter_prompt + gold_lines

    Explicit benchmark-facing factors:
    - regime / regime_level (including position_sweep, target_count_sweep,
      and the interaction sweeps)
    - target_count_override (optional manual override for experiments)
    - position_mode (primarily meaningful for position_sweep)
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
            position_mode=position_mode,  # type: ignore[arg-type]
            target_count_override=target_count_override,
        )
        rows.append(scene_to_dataset_row(scene))

    df = pd.DataFrame(rows)

    filename_parts = [regime, regime_level]
    if position_mode is not None and not (regime == "position_sweep" and position_mode == regime_level):
        filename_parts.append(position_mode)
    if target_count_override is not None:
        filename_parts.append(f"tc{target_count_override}")
    filename = "_".join(filename_parts) + ".csv"

    df.to_csv(output_dir / filename, index=False)
    return df
