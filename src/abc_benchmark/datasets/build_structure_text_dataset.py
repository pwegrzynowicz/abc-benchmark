from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.generation.structure_text_generator import (
    StructureTextSelectiveAttentionGenerator,
    scene_to_dataset_row,
)


def build_structure_text_dataset(
    output_dir: str | Path,
    *,
    regime: str,
    regime_level: str,
    count: int,
    start_seed: int = 0,
    serialization_style: str | None = None,
    target_count_override: int | None = None,
) -> pd.DataFrame:
    """Build a wide structure-text dataset.

    Each generated scene is exported once and supports both:
    - counting via count_prompt + gold_count
    - filtering via filter_prompt + gold_lines

    Explicit benchmark-facing factors:
    - regime / regime_level
    - structure_type
    - structure_depth
    - binding_distance
    - serialization_style
    - target_count
    - confound_count
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = StructureTextSelectiveAttentionGenerator()
    rows: list[dict[str, object]] = []

    for i in range(count):
        seed = start_seed + i
        scene = generator.generate(
            seed=seed,
            regime=regime,  # type: ignore[arg-type]
            regime_level=regime_level,
            target_count_override=target_count_override,
            serialization_style=serialization_style,  # type: ignore[arg-type]
        )
        rows.append(scene_to_dataset_row(scene))

    df = pd.DataFrame(rows)

    filename_parts = [regime, regime_level]
    if serialization_style is not None:
        filename_parts.append(serialization_style)
    if target_count_override is not None:
        filename_parts.append(f"tc{target_count_override}")
    filename = "_".join(filename_parts) + ".csv"

    df.to_csv(output_dir / filename, index=False)
    return df
