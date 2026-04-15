from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.selective_attention.structure_sensitive.text.generator import (
    StructureSensitiveTextGenerator, scene_to_dataset_row
)


def build_structure_sensitive_text_dataset(
    output_dir: str | Path,
    *,
    dimension: str,
    variant: str,
    count: int,
    start_seed: int = 0,
    target_count_override: int | None = None,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = StructureSensitiveTextGenerator()
    rows: list[dict[str, object]] = []

    for offset in range(count):
        seed = start_seed + offset
        scene = generator.generate(
            seed=seed,
            dimension=dimension,  # type: ignore[arg-type]
            variant=variant,
            target_count_override=target_count_override,
        )
        rows.append(scene_to_dataset_row(scene))

    dataframe = pd.DataFrame(rows)
    filename_parts = [dimension, variant]
    if target_count_override is not None:
        filename_parts.append(f"tc{target_count_override}")
    dataframe.to_csv(output_path / ("_".join(filename_parts) + ".csv"), index=False)
    return dataframe
