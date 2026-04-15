from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from abc_benchmark.selective_attention.structure_sensitive.visual.generator import (
    StructureSensitiveVisualGenerator,
    scene_to_scene_row,
)


DEFAULT_SLICES: list[tuple[str, str, int, int]] = [
    ("baseline_structure", "baseline", 20, 1000),

    ("grouping_principle", "proximity", 20, 1100),
    ("grouping_principle", "similarity", 20, 1200),
    ("grouping_principle", "common_region", 20, 1300),

    ("structure_type", "relation", 20, 3100),
    ("structure_type", "scope", 20, 3200),
    ("structure_type", "global_local", 20, 3300),

    ("combined", "easy", 20, 11000),
    ("combined", "medium", 20, 11100),
    ("combined", "hard", 20, 11200),
]

def build_structure_sensitive_visual_dataset(
    output_dir: str | Path,
    *,
    slices: Iterable[tuple[str, str, int, int]] = DEFAULT_SLICES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    plain_root = output_dir / "images" / "plain"
    ids_root = output_dir / "images" / "ids"
    plain_root.mkdir(parents=True, exist_ok=True)
    ids_root.mkdir(parents=True, exist_ok=True)

    generator = StructureSensitiveVisualGenerator()
    scene_rows: list[dict[str, object]] = []
    counting_rows: list[dict[str, object]] = []
    filtering_rows: list[dict[str, object]] = []

    for dimension, variant, count, start_seed in slices:
        for offset in range(count):
            seed = start_seed + offset
            print(f"Generating dimension={dimension} variant={variant} seed={seed}")
            scene = generator.generate(seed=seed, dimension=dimension, variant=variant)
            scene_row = scene_to_scene_row(scene)

            plain_rel_path = Path("images") / "plain" / dimension / variant / f"{scene.scene_id}.png"
            ids_rel_path = Path("images") / "ids" / dimension / variant / f"{scene.scene_id}.png"
            generator.render(scene, output_dir / plain_rel_path, show_item_ids=False)
            generator.render(scene, output_dir / ids_rel_path, show_item_ids=True)

            scene_rows.append(scene_row)
            counting_rows.append(
                {
                    **scene_row,
                    "task_type": "counting",
                    "image_path": str(plain_rel_path),
                    "label": scene.gold_count,
                }
            )
            filtering_rows.append(
                {
                    **scene_row,
                    "task_type": "filtering",
                    "image_path": str(ids_rel_path),
                    "label": json.dumps(scene.gold_indices),
                }
            )

    scenes_df = pd.DataFrame(scene_rows)
    counting_df = pd.DataFrame(counting_rows)
    filtering_df = pd.DataFrame(filtering_rows)

    scenes_df.to_csv(output_dir / "structure_sensitive_visual_scenes.csv", index=False)
    counting_df.to_csv(output_dir / "structure_sensitive_visual_counting.csv", index=False)
    filtering_df.to_csv(output_dir / "structure_sensitive_visual_filtering.csv", index=False)
    return scenes_df, counting_df, filtering_df
