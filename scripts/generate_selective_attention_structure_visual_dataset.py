from __future__ import annotations

import pandas as pd

from abc_benchmark.selective_attention.structure_sensitive.visual.build_dataset import (
    DEFAULT_SLICES,
    build_structure_sensitive_visual_dataset,
)

OUT_DIR = "artifacts/datasets/selective_attention/structure_sensitive/visual"
SLICES = DEFAULT_SLICES


def main() -> None:
    scenes_df, counting_df, filtering_df = build_structure_sensitive_visual_dataset(
        output_dir=OUT_DIR,
        slices=SLICES,
    )

    print(scenes_df[["dimension", "variant"]].value_counts())
    print(counting_df[["task_type"]].value_counts())
    print(filtering_df[["task_type"]].value_counts())
    print(scenes_df.head())


if __name__ == "__main__":
    main()
