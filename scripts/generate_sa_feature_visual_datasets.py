from __future__ import annotations

from abc_benchmark.selective_attention.feature_sensitive.visual.dataset import (
    build_feature_sensitive_visual_dataset,
)

OUT_DIR = "artifacts/datasets/selective_attention/feature_sensitive/visual"

SLICES: list[tuple[str, str, int, int]] = [
    ("baseline", "baseline", 30, 1000),
    ("set_size", "xs", 30, 1100),
    ("set_size", "s", 30, 1200),
    ("set_size", "m", 30, 1300),
    ("set_size", "l", 30, 1400),
    ("rule_arity", "color_only", 30, 2000),
    ("rule_arity", "shape_only", 30, 2050),
    ("rule_arity", "color_shape", 30, 2100),
    ("rule_arity", "color_shape_size", 30, 2200),
    ("confound", "low", 30, 3000),
    ("confound", "high", 30, 3100),
    ("spatial_density", "sparse", 30, 4000),
    ("spatial_density", "crowded", 30, 4100),
    ("target_count", "0", 30, 5000),
    ("target_count", "3", 30, 5100),
    ("layout_regularity", "random", 30, 6000),
    ("layout_regularity", "grid", 30, 6100),
    ("layout_regularity", "clustered", 30, 6200),
    ("combined", "easy", 30, 7000),
    ("combined", "medium", 30, 7100),
    ("combined", "hard", 30, 7200),
]


def main() -> None:
    result = build_feature_sensitive_visual_dataset(
        OUT_DIR,
        slices=SLICES,
    )

    print("Scenes:", len(result.scenes_df))
    print("Counting rows:", len(result.counting_df))
    print("Filtering rows:", len(result.filtering_df))
    print(result.scenes_df[["dimension", "variant"]].value_counts().sort_index())
    print(result.scenes_df.head())


if __name__ == "__main__":
    main()
