from __future__ import annotations

from abc_benchmark.selective_attention.feature_sensitive.visual.build_dataset import (
    build_feature_sensitive_visual_dataset,
)

OUT_DIR = "artifacts/datasets/selective_attention/feature_sensitive/visual"

SLICES: list[tuple[str, str, int, int]] = [
    ("baseline", "baseline", 20, 1000),
    ("set_size", "xs", 20, 1100),
    ("set_size", "s", 20, 1200),
    ("set_size", "m", 20, 1300),
    ("set_size", "l", 20, 1400),
    ("rule_arity", "color_only", 20, 2000),
    ("rule_arity", "shape_only", 20, 2050),
    ("rule_arity", "color_shape", 20, 2100),
    ("rule_arity", "color_shape_size", 20, 2200),
    ("confound", "low", 20, 3000),
    ("confound", "high", 20, 3100),
    ("spatial_density", "sparse", 20, 4000),
    ("spatial_density", "crowded", 20, 4100),
    ("target_count", "0", 20, 5000),
    ("target_count", "3", 20, 5100),
    ("layout_regularity", "random", 20, 6000),
    ("layout_regularity", "grid", 20, 6100),
    ("layout_regularity", "clustered", 20, 6200),
    ("combined", "easy", 20, 7000),
    ("combined", "medium", 20, 7100),
    ("combined", "hard", 20, 7200),
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
