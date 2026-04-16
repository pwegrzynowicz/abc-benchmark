from __future__ import annotations

from abc_benchmark.selective_attention.structure_sensitive.visual.dataset import (
    build_structure_sensitive_visual_dataset,
)

OUT_DIR = "artifacts/datasets/selective_attention/structure_sensitive/visual"

SLICES: list[tuple[str, str, int, int]] = [
    ("baseline", "simple", 30, 1000),
    ("principle", "proximity", 30, 2000),
    ("principle", "similarity", 30, 2100),
    ("principle", "continuity", 30, 2200),
    ("principle", "common_region", 30, 2300),
    ("target_count", "0", 30, 3000),
    ("target_count", "1", 30, 3100),
    ("target_count", "3", 30, 3200),
    ("target_count", "6", 30, 3300),
    ("combined", "easy", 30, 7000),
    ("combined", "medium", 30, 7100),
    ("combined", "hard", 30, 7200),
]


def main() -> None:
    result = build_structure_sensitive_visual_dataset(
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
