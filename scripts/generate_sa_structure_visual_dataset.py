from __future__ import annotations

from abc_benchmark.selective_attention.structure_sensitive.visual.dataset import (
    build_structure_sensitive_visual_dataset,
)

OUT_DIR = "artifacts/datasets/selective_attention/structure_sensitive/visual"

SLICES: list[tuple[str, str, int, int]] = [
    ("baseline", "simple", 20, 1000),
    ("principle", "proximity", 20, 2000),
    ("principle", "similarity", 20, 2100),
    ("principle", "continuity", 20, 2200),
    ("principle", "common_region", 20, 2300),
    ("combined", "easy", 20, 7000),
    ("combined", "medium", 20, 7100),
    ("combined", "hard", 20, 7200),
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
