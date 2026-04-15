from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.selective_attention.structure_sensitive.text.dataset import (
    build_structure_sensitive_text_dataset
)

SPECS: list[dict[str, object]] = [
    {"dimension": "baseline", "variant": "simple", "count": 20, "start_seed": 1000},
    {"dimension": "principle", "variant": "paragraph_proximity", "count": 20, "start_seed": 2000},
    {"dimension": "principle", "variant": "section_common_region", "count": 20, "start_seed": 2100},
    {"dimension": "principle", "variant": "format_similarity", "count": 20, "start_seed": 2200},
    {"dimension": "principle", "variant": "scope_indentation", "count": 20, "start_seed": 2300},
    {"dimension": "principle", "variant": "continuation_chain", "count": 20, "start_seed": 2400},
    {"dimension": "target_count", "variant": "0", "count": 20, "start_seed": 3000},
    {"dimension": "target_count", "variant": "1", "count": 20, "start_seed": 3100},
    {"dimension": "target_count", "variant": "3", "count": 20, "start_seed": 3200},
    {"dimension": "target_count", "variant": "6", "count": 20, "start_seed": 3300},
    {"dimension": "combined", "variant": "easy", "count": 20, "start_seed": 4000},
    {"dimension": "combined", "variant": "medium", "count": 20, "start_seed": 4100},
    {"dimension": "combined", "variant": "hard", "count": 20, "start_seed": 4200},
]


def _print_grouped_summary(dataframe: pd.DataFrame, title: str, group_columns: list[str]) -> None:
    print(f"\n=== {title} ===")
    print(dataframe.groupby(group_columns).size())


def main() -> None:
    output_dir = Path("artifacts/datasets/selective_attention/structure_sensitive/text")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes: list[pd.DataFrame] = []
    for spec in SPECS:
        dataframe = build_structure_sensitive_text_dataset(
            output_dir,
            dimension=str(spec["dimension"]),
            variant=str(spec["variant"]),
            count=int(spec["count"]),
            start_seed=int(spec.get("start_seed", 0)),
            target_count_override=(None if spec.get("target_count_override") is None else int(spec["target_count_override"])),
        )
        dataframes.append(dataframe)

    full_dataframe = pd.concat(dataframes, ignore_index=True)
    full_dataframe.to_csv(output_dir / "structure_sensitive_text_full.csv", index=False)

    _print_grouped_summary(full_dataframe, "by dimension, variant", ["dimension", "variant"])
    _print_grouped_summary(full_dataframe, "by principle", ["principle"])
    _print_grouped_summary(full_dataframe, "by render_style", ["render_style"])
    _print_grouped_summary(full_dataframe, "by target_in_anchor_group", ["target_in_anchor_group"])
    _print_grouped_summary(full_dataframe, "by target_outside_anchor_group", ["target_outside_anchor_group"])

    print("\n=== total rows ===")
    print(len(full_dataframe))

    print("\n=== columns ===")
    print(sorted(full_dataframe.columns.tolist()))


if __name__ == "__main__":
    main()
