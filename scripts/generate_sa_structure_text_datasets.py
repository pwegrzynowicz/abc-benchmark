from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.selective_attention.structure_sensitive.text.dataset import (
    build_structure_sensitive_text_dataset,
)


SPECS: list[dict[str, object]] = [
    {"dimension": "baseline_structure", "variant": "baseline", "count": 20, "start_seed": 1000},
    {"dimension": "structure_depth", "variant": "shallow", "count": 20, "start_seed": 2000},
    {"dimension": "structure_depth", "variant": "medium", "count": 20, "start_seed": 2100},
    {"dimension": "structure_depth", "variant": "nested", "count": 20, "start_seed": 2200},
    {"dimension": "binding_distance", "variant": "near", "count": 20, "start_seed": 3000},
    {"dimension": "binding_distance", "variant": "medium", "count": 20, "start_seed": 3100},
    {"dimension": "binding_distance", "variant": "far", "count": 20, "start_seed": 3200},
    {"dimension": "confound", "variant": "low", "count": 20, "start_seed": 4000},
    {"dimension": "confound", "variant": "medium", "count": 20, "start_seed": 4100},
    {"dimension": "confound", "variant": "high", "count": 20, "start_seed": 4200},
    {"dimension": "confound", "variant": "extreme", "count": 20, "start_seed": 4300},
    {"dimension": "confound_type", "variant": "grouping_leader_only", "count": 20, "start_seed": 4400},
    {"dimension": "confound_type", "variant": "grouping_follower_only", "count": 20, "start_seed": 4500},
    {"dimension": "confound_type", "variant": "grouping_cross_binding", "count": 20, "start_seed": 4600},
    {"dimension": "confound_type", "variant": "relation_role_reversal", "count": 20, "start_seed": 4700},
    {"dimension": "confound_type", "variant": "relation_leader_partial", "count": 20, "start_seed": 4800},
    {"dimension": "confound_type", "variant": "relation_follower_partial", "count": 20, "start_seed": 4900},
    {"dimension": "confound_type", "variant": "scope_wrong_scope_value", "count": 20, "start_seed": 5000},
    {"dimension": "confound_type", "variant": "scope_leader_partial", "count": 20, "start_seed": 5100},
    {"dimension": "confound_type", "variant": "scope_follower_partial", "count": 20, "start_seed": 5200},
    {"dimension": "confound_type", "variant": "global_local_wrong_pattern", "count": 20, "start_seed": 5300},
    {"dimension": "confound_type", "variant": "global_local_wrong_marker", "count": 20, "start_seed": 5400},
    {"dimension": "confound_type", "variant": "global_local_leader_color_only", "count": 20, "start_seed": 5500},
    {"dimension": "confound_type", "variant": "global_local_leader_shape_only", "count": 20, "start_seed": 5600},
    {"dimension": "target_count_x_structure_depth", "variant": "0_shallow", "count": 20, "start_seed": 5700},
    {"dimension": "target_count_x_structure_depth", "variant": "0_nested", "count": 20, "start_seed": 5800},
    {"dimension": "target_count_x_structure_depth", "variant": "3_shallow", "count": 20, "start_seed": 5900},
    {"dimension": "target_count_x_structure_depth", "variant": "3_nested", "count": 20, "start_seed": 6000},
    {"dimension": "serialization_style", "variant": "compact", "count": 20, "start_seed": 6100},
    {"dimension": "serialization_style", "variant": "tagged", "count": 20, "start_seed": 6200},
    {"dimension": "serialization_style", "variant": "nested", "count": 20, "start_seed": 6300},
    {"dimension": "combined", "variant": "easy", "count": 20, "start_seed": 7000},
    {"dimension": "combined", "variant": "medium", "count": 20, "start_seed": 7100},
    {"dimension": "combined", "variant": "hard", "count": 20, "start_seed": 7200},
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
            serialization_style=(
                None if spec.get("serialization_style") is None else str(spec["serialization_style"])
            ),
            target_count_override=(
                None if spec.get("target_count_override") is None else int(spec["target_count_override"])
            ),
        )
        dataframes.append(dataframe)

    full_dataframe = pd.concat(dataframes, ignore_index=True)
    full_dataframe.to_csv(output_dir / "structure_sensitive_text_full.csv", index=False)

    _print_grouped_summary(full_dataframe, "by dimension, variant", ["dimension", "variant"])
    _print_grouped_summary(full_dataframe, "by structure_type", ["structure_type"])
    _print_grouped_summary(full_dataframe, "by structure_depth", ["structure_depth"])
    _print_grouped_summary(full_dataframe, "by binding_distance", ["binding_distance"])
    _print_grouped_summary(full_dataframe, "by serialization_style", ["serialization_style"])
    _print_grouped_summary(full_dataframe, "by target_count", ["target_count"])
    _print_grouped_summary(full_dataframe, "by confound_count", ["confound_count"])
    _print_grouped_summary(full_dataframe, "by confound_type", ["confound_type"])
    _print_grouped_summary(full_dataframe, "by structure_type, confound_type", ["structure_type", "confound_type"])

    print("\n=== total rows ===")
    print(len(full_dataframe))

    print("\n=== columns ===")
    print(sorted(full_dataframe.columns.tolist()))


if __name__ == "__main__":
    main()
