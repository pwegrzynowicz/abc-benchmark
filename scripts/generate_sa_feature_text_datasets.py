from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.selective_attention.feature_sensitive.text.dataset import (
    build_feature_sensitive_text_dataset,
)


SPECS: list[dict[str, object]] = [
    {"dimension": "baseline", "variant": "baseline", "count": 20, "start_seed": 1000},
    {"dimension": "set_size", "variant": "xs", "count": 20, "start_seed": 2000},
    {"dimension": "set_size", "variant": "s", "count": 20, "start_seed": 2100},
    {"dimension": "set_size", "variant": "m", "count": 20, "start_seed": 2200},
    {"dimension": "set_size", "variant": "l", "count": 20, "start_seed": 2300},
    {"dimension": "rule_arity", "variant": "1f", "count": 20, "start_seed": 3000},
    {"dimension": "rule_arity", "variant": "2f", "count": 20, "start_seed": 3100},
    {"dimension": "rule_arity", "variant": "3f", "count": 20, "start_seed": 3200},
    {"dimension": "rule_arity", "variant": "4f", "count": 20, "start_seed": 3300},
    {"dimension": "noise_width", "variant": "n0", "count": 20, "start_seed": 4000},
    {"dimension": "noise_width", "variant": "n1", "count": 20, "start_seed": 4100},
    {"dimension": "noise_width", "variant": "n2", "count": 20, "start_seed": 4200},
    {"dimension": "confound", "variant": "low", "count": 20, "start_seed": 5000},
    {"dimension": "confound", "variant": "medium", "count": 20, "start_seed": 5100},
    {"dimension": "confound", "variant": "high", "count": 20, "start_seed": 5200},
    {"dimension": "confound", "variant": "extreme", "count": 20, "start_seed": 5300},
    {"dimension": "position", "variant": "random", "count": 20, "start_seed": 5400},
    {"dimension": "position", "variant": "front_loaded", "count": 20, "start_seed": 5500},
    {"dimension": "position", "variant": "back_loaded", "count": 20, "start_seed": 5600},
    {"dimension": "position", "variant": "clustered", "count": 20, "start_seed": 5700},
    {"dimension": "target_count", "variant": "0", "count": 20, "start_seed": 5800},
    {"dimension": "target_count", "variant": "1", "count": 20, "start_seed": 5900},
    {"dimension": "target_count", "variant": "3", "count": 20, "start_seed": 6000},
    {"dimension": "target_count", "variant": "5", "count": 20, "start_seed": 6100},
    {"dimension": "target_count_x_confound", "variant": "0_low", "count": 20, "start_seed": 6200},
    {"dimension": "target_count_x_confound", "variant": "0_medium", "count": 20, "start_seed": 6300},
    {"dimension": "target_count_x_confound", "variant": "0_extreme", "count": 20, "start_seed": 6400},
    {"dimension": "target_count_x_confound", "variant": "3_low", "count": 20, "start_seed": 6500},
    {"dimension": "target_count_x_confound", "variant": "3_medium", "count": 20, "start_seed": 6600},
    {"dimension": "target_count_x_confound", "variant": "3_extreme", "count": 20, "start_seed": 6700},
    {"dimension": "target_count_x_rule_arity", "variant": "0_1f", "count": 20, "start_seed": 6800},
    {"dimension": "target_count_x_rule_arity", "variant": "0_2f", "count": 20, "start_seed": 6900},
    {"dimension": "target_count_x_rule_arity", "variant": "0_4f", "count": 20, "start_seed": 7000},
    {"dimension": "target_count_x_rule_arity", "variant": "3_1f", "count": 20, "start_seed": 7100},
    {"dimension": "target_count_x_rule_arity", "variant": "3_2f", "count": 20, "start_seed": 7200},
    {"dimension": "target_count_x_rule_arity", "variant": "3_4f", "count": 20, "start_seed": 7300},
    {"dimension": "combined", "variant": "easy", "count": 20, "start_seed": 7400},
    {"dimension": "combined", "variant": "medium", "count": 20, "start_seed": 7500},
    {"dimension": "combined", "variant": "hard", "count": 20, "start_seed": 7600},
]


def _print_grouped_summary(dataframe: pd.DataFrame, title: str, group_columns: list[str]) -> None:
    print(f"\n=== {title} ===")
    print(dataframe.groupby(group_columns).size())


def main() -> None:
    output_dir = Path("artifacts/datasets/selective_attention/feature_sensitive/text")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes: list[pd.DataFrame] = []
    for spec in SPECS:
        dataframe = build_feature_sensitive_text_dataset(
            output_dir,
            dimension=str(spec["dimension"]),
            variant=str(spec["variant"]),
            count=int(spec["count"]),
            start_seed=int(spec.get("start_seed", 0)),
            position_mode=None if spec.get("position_mode") is None else str(spec["position_mode"]),
            target_count_override=None
            if spec.get("target_count_override") is None
            else int(spec["target_count_override"]),
        )
        dataframes.append(dataframe)

    full_dataframe = pd.concat(dataframes, ignore_index=True)
    full_dataframe.to_csv(output_dir / "feature_sensitive_text_full.csv", index=False)

    _print_grouped_summary(full_dataframe, "by dimension, variant", ["dimension", "variant"])
    _print_grouped_summary(full_dataframe, "by target_feature_count", ["target_feature_count"])
    _print_grouped_summary(full_dataframe, "by num_records", ["num_records"])
    _print_grouped_summary(full_dataframe, "by target_count", ["target_count"])
    _print_grouped_summary(full_dataframe, "by position_mode", ["position_mode"])
    _print_grouped_summary(full_dataframe, "by dimension, target_count", ["dimension", "target_count"])

    print("\n=== total rows ===")
    print(len(full_dataframe))

    print("\n=== columns ===")
    print(sorted(full_dataframe.columns.tolist()))


if __name__ == "__main__":
    main()
