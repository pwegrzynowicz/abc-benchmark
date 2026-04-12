
from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.datasets.build_feature_text_dataset_v2 import build_feature_text_dataset_v2


SPECS: list[dict[str, object]] = [
    {"regime": "baseline", "regime_level": "baseline", "count": 20, "start_seed": 1000},

    {"regime": "set_size_sweep", "regime_level": "xs", "count": 20, "start_seed": 2000},
    {"regime": "set_size_sweep", "regime_level": "s", "count": 20, "start_seed": 2100},
    {"regime": "set_size_sweep", "regime_level": "m", "count": 20, "start_seed": 2200},
    {"regime": "set_size_sweep", "regime_level": "l", "count": 20, "start_seed": 2300},

    {"regime": "rule_arity_sweep", "regime_level": "1f", "count": 20, "start_seed": 3000},
    {"regime": "rule_arity_sweep", "regime_level": "2f", "count": 20, "start_seed": 3100},
    {"regime": "rule_arity_sweep", "regime_level": "3f", "count": 20, "start_seed": 3200},
    {"regime": "rule_arity_sweep", "regime_level": "4f", "count": 20, "start_seed": 3300},

    {"regime": "noise_width_sweep", "regime_level": "n0", "count": 20, "start_seed": 4000},
    {"regime": "noise_width_sweep", "regime_level": "n1", "count": 20, "start_seed": 4100},
    {"regime": "noise_width_sweep", "regime_level": "n2", "count": 20, "start_seed": 4200},

    {"regime": "confound_sweep", "regime_level": "low", "count": 20, "start_seed": 5000},
    {"regime": "confound_sweep", "regime_level": "medium", "count": 20, "start_seed": 5100},
    {"regime": "confound_sweep", "regime_level": "high", "count": 20, "start_seed": 5200},
    {"regime": "confound_sweep", "regime_level": "extreme", "count": 20, "start_seed": 5300},

    {"regime": "position_sweep", "regime_level": "random", "count": 20, "start_seed": 5400},
    {"regime": "position_sweep", "regime_level": "front_loaded", "count": 20, "start_seed": 5500},
    {"regime": "position_sweep", "regime_level": "back_loaded", "count": 20, "start_seed": 5600},
    {"regime": "position_sweep", "regime_level": "clustered", "count": 20, "start_seed": 5700},

    {"regime": "target_count_sweep", "regime_level": "0", "count": 20, "start_seed": 5800},
    {"regime": "target_count_sweep", "regime_level": "1", "count": 20, "start_seed": 5900},
    {"regime": "target_count_sweep", "regime_level": "3", "count": 20, "start_seed": 6000},
    {"regime": "target_count_sweep", "regime_level": "5", "count": 20, "start_seed": 6100},

    {"regime": "target_count_x_confound_sweep", "regime_level": "0_low", "count": 20, "start_seed": 6200},
    {"regime": "target_count_x_confound_sweep", "regime_level": "0_medium", "count": 20, "start_seed": 6300},
    {"regime": "target_count_x_confound_sweep", "regime_level": "0_extreme", "count": 20, "start_seed": 6400},
    {"regime": "target_count_x_confound_sweep", "regime_level": "3_low", "count": 20, "start_seed": 6500},
    {"regime": "target_count_x_confound_sweep", "regime_level": "3_medium", "count": 20, "start_seed": 6600},
    {"regime": "target_count_x_confound_sweep", "regime_level": "3_extreme", "count": 20, "start_seed": 6700},

    {"regime": "target_count_x_rule_arity_sweep", "regime_level": "0_1f", "count": 20, "start_seed": 6800},
    {"regime": "target_count_x_rule_arity_sweep", "regime_level": "0_2f", "count": 20, "start_seed": 6900},
    {"regime": "target_count_x_rule_arity_sweep", "regime_level": "0_4f", "count": 20, "start_seed": 7000},
    {"regime": "target_count_x_rule_arity_sweep", "regime_level": "3_1f", "count": 20, "start_seed": 7100},
    {"regime": "target_count_x_rule_arity_sweep", "regime_level": "3_2f", "count": 20, "start_seed": 7200},
    {"regime": "target_count_x_rule_arity_sweep", "regime_level": "3_4f", "count": 20, "start_seed": 7300},

    {"regime": "combined", "regime_level": "easy", "count": 20, "start_seed": 7400},
    {"regime": "combined", "regime_level": "medium", "count": 20, "start_seed": 7500},
    {"regime": "combined", "regime_level": "hard", "count": 20, "start_seed": 7600},
]


def main() -> None:
    out_dir = Path("artifacts/datasets/selective_attention/feature_text_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []
    for spec in SPECS:
        df = build_feature_text_dataset_v2(
            out_dir,
            regime=str(spec["regime"]),
            regime_level=str(spec["regime_level"]),
            count=int(spec["count"]),
            start_seed=int(spec.get("start_seed", 0)),
            position_mode=(
                None
                if spec.get("position_mode") is None
                else str(spec["position_mode"])
            ),
            target_count_override=(
                None
                if spec.get("target_count_override") is None
                else int(spec["target_count_override"])
            ),
        )
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(out_dir / "feature_text_v2_full.csv", index=False)

    print("=== feature_text_v2_full: by regime, regime_level ===")
    print(full_df.groupby(["regime", "regime_level"]).size())

    print("\n=== feature_text_v2_full: by position_mode ===")
    print(full_df.groupby(["position_mode"]).size())

    print("\n=== feature_text_v2_full: by target_count ===")
    print(full_df.groupby(["target_count"]).size())

    print("\n=== feature_text_v2_full: total rows ===")
    print(len(full_df))

    print("\n=== feature_text_v2_full: columns ===")
    print(sorted(full_df.columns.tolist()))

    print("\n=== feature_text_v2_full: sample ===")
    print(full_df.head())


if __name__ == "__main__":
    main()
