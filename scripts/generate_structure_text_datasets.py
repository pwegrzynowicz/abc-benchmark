from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.datasets.build_structure_text_dataset import (
    build_structure_text_dataset,
)


SPECS: list[dict[str, object]] = [
    {"regime": "baseline_structure", "regime_level": "baseline", "count": 20, "start_seed": 1000},

    {"regime": "structure_depth_sweep", "regime_level": "shallow", "count": 20, "start_seed": 2000},
    {"regime": "structure_depth_sweep", "regime_level": "medium", "count": 20, "start_seed": 2100},
    {"regime": "structure_depth_sweep", "regime_level": "nested", "count": 20, "start_seed": 2200},

    {"regime": "binding_distance_sweep", "regime_level": "near", "count": 20, "start_seed": 3000},
    {"regime": "binding_distance_sweep", "regime_level": "medium", "count": 20, "start_seed": 3100},
    {"regime": "binding_distance_sweep", "regime_level": "far", "count": 20, "start_seed": 3200},

    {"regime": "confound_sweep", "regime_level": "low", "count": 20, "start_seed": 4000},
    {"regime": "confound_sweep", "regime_level": "medium", "count": 20, "start_seed": 4100},
    {"regime": "confound_sweep", "regime_level": "high", "count": 20, "start_seed": 4200},
    {"regime": "confound_sweep", "regime_level": "extreme", "count": 20, "start_seed": 4300},

    {"regime": "target_count_x_structure_depth", "regime_level": "0_shallow", "count": 20, "start_seed": 5000},
    {"regime": "target_count_x_structure_depth", "regime_level": "0_nested", "count": 20, "start_seed": 5100},
    {"regime": "target_count_x_structure_depth", "regime_level": "3_shallow", "count": 20, "start_seed": 5200},
    {"regime": "target_count_x_structure_depth", "regime_level": "3_nested", "count": 20, "start_seed": 5300},

    {"regime": "serialization_style_sweep", "regime_level": "compact", "count": 20, "start_seed": 6000},
    {"regime": "serialization_style_sweep", "regime_level": "tagged", "count": 20, "start_seed": 6100},
    {"regime": "serialization_style_sweep", "regime_level": "nested", "count": 20, "start_seed": 6200},

    {"regime": "combined", "regime_level": "easy", "count": 20, "start_seed": 7000},
    {"regime": "combined", "regime_level": "medium", "count": 20, "start_seed": 7100},
    {"regime": "combined", "regime_level": "hard", "count": 20, "start_seed": 7200},
]


def main() -> None:
    out_dir = Path("artifacts/datasets/selective_attention/structure_text")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []
    for spec in SPECS:
        df = build_structure_text_dataset(
            out_dir,
            regime=str(spec["regime"]),
            regime_level=str(spec["regime_level"]),
            count=int(spec["count"]),
            start_seed=int(spec.get("start_seed", 0)),
            serialization_style=(
                None
                if spec.get("serialization_style") is None
                else str(spec["serialization_style"])
            ),
            target_count_override=(
                None
                if spec.get("target_count_override") is None
                else int(spec["target_count_override"])
            ),
        )
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(out_dir / "structure_text_full.csv", index=False)

    print("=== structure_text_full: by regime, regime_level ===")
    print(full_df.groupby(["regime", "regime_level"]).size())

    print("\n=== structure_text_full: by structure_type ===")
    print(full_df.groupby(["structure_type"]).size())

    print("\n=== structure_text_full: by structure_depth ===")
    print(full_df.groupby(["structure_depth"]).size())

    print("\n=== structure_text_full: by binding_distance ===")
    print(full_df.groupby(["binding_distance"]).size())

    print("\n=== structure_text_full: by serialization_style ===")
    print(full_df.groupby(["serialization_style"]).size())

    print("\n=== structure_text_full: by target_count ===")
    print(full_df.groupby(["target_count"]).size())

    print("\n=== structure_text_full: total rows ===")
    print(len(full_df))

    print("\n=== structure_text_full: columns ===")
    print(sorted(full_df.columns.tolist()))

    print("\n=== structure_text_full: sample ===")
    print(full_df.head())


if __name__ == "__main__":
    main()
