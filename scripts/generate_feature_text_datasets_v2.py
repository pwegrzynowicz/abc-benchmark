from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.datasets.build_feature_text_dataset_v2 import build_feature_text_dataset_v2


def main() -> None:
    out_dir = Path("artifacts/datasets/feature_text_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        # Baseline
        ("baseline", "baseline", 20, 1000),

        # Single-factor sweeps
        ("set_size_sweep", "xs", 20, 2000),
        ("set_size_sweep", "s", 20, 2100),
        ("set_size_sweep", "m", 20, 2200),
        ("set_size_sweep", "l", 20, 2300),

        ("rule_arity_sweep", "1f", 20, 3000),
        ("rule_arity_sweep", "2f", 20, 3100),
        ("rule_arity_sweep", "3f", 20, 3200),
        ("rule_arity_sweep", "4f", 20, 3300),

        ("noise_width_sweep", "n0", 20, 4000),
        ("noise_width_sweep", "n1", 20, 4100),
        ("noise_width_sweep", "n2", 20, 4200),

        ("confound_sweep", "low", 20, 5000),
        ("confound_sweep", "medium", 20, 5100),
        ("confound_sweep", "high", 20, 5200),
        ("confound_sweep", "extreme", 20, 5300),

        # Combined
        ("combined", "easy", 20, 6000),
        ("combined", "medium", 20, 6100),
        ("combined", "hard", 20, 6200),
    ]

    dfs: list[pd.DataFrame] = []
    for regime, regime_level, count, start_seed in specs:
        df = build_feature_text_dataset_v2(
            out_dir,
            regime=regime,
            regime_level=regime_level,
            count=count,
            start_seed=start_seed,
        )
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(out_dir / "feature_text_v2_full.csv", index=False)

    print("=== feature_text_v2_full: by regime, regime_level ===")
    print(full_df.groupby(["regime", "regime_level"]).size())

    print("\n=== feature_text_v2_full: total rows ===")
    print(len(full_df))

    print("\n=== feature_text_v2_full: sample ===")
    print(full_df.head())


if __name__ == "__main__":
    main()
