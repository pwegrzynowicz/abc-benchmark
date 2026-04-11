from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.datasets.build_feature_text_dataset_v2 import build_feature_text_dataset_v2


def main() -> None:
    out_dir = Path("artifacts/datasets/feature_text_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        # Baseline
        ("baseline", "baseline", "count", 20, 1000),
        ("baseline", "baseline", "filter", 20, 1100),

        # Single-factor sweeps
        ("set_size_sweep", "xs", "count", 20, 2000),
        ("set_size_sweep", "s", "count", 20, 2100),
        ("set_size_sweep", "m", "count", 20, 2200),
        ("set_size_sweep", "l", "count", 20, 2300),

        ("rule_arity_sweep", "1f", "count", 20, 3000),
        ("rule_arity_sweep", "2f", "count", 20, 3100),
        ("rule_arity_sweep", "3f", "count", 20, 3200),
        ("rule_arity_sweep", "4f", "count", 20, 3300),

        ("noise_width_sweep", "n0", "count", 20, 4000),
        ("noise_width_sweep", "n1", "count", 20, 4100),
        ("noise_width_sweep", "n2", "count", 20, 4200),

        ("confound_sweep", "low", "count", 20, 5000),
        ("confound_sweep", "medium", "count", 20, 5100),
        ("confound_sweep", "high", "count", 20, 5200),
        ("confound_sweep", "extreme", "count", 20, 5300),

        # Combined
        ("combined", "easy", "count", 20, 6000),
        ("combined", "medium", "count", 20, 6100),
        ("combined", "hard", "count", 20, 6200),

        ("combined", "easy", "filter", 20, 7000),
        ("combined", "medium", "filter", 20, 7100),
        ("combined", "hard", "filter", 20, 7200),
    ]

    dfs: list[pd.DataFrame] = []
    for regime, regime_level, response_mode, count, start_seed in specs:
        df = build_feature_text_dataset_v2(
            out_dir,
            regime=regime,
            regime_level=regime_level,
            count=count,
            response_mode=response_mode,
            start_seed=start_seed,
        )
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(out_dir / "feature_text_v2_full.csv", index=False)

    print(full_df.groupby(["regime", "regime_level", "response_mode"]).size())
    print(full_df.head())


if __name__ == "__main__":
    main()
