import pandas as pd

from abc_benchmark.datasets.build_feature_text_dataset import build_feature_text_dataset

out_dir = "artifacts/datasets/feature_text"

dfs = [
    build_feature_text_dataset(out_dir, "easy", 20, start_seed=1000),
    build_feature_text_dataset(out_dir, "medium", 20, start_seed=2000),
    build_feature_text_dataset(out_dir, "hard", 20, start_seed=3000),
]

full_df = pd.concat(dfs, ignore_index=True)
full_df.to_csv(f"{out_dir}/feature_count_full.csv", index=False)

print(full_df["difficulty"].value_counts())
print(full_df.head())
