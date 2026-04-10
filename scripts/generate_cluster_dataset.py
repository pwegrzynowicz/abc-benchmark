import pandas as pd
from abc_benchmark.datasets.build_cluster_dataset import build_cluster_dataset

out_dir = "artifacts/datasets"

dfs = [
    build_cluster_dataset(out_dir, "easy", 20, 1000),
    build_cluster_dataset(out_dir, "medium", 20, 2000),
    build_cluster_dataset(out_dir, "hard", 20, 3000),
]

full_df = pd.concat(dfs, ignore_index=True)
full_df.to_csv(f"{out_dir}/cluster_count_full.csv", index=False)

print(full_df["difficulty"].value_counts())
print(full_df.head())