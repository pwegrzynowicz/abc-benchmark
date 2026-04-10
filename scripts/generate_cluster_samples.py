from pathlib import Path

from abc_benchmark.generation.cluster_generator import generate_sample_set

if __name__ == "__main__":
    generate_sample_set(Path("artifacts/sample_scenes"), count_per_difficulty=50, start_seed=100)
