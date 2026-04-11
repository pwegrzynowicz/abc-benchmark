from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.generation.feature_text_generator import make_generator


def build_feature_text_dataset(
    output_dir: str | Path,
    difficulty: str,
    count: int,
    start_seed: int = 0,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = make_generator(difficulty)
    rows: list[dict] = []

    for i in range(count):
        seed = start_seed + i
        scene = gen.generate(seed=seed, difficulty_name=difficulty)
        rows.append(
            {
                "seed": scene.seed,
                "difficulty": scene.difficulty,
                "instruction": scene.instruction,
                "text_input": scene.text_input,
                "prompt": scene.prompt,
                "gold_label": scene.gold_label,
                "target_color": scene.target_color,
                "target_shape": scene.target_shape,
                "target_marker": scene.target_marker,
                "num_records": scene.num_records,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{difficulty}.csv", index=False)
    return df