from __future__ import annotations

from pathlib import Path

import pandas as pd

from abc_benchmark.generation.cluster_generator import make_generator


def build_cluster_dataset(output_dir: str | Path, difficulty: str, count: int, start_seed: int = 0) -> pd.DataFrame:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images" / difficulty
    image_dir.mkdir(parents=True, exist_ok=True)

    gen = make_generator(difficulty)
    rows = []

    for i in range(count):
        seed = start_seed + i
        scene = gen.generate(seed=seed, difficulty_name=difficulty)
        image_path = image_dir / f"{difficulty}_{seed}.png"
        gen.render(scene, image_path)
        rows.append(
            {
                "seed": scene.seed,
                "difficulty": scene.difficulty,
                "prompt": scene.prompt,
                "image_path": str(Path("images") / difficulty / f"{difficulty}_{seed}.png"),
                "gold_label": scene.gold_label,
                "target_shape": scene.target_shape,
                "target_color": scene.target_color,
                "target_cluster_id": scene.target_cluster_id,
                "num_items": len(scene.items),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{difficulty}.csv", index=False)
    return df
