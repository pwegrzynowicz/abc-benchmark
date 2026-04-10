from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


def build_dataset(
    output_dir: str | Path,
    difficulty: str,
    count: int,
    make_generator: Callable[[str], object],
    row_builder: Callable[[object, str, int], dict],
    start_seed: int = 0,
    csv_name: str | None = None,
) -> pd.DataFrame:
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
        rows.append(row_builder(scene, difficulty, seed))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / (csv_name or f"{difficulty}.csv"), index=False)
    return df