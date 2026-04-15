from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from abc_benchmark.selective_attention.feature_sensitive.visual.generator import (
    DimensionName,
    FeatureSensitiveVisualGenerator,
    FeatureSensitiveVisualScene,
    scene_to_scene_row,
)


@dataclass(frozen=True)
class VisualSliceSpec:
    dimension: DimensionName
    variant: str
    count: int
    start_seed: int


@dataclass(frozen=True)
class BuildResult:
    scenes_df: pd.DataFrame
    counting_df: pd.DataFrame
    filtering_df: pd.DataFrame


SCENES_CSV_NAME = "feature_sensitive_visual_scenes.csv"
COUNTING_CSV_NAME = "feature_sensitive_visual_counting.csv"
FILTERING_CSV_NAME = "feature_sensitive_visual_filtering.csv"


def build_feature_sensitive_visual_dataset(
    output_dir: str | Path,
    *,
    slices: Sequence[VisualSliceSpec | tuple[str, str, int, int]],
) -> BuildResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = FeatureSensitiveVisualGenerator()
    scenes: list[FeatureSensitiveVisualScene] = []
    for slice_spec in _normalize_slices(slices):
        scenes.extend(_generate_scenes_for_slice(generator, slice_spec, output_dir=output_dir))

    scenes_rows = [scene_to_scene_row(scene) for scene in scenes]
    counting_rows = [_make_task_row(scene, task_type="counting") for scene in scenes]
    filtering_rows = [_make_task_row(scene, task_type="filtering") for scene in scenes]

    scenes_df = pd.DataFrame(scenes_rows)
    counting_df = pd.DataFrame(counting_rows)
    filtering_df = pd.DataFrame(filtering_rows)

    scenes_df.to_csv(output_dir / SCENES_CSV_NAME, index=False)
    counting_df.to_csv(output_dir / COUNTING_CSV_NAME, index=False)
    filtering_df.to_csv(output_dir / FILTERING_CSV_NAME, index=False)

    return BuildResult(
        scenes_df=scenes_df,
        counting_df=counting_df,
        filtering_df=filtering_df,
    )


def _normalize_slices(
    slices: Sequence[VisualSliceSpec | tuple[str, str, int, int]],
) -> list[VisualSliceSpec]:
    normalized: list[VisualSliceSpec] = []
    for entry in slices:
        if isinstance(entry, VisualSliceSpec):
            normalized.append(entry)
            continue
        dimension, variant, count, start_seed = entry
        normalized.append(
            VisualSliceSpec(
                dimension=dimension,  # type: ignore[arg-type]
                variant=variant,
                count=count,
                start_seed=start_seed,
            )
        )
    return normalized


def _generate_scenes_for_slice(
    generator: FeatureSensitiveVisualGenerator,
    slice_spec: VisualSliceSpec,
    *,
    output_dir: Path,
) -> Iterable[FeatureSensitiveVisualScene]:
    for offset in range(slice_spec.count):
        seed = slice_spec.start_seed + offset
        scene = generator.generate(
            seed=seed,
            dimension=slice_spec.dimension,
            variant=slice_spec.variant,
        )
        _render_scene_variants(generator, scene, output_dir=output_dir)
        yield scene


def _render_scene_variants(
    generator: FeatureSensitiveVisualGenerator,
    scene: FeatureSensitiveVisualScene,
    *,
    output_dir: Path,
) -> None:
    generator.render(
        scene,
        output_dir / _scene_image_path(scene, image_family="plain"),
        show_item_ids=False,
    )
    generator.render(
        scene,
        output_dir / _scene_image_path(scene, image_family="ids"),
        show_item_ids=True,
    )


def _scene_image_path(
    scene: FeatureSensitiveVisualScene,
    *,
    image_family: str,
) -> Path:
    return Path("images") / image_family / scene.dimension / scene.variant / f"{scene.scene_id}.png"


def _make_task_row(
    scene: FeatureSensitiveVisualScene,
    *,
    task_type: str,
) -> dict[str, object]:
    row = scene_to_scene_row(scene)
    row["task_type"] = task_type

    if task_type == "counting":
        row["image_path"] = str(_scene_image_path(scene, image_family="plain"))
        row["prompt"] = scene.count_prompt
        row["label"] = scene.gold_count
        return row

    if task_type == "filtering":
        row["image_path"] = str(_scene_image_path(scene, image_family="ids"))
        row["prompt"] = scene.filter_prompt
        row["label"] = row["gold_indices"]
        return row

    raise ValueError(f"Unsupported task_type: {task_type}")
