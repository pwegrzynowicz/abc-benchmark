from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from abc_benchmark.generation.common import (
    COLORS,
    SHAPES,
    ColorName,
    GenerationError,
    ItemSpec,
    SceneSpec,
    Shape,
    distance,
    render_scene,
)


@dataclass(frozen=True)
class DifficultyConfig:
    width: int = 640
    height: int = 480
    margin: int = 36
    num_items_min: int = 10
    num_items_max: int = 16
    object_radius: int = 10
    min_object_gap: float = 6.0
    max_attempts: int = 1000
    target_count_min: int = 1
    target_count_max: int = 5
    min_same_color_distractors: int = 1
    min_same_shape_distractors: int = 1


EASY = DifficultyConfig(
    num_items_min=10,
    num_items_max=13,
    min_object_gap=8.0,
    target_count_min=1,
    target_count_max=4,
    min_same_color_distractors=1,
    min_same_shape_distractors=1,
)

MEDIUM = DifficultyConfig(
    num_items_min=13,
    num_items_max=16,
    min_object_gap=6.0,
    target_count_min=1,
    target_count_max=5,
    min_same_color_distractors=2,
    min_same_shape_distractors=2,
)

HARD = DifficultyConfig(
    num_items_min=16,
    num_items_max=20,
    min_object_gap=4.0,
    target_count_min=2,
    target_count_max=6,
    min_same_color_distractors=3,
    min_same_shape_distractors=3,
)


class FeatureConstrainedCountingGenerator:
    def __init__(self, config: DifficultyConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()

    def generate(self, seed: int | None = None, difficulty_name: str = "custom") -> SceneSpec:
        local_rng = random.Random(seed) if seed is not None else self.rng

        for _ in range(self.config.max_attempts):
            target_shape = local_rng.choice(SHAPES)
            target_color = local_rng.choice(COLORS)

            num_items = local_rng.randint(self.config.num_items_min, self.config.num_items_max)
            max_targets = min(self.config.target_count_max, num_items)
            min_targets = min(self.config.target_count_min, max_targets)
            if min_targets > max_targets:
                continue

            target_count = local_rng.randint(min_targets, max_targets)

            min_required = (
                target_count
                + self.config.min_same_color_distractors
                + self.config.min_same_shape_distractors
            )
            if min_required > num_items:
                continue

            positions = self._sample_positions(num_items, local_rng)
            if positions is None:
                continue

            items = self._assign_features(
                positions=positions,
                target_shape=target_shape,
                target_color=target_color,
                target_count=target_count,
                rng=local_rng,
            )

            if not self._passes_geometry_checks(items):
                continue
            if not self._passes_anti_shortcut_constraints(
                items=items,
                target_shape=target_shape,
                target_color=target_color,
            ):
                continue

            gold = self._count_targets(items, target_shape, target_color)
            prompt = f"Count the {target_color} {target_shape}s. Respond with a number only."

            return SceneSpec(
                seed=seed if seed is not None else -1,
                difficulty=difficulty_name,
                width=self.config.width,
                height=self.config.height,
                target_shape=target_shape,
                target_color=target_color,
                target_cluster_id=None,
                gold_label=gold,
                items=items,
                prompt=prompt,
            )

        raise GenerationError("Failed to generate a valid feature-counting scene within max_attempts")

    def render(self, scene: SceneSpec, output_path: str | Path | None = None) -> Image.Image:
        return render_scene(
            scene,
            object_radius=self.config.object_radius,
            output_path=output_path,
        )

    def generate_many(
        self,
        count: int,
        difficulty_name: str = "custom",
        start_seed: int = 0,
    ) -> list[SceneSpec]:
        return [
            self.generate(seed=start_seed + i, difficulty_name=difficulty_name)
            for i in range(count)
        ]

    def _sample_positions(
        self,
        num_items: int,
        rng: random.Random,
    ) -> list[tuple[float, float]] | None:
        positions: list[tuple[float, float]] = []
        for _ in range(num_items):
            placed = False
            for _ in range(300):
                x = rng.uniform(
                    self.config.margin + self.config.object_radius,
                    self.config.width - self.config.margin - self.config.object_radius,
                )
                y = rng.uniform(
                    self.config.margin + self.config.object_radius,
                    self.config.height - self.config.margin - self.config.object_radius - 36,
                )

                if any(
                    distance((x, y), p) < (2 * self.config.object_radius + self.config.min_object_gap)
                    for p in positions
                ):
                    continue

                positions.append((x, y))
                placed = True
                break

            if not placed:
                return None

        return positions

    def _assign_features(
        self,
        positions: list[tuple[float, float]],
        target_shape: Shape,
        target_color: ColorName,
        target_count: int,
        rng: random.Random,
    ) -> list[ItemSpec]:
        num_items = len(positions)
        indices = list(range(num_items))
        rng.shuffle(indices)

        num_same_color = self.config.min_same_color_distractors
        num_same_shape = self.config.min_same_shape_distractors
        remaining = num_items - target_count - num_same_color - num_same_shape

        exact_target_indices = indices[:target_count]
        same_color_indices = indices[target_count : target_count + num_same_color]
        same_shape_indices = indices[
            target_count + num_same_color : target_count + num_same_color + num_same_shape
        ]
        filler_indices = indices[target_count + num_same_color + num_same_shape :]

        items: list[ItemSpec | None] = [None] * num_items

        for idx in exact_target_indices:
            x, y = positions[idx]
            items[idx] = ItemSpec(
                x=x,
                y=y,
                shape=target_shape,
                color=target_color,
                cluster_id=0,
                is_anchor=False,
            )

        for idx in same_color_indices:
            x, y = positions[idx]
            wrong_shape = rng.choice([s for s in SHAPES if s != target_shape])
            items[idx] = ItemSpec(
                x=x,
                y=y,
                shape=wrong_shape,
                color=target_color,
                cluster_id=0,
                is_anchor=False,
            )

        for idx in same_shape_indices:
            x, y = positions[idx]
            wrong_color = rng.choice([c for c in COLORS if c != target_color])
            items[idx] = ItemSpec(
                x=x,
                y=y,
                shape=target_shape,
                color=wrong_color,
                cluster_id=0,
                is_anchor=False,
            )

        non_target_candidates = [
            (shape, color)
            for shape in SHAPES
            for color in COLORS
            if shape != target_shape and color != target_color
        ]

        for idx in filler_indices:
            x, y = positions[idx]
            shape, color = rng.choice(non_target_candidates)
            items[idx] = ItemSpec(
                x=x,
                y=y,
                shape=shape,
                color=color,
                cluster_id=0,
                is_anchor=False,
            )

        return [item for item in items if item is not None]

    def _passes_geometry_checks(self, items: list[ItemSpec]) -> bool:
        if any(it.is_anchor for it in items):
            return False

        for i, a in enumerate(items):
            if not self._within_bounds((a.x, a.y)):
                return False
            for b in items[i + 1 :]:
                if distance((a.x, a.y), (b.x, b.y)) < (
                    2 * self.config.object_radius + self.config.min_object_gap
                ):
                    return False

        return True

    def _passes_anti_shortcut_constraints(
        self,
        items: list[ItemSpec],
        target_shape: Shape,
        target_color: ColorName,
    ) -> bool:
        correct = self._count_targets(items, target_shape, target_color)

        same_color_wrong_shape = sum(
            1
            for it in items
            if it.color == target_color and it.shape != target_shape
        )
        same_shape_wrong_color = sum(
            1
            for it in items
            if it.shape == target_shape and it.color != target_color
        )
        unrelated = sum(
            1
            for it in items
            if it.shape != target_shape and it.color != target_color
        )

        if correct < self.config.target_count_min or correct > self.config.target_count_max:
            return False
        if same_color_wrong_shape < self.config.min_same_color_distractors:
            return False
        if same_shape_wrong_color < self.config.min_same_shape_distractors:
            return False
        if unrelated < 1:
            return False
        if correct >= len(items):
            return False

        return True

    @staticmethod
    def _count_targets(
        items: list[ItemSpec],
        target_shape: Shape,
        target_color: ColorName,
    ) -> int:
        return sum(
            1
            for it in items
            if it.shape == target_shape and it.color == target_color
        )

    def _within_bounds(self, point: tuple[float, float]) -> bool:
        x, y = point
        r = self.config.object_radius
        return (
            self.config.margin + r <= x <= self.config.width - self.config.margin - r
            and self.config.margin + r <= y <= self.config.height - self.config.margin - r - 36
        )


def make_generator(difficulty: str) -> FeatureConstrainedCountingGenerator:
    configs = {"easy": EASY, "medium": MEDIUM, "hard": HARD}
    try:
        return FeatureConstrainedCountingGenerator(configs[difficulty.lower()])
    except KeyError as exc:
        raise ValueError(f"Unknown difficulty: {difficulty}") from exc


def generate_sample_set(
    output_dir: str | Path,
    count_per_difficulty: int = 5,
    start_seed: int = 0,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seed = start_seed
    for difficulty in ("easy", "medium", "hard"):
        gen = make_generator(difficulty)
        i = 0
        attempts = 0
        while i < count_per_difficulty and attempts < count_per_difficulty * 20:
            try:
                scene = gen.generate(seed=seed, difficulty_name=difficulty)
                filename = output_path / f"{difficulty}_{i:03d}_gold_{scene.gold_label}.png"
                gen.render(scene, filename)
                i += 1
            except GenerationError:
                pass
            seed += 1
            attempts += 1

        if i < count_per_difficulty:
            raise GenerationError(
                f"Only generated {i}/{count_per_difficulty} scenes for difficulty={difficulty}"
            )