from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Literal

ColorName = Literal["red", "blue", "green", "yellow"]
Shape = Literal["circle", "square", "triangle"]
MarkerValue = Literal["0", "1"]

COLORS: tuple[ColorName, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[Shape, ...] = ("circle", "square", "triangle")
MARKERS: tuple[MarkerValue, ...] = ("0", "1")


@dataclass(frozen=True)
class DifficultyConfig:
    num_records_min: int = 12
    num_records_max: int = 18
    target_count_min: int = 1
    target_count_max: int = 4
    use_marker_in_target: bool = False
    min_same_color_wrong_shape: int = 1
    min_same_shape_wrong_color: int = 1
    min_same_color_shape_wrong_marker: int = 0
    min_unrelated: int = 2
    max_attempts: int = 1000


EASY = DifficultyConfig(
    num_records_min=12,
    num_records_max=16,
    target_count_min=1,
    target_count_max=4,
    use_marker_in_target=False,
    min_same_color_wrong_shape=1,
    min_same_shape_wrong_color=1,
    min_same_color_shape_wrong_marker=0,
    min_unrelated=2,
)

MEDIUM = DifficultyConfig(
    num_records_min=16,
    num_records_max=22,
    target_count_min=1,
    target_count_max=5,
    use_marker_in_target=True,
    min_same_color_wrong_shape=2,
    min_same_shape_wrong_color=2,
    min_same_color_shape_wrong_marker=1,
    min_unrelated=2,
)

HARD = DifficultyConfig(
    num_records_min=22,
    num_records_max=30,
    target_count_min=2,
    target_count_max=6,
    use_marker_in_target=True,
    min_same_color_wrong_shape=3,
    min_same_shape_wrong_color=3,
    min_same_color_shape_wrong_marker=2,
    min_unrelated=3,
)


@dataclass(frozen=True)
class RecordSpec:
    color: ColorName
    shape: Shape
    marker: MarkerValue

    def to_text(self) -> str:
        return f"{self.color} | {self.shape} | {self.marker}"


@dataclass
class TextSceneSpec:
    seed: int
    difficulty: str
    instruction: str
    text_input: str
    prompt: str
    gold_label: int
    target_color: ColorName
    target_shape: Shape
    target_marker: MarkerValue | None
    num_records: int
    records: list[RecordSpec]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["records"] = [asdict(r) for r in self.records]
        return payload


class GenerationError(RuntimeError):
    pass


class FeatureTextSelectiveAttentionGenerator:
    def __init__(self, config: DifficultyConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()

    def generate(self, seed: int | None = None, difficulty_name: str = "custom") -> TextSceneSpec:
        local_rng = random.Random(seed) if seed is not None else self.rng

        for _ in range(self.config.max_attempts):
            target_color = local_rng.choice(COLORS)
            target_shape = local_rng.choice(SHAPES)
            target_marker: MarkerValue | None = (
                local_rng.choice(MARKERS) if self.config.use_marker_in_target else None
            )

            num_records = local_rng.randint(
                self.config.num_records_min,
                self.config.num_records_max,
            )

            max_targets = min(self.config.target_count_max, num_records)
            min_targets = min(self.config.target_count_min, max_targets)
            if min_targets > max_targets:
                continue

            target_count = local_rng.randint(min_targets, max_targets)

            min_required = (
                target_count
                + self.config.min_same_color_wrong_shape
                + self.config.min_same_shape_wrong_color
                + self.config.min_same_color_shape_wrong_marker
                + self.config.min_unrelated
            )
            if min_required > num_records:
                continue

            records = self._build_records(
                rng=local_rng,
                num_records=num_records,
                target_color=target_color,
                target_shape=target_shape,
                target_marker=target_marker,
                target_count=target_count,
            )

            if not self._passes_anti_shortcut_constraints(
                records=records,
                target_color=target_color,
                target_shape=target_shape,
                target_marker=target_marker,
            ):
                continue

            instruction = self._build_instruction(
                target_color=target_color,
                target_shape=target_shape,
                target_marker=target_marker,
            )
            text_input = "\n".join(record.to_text() for record in records)
            prompt = f"{instruction}\n\n{text_input}"
            gold_label = self._count_targets(
                records=records,
                target_color=target_color,
                target_shape=target_shape,
                target_marker=target_marker,
            )

            return TextSceneSpec(
                seed=seed if seed is not None else -1,
                difficulty=difficulty_name,
                instruction=instruction,
                text_input=text_input,
                prompt=prompt,
                gold_label=gold_label,
                target_color=target_color,
                target_shape=target_shape,
                target_marker=target_marker,
                num_records=len(records),
                records=records,
            )

        raise GenerationError("Failed to generate a valid text selective-attention scene within max_attempts")

    def generate_many(
        self,
        count: int,
        difficulty_name: str = "custom",
        start_seed: int = 0,
    ) -> list[TextSceneSpec]:
        return [
            self.generate(seed=start_seed + i, difficulty_name=difficulty_name)
            for i in range(count)
        ]

    def _build_records(
        self,
        rng: random.Random,
        num_records: int,
        target_color: ColorName,
        target_shape: Shape,
        target_marker: MarkerValue | None,
        target_count: int,
    ) -> list[RecordSpec]:
        records: list[RecordSpec] = []

        for _ in range(target_count):
            records.append(
                RecordSpec(
                    color=target_color,
                    shape=target_shape,
                    marker=target_marker if target_marker is not None else rng.choice(MARKERS),
                )
            )

        for _ in range(self.config.min_same_color_wrong_shape):
            wrong_shape = rng.choice([shape for shape in SHAPES if shape != target_shape])
            marker = (
                rng.choice(MARKERS)
                if target_marker is None
                else rng.choice(MARKERS)
            )
            records.append(
                RecordSpec(
                    color=target_color,
                    shape=wrong_shape,
                    marker=marker,
                )
            )

        for _ in range(self.config.min_same_shape_wrong_color):
            wrong_color = rng.choice([color for color in COLORS if color != target_color])
            marker = (
                rng.choice(MARKERS)
                if target_marker is None
                else rng.choice(MARKERS)
            )
            records.append(
                RecordSpec(
                    color=wrong_color,
                    shape=target_shape,
                    marker=marker,
                )
            )

        for _ in range(self.config.min_same_color_shape_wrong_marker):
            if target_marker is None:
                break
            wrong_marker = "1" if target_marker == "0" else "0"
            records.append(
                RecordSpec(
                    color=target_color,
                    shape=target_shape,
                    marker=wrong_marker,
                )
            )

        remaining = num_records - len(records)
        filler_candidates = self._filler_candidates(
            target_color=target_color,
            target_shape=target_shape,
            target_marker=target_marker,
        )
        for _ in range(remaining):
            records.append(rng.choice(filler_candidates))

        rng.shuffle(records)
        return records

    def _filler_candidates(
        self,
        target_color: ColorName,
        target_shape: Shape,
        target_marker: MarkerValue | None,
    ) -> list[RecordSpec]:
        candidates: list[RecordSpec] = []
        for color in COLORS:
            for shape in SHAPES:
                for marker in MARKERS:
                    record = RecordSpec(color=color, shape=shape, marker=marker)
                    if self._matches_target(
                        record=record,
                        target_color=target_color,
                        target_shape=target_shape,
                        target_marker=target_marker,
                    ):
                        continue
                    candidates.append(record)
        return candidates

    def _passes_anti_shortcut_constraints(
        self,
        records: list[RecordSpec],
        target_color: ColorName,
        target_shape: Shape,
        target_marker: MarkerValue | None,
    ) -> bool:
        correct = self._count_targets(
            records=records,
            target_color=target_color,
            target_shape=target_shape,
            target_marker=target_marker,
        )

        same_color_wrong_shape = sum(
            1
            for record in records
            if record.color == target_color
            and record.shape != target_shape
        )

        same_shape_wrong_color = sum(
            1
            for record in records
            if record.shape == target_shape
            and record.color != target_color
        )

        same_color_shape_wrong_marker = 0
        if target_marker is not None:
            same_color_shape_wrong_marker = sum(
                1
                for record in records
                if record.color == target_color
                and record.shape == target_shape
                and record.marker != target_marker
            )

        unrelated = sum(
            1
            for record in records
            if record.color != target_color
            and record.shape != target_shape
        )

        if correct < self.config.target_count_min or correct > self.config.target_count_max:
            return False
        if same_color_wrong_shape < self.config.min_same_color_wrong_shape:
            return False
        if same_shape_wrong_color < self.config.min_same_shape_wrong_color:
            return False
        if target_marker is not None and same_color_shape_wrong_marker < self.config.min_same_color_shape_wrong_marker:
            return False
        if unrelated < self.config.min_unrelated:
            return False
        if correct >= len(records):
            return False

        return True

    def _build_instruction(
        self,
        target_color: ColorName,
        target_shape: Shape,
        target_marker: MarkerValue | None,
    ) -> str:
        if target_marker is None:
            return (
                f"Count the entries that are {target_color} {target_shape}s. "
                "Respond with a number only."
            )
        return (
            f"Count the entries that are {target_color} {target_shape}s with marker={target_marker}. "
            "Respond with a number only."
        )

    @staticmethod
    def _matches_target(
        record: RecordSpec,
        target_color: ColorName,
        target_shape: Shape,
        target_marker: MarkerValue | None,
    ) -> bool:
        if record.color != target_color:
            return False
        if record.shape != target_shape:
            return False
        if target_marker is not None and record.marker != target_marker:
            return False
        return True

    def _count_targets(
        self,
        records: list[RecordSpec],
        target_color: ColorName,
        target_shape: Shape,
        target_marker: MarkerValue | None,
    ) -> int:
        return sum(
            1
            for record in records
            if self._matches_target(
                record=record,
                target_color=target_color,
                target_shape=target_shape,
                target_marker=target_marker,
            )
        )


def make_generator(difficulty: str) -> FeatureTextSelectiveAttentionGenerator:
    configs = {
        "easy": EASY,
        "medium": MEDIUM,
        "hard": HARD,
    }
    try:
        return FeatureTextSelectiveAttentionGenerator(configs[difficulty.lower()])
    except KeyError as exc:
        raise ValueError(f"Unknown difficulty: {difficulty}") from exc