
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from typing import Literal

ColorName = Literal["red", "blue", "green", "yellow"]
Shape = Literal["circle", "square", "triangle"]
MarkerValue = Literal["0", "1"]
SizeValue = Literal["small", "medium", "large"]
PatternValue = Literal["solid", "striped", "dotted"]
ZoneValue = Literal["A", "B", "C", "D"]
CodeValue = Literal["X1", "X2", "X3", "X4"]

RegimeName = Literal[
    "baseline",
    "set_size_sweep",
    "rule_arity_sweep",
    "noise_width_sweep",
    "confound_sweep",
    "position_sweep",
    "target_count_sweep",
    "target_count_x_confound_sweep",
    "target_count_x_rule_arity_sweep",
    "combined",
]
PositionMode = Literal["random", "front_loaded", "back_loaded", "clustered"]

COLORS: tuple[ColorName, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[Shape, ...] = ("circle", "square", "triangle")
MARKERS: tuple[MarkerValue, ...] = ("0", "1")
SIZES: tuple[SizeValue, ...] = ("small", "medium", "large")
PATTERNS: tuple[PatternValue, ...] = ("solid", "striped", "dotted")
ZONES: tuple[ZoneValue, ...] = ("A", "B", "C", "D")
CODES: tuple[CodeValue, ...] = ("X1", "X2", "X3", "X4")

FIELD_VALUES: dict[str, tuple[str, ...]] = {
    "color": COLORS,
    "shape": SHAPES,
    "marker": MARKERS,
    "size": SIZES,
    "pattern": PATTERNS,
    "zone": ZONES,
    "code": CODES,
}

FIELD_ORDER: tuple[str, ...] = (
    "color",
    "shape",
    "marker",
    "size",
    "pattern",
    "zone",
    "code",
)


@dataclass(frozen=True)
class FeatureTextFactors:
    regime: RegimeName
    regime_level: str
    num_records: int
    target_feature_count: int
    target_count: int
    active_fields: tuple[str, ...]
    target_fields: tuple[str, ...]
    irrelevant_noise_fields: tuple[str, ...]
    same_color_wrong_shape_count: int
    same_shape_wrong_color_count: int
    same_core_wrong_marker_count: int
    same_core_wrong_size_count: int
    same_core_wrong_pattern_count: int
    unrelated_count: int


@dataclass(frozen=True)
class RecordSpec:
    color: ColorName
    shape: Shape
    marker: MarkerValue
    size: SizeValue
    pattern: PatternValue
    zone: ZoneValue
    code: CodeValue

    def value(self, field: str) -> str:
        return str(getattr(self, field))

    def render(self, active_fields: tuple[str, ...]) -> str:
        return " | ".join(self.value(field) for field in active_fields)


@dataclass
class TextSceneSpec:
    seed: int
    regime: RegimeName
    regime_level: str
    position_mode: PositionMode

    count_instruction: str
    filter_instruction: str
    text_input: str
    count_prompt: str
    filter_prompt: str

    gold_count: int
    gold_lines: list[int]

    target_definition: dict[str, str]
    factors: FeatureTextFactors
    records: list[RecordSpec]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["records"] = [asdict(r) for r in self.records]
        payload["target_definition"] = dict(self.target_definition)
        return payload


class GenerationError(RuntimeError):
    pass


class FeatureTextSelectiveAttentionGeneratorV2:
    def __init__(
        self,
        rng: random.Random | None = None,
        max_attempts: int = 1000,
    ) -> None:
        self.rng = rng or random.Random()
        self.max_attempts = max_attempts

    def generate(
        self,
        *,
        seed: int | None = None,
        factors: FeatureTextFactors | None = None,
        regime: RegimeName = "combined",
        regime_level: str = "medium",
        position_mode: PositionMode | None = None,
        target_count_override: int | None = None,
    ) -> TextSceneSpec:
        local_rng = random.Random(seed) if seed is not None else self.rng

        if factors is None:
            factors = self.sample_factors(
                rng=local_rng,
                regime=regime,
                level=regime_level,
            )

        if target_count_override is not None:
            factors = self._override_target_count(factors, target_count_override)

        resolved_position_mode = self._resolve_position_mode(regime, regime_level, position_mode)

        for _ in range(self.max_attempts):
            target_definition = self._sample_target_spec(local_rng, factors.target_fields)
            records = self._build_records(local_rng, factors, target_definition)
            records = self._arrange_records(
                local_rng,
                records,
                factors,
                target_definition,
                resolved_position_mode,
            )

            if not self._passes_constraints(records, factors, target_definition):
                continue

            text_input = "\n".join(record.render(factors.active_fields) for record in records)
            count_instruction = self._build_count_instruction(factors.target_fields, target_definition)
            filter_instruction = self._build_filter_instruction(factors.target_fields, target_definition)
            count_prompt = f"{count_instruction}\n\n{text_input}"
            filter_prompt = f"{filter_instruction}\n\n{text_input}"
            gold_lines = self._matching_line_numbers(records, factors.target_fields, target_definition)
            gold_count = len(gold_lines)

            return TextSceneSpec(
                seed=seed if seed is not None else -1,
                regime=factors.regime,
                regime_level=factors.regime_level,
                position_mode=resolved_position_mode,
                count_instruction=count_instruction,
                filter_instruction=filter_instruction,
                text_input=text_input,
                count_prompt=count_prompt,
                filter_prompt=filter_prompt,
                gold_count=gold_count,
                gold_lines=gold_lines,
                target_definition=target_definition,
                factors=factors,
                records=records,
            )

        raise GenerationError("Failed to generate a valid feature-text scene within max_attempts")

    def generate_many(
        self,
        *,
        count: int,
        start_seed: int = 0,
        regime: RegimeName = "combined",
        regime_level: str = "medium",
        position_mode: PositionMode | None = None,
        target_count_override: int | None = None,
    ) -> list[TextSceneSpec]:
        return [
            self.generate(
                seed=start_seed + i,
                regime=regime,
                regime_level=regime_level,
                position_mode=position_mode,
                target_count_override=target_count_override,
            )
            for i in range(count)
        ]

    def sample_factors(
        self,
        *,
        rng: random.Random,
        regime: RegimeName,
        level: str,
    ) -> FeatureTextFactors:
        if regime == "baseline":
            return self._sample_baseline_factors(rng)
        if regime == "set_size_sweep":
            return self._sample_set_size_sweep_factors(rng, level)
        if regime == "rule_arity_sweep":
            return self._sample_rule_arity_sweep_factors(rng, level)
        if regime == "noise_width_sweep":
            return self._sample_noise_width_sweep_factors(rng, level)
        if regime == "confound_sweep":
            return self._sample_confound_sweep_factors(rng, level)
        if regime == "position_sweep":
            return self._sample_position_sweep_factors(rng, level)
        if regime == "target_count_sweep":
            return self._sample_target_count_sweep_factors(rng, level)
        if regime == "target_count_x_confound_sweep":
            return self._sample_target_count_x_confound_sweep_factors(rng, level)
        if regime == "target_count_x_rule_arity_sweep":
            return self._sample_target_count_x_rule_arity_sweep_factors(rng, level)
        if regime == "combined":
            return self._sample_combined_factors(rng, level)
        raise ValueError(f"Unknown regime: {regime}")

    def _sample_baseline_factors(self, rng: random.Random) -> FeatureTextFactors:
        num_records = 8
        target_count = rng.randint(1, 2)
        return FeatureTextFactors(
            regime="baseline",
            regime_level="baseline",
            num_records=num_records,
            target_feature_count=2,
            target_count=target_count,
            active_fields=("color", "shape", "marker"),
            target_fields=("color", "shape"),
            irrelevant_noise_fields=("marker",),
            same_color_wrong_shape_count=1,
            same_shape_wrong_color_count=1,
            same_core_wrong_marker_count=0,
            same_core_wrong_size_count=0,
            same_core_wrong_pattern_count=0,
            unrelated_count=max(2, num_records - target_count - 2),
        )

    def _sample_set_size_sweep_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        mapping = {"xs": 8, "s": 16, "m": 24, "l": 32}
        if level not in mapping:
            raise ValueError(f"Unknown set_size_sweep level: {level}")
        num_records = mapping[level]
        target_count = max(1, min(4, num_records // 8))
        unrelated = max(2, num_records - target_count - 2)
        return FeatureTextFactors(
            regime="set_size_sweep",
            regime_level=level,
            num_records=num_records,
            target_feature_count=2,
            target_count=target_count,
            active_fields=("color", "shape", "marker"),
            target_fields=("color", "shape"),
            irrelevant_noise_fields=("marker",),
            same_color_wrong_shape_count=1,
            same_shape_wrong_color_count=1,
            same_core_wrong_marker_count=0,
            same_core_wrong_size_count=0,
            same_core_wrong_pattern_count=0,
            unrelated_count=unrelated,
        )

    def _sample_rule_arity_sweep_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        mapping = {
            "1f": ("color",),
            "2f": ("color", "shape"),
            "3f": ("color", "shape", "marker"),
            "4f": ("color", "shape", "marker", "size"),
        }
        if level not in mapping:
            raise ValueError(f"Unknown rule_arity_sweep level: {level}")
        target_fields = mapping[level]
        active_fields = ("color", "shape", "marker", "size", "pattern")
        irrelevant_noise_fields = tuple(f for f in active_fields if f not in target_fields)
        num_records = 20
        target_count = 3
        scws = 2 if {"color", "shape"}.issubset(target_fields) else 0
        sswc = 2 if {"color", "shape"}.issubset(target_fields) else 0
        scm = 1 if "marker" in target_fields else 0
        scs = 1 if "size" in target_fields else 0
        scp = 1 if "pattern" in target_fields else 0
        unrelated_count = num_records - target_count - scws - sswc - scm - scs - scp
        return FeatureTextFactors(
            regime="rule_arity_sweep",
            regime_level=level,
            num_records=num_records,
            target_feature_count=len(target_fields),
            target_count=target_count,
            active_fields=active_fields,
            target_fields=target_fields,
            irrelevant_noise_fields=irrelevant_noise_fields,
            same_color_wrong_shape_count=scws,
            same_shape_wrong_color_count=sswc,
            same_core_wrong_marker_count=scm,
            same_core_wrong_size_count=scs,
            same_core_wrong_pattern_count=scp,
            unrelated_count=unrelated_count,
        )

    def _sample_noise_width_sweep_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        mapping = {
            "n0": ("color", "shape", "marker"),
            "n1": ("color", "shape", "marker", "size", "pattern"),
            "n2": ("color", "shape", "marker", "size", "pattern", "zone", "code"),
        }
        if level not in mapping:
            raise ValueError(f"Unknown noise_width_sweep level: {level}")
        active_fields = mapping[level]
        target_fields = ("color", "shape")
        irrelevant_noise_fields = tuple(f for f in active_fields if f not in target_fields)
        num_records = 20
        target_count = 3
        return FeatureTextFactors(
            regime="noise_width_sweep",
            regime_level=level,
            num_records=num_records,
            target_feature_count=2,
            target_count=target_count,
            active_fields=active_fields,
            target_fields=target_fields,
            irrelevant_noise_fields=irrelevant_noise_fields,
            same_color_wrong_shape_count=2,
            same_shape_wrong_color_count=2,
            same_core_wrong_marker_count=0,
            same_core_wrong_size_count=0,
            same_core_wrong_pattern_count=0,
            unrelated_count=num_records - target_count - 4,
        )

    def _sample_confound_sweep_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        mapping = {
            "low": (1, 1, 1, 1, 0, 16),
            "medium": (2, 2, 2, 2, 0, 20),
            "high": (4, 4, 3, 3, 0, 28),
            "extreme": (6, 6, 4, 4, 0, 36),
        }
        if level not in mapping:
            raise ValueError(f"Unknown confound_sweep level: {level}")
        scws, sswc, scm, scs, scp, num_records = mapping[level]
        target_count = 3
        unrelated_count = num_records - target_count - scws - sswc - scm - scs - scp
        return FeatureTextFactors(
            regime="confound_sweep",
            regime_level=level,
            num_records=num_records,
            target_feature_count=4,
            target_count=target_count,
            active_fields=("color", "shape", "marker", "size", "pattern"),
            target_fields=("color", "shape", "marker", "size"),
            irrelevant_noise_fields=("pattern",),
            same_color_wrong_shape_count=scws,
            same_shape_wrong_color_count=sswc,
            same_core_wrong_marker_count=scm,
            same_core_wrong_size_count=scs,
            same_core_wrong_pattern_count=scp,
            unrelated_count=max(2, unrelated_count),
        )

    def _sample_position_sweep_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        if level not in {"random", "front_loaded", "back_loaded", "clustered"}:
            raise ValueError(f"Unknown position_sweep level: {level}")
        num_records = 20
        target_count = 3
        return FeatureTextFactors(
            regime="position_sweep",
            regime_level=level,
            num_records=num_records,
            target_feature_count=2,
            target_count=target_count,
            active_fields=("color", "shape", "marker", "size"),
            target_fields=("color", "shape"),
            irrelevant_noise_fields=("marker", "size"),
            same_color_wrong_shape_count=3,
            same_shape_wrong_color_count=3,
            same_core_wrong_marker_count=0,
            same_core_wrong_size_count=0,
            same_core_wrong_pattern_count=0,
            unrelated_count=11,
        )


    def _sample_target_count_sweep_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        mapping = {
            "0": 0,
            "1": 1,
            "3": 3,
            "5": 5,
        }
        if level not in mapping:
            raise ValueError(f"Unknown target_count_sweep level: {level}")

        target_count = mapping[level]
        num_records = 20
        same_color_wrong_shape_count = 3
        same_shape_wrong_color_count = 3
        unrelated_count = num_records - target_count - same_color_wrong_shape_count - same_shape_wrong_color_count

        if unrelated_count < 2:
            raise GenerationError("target_count_sweep configuration leaves too few unrelated records")

        return FeatureTextFactors(
            regime="target_count_sweep",
            regime_level=level,
            num_records=num_records,
            target_feature_count=2,
            target_count=target_count,
            active_fields=("color", "shape", "marker", "size"),
            target_fields=("color", "shape"),
            irrelevant_noise_fields=("marker", "size"),
            same_color_wrong_shape_count=same_color_wrong_shape_count,
            same_shape_wrong_color_count=same_shape_wrong_color_count,
            same_core_wrong_marker_count=0,
            same_core_wrong_size_count=0,
            same_core_wrong_pattern_count=0,
            unrelated_count=unrelated_count,
        )


    def _sample_target_count_x_confound_sweep_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        mapping = {
            "0_low": (0, 1, 1, 1, 1, 16),
            "0_medium": (0, 2, 2, 2, 2, 20),
            "0_extreme": (0, 6, 6, 4, 4, 36),
            "3_low": (3, 1, 1, 1, 1, 16),
            "3_medium": (3, 2, 2, 2, 2, 20),
            "3_extreme": (3, 6, 6, 4, 4, 36),
        }
        if level not in mapping:
            raise ValueError(f"Unknown target_count_x_confound_sweep level: {level}")

        target_count, scws, sswc, scm, scs, num_records = mapping[level]
        unrelated_count = num_records - target_count - scws - sswc - scm - scs
        if unrelated_count < 2:
            raise GenerationError("target_count_x_confound_sweep configuration leaves too few unrelated records")

        return FeatureTextFactors(
            regime="target_count_x_confound_sweep",
            regime_level=level,
            num_records=num_records,
            target_feature_count=4,
            target_count=target_count,
            active_fields=("color", "shape", "marker", "size", "pattern"),
            target_fields=("color", "shape", "marker", "size"),
            irrelevant_noise_fields=("pattern",),
            same_color_wrong_shape_count=scws,
            same_shape_wrong_color_count=sswc,
            same_core_wrong_marker_count=scm,
            same_core_wrong_size_count=scs,
            same_core_wrong_pattern_count=0,
            unrelated_count=unrelated_count,
        )

    def _sample_target_count_x_rule_arity_sweep_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        mapping = {
            "0_1f": (0, ("color",)),
            "0_2f": (0, ("color", "shape")),
            "0_4f": (0, ("color", "shape", "marker", "size")),
            "3_1f": (3, ("color",)),
            "3_2f": (3, ("color", "shape")),
            "3_4f": (3, ("color", "shape", "marker", "size")),
        }
        if level not in mapping:
            raise ValueError(f"Unknown target_count_x_rule_arity_sweep level: {level}")

        target_count, target_fields = mapping[level]
        active_fields = ("color", "shape", "marker", "size", "pattern")
        irrelevant_noise_fields = tuple(f for f in active_fields if f not in target_fields)
        num_records = 20

        same_color_wrong_shape_count = 2 if {"color", "shape"}.issubset(target_fields) else 0
        same_shape_wrong_color_count = 2 if {"color", "shape"}.issubset(target_fields) else 0
        same_core_wrong_marker_count = 1 if "marker" in target_fields else 0
        same_core_wrong_size_count = 1 if "size" in target_fields else 0
        same_core_wrong_pattern_count = 0

        unrelated_count = (
            num_records
            - target_count
            - same_color_wrong_shape_count
            - same_shape_wrong_color_count
            - same_core_wrong_marker_count
            - same_core_wrong_size_count
            - same_core_wrong_pattern_count
        )
        if unrelated_count < 2:
            raise GenerationError("target_count_x_rule_arity_sweep configuration leaves too few unrelated records")

        return FeatureTextFactors(
            regime="target_count_x_rule_arity_sweep",
            regime_level=level,
            num_records=num_records,
            target_feature_count=len(target_fields),
            target_count=target_count,
            active_fields=active_fields,
            target_fields=target_fields,
            irrelevant_noise_fields=irrelevant_noise_fields,
            same_color_wrong_shape_count=same_color_wrong_shape_count,
            same_shape_wrong_color_count=same_shape_wrong_color_count,
            same_core_wrong_marker_count=same_core_wrong_marker_count,
            same_core_wrong_size_count=same_core_wrong_size_count,
            same_core_wrong_pattern_count=same_core_wrong_pattern_count,
            unrelated_count=unrelated_count,
        )

    def _sample_combined_factors(self, rng: random.Random, level: str) -> FeatureTextFactors:
        if level == "easy":
            return FeatureTextFactors(
                regime="combined",
                regime_level="easy",
                num_records=16,
                target_feature_count=2,
                target_count=3,
                active_fields=("color", "shape", "marker", "size"),
                target_fields=("color", "shape"),
                irrelevant_noise_fields=("marker", "size"),
                same_color_wrong_shape_count=2,
                same_shape_wrong_color_count=2,
                same_core_wrong_marker_count=0,
                same_core_wrong_size_count=0,
                same_core_wrong_pattern_count=0,
                unrelated_count=9,
            )
        if level == "medium":
            return FeatureTextFactors(
                regime="combined",
                regime_level="medium",
                num_records=24,
                target_feature_count=3,
                target_count=4,
                active_fields=("color", "shape", "marker", "size", "pattern"),
                target_fields=("color", "shape", "marker"),
                irrelevant_noise_fields=("size", "pattern"),
                same_color_wrong_shape_count=3,
                same_shape_wrong_color_count=3,
                same_core_wrong_marker_count=2,
                same_core_wrong_size_count=0,
                same_core_wrong_pattern_count=0,
                unrelated_count=12,
            )
        if level == "hard":
            return FeatureTextFactors(
                regime="combined",
                regime_level="hard",
                num_records=32,
                target_feature_count=4,
                target_count=5,
                active_fields=("color", "shape", "marker", "size", "pattern", "zone", "code"),
                target_fields=("color", "shape", "marker", "size"),
                irrelevant_noise_fields=("pattern", "zone", "code"),
                same_color_wrong_shape_count=5,
                same_shape_wrong_color_count=5,
                same_core_wrong_marker_count=3,
                same_core_wrong_size_count=3,
                same_core_wrong_pattern_count=0,
                unrelated_count=9,
            )
        raise ValueError(f"Unknown combined level: {level}")

    def _resolve_position_mode(
        self,
        regime: RegimeName,
        regime_level: str,
        position_mode: PositionMode | None,
    ) -> PositionMode:
        if position_mode is not None:
            return position_mode
        if regime == "position_sweep":
            return regime_level  # type: ignore[return-value]
        return "random"

    def _override_target_count(self, factors: FeatureTextFactors, target_count_override: int) -> FeatureTextFactors:
        if target_count_override < 0:
            raise ValueError("target_count_override must be >= 0")
        delta = target_count_override - factors.target_count
        return replace(
            factors,
            target_count=target_count_override,
            num_records=factors.num_records + max(0, delta),
            unrelated_count=factors.unrelated_count + max(0, -delta),
        )

    def _sample_target_spec(self, rng: random.Random, target_fields: tuple[str, ...]) -> dict[str, str]:
        return {field: rng.choice(FIELD_VALUES[field]) for field in target_fields}

    def _build_records(self, rng: random.Random, factors: FeatureTextFactors, target_definition: dict[str, str]) -> list[RecordSpec]:
        records: list[RecordSpec] = []
        records.extend(self._build_exact_targets(rng, factors, target_definition))
        records.extend(self._build_same_color_wrong_shape(rng, factors, target_definition))
        records.extend(self._build_same_shape_wrong_color(rng, factors, target_definition))
        records.extend(self._build_same_core_wrong_field(rng, factors, target_definition, field="marker", count=factors.same_core_wrong_marker_count))
        records.extend(self._build_same_core_wrong_field(rng, factors, target_definition, field="size", count=factors.same_core_wrong_size_count))
        records.extend(self._build_same_core_wrong_field(rng, factors, target_definition, field="pattern", count=factors.same_core_wrong_pattern_count))

        remaining = factors.num_records - len(records)
        if remaining < factors.unrelated_count:
            raise GenerationError("Factor bundle over-allocates records before unrelated filler")

        records.extend(self._build_unrelated_records(rng, factors, target_definition, factors.unrelated_count))
        remaining = factors.num_records - len(records)
        if remaining > 0:
            records.extend(self._build_extra_filler(rng, factors, target_definition, remaining))
        return records

    def _arrange_records(
        self,
        rng: random.Random,
        records: list[RecordSpec],
        factors: FeatureTextFactors,
        target_definition: dict[str, str],
        position_mode: PositionMode,
    ) -> list[RecordSpec]:
        arranged = list(records)
        if position_mode == "random" or factors.target_count == 0:
            rng.shuffle(arranged)
            return arranged

        targets = [r for r in arranged if self._matches_target(r, factors.target_fields, target_definition)]
        non_targets = [r for r in arranged if not self._matches_target(r, factors.target_fields, target_definition)]
        rng.shuffle(targets)
        rng.shuffle(non_targets)

        if position_mode == "front_loaded":
            return targets + non_targets
        if position_mode == "back_loaded":
            return non_targets + targets
        if position_mode == "clustered":
            start = 0 if not non_targets else rng.randint(0, len(non_targets))
            return non_targets[:start] + targets + non_targets[start:]

        rng.shuffle(arranged)
        return arranged

    def _build_exact_targets(self, rng: random.Random, factors: FeatureTextFactors, target_definition: dict[str, str]) -> list[RecordSpec]:
        return [self._random_record(rng, factors.active_fields, fixed=target_definition) for _ in range(factors.target_count)]

    def _build_same_color_wrong_shape(self, rng: random.Random, factors: FeatureTextFactors, target_definition: dict[str, str]) -> list[RecordSpec]:
        count = factors.same_color_wrong_shape_count
        if count <= 0 or "color" not in factors.target_fields or "shape" not in factors.target_fields:
            return []
        out = []
        for _ in range(count):
            fixed = {"color": target_definition["color"]}
            fixed["shape"] = rng.choice([x for x in SHAPES if x != target_definition["shape"]])
            if "marker" in factors.target_fields:
                fixed["marker"] = target_definition["marker"]
            if "size" in factors.target_fields:
                fixed["size"] = target_definition["size"]
            if "pattern" in factors.target_fields:
                fixed["pattern"] = target_definition["pattern"]
            out.append(self._random_record(rng, factors.active_fields, fixed=fixed))
        return out

    def _build_same_shape_wrong_color(self, rng: random.Random, factors: FeatureTextFactors, target_definition: dict[str, str]) -> list[RecordSpec]:
        count = factors.same_shape_wrong_color_count
        if count <= 0 or "shape" not in factors.target_fields or "color" not in factors.target_fields:
            return []
        out = []
        for _ in range(count):
            fixed = {"shape": target_definition["shape"]}
            fixed["color"] = rng.choice([x for x in COLORS if x != target_definition["color"]])
            if "marker" in factors.target_fields:
                fixed["marker"] = target_definition["marker"]
            if "size" in factors.target_fields:
                fixed["size"] = target_definition["size"]
            if "pattern" in factors.target_fields:
                fixed["pattern"] = target_definition["pattern"]
            out.append(self._random_record(rng, factors.active_fields, fixed=fixed))
        return out

    def _build_same_core_wrong_field(self, rng: random.Random, factors: FeatureTextFactors, target_definition: dict[str, str], *, field: str, count: int) -> list[RecordSpec]:
        if count <= 0 or field not in factors.target_fields:
            return []
        out = []
        for _ in range(count):
            fixed = {}
            for target_field in factors.target_fields:
                if target_field == field:
                    continue
                fixed[target_field] = target_definition[target_field]
            fixed[field] = rng.choice([x for x in FIELD_VALUES[field] if x != target_definition[field]])
            out.append(self._random_record(rng, factors.active_fields, fixed=fixed))
        return out

    def _build_unrelated_records(self, rng: random.Random, factors: FeatureTextFactors, target_definition: dict[str, str], count: int) -> list[RecordSpec]:
        out = []
        for _ in range(count):
            for _ in range(100):
                record = self._random_record(rng, factors.active_fields, fixed={})
                if self._is_unrelated(record, factors.target_fields, target_definition):
                    out.append(record)
                    break
            else:
                raise GenerationError("Failed to sample enough unrelated records")
        return out

    def _build_extra_filler(self, rng: random.Random, factors: FeatureTextFactors, target_definition: dict[str, str], count: int) -> list[RecordSpec]:
        out = []
        for _ in range(count):
            for _ in range(100):
                record = self._random_record(rng, factors.active_fields, fixed={})
                if not self._matches_target(record, factors.target_fields, target_definition):
                    out.append(record)
                    break
            else:
                raise GenerationError("Failed to sample enough filler records")
        return out

    def _random_record(self, rng: random.Random, active_fields: tuple[str, ...], *, fixed: dict[str, str]) -> RecordSpec:
        values: dict[str, str] = {}
        for field in FIELD_ORDER:
            values[field] = fixed[field] if field in fixed else rng.choice(FIELD_VALUES[field])
        return RecordSpec(
            color=values["color"],  # type: ignore[arg-type]
            shape=values["shape"],  # type: ignore[arg-type]
            marker=values["marker"],  # type: ignore[arg-type]
            size=values["size"],  # type: ignore[arg-type]
            pattern=values["pattern"],  # type: ignore[arg-type]
            zone=values["zone"],  # type: ignore[arg-type]
            code=values["code"],  # type: ignore[arg-type]
        )

    def _matches_target(self, record: RecordSpec, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> bool:
        return all(record.value(field) == target_definition[field] for field in target_fields)

    def _is_unrelated(self, record: RecordSpec, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> bool:
        return all(record.value(field) != target_definition[field] for field in target_fields)

    def _matching_line_numbers(self, records: list[RecordSpec], target_fields: tuple[str, ...], target_definition: dict[str, str]) -> list[int]:
        return [i + 1 for i, record in enumerate(records) if self._matches_target(record, target_fields, target_definition)]

    def _passes_constraints(self, records: list[RecordSpec], factors: FeatureTextFactors, target_definition: dict[str, str]) -> bool:
        gold_lines = self._matching_line_numbers(records, factors.target_fields, target_definition)
        if len(gold_lines) != factors.target_count:
            return False

        if factors.target_count == 0:
            for field in factors.target_fields:
                if self._simplified_rule_count(records, factors.target_fields, target_definition, drop_field=field) <= 0:
                    return False
        else:
            for field in factors.target_fields:
                if self._simplified_rule_count(records, factors.target_fields, target_definition, drop_field=field) == factors.target_count:
                    return False

        if factors.same_color_wrong_shape_count > 0 and self._count_same_color_wrong_shape(records, factors, target_definition) < factors.same_color_wrong_shape_count:
            return False
        if factors.same_shape_wrong_color_count > 0 and self._count_same_shape_wrong_color(records, factors, target_definition) < factors.same_shape_wrong_color_count:
            return False
        if factors.same_core_wrong_marker_count > 0 and self._count_same_core_wrong_field(records, factors, target_definition, field="marker") < factors.same_core_wrong_marker_count:
            return False
        if factors.same_core_wrong_size_count > 0 and self._count_same_core_wrong_field(records, factors, target_definition, field="size") < factors.same_core_wrong_size_count:
            return False
        if factors.same_core_wrong_pattern_count > 0 and self._count_same_core_wrong_field(records, factors, target_definition, field="pattern") < factors.same_core_wrong_pattern_count:
            return False
        if self._count_unrelated(records, factors.target_fields, target_definition) < factors.unrelated_count:
            return False
        return True

    def _simplified_rule_count(self, records: list[RecordSpec], target_fields: tuple[str, ...], target_definition: dict[str, str], *, drop_field: str) -> int:
        reduced_fields = tuple(field for field in target_fields if field != drop_field)
        return sum(1 for record in records if all(record.value(field) == target_definition[field] for field in reduced_fields))

    def _count_same_color_wrong_shape(self, records: list[RecordSpec], factors: FeatureTextFactors, target_definition: dict[str, str]) -> int:
        if "color" not in factors.target_fields or "shape" not in factors.target_fields:
            return 0
        count = 0
        for record in records:
            if record.color != target_definition["color"] or record.shape == target_definition["shape"]:
                continue
            if "marker" in factors.target_fields and record.marker != target_definition["marker"]:
                continue
            if "size" in factors.target_fields and record.size != target_definition["size"]:
                continue
            if "pattern" in factors.target_fields and record.pattern != target_definition["pattern"]:
                continue
            count += 1
        return count

    def _count_same_shape_wrong_color(self, records: list[RecordSpec], factors: FeatureTextFactors, target_definition: dict[str, str]) -> int:
        if "shape" not in factors.target_fields or "color" not in factors.target_fields:
            return 0
        count = 0
        for record in records:
            if record.shape != target_definition["shape"] or record.color == target_definition["color"]:
                continue
            if "marker" in factors.target_fields and record.marker != target_definition["marker"]:
                continue
            if "size" in factors.target_fields and record.size != target_definition["size"]:
                continue
            if "pattern" in factors.target_fields and record.pattern != target_definition["pattern"]:
                continue
            count += 1
        return count

    def _count_same_core_wrong_field(self, records: list[RecordSpec], factors: FeatureTextFactors, target_definition: dict[str, str], *, field: str) -> int:
        if field not in factors.target_fields:
            return 0
        count = 0
        for record in records:
            ok = True
            for target_field in factors.target_fields:
                value = record.value(target_field)
                if target_field == field:
                    if value == target_definition[target_field]:
                        ok = False
                        break
                elif value != target_definition[target_field]:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    def _count_unrelated(self, records: list[RecordSpec], target_fields: tuple[str, ...], target_definition: dict[str, str]) -> int:
        return sum(1 for record in records if self._is_unrelated(record, target_fields, target_definition))

    def _field_description(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        return ", ".join(f"{field}={target_definition[field]}" for field in target_fields)

    def _build_count_instruction(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        desc = self._field_description(target_fields, target_definition)
        return (
            f"Count the entries matching all of the following attributes: {desc}.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n"
            '- "count" must be an integer\n'
            "- Do not include any extra keys\n"
            "- Return only the JSON object"
        )

    def _build_filter_instruction(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        desc = self._field_description(target_fields, target_definition)
        return (
            f"Return the line numbers (1-based) of the entries matching all of the following attributes: {desc}.\n"
            'Respond with a JSON object of the form {"lines": [<sorted unique integers>]}.\n'
            "Rules:\n"
            "- Use 1-based indexing\n"
            "- Sort ascending\n"
            "- Do not include duplicates\n"
            "- Return only the JSON object"
        )


def scene_to_dataset_row(scene: TextSceneSpec) -> dict:
    factors = scene.factors
    return {
        "seed": scene.seed,
        "regime": scene.regime,
        "regime_level": scene.regime_level,
        "position_mode": scene.position_mode,
        "count_instruction": scene.count_instruction,
        "filter_instruction": scene.filter_instruction,
        "text_input": scene.text_input,
        "count_prompt": scene.count_prompt,
        "filter_prompt": scene.filter_prompt,
        "gold_count": scene.gold_count,
        "gold_lines": json.dumps(scene.gold_lines),
        "target_definition": json.dumps(scene.target_definition, sort_keys=True),
        "num_records": factors.num_records,
        "target_feature_count": factors.target_feature_count,
        "target_count": factors.target_count,
        "active_fields": json.dumps(list(factors.active_fields)),
        "target_fields": json.dumps(list(factors.target_fields)),
        "irrelevant_noise_fields": json.dumps(list(factors.irrelevant_noise_fields)),
        "same_color_wrong_shape_count": factors.same_color_wrong_shape_count,
        "same_shape_wrong_color_count": factors.same_shape_wrong_color_count,
        "same_core_wrong_marker_count": factors.same_core_wrong_marker_count,
        "same_core_wrong_size_count": factors.same_core_wrong_size_count,
        "same_core_wrong_pattern_count": factors.same_core_wrong_pattern_count,
        "unrelated_count": factors.unrelated_count,
    }
