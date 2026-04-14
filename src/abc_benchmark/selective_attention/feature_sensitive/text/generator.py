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

FamilyName = Literal["selective_attention"]
AttentionalBasisName = Literal["feature_sensitive"]
ModalityName = Literal["text"]
DimensionName = Literal[
    "baseline",
    "set_size",
    "rule_arity",
    "noise_width",
    "confound",
    "position",
    "target_count",
    "target_count_x_confound",
    "target_count_x_rule_arity",
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
class FeatureSensitiveTextFactors:
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
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
class FeatureSensitiveTextScene:
    seed: int
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    position_mode: PositionMode
    count_instruction: str
    filter_instruction: str
    text_input: str
    count_prompt: str
    filter_prompt: str
    gold_count: int
    gold_lines: list[int]
    target_definition: dict[str, str]
    factors: FeatureSensitiveTextFactors
    records: list[RecordSpec]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["records"] = [asdict(record) for record in self.records]
        payload["target_definition"] = dict(self.target_definition)
        return payload


class GenerationError(RuntimeError):
    pass


class FeatureSensitiveTextGenerator:
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
        factors: FeatureSensitiveTextFactors | None = None,
        dimension: DimensionName = "combined",
        variant: str = "medium",
        position_mode: PositionMode | None = None,
        target_count_override: int | None = None,
    ) -> FeatureSensitiveTextScene:
        local_rng = random.Random(seed) if seed is not None else self.rng

        if factors is None:
            factors = self.sample_factors(rng=local_rng, dimension=dimension, variant=variant)

        if target_count_override is not None:
            factors = self._override_target_count(factors, target_count_override)

        resolved_position_mode = self._resolve_position_mode(dimension, variant, position_mode)

        for _ in range(self.max_attempts):
            target_definition = self._sample_target_spec(local_rng, factors.target_fields)
            records = self._build_records(local_rng, factors, target_definition)
            arranged_records = self._arrange_records(
                local_rng,
                records,
                factors,
                target_definition,
                resolved_position_mode,
            )

            if not self._passes_constraints(arranged_records, factors, target_definition):
                continue

            text_input = "\n".join(record.render(factors.active_fields) for record in arranged_records)
            count_instruction = self._build_count_instruction(factors.target_fields, target_definition)
            filter_instruction = self._build_filter_instruction(factors.target_fields, target_definition)
            count_prompt = f"{count_instruction}\n\n{text_input}"
            filter_prompt = f"{filter_instruction}\n\n{text_input}"
            gold_lines = self._matching_line_numbers(arranged_records, factors.target_fields, target_definition)

            return FeatureSensitiveTextScene(
                seed=seed if seed is not None else -1,
                family=factors.family,
                attentional_basis=factors.attentional_basis,
                modality=factors.modality,
                dimension=factors.dimension,
                variant=factors.variant,
                position_mode=resolved_position_mode,
                count_instruction=count_instruction,
                filter_instruction=filter_instruction,
                text_input=text_input,
                count_prompt=count_prompt,
                filter_prompt=filter_prompt,
                gold_count=len(gold_lines),
                gold_lines=gold_lines,
                target_definition=target_definition,
                factors=factors,
                records=arranged_records,
            )

        raise GenerationError("Failed to generate a valid feature-sensitive text scene within max_attempts")

    def sample_factors(
        self,
        *,
        rng: random.Random,
        dimension: DimensionName,
        variant: str,
    ) -> FeatureSensitiveTextFactors:
        if dimension == "baseline":
            return self._sample_baseline_factors(rng)
        if dimension == "set_size":
            return self._sample_set_size_factors(rng, variant)
        if dimension == "rule_arity":
            return self._sample_rule_arity_factors(variant)
        if dimension == "noise_width":
            return self._sample_noise_width_factors(variant)
        if dimension == "confound":
            return self._sample_confound_factors(variant)
        if dimension == "position":
            return self._sample_position_factors(variant)
        if dimension == "target_count":
            return self._sample_target_count_factors(variant)
        if dimension == "target_count_x_confound":
            return self._sample_target_count_x_confound_factors(variant)
        if dimension == "target_count_x_rule_arity":
            return self._sample_target_count_x_rule_arity_factors(variant)
        if dimension == "combined":
            return self._sample_combined_factors(variant)
        raise ValueError(f"Unknown dimension: {dimension}")

    def _base_factors(
        self,
        *,
        dimension: DimensionName,
        variant: str,
        num_records: int,
        target_feature_count: int,
        target_count: int,
        active_fields: tuple[str, ...],
        target_fields: tuple[str, ...],
        irrelevant_noise_fields: tuple[str, ...],
        same_color_wrong_shape_count: int,
        same_shape_wrong_color_count: int,
        same_core_wrong_marker_count: int,
        same_core_wrong_size_count: int,
        same_core_wrong_pattern_count: int,
        unrelated_count: int,
    ) -> FeatureSensitiveTextFactors:
        return FeatureSensitiveTextFactors(
            family="selective_attention",
            attentional_basis="feature_sensitive",
            modality="text",
            dimension=dimension,
            variant=variant,
            num_records=num_records,
            target_feature_count=target_feature_count,
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

    def _sample_baseline_factors(self, rng: random.Random) -> FeatureSensitiveTextFactors:
        num_records = 8
        target_count = rng.randint(1, 2)
        return self._base_factors(
            dimension="baseline",
            variant="baseline",
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

    def _sample_set_size_factors(self, rng: random.Random, variant: str) -> FeatureSensitiveTextFactors:
        mapping = {"xs": 8, "s": 16, "m": 24, "l": 32}
        if variant not in mapping:
            raise ValueError(f"Unknown set_size variant: {variant}")
        num_records = mapping[variant]
        target_count = max(1, min(4, num_records // 8))
        return self._base_factors(
            dimension="set_size",
            variant=variant,
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

    def _sample_rule_arity_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        mapping = {
            "1f": ("color",),
            "2f": ("color", "shape"),
            "3f": ("color", "shape", "marker"),
            "4f": ("color", "shape", "marker", "size"),
        }
        if variant not in mapping:
            raise ValueError(f"Unknown rule_arity variant: {variant}")
        target_fields = mapping[variant]
        active_fields = ("color", "shape", "marker", "size", "pattern")
        return self._base_factors(
            dimension="rule_arity",
            variant=variant,
            num_records=20,
            target_feature_count=len(target_fields),
            target_count=3,
            active_fields=active_fields,
            target_fields=target_fields,
            irrelevant_noise_fields=tuple(field for field in active_fields if field not in target_fields),
            same_color_wrong_shape_count=2 if {"color", "shape"}.issubset(target_fields) else 0,
            same_shape_wrong_color_count=2 if {"color", "shape"}.issubset(target_fields) else 0,
            same_core_wrong_marker_count=1 if "marker" in target_fields else 0,
            same_core_wrong_size_count=1 if "size" in target_fields else 0,
            same_core_wrong_pattern_count=1 if "pattern" in target_fields else 0,
            unrelated_count=20
            - 3
            - (2 if {"color", "shape"}.issubset(target_fields) else 0)
            - (2 if {"color", "shape"}.issubset(target_fields) else 0)
            - (1 if "marker" in target_fields else 0)
            - (1 if "size" in target_fields else 0)
            - (1 if "pattern" in target_fields else 0),
        )

    def _sample_noise_width_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        mapping = {
            "n0": ("color", "shape", "marker"),
            "n1": ("color", "shape", "marker", "size", "pattern"),
            "n2": ("color", "shape", "marker", "size", "pattern", "zone", "code"),
        }
        if variant not in mapping:
            raise ValueError(f"Unknown noise_width variant: {variant}")
        active_fields = mapping[variant]
        target_fields = ("color", "shape")
        return self._base_factors(
            dimension="noise_width",
            variant=variant,
            num_records=20,
            target_feature_count=2,
            target_count=3,
            active_fields=active_fields,
            target_fields=target_fields,
            irrelevant_noise_fields=tuple(field for field in active_fields if field not in target_fields),
            same_color_wrong_shape_count=2,
            same_shape_wrong_color_count=2,
            same_core_wrong_marker_count=0,
            same_core_wrong_size_count=0,
            same_core_wrong_pattern_count=0,
            unrelated_count=13,
        )

    def _sample_confound_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        mapping = {
            "low": (1, 1, 1, 1, 0, 16),
            "medium": (2, 2, 2, 2, 0, 20),
            "high": (4, 4, 3, 3, 0, 28),
            "extreme": (6, 6, 4, 4, 0, 36),
        }
        if variant not in mapping:
            raise ValueError(f"Unknown confound variant: {variant}")
        scws, sswc, scm, scs, scp, num_records = mapping[variant]
        return self._base_factors(
            dimension="confound",
            variant=variant,
            num_records=num_records,
            target_feature_count=4,
            target_count=3,
            active_fields=("color", "shape", "marker", "size", "pattern"),
            target_fields=("color", "shape", "marker", "size"),
            irrelevant_noise_fields=("pattern",),
            same_color_wrong_shape_count=scws,
            same_shape_wrong_color_count=sswc,
            same_core_wrong_marker_count=scm,
            same_core_wrong_size_count=scs,
            same_core_wrong_pattern_count=scp,
            unrelated_count=max(2, num_records - 3 - scws - sswc - scm - scs - scp),
        )

    def _sample_position_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        if variant not in {"random", "front_loaded", "back_loaded", "clustered"}:
            raise ValueError(f"Unknown position variant: {variant}")
        return self._base_factors(
            dimension="position",
            variant=variant,
            num_records=20,
            target_feature_count=2,
            target_count=3,
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

    def _sample_target_count_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        mapping = {"0": 0, "1": 1, "3": 3, "5": 5}
        if variant not in mapping:
            raise ValueError(f"Unknown target_count variant: {variant}")
        target_count = mapping[variant]
        num_records = 20
        same_color_wrong_shape_count = 3
        same_shape_wrong_color_count = 3
        unrelated_count = num_records - target_count - same_color_wrong_shape_count - same_shape_wrong_color_count
        if unrelated_count < 2:
            raise GenerationError("target_count configuration leaves too few unrelated records")
        return self._base_factors(
            dimension="target_count",
            variant=variant,
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

    def _sample_target_count_x_confound_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        mapping = {
            "0_low": (0, 1, 1, 1, 1, 16),
            "0_medium": (0, 2, 2, 2, 2, 20),
            "0_extreme": (0, 6, 6, 4, 4, 36),
            "3_low": (3, 1, 1, 1, 1, 16),
            "3_medium": (3, 2, 2, 2, 2, 20),
            "3_extreme": (3, 6, 6, 4, 4, 36),
        }
        if variant not in mapping:
            raise ValueError(f"Unknown target_count_x_confound variant: {variant}")
        target_count, scws, sswc, scm, scs, num_records = mapping[variant]
        unrelated_count = num_records - target_count - scws - sswc - scm - scs
        if unrelated_count < 2:
            raise GenerationError("target_count_x_confound configuration leaves too few unrelated records")
        return self._base_factors(
            dimension="target_count_x_confound",
            variant=variant,
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

    def _sample_target_count_x_rule_arity_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        mapping = {
            "0_1f": (0, ("color",)),
            "0_2f": (0, ("color", "shape")),
            "0_4f": (0, ("color", "shape", "marker", "size")),
            "3_1f": (3, ("color",)),
            "3_2f": (3, ("color", "shape")),
            "3_4f": (3, ("color", "shape", "marker", "size")),
        }
        if variant not in mapping:
            raise ValueError(f"Unknown target_count_x_rule_arity variant: {variant}")
        target_count, target_fields = mapping[variant]
        active_fields = ("color", "shape", "marker", "size", "pattern")
        same_color_wrong_shape_count = 2 if {"color", "shape"}.issubset(target_fields) else 0
        same_shape_wrong_color_count = 2 if {"color", "shape"}.issubset(target_fields) else 0
        same_core_wrong_marker_count = 1 if "marker" in target_fields else 0
        same_core_wrong_size_count = 1 if "size" in target_fields else 0
        unrelated_count = (
            20
            - target_count
            - same_color_wrong_shape_count
            - same_shape_wrong_color_count
            - same_core_wrong_marker_count
            - same_core_wrong_size_count
        )
        if unrelated_count < 2:
            raise GenerationError("target_count_x_rule_arity configuration leaves too few unrelated records")
        return self._base_factors(
            dimension="target_count_x_rule_arity",
            variant=variant,
            num_records=20,
            target_feature_count=len(target_fields),
            target_count=target_count,
            active_fields=active_fields,
            target_fields=target_fields,
            irrelevant_noise_fields=tuple(field for field in active_fields if field not in target_fields),
            same_color_wrong_shape_count=same_color_wrong_shape_count,
            same_shape_wrong_color_count=same_shape_wrong_color_count,
            same_core_wrong_marker_count=same_core_wrong_marker_count,
            same_core_wrong_size_count=same_core_wrong_size_count,
            same_core_wrong_pattern_count=0,
            unrelated_count=unrelated_count,
        )

    def _sample_combined_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        if variant == "easy":
            return self._base_factors(
                dimension="combined",
                variant="easy",
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
        if variant == "medium":
            return self._base_factors(
                dimension="combined",
                variant="medium",
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
        if variant == "hard":
            return self._base_factors(
                dimension="combined",
                variant="hard",
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
        raise ValueError(f"Unknown combined variant: {variant}")

    def _resolve_position_mode(
        self,
        dimension: DimensionName,
        variant: str,
        position_mode: PositionMode | None,
    ) -> PositionMode:
        if position_mode is not None:
            return position_mode
        if dimension == "position":
            return variant  # type: ignore[return-value]
        return "random"

    def _override_target_count(
        self,
        factors: FeatureSensitiveTextFactors,
        target_count_override: int,
    ) -> FeatureSensitiveTextFactors:
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

    def _build_records(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> list[RecordSpec]:
        records: list[RecordSpec] = []
        records.extend(self._build_exact_targets(rng, factors, target_definition))
        records.extend(self._build_same_color_wrong_shape(rng, factors, target_definition))
        records.extend(self._build_same_shape_wrong_color(rng, factors, target_definition))
        records.extend(
            self._build_same_core_wrong_field(
                rng,
                factors,
                target_definition,
                field="marker",
                count=factors.same_core_wrong_marker_count,
            )
        )
        records.extend(
            self._build_same_core_wrong_field(
                rng,
                factors,
                target_definition,
                field="size",
                count=factors.same_core_wrong_size_count,
            )
        )
        records.extend(
            self._build_same_core_wrong_field(
                rng,
                factors,
                target_definition,
                field="pattern",
                count=factors.same_core_wrong_pattern_count,
            )
        )

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
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        position_mode: PositionMode,
    ) -> list[RecordSpec]:
        arranged = list(records)
        if position_mode == "random" or factors.target_count == 0:
            rng.shuffle(arranged)
            return arranged

        targets = [record for record in arranged if self._matches_target(record, factors.target_fields, target_definition)]
        non_targets = [record for record in arranged if not self._matches_target(record, factors.target_fields, target_definition)]
        rng.shuffle(targets)
        rng.shuffle(non_targets)

        if position_mode == "front_loaded":
            return targets + non_targets
        if position_mode == "back_loaded":
            return non_targets + targets
        if position_mode == "clustered":
            start_index = 0 if not non_targets else rng.randint(0, len(non_targets))
            return non_targets[:start_index] + targets + non_targets[start_index:]

        rng.shuffle(arranged)
        return arranged

    def _build_exact_targets(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> list[RecordSpec]:
        return [self._random_record(rng, fixed=target_definition) for _ in range(factors.target_count)]

    def _build_same_color_wrong_shape(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> list[RecordSpec]:
        if factors.same_color_wrong_shape_count <= 0 or "color" not in factors.target_fields or "shape" not in factors.target_fields:
            return []

        records: list[RecordSpec] = []
        for _ in range(factors.same_color_wrong_shape_count):
            fixed = {"color": target_definition["color"]}
            fixed["shape"] = rng.choice([shape for shape in SHAPES if shape != target_definition["shape"]])
            for field in ("marker", "size", "pattern"):
                if field in factors.target_fields:
                    fixed[field] = target_definition[field]
            records.append(self._random_record(rng, fixed=fixed))
        return records

    def _build_same_shape_wrong_color(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> list[RecordSpec]:
        if factors.same_shape_wrong_color_count <= 0 or "shape" not in factors.target_fields or "color" not in factors.target_fields:
            return []

        records: list[RecordSpec] = []
        for _ in range(factors.same_shape_wrong_color_count):
            fixed = {"shape": target_definition["shape"]}
            fixed["color"] = rng.choice([color for color in COLORS if color != target_definition["color"]])
            for field in ("marker", "size", "pattern"):
                if field in factors.target_fields:
                    fixed[field] = target_definition[field]
            records.append(self._random_record(rng, fixed=fixed))
        return records

    def _build_same_core_wrong_field(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        *,
        field: str,
        count: int,
    ) -> list[RecordSpec]:
        if count <= 0 or field not in factors.target_fields:
            return []

        records: list[RecordSpec] = []
        for _ in range(count):
            fixed: dict[str, str] = {}
            for target_field in factors.target_fields:
                if target_field != field:
                    fixed[target_field] = target_definition[target_field]
            fixed[field] = rng.choice([value for value in FIELD_VALUES[field] if value != target_definition[field]])
            records.append(self._random_record(rng, fixed=fixed))
        return records

    def _build_unrelated_records(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        count: int,
    ) -> list[RecordSpec]:
        records: list[RecordSpec] = []
        for _ in range(count):
            for _ in range(100):
                record = self._random_record(rng, fixed={})
                if self._is_unrelated(record, factors.target_fields, target_definition):
                    records.append(record)
                    break
            else:
                raise GenerationError("Failed to sample enough unrelated records")
        return records

    def _build_extra_filler(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        count: int,
    ) -> list[RecordSpec]:
        records: list[RecordSpec] = []
        for _ in range(count):
            for _ in range(100):
                record = self._random_record(rng, fixed={})
                if not self._matches_target(record, factors.target_fields, target_definition):
                    records.append(record)
                    break
            else:
                raise GenerationError("Failed to sample enough filler records")
        return records

    def _random_record(self, rng: random.Random, *, fixed: dict[str, str]) -> RecordSpec:
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

    def _matches_target(
        self,
        record: RecordSpec,
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
    ) -> bool:
        return all(record.value(field) == target_definition[field] for field in target_fields)

    def _is_unrelated(
        self,
        record: RecordSpec,
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
    ) -> bool:
        return all(record.value(field) != target_definition[field] for field in target_fields)

    def _matching_line_numbers(
        self,
        records: list[RecordSpec],
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
    ) -> list[int]:
        return [
            index + 1
            for index, record in enumerate(records)
            if self._matches_target(record, target_fields, target_definition)
        ]

    def _passes_constraints(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> bool:
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

        if self._count_same_color_wrong_shape(records, factors, target_definition) < factors.same_color_wrong_shape_count:
            return False
        if self._count_same_shape_wrong_color(records, factors, target_definition) < factors.same_shape_wrong_color_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, field="marker") < factors.same_core_wrong_marker_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, field="size") < factors.same_core_wrong_size_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, field="pattern") < factors.same_core_wrong_pattern_count:
            return False
        if self._count_unrelated(records, factors.target_fields, target_definition) < factors.unrelated_count:
            return False
        return True

    def _simplified_rule_count(
        self,
        records: list[RecordSpec],
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
        *,
        drop_field: str,
    ) -> int:
        reduced_fields = tuple(field for field in target_fields if field != drop_field)
        return sum(
            1
            for record in records
            if all(record.value(field) == target_definition[field] for field in reduced_fields)
        )

    def _count_same_color_wrong_shape(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> int:
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

    def _count_same_shape_wrong_color(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> int:
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

    def _count_same_core_wrong_field(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        *,
        field: str,
    ) -> int:
        if field not in factors.target_fields:
            return 0
        count = 0
        for record in records:
            is_match = True
            for target_field in factors.target_fields:
                value = record.value(target_field)
                if target_field == field:
                    if value == target_definition[target_field]:
                        is_match = False
                        break
                elif value != target_definition[target_field]:
                    is_match = False
                    break
            if is_match:
                count += 1
        return count

    def _count_unrelated(
        self,
        records: list[RecordSpec],
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
    ) -> int:
        return sum(1 for record in records if self._is_unrelated(record, target_fields, target_definition))

    def _field_description(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        return ", ".join(f"{field}={target_definition[field]}" for field in target_fields)

    def _build_count_instruction(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        description = self._field_description(target_fields, target_definition)
        return (
            f"Count the entries matching all of the following attributes: {description}.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n"
            '- "count" must be an integer\n'
            "- Do not include any extra keys\n"
            "- Return only the JSON object"
        )

    def _build_filter_instruction(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        description = self._field_description(target_fields, target_definition)
        return (
            f"Return the line numbers (1-based) of the entries matching all of the following attributes: {description}.\n"
            'Respond with a JSON object of the form {"lines": [<sorted unique integers>]}.\n'
            "Rules:\n"
            "- Use 1-based indexing\n"
            "- Sort ascending\n"
            "- Do not include duplicates\n"
            "- Return only the JSON object"
        )


def scene_to_dataset_row(scene: FeatureSensitiveTextScene) -> dict[str, object]:
    factors = scene.factors
    return {
        "seed": scene.seed,
        "family": scene.family,
        "attentional_basis": scene.attentional_basis,
        "modality": scene.modality,
        "dimension": scene.dimension,
        "variant": scene.variant,
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
