from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from typing import Literal

ColorName = Literal["red", "blue", "green", "yellow"]
Shape = Literal["circle", "square", "triangle"]
MarkerValue = Literal["0", "1"]
SizeValue = Literal["small", "medium", "large", "xl"]
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
    "adversarial_confound",
    "negation",
    "disjunction",
]
PositionMode = Literal["random", "front_loaded", "back_loaded", "clustered"]

COLORS: tuple[ColorName, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[Shape, ...] = ("circle", "square", "triangle")
MARKERS: tuple[MarkerValue, ...] = ("0", "1")
SIZES: tuple[SizeValue, ...] = ("small", "medium", "large", "xl")
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

FIELD_ORDER: tuple[str, ...] = ("color", "shape", "marker", "size", "pattern", "zone", "code")
CORE_FIELDS: tuple[str, ...] = ("color", "shape", "marker", "size", "pattern")


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
    forbidden_field: str | None = None
    secondary_target_fields: tuple[str, ...] | None = None
    secondary_target_count: int | None = None
    relation: str | None = None
    forbidden_feature_violation_count: int = 0
    same_core_wrong_color_count: int = 0
    same_core_wrong_shape_count: int = 0
    near_miss_count: int = 0


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
    secondary_target_definition: dict[str, str] | None = None
    forbidden_value: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["records"] = [asdict(record) for record in self.records]
        payload["target_definition"] = dict(self.target_definition)
        if self.secondary_target_definition is not None:
            payload["secondary_target_definition"] = dict(self.secondary_target_definition)
        return payload


class GenerationError(RuntimeError):
    pass


class FeatureSensitiveTextGenerator:
    def __init__(self, rng: random.Random | None = None, max_attempts: int = 1000) -> None:
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
        if target_count_override is not None and factors.dimension not in {"disjunction"}:
            factors = self._override_target_count(factors, target_count_override)
        resolved_position_mode = self._resolve_position_mode(dimension, variant, position_mode)

        for _ in range(self.max_attempts):
            if factors.dimension == "negation":
                scene = self._generate_negation_scene(local_rng, factors, seed, resolved_position_mode)
            elif factors.dimension == "disjunction":
                scene = self._generate_disjunction_scene(local_rng, factors, seed, resolved_position_mode)
            else:
                scene = self._generate_standard_scene(local_rng, factors, seed, resolved_position_mode)
            if scene is not None:
                return scene
        raise GenerationError("Failed to generate a valid feature-sensitive text scene within max_attempts")

    def sample_factors(self, *, rng: random.Random, dimension: DimensionName, variant: str) -> FeatureSensitiveTextFactors:
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
        if dimension == "adversarial_confound":
            return self._sample_adversarial_confound_factors(variant)
        if dimension == "negation":
            return self._sample_negation_factors(variant)
        if dimension == "disjunction":
            return self._sample_disjunction_factors(variant)
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
        forbidden_field: str | None = None,
        secondary_target_fields: tuple[str, ...] | None = None,
        secondary_target_count: int | None = None,
        relation: str | None = None,
        forbidden_feature_violation_count: int = 0,
        same_core_wrong_color_count: int = 0,
        same_core_wrong_shape_count: int = 0,
        near_miss_count: int = 0,
    ) -> FeatureSensitiveTextFactors:
        allocated = (
            target_count
            + same_color_wrong_shape_count
            + same_shape_wrong_color_count
            + same_core_wrong_color_count
            + same_core_wrong_shape_count
            + same_core_wrong_marker_count
            + same_core_wrong_size_count
            + same_core_wrong_pattern_count
            + forbidden_feature_violation_count
            + near_miss_count
        )

        if secondary_target_count is not None:
            allocated += secondary_target_count

        adjusted_unrelated = min(unrelated_count, max(0, num_records - allocated))

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
            unrelated_count=adjusted_unrelated,
            forbidden_field=forbidden_field,
            secondary_target_fields=secondary_target_fields,
            secondary_target_count=secondary_target_count,
            relation=relation,
            forbidden_feature_violation_count=forbidden_feature_violation_count,
            same_core_wrong_color_count=same_core_wrong_color_count,
            same_core_wrong_shape_count=same_core_wrong_shape_count,
            near_miss_count=near_miss_count,
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
        return self._base_factors(
            dimension="noise_width",
            variant=variant,
            num_records=20,
            target_feature_count=2,
            target_count=3,
            active_fields=active_fields,
            target_fields=("color", "shape"),
            irrelevant_noise_fields=tuple(field for field in active_fields if field not in {"color", "shape"}),
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
        scws = 3
        sswc = 3
        unrelated = num_records - target_count - scws - sswc
        if unrelated < 2:
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
            same_color_wrong_shape_count=scws,
            same_shape_wrong_color_count=sswc,
            same_core_wrong_marker_count=0,
            same_core_wrong_size_count=0,
            same_core_wrong_pattern_count=0,
            unrelated_count=unrelated,
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
        unrelated = num_records - target_count - scws - sswc - scm - scs
        if unrelated < 2:
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
            unrelated_count=unrelated,
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
        scws = 2 if {"color", "shape"}.issubset(target_fields) else 0
        sswc = 2 if {"color", "shape"}.issubset(target_fields) else 0
        scm = 1 if "marker" in target_fields else 0
        scs = 1 if "size" in target_fields else 0
        unrelated = 20 - target_count - scws - sswc - scm - scs
        if unrelated < 2:
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
            same_color_wrong_shape_count=scws,
            same_shape_wrong_color_count=sswc,
            same_core_wrong_marker_count=scm,
            same_core_wrong_size_count=scs,
            same_core_wrong_pattern_count=0,
            unrelated_count=unrelated,
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
                near_miss_count=3,
            )
        raise ValueError(f"Unknown combined variant: {variant}")

    def _sample_adversarial_confound_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        if variant == "dense":
            return self._base_factors(
                dimension="adversarial_confound",
                variant=variant,
                num_records=32,
                target_feature_count=5,
                target_count=3,
                active_fields=("color", "shape", "marker", "size", "pattern", "zone", "code"),
                target_fields=("color", "shape", "marker", "size", "pattern"),
                irrelevant_noise_fields=("zone", "code"),
                same_color_wrong_shape_count=0,
                same_shape_wrong_color_count=0,
                same_core_wrong_marker_count=4,
                same_core_wrong_size_count=4,
                same_core_wrong_pattern_count=4,
                unrelated_count=9,
                same_core_wrong_color_count=4,
                same_core_wrong_shape_count=4,
            )
        if variant == "extreme":
            return self._base_factors(
                dimension="adversarial_confound",
                variant=variant,
                num_records=48,
                target_feature_count=5,
                target_count=3,
                active_fields=("color", "shape", "marker", "size", "pattern", "zone", "code"),
                target_fields=("color", "shape", "marker", "size", "pattern"),
                irrelevant_noise_fields=("zone", "code"),
                same_color_wrong_shape_count=0,
                same_shape_wrong_color_count=0,
                same_core_wrong_marker_count=6,
                same_core_wrong_size_count=6,
                same_core_wrong_pattern_count=6,
                unrelated_count=15,
                same_core_wrong_color_count=6,
                same_core_wrong_shape_count=6,
                near_miss_count=4,
            )
        raise ValueError(f"Unknown adversarial_confound variant: {variant}")

    def _sample_negation_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        if variant == "easy":
            return self._base_factors(
                dimension="negation",
                variant=variant,
                num_records=20,
                target_feature_count=3,
                target_count=3,
                active_fields=("color", "shape", "marker", "pattern", "size"),
                target_fields=("color", "shape", "marker"),
                irrelevant_noise_fields=("size",),
                forbidden_field="pattern",
                same_color_wrong_shape_count=2,
                same_shape_wrong_color_count=2,
                same_core_wrong_marker_count=2,
                same_core_wrong_size_count=0,
                same_core_wrong_pattern_count=0,
                unrelated_count=8,
                forbidden_feature_violation_count=3,
            )
        if variant == "hard":
            return self._base_factors(
                dimension="negation",
                variant=variant,
                num_records=32,
                target_feature_count=4,
                target_count=3,
                active_fields=("color", "shape", "marker", "size", "pattern", "zone", "code"),
                target_fields=("color", "shape", "marker", "size"),
                irrelevant_noise_fields=("zone", "code"),
                forbidden_field="pattern",
                same_color_wrong_shape_count=3,
                same_shape_wrong_color_count=3,
                same_core_wrong_marker_count=3,
                same_core_wrong_size_count=3,
                same_core_wrong_pattern_count=0,
                unrelated_count=13,
                forbidden_feature_violation_count=4,
                near_miss_count=2,
            )
        raise ValueError(f"Unknown negation variant: {variant}")

    def _sample_disjunction_factors(self, variant: str) -> FeatureSensitiveTextFactors:
        if variant == "easy":
            return self._base_factors(
                dimension="disjunction",
                variant=variant,
                num_records=20,
                target_feature_count=2,
                target_count=2,
                active_fields=("color", "shape", "marker", "size"),
                target_fields=("color", "shape"),
                secondary_target_fields=("color", "shape"),
                secondary_target_count=2,
                irrelevant_noise_fields=("marker", "size"),
                relation="or",
                same_color_wrong_shape_count=2,
                same_shape_wrong_color_count=2,
                same_core_wrong_marker_count=0,
                same_core_wrong_size_count=0,
                same_core_wrong_pattern_count=0,
                unrelated_count=8,
            )
        if variant == "hard":
            return self._base_factors(
                dimension="disjunction",
                variant=variant,
                num_records=32,
                target_feature_count=4,
                target_count=4,
                active_fields=("color", "shape", "marker", "size", "pattern", "zone", "code"),
                target_fields=("color", "shape", "marker", "size"),
                secondary_target_fields=("color", "shape", "marker", "size"),
                secondary_target_count=1,
                irrelevant_noise_fields=("pattern", "zone", "code"),
                relation="or",
                same_color_wrong_shape_count=0,
                same_shape_wrong_color_count=0,
                same_core_wrong_marker_count=3,
                same_core_wrong_size_count=3,
                same_core_wrong_pattern_count=0,
                unrelated_count=9,
                same_core_wrong_color_count=3,
                same_core_wrong_shape_count=3,
                near_miss_count=4,
            )
        raise ValueError(f"Unknown disjunction variant: {variant}")

    def _resolve_position_mode(self, dimension: DimensionName, variant: str, position_mode: PositionMode | None) -> PositionMode:
        if position_mode is not None:
            return position_mode
        if dimension == "position":
            return variant  # type: ignore[return-value]
        return "random"

    def _override_target_count(self, factors: FeatureSensitiveTextFactors, target_count_override: int) -> FeatureSensitiveTextFactors:
        if target_count_override < 0:
            raise ValueError("target_count_override must be >= 0")
        delta = target_count_override - factors.target_count
        return replace(
            factors,
            target_count=target_count_override,
            num_records=factors.num_records + max(0, delta),
            unrelated_count=factors.unrelated_count + max(0, -delta),
        )

    def _generate_standard_scene(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        seed: int | None,
        position_mode: PositionMode,
    ) -> FeatureSensitiveTextScene | None:
        target_definition = self._sample_target_spec(rng, factors.target_fields)
        records = self._build_standard_records(rng, factors, target_definition)
        arranged = self._arrange_records(
            rng,
            records,
            lambda rec: self._matches_target(rec, factors.target_fields, target_definition),
            position_mode,
        )
        if not self._passes_standard_constraints(arranged, factors, target_definition):
            return None
        return self._finalize_scene(
            seed,
            factors,
            position_mode,
            arranged,
            target_definition=target_definition,
            gold_lines=self._matching_line_numbers(arranged, factors.target_fields, target_definition),
            count_instruction=self._build_count_instruction(factors.target_fields, target_definition),
            filter_instruction=self._build_filter_instruction(factors.target_fields, target_definition),
        )

    def _generate_negation_scene(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        seed: int | None,
        position_mode: PositionMode,
    ) -> FeatureSensitiveTextScene | None:
        if not factors.forbidden_field:
            raise GenerationError("negation factors require forbidden_field")
        target_definition = self._sample_target_spec(rng, factors.target_fields)
        forbidden_value = rng.choice(FIELD_VALUES[factors.forbidden_field])
        records = self._build_negation_records(rng, factors, target_definition, factors.forbidden_field, forbidden_value)
        matcher = lambda rec: self._matches_negation_rule(
            rec,
            factors.target_fields,
            target_definition,
            factors.forbidden_field,
            forbidden_value,
        )
        arranged = self._arrange_records(rng, records, matcher, position_mode)
        gold_lines = [i + 1 for i, rec in enumerate(arranged) if matcher(rec)]
        if not self._passes_negation_constraints(
            arranged,
            factors,
            target_definition,
            factors.forbidden_field,
            forbidden_value,
            gold_lines,
        ):
            return None
        return self._finalize_scene(
            seed,
            factors,
            position_mode,
            arranged,
            target_definition=target_definition,
            forbidden_value=forbidden_value,
            gold_lines=gold_lines,
            count_instruction=self._build_negation_count_instruction(
                factors.target_fields,
                target_definition,
                factors.forbidden_field,
                forbidden_value,
            ),
            filter_instruction=self._build_negation_filter_instruction(
                factors.target_fields,
                target_definition,
                factors.forbidden_field,
                forbidden_value,
            ),
        )

    def _generate_disjunction_scene(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        seed: int | None,
        position_mode: PositionMode,
    ) -> FeatureSensitiveTextScene | None:
        if factors.secondary_target_fields is None or factors.secondary_target_count is None:
            raise GenerationError("disjunction factors require secondary target fields and count")
        definition_a = self._sample_target_spec(rng, factors.target_fields)
        definition_b = self._sample_secondary_target_spec(rng, factors.secondary_target_fields, definition_a)
        records = self._build_disjunction_records(rng, factors, definition_a, definition_b)
        matcher_a = lambda rec: self._matches_target(rec, factors.target_fields, definition_a)
        matcher_b = lambda rec: self._matches_target(rec, factors.secondary_target_fields or (), definition_b)
        matcher = lambda rec: matcher_a(rec) or matcher_b(rec)
        arranged = self._arrange_records(rng, records, matcher, position_mode)
        a_lines = [i + 1 for i, rec in enumerate(arranged) if matcher_a(rec)]
        b_lines = [i + 1 for i, rec in enumerate(arranged) if matcher_b(rec)]
        gold_lines = sorted(set(a_lines) | set(b_lines))
        if not self._passes_disjunction_constraints(arranged, factors, definition_a, definition_b, a_lines, b_lines, gold_lines):
            return None
        return self._finalize_scene(
            seed,
            factors,
            position_mode,
            arranged,
            target_definition=definition_a,
            secondary_target_definition=definition_b,
            gold_lines=gold_lines,
            count_instruction=self._build_disjunction_count_instruction(
                factors.target_fields,
                definition_a,
                factors.secondary_target_fields,
                definition_b,
            ),
            filter_instruction=self._build_disjunction_filter_instruction(
                factors.target_fields,
                definition_a,
                factors.secondary_target_fields,
                definition_b,
            ),
        )

    def _finalize_scene(
        self,
        seed: int | None,
        factors: FeatureSensitiveTextFactors,
        position_mode: PositionMode,
        records: list[RecordSpec],
        target_definition: dict[str, str],
        gold_lines: list[int],
        count_instruction: str,
        filter_instruction: str,
        secondary_target_definition: dict[str, str] | None = None,
        forbidden_value: str | None = None,
    ) -> FeatureSensitiveTextScene:
        text_input = "\n".join(rec.render(factors.active_fields) for rec in records)
        return FeatureSensitiveTextScene(
            seed=seed if seed is not None else -1,
            family=factors.family,
            attentional_basis=factors.attentional_basis,
            modality=factors.modality,
            dimension=factors.dimension,
            variant=factors.variant,
            position_mode=position_mode,
            count_instruction=count_instruction,
            filter_instruction=filter_instruction,
            text_input=text_input,
            count_prompt=f"{count_instruction}\n\n{text_input}",
            filter_prompt=f"{filter_instruction}\n\n{text_input}",
            gold_count=len(gold_lines),
            gold_lines=gold_lines,
            target_definition=target_definition,
            factors=factors,
            records=records,
            secondary_target_definition=secondary_target_definition,
            forbidden_value=forbidden_value,
        )

    def _build_standard_records(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> list[RecordSpec]:
        records: list[RecordSpec] = []
        matcher = self._standard_matcher(factors, target_definition)

        records.extend(self._build_exact_targets_for_definition(rng, target_definition, factors.target_count))
        records.extend(self._build_same_color_wrong_shape(rng, factors, target_definition))
        records.extend(self._build_same_shape_wrong_color(rng, factors, target_definition))
        records.extend(self._build_same_core_wrong_field_any(rng, factors, target_definition, "color", factors.same_core_wrong_color_count))
        records.extend(self._build_same_core_wrong_field_any(rng, factors, target_definition, "shape", factors.same_core_wrong_shape_count))
        records.extend(self._build_same_core_wrong_field_any(rng, factors, target_definition, "marker", factors.same_core_wrong_marker_count))
        records.extend(self._build_same_core_wrong_field_any(rng, factors, target_definition, "size", factors.same_core_wrong_size_count))
        records.extend(self._build_same_core_wrong_field_any(rng, factors, target_definition, "pattern", factors.same_core_wrong_pattern_count))
        records.extend(self._build_near_miss_records(rng, factors, target_definition, factors.near_miss_count, [matcher]))
        records.extend(self._build_unrelated_records(rng, factors, [matcher], factors.unrelated_count))
        remaining = factors.num_records - len(records)
        if remaining < 0:
            raise GenerationError("Factor bundle over-allocates records")
        records.extend(self._build_extra_filler(rng, factors, [matcher], remaining))
        return records

    def _build_negation_records(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        forbidden_field: str,
        forbidden_value: str,
    ) -> list[RecordSpec]:
        records: list[RecordSpec] = []
        positive_fixed = dict(target_definition)
        for _ in range(factors.target_count):
            fixed = dict(positive_fixed)
            fixed[forbidden_field] = rng.choice([v for v in FIELD_VALUES[forbidden_field] if v != forbidden_value])
            records.append(self._random_record(rng, fixed=fixed))

        records.extend(self._build_same_color_wrong_shape(rng, factors, target_definition))
        records.extend(self._build_same_shape_wrong_color(rng, factors, target_definition))
        records.extend(self._build_same_core_wrong_field_any(rng, factors, target_definition, "marker", factors.same_core_wrong_marker_count))
        records.extend(self._build_same_core_wrong_field_any(rng, factors, target_definition, "size", factors.same_core_wrong_size_count))

        matcher = lambda rec: self._matches_negation_rule(
            rec,
            factors.target_fields,
            target_definition,
            forbidden_field,
            forbidden_value,
        )

        records.extend(
            self._build_near_miss_records(
                rng,
                factors,
                target_definition,
                factors.near_miss_count,
                [matcher],
                forbidden_field=forbidden_field,
                forbidden_value=forbidden_value,
            )
        )
        records.extend(self._build_forbidden_feature_violations(rng, factors, target_definition, forbidden_field, forbidden_value))
        records.extend(self._build_unrelated_records(rng, factors, [matcher], factors.unrelated_count))
        remaining = factors.num_records - len(records)
        if remaining < 0:
            raise GenerationError("Negation factor bundle over-allocates records")
        records.extend(self._build_extra_filler(rng, factors, [matcher], remaining))
        return records

    def _build_disjunction_records(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        definition_a: dict[str, str],
        definition_b: dict[str, str],
    ) -> list[RecordSpec]:
        records: list[RecordSpec] = []
        records.extend(self._build_exact_targets_for_definition(rng, definition_a, factors.target_count))
        records.extend(self._build_exact_targets_for_definition(rng, definition_b, factors.secondary_target_count or 0))

        matcher_a = self._standard_matcher(factors, definition_a)
        matcher_b = lambda rec: self._matches_target(rec, factors.secondary_target_fields or (), definition_b)

        if factors.variant == "easy":
            records.extend(self._build_same_color_wrong_shape(rng, factors, definition_a))
            records.extend(self._build_same_shape_wrong_color(rng, factors, definition_a))
            records.extend(self._build_same_color_wrong_shape(rng, factors, definition_b))
            records.extend(self._build_same_shape_wrong_color(rng, factors, definition_b))
        else:
            for field, count in [
                ("color", factors.same_core_wrong_color_count),
                ("shape", factors.same_core_wrong_shape_count),
                ("marker", factors.same_core_wrong_marker_count),
                ("size", factors.same_core_wrong_size_count),
            ]:
                count_a = (count + 1) // 2
                count_b = count // 2
                records.extend(self._build_same_core_wrong_field_any(rng, factors, definition_a, field, count_a))
                records.extend(self._build_same_core_wrong_field_any(rng, factors, definition_b, field, count_b))

            near_a = max(1, factors.near_miss_count // 2)
            near_b = max(1, factors.near_miss_count - near_a)
            records.extend(self._build_near_miss_records(rng, factors, definition_a, near_a, [matcher_a, matcher_b]))
            records.extend(self._build_near_miss_records(rng, factors, definition_b, near_b, [matcher_a, matcher_b]))

        records.extend(self._build_unrelated_records(rng, factors, [matcher_a, matcher_b], factors.unrelated_count))
        remaining = factors.num_records - len(records)
        if remaining < 0:
            raise GenerationError("Disjunction factor bundle over-allocates records")
        records.extend(self._build_extra_filler(rng, factors, [matcher_a, matcher_b], remaining))
        return records

    def _build_exact_targets_for_definition(self, rng: random.Random, definition: dict[str, str], count: int) -> list[RecordSpec]:
        return [self._random_record(rng, fixed=definition) for _ in range(count)]

    def _build_same_color_wrong_shape(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> list[RecordSpec]:
        if factors.same_color_wrong_shape_count <= 0 or "color" not in factors.target_fields or "shape" not in factors.target_fields:
            return []
        records = []
        for _ in range(factors.same_color_wrong_shape_count):
            fixed = {
                "color": target_definition["color"],
                "shape": rng.choice([s for s in SHAPES if s != target_definition["shape"]]),
            }
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
        if factors.same_shape_wrong_color_count <= 0 or "color" not in factors.target_fields or "shape" not in factors.target_fields:
            return []
        records = []
        for _ in range(factors.same_shape_wrong_color_count):
            fixed = {
                "shape": target_definition["shape"],
                "color": rng.choice([c for c in COLORS if c != target_definition["color"]]),
            }
            for field in ("marker", "size", "pattern"):
                if field in factors.target_fields:
                    fixed[field] = target_definition[field]
            records.append(self._random_record(rng, fixed=fixed))
        return records

    def _build_same_core_wrong_field_any(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        field: str,
        count: int,
    ) -> list[RecordSpec]:
        if count <= 0 or field not in factors.target_fields:
            return []
        records = []
        for _ in range(count):
            fixed = {f: target_definition[f] for f in factors.target_fields}
            fixed[field] = rng.choice([v for v in FIELD_VALUES[field] if v != target_definition[field]])
            records.append(self._random_record(rng, fixed=fixed))
        return records

    def _build_near_miss_records(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        count: int,
        matchers: list,
        forbidden_field: str | None = None,
        forbidden_value: str | None = None,
    ) -> list[RecordSpec]:
        if count <= 0:
            return []

        preferred_fields = tuple(field for field in ("size", "shape", "color") if field in factors.target_fields)
        candidate_fields = list(preferred_fields if preferred_fields else factors.target_fields)
        records: list[RecordSpec] = []
        seen: set[tuple[str, ...]] = set()

        for _ in range(count):
            built = False
            for _attempt in range(200):
                field = rng.choice(candidate_fields)
                fixed = {f: target_definition[f] for f in factors.target_fields}
                fixed[field] = rng.choice([v for v in FIELD_VALUES[field] if v != target_definition[field]])

                if forbidden_field is not None and forbidden_value is not None:
                    if forbidden_field in FIELD_VALUES:
                        fixed[forbidden_field] = rng.choice([v for v in FIELD_VALUES[forbidden_field] if v != forbidden_value])

                record = self._random_record(rng, fixed=fixed)
                key = tuple(record.value(field_name) for field_name in FIELD_ORDER)
                if key in seen:
                    continue
                if any(matcher(record) for matcher in matchers):
                    continue
                seen.add(key)
                records.append(record)
                built = True
                break
            if not built:
                raise GenerationError("Failed to build near-miss records")
        return records

    def _build_forbidden_feature_violations(
        self,
        rng: random.Random,
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        forbidden_field: str,
        forbidden_value: str,
    ) -> list[RecordSpec]:
        records = []
        for _ in range(factors.forbidden_feature_violation_count):
            fixed = dict(target_definition)
            fixed[forbidden_field] = forbidden_value
            records.append(self._random_record(rng, fixed=fixed))
        return records

    def _build_unrelated_records(self, rng: random.Random, factors: FeatureSensitiveTextFactors, matchers: list, count: int) -> list[RecordSpec]:
        records = []
        for _ in range(count):
            for _ in range(200):
                record = self._random_record(rng, fixed={})
                if all(not matcher(record) for matcher in matchers) and self._is_unrelated_to_definitions(record, factors, matchers):
                    records.append(record)
                    break
            else:
                raise GenerationError("Failed to sample enough unrelated records")
        return records

    def _build_extra_filler(self, rng: random.Random, factors: FeatureSensitiveTextFactors, matchers: list, count: int) -> list[RecordSpec]:
        records = []
        for _ in range(count):
            for _ in range(200):
                record = self._random_record(rng, fixed={})
                if all(not matcher(record) for matcher in matchers):
                    records.append(record)
                    break
            else:
                raise GenerationError("Failed to sample enough filler records")
        return records

    def _is_unrelated_to_definitions(self, record: RecordSpec, factors: FeatureSensitiveTextFactors, matchers: list) -> bool:
        return all(not matcher(record) for matcher in matchers)

    def _arrange_records(self, rng: random.Random, records: list[RecordSpec], matcher, position_mode: PositionMode) -> list[RecordSpec]:
        arranged = list(records)
        if position_mode == "random":
            rng.shuffle(arranged)
            return arranged
        targets = [record for record in arranged if matcher(record)]
        non_targets = [record for record in arranged if not matcher(record)]
        rng.shuffle(targets)
        rng.shuffle(non_targets)
        if not targets:
            rng.shuffle(arranged)
            return arranged
        if position_mode == "front_loaded":
            return targets + non_targets
        if position_mode == "back_loaded":
            return non_targets + targets
        if position_mode == "clustered":
            start = 0 if not non_targets else rng.randint(0, len(non_targets))
            return non_targets[:start] + targets + non_targets[start:]
        rng.shuffle(arranged)
        return arranged

    def _sample_target_spec(self, rng: random.Random, target_fields: tuple[str, ...]) -> dict[str, str]:
        spec: dict[str, str] = {}
        for field in target_fields:
            if field == "size":
                spec[field] = rng.choices(
                    population=["small", "medium", "large", "xl"],
                    weights=[30, 20, 20, 30],
                    k=1,
                )[0]
            else:
                spec[field] = rng.choice(FIELD_VALUES[field])
        return spec

    def _sample_secondary_target_spec(
        self,
        rng: random.Random,
        target_fields: tuple[str, ...],
        primary: dict[str, str],
    ) -> dict[str, str]:
        for _ in range(200):
            candidate = self._sample_target_spec(rng, target_fields)
            if candidate != primary and any(candidate.get(field) != primary.get(field) for field in target_fields if field in primary):
                return candidate
        raise GenerationError("Failed to sample disjoint secondary target definition")

    def _sample_field_value(self, rng: random.Random, field: str) -> str:
        if field == "size":
            return rng.choices(
                population=["small", "medium", "large", "xl"],
                weights=[40, 30, 20, 10],
                k=1,
            )[0]
        return rng.choice(FIELD_VALUES[field])

    def _random_record(self, rng: random.Random, *, fixed: dict[str, str]) -> RecordSpec:
        values: dict[str, str] = {}
        for field in FIELD_ORDER:
            values[field] = fixed[field] if field in fixed else self._sample_field_value(rng, field)
        return RecordSpec(
            color=values["color"],
            shape=values["shape"],
            marker=values["marker"],
            size=values["size"],
            pattern=values["pattern"],
            zone=values["zone"],
            code=values["code"],
        )  # type: ignore[arg-type]

    def _standard_matcher(self, factors: FeatureSensitiveTextFactors, target_definition: dict[str, str]):
        return lambda record: self._matches_target(record, factors.target_fields, target_definition)

    def _matches_target(self, record: RecordSpec, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> bool:
        return all(record.value(field) == target_definition[field] for field in target_fields)

    def _matches_negation_rule(
        self,
        record: RecordSpec,
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
        forbidden_field: str,
        forbidden_value: str,
    ) -> bool:
        return self._matches_target(record, target_fields, target_definition) and record.value(forbidden_field) != forbidden_value

    def _matching_line_numbers(self, records: list[RecordSpec], target_fields: tuple[str, ...], target_definition: dict[str, str]) -> list[int]:
        return [i + 1 for i, record in enumerate(records) if self._matches_target(record, target_fields, target_definition)]

    def _passes_standard_constraints(
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
                if self._simplified_rule_count(records, factors.target_fields, target_definition, field) <= 0:
                    return False
        else:
            for field in factors.target_fields:
                if self._simplified_rule_count(records, factors.target_fields, target_definition, field) == factors.target_count:
                    return False
        if self._count_same_color_wrong_shape(records, factors, target_definition) < factors.same_color_wrong_shape_count:
            return False
        if self._count_same_shape_wrong_color(records, factors, target_definition) < factors.same_shape_wrong_color_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, "color") < factors.same_core_wrong_color_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, "shape") < factors.same_core_wrong_shape_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, "marker") < factors.same_core_wrong_marker_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, "size") < factors.same_core_wrong_size_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, "pattern") < factors.same_core_wrong_pattern_count:
            return False
        return True

    def _passes_negation_constraints(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        forbidden_field: str,
        forbidden_value: str,
        gold_lines: list[int],
    ) -> bool:
        if len(gold_lines) != factors.target_count:
            return False
        for field in factors.target_fields:
            if self._simplified_rule_count(records, factors.target_fields, target_definition, field) == factors.target_count:
                return False
        if self._count_same_color_wrong_shape(records, factors, target_definition) < factors.same_color_wrong_shape_count:
            return False
        if self._count_same_shape_wrong_color(records, factors, target_definition) < factors.same_shape_wrong_color_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, "marker") < factors.same_core_wrong_marker_count:
            return False
        if self._count_same_core_wrong_field(records, factors, target_definition, "size") < factors.same_core_wrong_size_count:
            return False
        violations = sum(
            1
            for rec in records
            if self._matches_target(rec, factors.target_fields, target_definition) and rec.value(forbidden_field) == forbidden_value
        )
        return violations >= factors.forbidden_feature_violation_count

    def _passes_disjunction_constraints(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        definition_a: dict[str, str],
        definition_b: dict[str, str],
        a_lines: list[int],
        b_lines: list[int],
        gold_lines: list[int],
    ) -> bool:
        if len(gold_lines) != factors.target_count + (factors.secondary_target_count or 0):
            return False
        if set(a_lines) & set(b_lines):
            return False
        if factors.variant == "easy":
            if self._count_same_color_wrong_shape(records, factors, definition_a) < factors.same_color_wrong_shape_count:
                return False
            if self._count_same_shape_wrong_color(records, factors, definition_a) < factors.same_shape_wrong_color_count:
                return False
        else:
            for field, needed in [
                ("color", factors.same_core_wrong_color_count),
                ("shape", factors.same_core_wrong_shape_count),
                ("marker", factors.same_core_wrong_marker_count),
                ("size", factors.same_core_wrong_size_count),
            ]:
                total = self._count_same_core_wrong_field(records, factors, definition_a, field) + self._count_same_core_wrong_field(records, factors, definition_b, field)
                if total < needed:
                    return False
        return True

    def _simplified_rule_count(
        self,
        records: list[RecordSpec],
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
        drop_field: str,
    ) -> int:
        reduced = tuple(field for field in target_fields if field != drop_field)
        return sum(1 for rec in records if all(rec.value(field) == target_definition[field] for field in reduced))

    def _count_same_color_wrong_shape(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> int:
        if "color" not in factors.target_fields or "shape" not in factors.target_fields:
            return 0
        count = 0
        for rec in records:
            if rec.color != target_definition["color"] or rec.shape == target_definition["shape"]:
                continue
            if any(rec.value(field) != target_definition[field] for field in factors.target_fields if field in {"marker", "size", "pattern"}):
                continue
            count += 1
        return count

    def _count_same_shape_wrong_color(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> int:
        if "color" not in factors.target_fields or "shape" not in factors.target_fields:
            return 0
        count = 0
        for rec in records:
            if rec.shape != target_definition["shape"] or rec.color == target_definition["color"]:
                continue
            if any(rec.value(field) != target_definition[field] for field in factors.target_fields if field in {"marker", "size", "pattern"}):
                continue
            count += 1
        return count

    def _count_same_core_wrong_field(
        self,
        records: list[RecordSpec],
        factors: FeatureSensitiveTextFactors,
        target_definition: dict[str, str],
        field: str,
    ) -> int:
        if field not in factors.target_fields:
            return 0
        count = 0
        for rec in records:
            ok = True
            for f in factors.target_fields:
                if f == field:
                    if rec.value(f) == target_definition[f]:
                        ok = False
                        break
                elif rec.value(f) != target_definition[f]:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    def _field_description(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        return ", ".join(f"{field}={target_definition[field]}" for field in target_fields)

    def _build_count_instruction(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        return (
            f"Count the entries matching all of the following attributes: {self._field_description(target_fields, target_definition)}.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n- \"count\" must be an integer\n- Do not include any extra keys\n- Return only the JSON object"
        )

    def _build_filter_instruction(self, target_fields: tuple[str, ...], target_definition: dict[str, str]) -> str:
        return (
            f"Return the line numbers (1-based) of the entries matching all of the following attributes: {self._field_description(target_fields, target_definition)}.\n"
            'Respond with a JSON object of the form {"lines": [<sorted unique integers>]}.\n'
            "Rules:\n- Use 1-based indexing\n- Sort ascending\n- Do not include duplicates\n- Return only the JSON object"
        )

    def _build_negation_count_instruction(
        self,
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
        forbidden_field: str,
        forbidden_value: str,
    ) -> str:
        return (
            f"Count the entries matching all of the following attributes: {self._field_description(target_fields, target_definition)}, but not {forbidden_field}={forbidden_value}.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n- \"count\" must be an integer\n- Do not include any extra keys\n- Return only the JSON object"
        )

    def _build_negation_filter_instruction(
        self,
        target_fields: tuple[str, ...],
        target_definition: dict[str, str],
        forbidden_field: str,
        forbidden_value: str,
    ) -> str:
        return (
            f"Return the line numbers (1-based) of the entries matching all of the following attributes: {self._field_description(target_fields, target_definition)}, but not {forbidden_field}={forbidden_value}.\n"
            'Respond with a JSON object of the form {"lines": [<sorted unique integers>]}.\n'
            "Rules:\n- Use 1-based indexing\n- Sort ascending\n- Do not include duplicates\n- Return only the JSON object"
        )

    def _build_disjunction_count_instruction(
        self,
        fields_a: tuple[str, ...],
        def_a: dict[str, str],
        fields_b: tuple[str, ...] | None,
        def_b: dict[str, str],
    ) -> str:
        return (
            "Count the entries matching either of the following attribute sets:\n"
            f"1. {self._field_description(fields_a, def_a)}\n"
            f"2. {self._field_description(fields_b or (), def_b)}\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n- \"count\" must be an integer\n- Do not include any extra keys\n- Return only the JSON object"
        )

    def _build_disjunction_filter_instruction(
        self,
        fields_a: tuple[str, ...],
        def_a: dict[str, str],
        fields_b: tuple[str, ...] | None,
        def_b: dict[str, str],
    ) -> str:
        return (
            "Return the line numbers (1-based) of the entries matching either of the following attribute sets:\n"
            f"1. {self._field_description(fields_a, def_a)}\n"
            f"2. {self._field_description(fields_b or (), def_b)}\n"
            'Respond with a JSON object of the form {"lines": [<sorted unique integers>]}.\n'
            "Rules:\n- Use 1-based indexing\n- Sort ascending\n- Do not include duplicates\n- Return only the JSON object"
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
        "secondary_target_definition": None if scene.secondary_target_definition is None else json.dumps(scene.secondary_target_definition, sort_keys=True),
        "secondary_target_fields": None if factors.secondary_target_fields is None else json.dumps(list(factors.secondary_target_fields)),
        "secondary_target_count": factors.secondary_target_count,
        "relation": factors.relation,
        "forbidden_field": factors.forbidden_field,
        "forbidden_value": scene.forbidden_value,
        "num_records": factors.num_records,
        "target_feature_count": factors.target_feature_count,
        "target_count": factors.target_count,
        "active_fields": json.dumps(list(factors.active_fields)),
        "target_fields": json.dumps(list(factors.target_fields)),
        "irrelevant_noise_fields": json.dumps(list(factors.irrelevant_noise_fields)),
        "same_color_wrong_shape_count": factors.same_color_wrong_shape_count,
        "same_shape_wrong_color_count": factors.same_shape_wrong_color_count,
        "same_core_wrong_color_count": factors.same_core_wrong_color_count,
        "same_core_wrong_shape_count": factors.same_core_wrong_shape_count,
        "same_core_wrong_marker_count": factors.same_core_wrong_marker_count,
        "same_core_wrong_size_count": factors.same_core_wrong_size_count,
        "same_core_wrong_pattern_count": factors.same_core_wrong_pattern_count,
        "forbidden_feature_violation_count": factors.forbidden_feature_violation_count,
        "near_miss_count": factors.near_miss_count,
        "unrelated_count": factors.unrelated_count,
    }