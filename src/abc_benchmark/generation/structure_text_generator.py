from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from typing import Literal, Sequence

EntityType = Literal["alpha", "beta", "gamma", "delta"]
ColorValue = Literal["red", "blue", "green", "yellow"]
ShapeValue = Literal["circle", "square", "triangle"]
MarkerValue = Literal["0", "1"]
SizeValue = Literal["small", "medium", "large"]
PatternValue = Literal["solid", "striped", "dotted"]

StructureType = Literal["grouping", "relation", "scope", "global_local"]
RegimeName = Literal[
    "baseline_structure",
    "structure_depth_sweep",
    "binding_distance_sweep",
    "confound_sweep",
    "target_count_x_structure_depth",
    "serialization_style_sweep",
    "combined",
]
SerializationStyle = Literal["compact", "tagged", "nested"]

ENTITY_TYPES: tuple[EntityType, ...] = ("alpha", "beta", "gamma", "delta")
COLORS: tuple[ColorValue, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[ShapeValue, ...] = ("circle", "square", "triangle")
MARKERS: tuple[MarkerValue, ...] = ("0", "1")
SIZES: tuple[SizeValue, ...] = ("small", "medium", "large")
PATTERNS: tuple[PatternValue, ...] = ("solid", "striped", "dotted")


@dataclass(frozen=True)
class ObjectSpec:
    entity: EntityType
    color: ColorValue
    shape: ShapeValue
    marker: MarkerValue
    size: SizeValue
    pattern: PatternValue

    def value(self, field: str) -> str:
        return str(getattr(self, field))


@dataclass(frozen=True)
class SceneRecord:
    leader: ObjectSpec
    follower: ObjectSpec
    tag: str


@dataclass(frozen=True)
class StructureTextFactors:
    regime: RegimeName
    regime_level: str
    structure_type: StructureType
    structure_depth: int
    binding_distance: str
    serialization_style: SerializationStyle
    num_records: int
    target_count: int
    confound_count: int
    unrelated_count: int
    line_length_noise: int


@dataclass
class StructureSceneSpec:
    seed: int
    regime: RegimeName
    regime_level: str
    structure_type: StructureType
    count_instruction: str
    filter_instruction: str
    text_input: str
    count_prompt: str
    filter_prompt: str
    gold_count: int
    gold_lines: list[int]
    target_definition: dict[str, str]
    factors: StructureTextFactors
    records: list[SceneRecord]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["records"] = [
            {
                "leader": asdict(record.leader),
                "follower": asdict(record.follower),
                "tag": record.tag,
            }
            for record in self.records
        ]
        return payload


class GenerationError(RuntimeError):
    pass


class StructureTextSelectiveAttentionGenerator:
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
        factors: StructureTextFactors | None = None,
        regime: RegimeName = "combined",
        regime_level: str = "medium",
        target_count_override: int | None = None,
        serialization_style: SerializationStyle | None = None,
    ) -> StructureSceneSpec:
        local_rng = random.Random(seed) if seed is not None else self.rng

        if factors is None:
            factors = self.sample_factors(rng=local_rng, regime=regime, level=regime_level)

        if target_count_override is not None:
            factors = self._override_target_count(factors, target_count_override)
        if serialization_style is not None:
            factors = replace(factors, serialization_style=serialization_style)

        for _ in range(self.max_attempts):
            target_definition = self._sample_target_definition(local_rng)
            records = self._build_records(local_rng, factors, target_definition)
            local_rng.shuffle(records)

            if not self._passes_constraints(records, factors, target_definition):
                continue

            text_input = self._render_records(records, factors)
            count_instruction = self._build_count_instruction(factors, target_definition)
            filter_instruction = self._build_filter_instruction(factors, target_definition)
            count_prompt = f"{count_instruction}\n\n{text_input}"
            filter_prompt = f"{filter_instruction}\n\n{text_input}"
            gold_lines = self._matching_line_numbers(records, factors, target_definition)

            return StructureSceneSpec(
                seed=seed if seed is not None else -1,
                regime=factors.regime,
                regime_level=factors.regime_level,
                structure_type=factors.structure_type,
                count_instruction=count_instruction,
                filter_instruction=filter_instruction,
                text_input=text_input,
                count_prompt=count_prompt,
                filter_prompt=filter_prompt,
                gold_count=len(gold_lines),
                gold_lines=gold_lines,
                target_definition=target_definition,
                factors=factors,
                records=records,
            )

        raise GenerationError("Failed to generate a valid structure-text scene within max_attempts")

    def generate_many(
        self,
        *,
        count: int,
        start_seed: int = 0,
        regime: RegimeName = "combined",
        regime_level: str = "medium",
        target_count_override: int | None = None,
        serialization_style: SerializationStyle | None = None,
    ) -> list[StructureSceneSpec]:
        return [
            self.generate(
                seed=start_seed + i,
                regime=regime,
                regime_level=regime_level,
                target_count_override=target_count_override,
                serialization_style=serialization_style,
            )
            for i in range(count)
        ]

    def sample_factors(
        self,
        *,
        rng: random.Random,
        regime: RegimeName,
        level: str,
    ) -> StructureTextFactors:
        if regime == "baseline_structure":
            return StructureTextFactors(
                regime="baseline_structure",
                regime_level="baseline",
                structure_type="grouping",
                structure_depth=1,
                binding_distance="near",
                serialization_style="compact",
                num_records=10,
                target_count=rng.randint(1, 2),
                confound_count=3,
                unrelated_count=5,
                line_length_noise=0,
            )
        if regime == "structure_depth_sweep":
            mapping = {
                "shallow": ("grouping", 1, "near", "compact", 12, 2, 4, 6, 0),
                "medium": ("scope", 2, "near", "tagged", 14, 2, 5, 7, 1),
                "nested": ("global_local", 3, "far", "nested", 16, 2, 6, 8, 2),
            }
            return self._factors_from_mapping(regime, level, mapping)
        if regime == "binding_distance_sweep":
            mapping = {
                "near": ("relation", 1, "near", "compact", 14, 3, 4, 7, 0),
                "medium": ("relation", 2, "medium", "tagged", 14, 3, 5, 6, 1),
                "far": ("relation", 2, "far", "nested", 14, 3, 6, 5, 2),
            }
            return self._factors_from_mapping(regime, level, mapping)
        if regime == "confound_sweep":
            mapping = {
                "low": ("grouping", 1, "near", "compact", 14, 3, 3, 8, 0),
                "medium": ("grouping", 2, "medium", "tagged", 18, 3, 6, 9, 1),
                "high": ("scope", 2, "medium", "tagged", 24, 3, 10, 11, 1),
                "extreme": ("global_local", 3, "far", "nested", 30, 3, 15, 12, 2),
            }
            return self._factors_from_mapping(regime, level, mapping)
        if regime == "target_count_x_structure_depth":
            mapping = {
                "0_shallow": ("grouping", 1, "near", "compact", 12, 0, 5, 7, 0),
                "0_nested": ("global_local", 3, "far", "nested", 16, 0, 8, 8, 2),
                "3_shallow": ("grouping", 1, "near", "compact", 12, 3, 4, 5, 0),
                "3_nested": ("global_local", 3, "far", "nested", 18, 3, 8, 7, 2),
            }
            return self._factors_from_mapping(regime, level, mapping)
        if regime == "serialization_style_sweep":
            mapping = {
                "compact": ("grouping", 1, "near", "compact", 14, 3, 5, 6, 0),
                "tagged": ("grouping", 2, "medium", "tagged", 14, 3, 5, 6, 1),
                "nested": ("grouping", 2, "medium", "nested", 14, 3, 5, 6, 2),
            }
            return self._factors_from_mapping(regime, level, mapping)
        if regime == "combined":
            mapping = {
                "easy": ("grouping", 1, "near", "compact", 16, 3, 5, 8, 0),
                "medium": ("scope", 2, "medium", "tagged", 22, 4, 8, 10, 1),
                "hard": ("global_local", 3, "far", "nested", 30, 5, 13, 12, 2),
            }
            return self._factors_from_mapping(regime, level, mapping)
        raise ValueError(f"Unknown regime: {regime}")

    def _factors_from_mapping(
        self,
        regime: RegimeName,
        level: str,
        mapping: dict[str, tuple[StructureType, int, str, SerializationStyle, int, int, int, int, int]],
    ) -> StructureTextFactors:
        if level not in mapping:
            raise ValueError(f"Unknown level {level!r} for regime {regime}")
        structure_type, structure_depth, binding_distance, serialization_style, num_records, target_count, confound_count, unrelated_count, line_length_noise = mapping[level]
        return StructureTextFactors(
            regime=regime,
            regime_level=level,
            structure_type=structure_type,
            structure_depth=structure_depth,
            binding_distance=binding_distance,
            serialization_style=serialization_style,
            num_records=num_records,
            target_count=target_count,
            confound_count=confound_count,
            unrelated_count=unrelated_count,
            line_length_noise=line_length_noise,
        )

    def _override_target_count(
        self,
        factors: StructureTextFactors,
        target_count_override: int,
    ) -> StructureTextFactors:
        if target_count_override < 0:
            raise ValueError("target_count_override must be >= 0")
        delta = target_count_override - factors.target_count
        return replace(
            factors,
            target_count=target_count_override,
            num_records=factors.num_records + max(0, delta),
            unrelated_count=factors.unrelated_count + max(0, -delta),
        )

    def _sample_target_definition(self, rng: random.Random) -> dict[str, str]:
        return {
            "leader_color": rng.choice(COLORS),
            "leader_shape": rng.choice(SHAPES),
            "follower_marker": rng.choice(MARKERS),
            "scope_size": rng.choice(SIZES),
            "local_pattern": rng.choice(PATTERNS),
        }

    def _build_records(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> list[SceneRecord]:
        records: list[SceneRecord] = []
        records.extend(self._build_target_records(rng, factors, target_definition))
        records.extend(self._build_confound_records(rng, factors, target_definition))

        remaining = factors.num_records - len(records)
        if remaining < factors.unrelated_count:
            raise GenerationError("Factor bundle over-allocates records before unrelated filler")

        records.extend(self._build_unrelated_records(rng, factors, target_definition, factors.unrelated_count))
        remaining = factors.num_records - len(records)
        if remaining > 0:
            records.extend(self._build_extra_filler(rng, factors, target_definition, remaining))
        return records

    def _build_target_records(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> list[SceneRecord]:
        return [self._random_target_record(rng, factors, target_definition) for _ in range(factors.target_count)]

    def _build_confound_records(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> list[SceneRecord]:
        builders = {
            "grouping": self._random_grouping_confound,
            "relation": self._random_relation_confound,
            "scope": self._random_scope_confound,
            "global_local": self._random_global_local_confound,
        }
        build = builders[factors.structure_type]
        return [build(rng, factors, target_definition) for _ in range(factors.confound_count)]

    def _build_unrelated_records(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
        count: int,
    ) -> list[SceneRecord]:
        records: list[SceneRecord] = []
        for _ in range(count):
            for _ in range(100):
                record = self._random_unrelated_record(rng, factors)
                if not self._matches_target(record, factors, target_definition):
                    records.append(record)
                    break
            else:
                raise GenerationError("Failed to sample enough unrelated records")
        return records

    def _build_extra_filler(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
        count: int,
    ) -> list[SceneRecord]:
        records: list[SceneRecord] = []
        for _ in range(count):
            for _ in range(100):
                record = self._random_unrelated_record(rng, factors)
                if not self._matches_target(record, factors, target_definition):
                    records.append(record)
                    break
            else:
                raise GenerationError("Failed to sample enough filler records")
        return records

    def _random_target_record(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> SceneRecord:
        leader = self._random_object(
            rng,
            fixed={
                "color": target_definition["leader_color"],
                "shape": target_definition["leader_shape"],
                "size": target_definition["scope_size"],
            },
        )
        follower = self._random_object(
            rng,
            fixed={
                "marker": target_definition["follower_marker"],
                "pattern": target_definition["local_pattern"],
            },
        )
        return SceneRecord(leader=leader, follower=follower, tag=self._random_tag(rng, factors))

    def _random_grouping_confound(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> SceneRecord:
        confound_kind = rng.choice(["leader_match_only", "follower_match_only", "cross_boundaries"])
        if confound_kind == "leader_match_only":
            leader = self._random_object(
                rng,
                fixed={
                    "color": target_definition["leader_color"],
                    "shape": target_definition["leader_shape"],
                },
            )
            follower = self._random_object(
                rng,
                fixed={
                    "marker": self._different_choice(rng, MARKERS, target_definition["follower_marker"]),
                    "pattern": target_definition["local_pattern"],
                },
            )
        elif confound_kind == "follower_match_only":
            leader = self._random_object(
                rng,
                fixed={
                    "color": self._different_choice(rng, COLORS, target_definition["leader_color"]),
                    "shape": target_definition["leader_shape"],
                },
            )
            follower = self._random_object(
                rng,
                fixed={
                    "marker": target_definition["follower_marker"],
                    "pattern": target_definition["local_pattern"],
                },
            )
        else:
            leader = self._random_object(
                rng,
                fixed={
                    "color": target_definition["leader_color"],
                    "shape": self._different_choice(rng, SHAPES, target_definition["leader_shape"]),
                },
            )
            follower = self._random_object(
                rng,
                fixed={
                    "marker": target_definition["follower_marker"],
                    "pattern": self._different_choice(rng, PATTERNS, target_definition["local_pattern"]),
                },
            )
        return SceneRecord(leader=leader, follower=follower, tag=self._random_tag(rng, factors))

    def _random_relation_confound(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> SceneRecord:
        reverse_relation = rng.choice([True, False])
        if reverse_relation:
            leader = self._random_object(
                rng,
                fixed={
                    "marker": target_definition["follower_marker"],
                    "pattern": target_definition["local_pattern"],
                },
            )
            follower = self._random_object(
                rng,
                fixed={
                    "color": target_definition["leader_color"],
                    "shape": target_definition["leader_shape"],
                },
            )
        else:
            leader = self._random_object(
                rng,
                fixed={
                    "color": target_definition["leader_color"],
                    "shape": self._different_choice(rng, SHAPES, target_definition["leader_shape"]),
                },
            )
            follower = self._random_object(
                rng,
                fixed={
                    "marker": target_definition["follower_marker"],
                    "pattern": target_definition["local_pattern"],
                },
            )
        return SceneRecord(leader=leader, follower=follower, tag=self._random_tag(rng, factors))

    def _random_scope_confound(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> SceneRecord:
        leader = self._random_object(
            rng,
            fixed={
                "color": target_definition["leader_color"],
                "shape": target_definition["leader_shape"],
                "size": self._different_choice(rng, SIZES, target_definition["scope_size"]),
            },
        )
        follower = self._random_object(
            rng,
            fixed={
                "marker": target_definition["follower_marker"],
                "pattern": target_definition["local_pattern"],
            },
        )
        return SceneRecord(leader=leader, follower=follower, tag=self._random_tag(rng, factors))

    def _random_global_local_confound(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> SceneRecord:
        leader = self._random_object(
            rng,
            fixed={
                "color": target_definition["leader_color"],
                "shape": target_definition["leader_shape"],
            },
        )
        follower = self._random_object(
            rng,
            fixed={
                "marker": target_definition["follower_marker"],
                "pattern": self._different_choice(rng, PATTERNS, target_definition["local_pattern"]),
                "size": target_definition["scope_size"],
            },
        )
        return SceneRecord(leader=leader, follower=follower, tag=self._random_tag(rng, factors))

    def _random_unrelated_record(
        self,
        rng: random.Random,
        factors: StructureTextFactors,
    ) -> SceneRecord:
        return SceneRecord(
            leader=self._random_object(rng, fixed={}),
            follower=self._random_object(rng, fixed={}),
            tag=self._random_tag(rng, factors),
        )

    def _random_object(self, rng: random.Random, fixed: dict[str, str]) -> ObjectSpec:
        values = {
            "entity": fixed.get("entity", rng.choice(ENTITY_TYPES)),
            "color": fixed.get("color", rng.choice(COLORS)),
            "shape": fixed.get("shape", rng.choice(SHAPES)),
            "marker": fixed.get("marker", rng.choice(MARKERS)),
            "size": fixed.get("size", rng.choice(SIZES)),
            "pattern": fixed.get("pattern", rng.choice(PATTERNS)),
        }
        return ObjectSpec(
            entity=values["entity"],  # type: ignore[arg-type]
            color=values["color"],  # type: ignore[arg-type]
            shape=values["shape"],  # type: ignore[arg-type]
            marker=values["marker"],  # type: ignore[arg-type]
            size=values["size"],  # type: ignore[arg-type]
            pattern=values["pattern"],  # type: ignore[arg-type]
        )

    def _random_tag(self, rng: random.Random, factors: StructureTextFactors) -> str:
        parts = [rng.choice(["G1", "G2", "G3", "G4"])]
        for _ in range(factors.line_length_noise):
            parts.append(rng.choice(["aux=left", "aux=right", "flag=on", "flag=off"]))
        return " ".join(parts)

    def _different_choice(self, rng: random.Random, values: Sequence[str], current: str) -> str:
        return rng.choice([value for value in values if value != current])

    def _render_records(self, records: list[SceneRecord], factors: StructureTextFactors) -> str:
        return "\n".join(self._render_line(index=i + 1, record=record, factors=factors) for i, record in enumerate(records))

    def _render_line(self, index: int, record: SceneRecord, factors: StructureTextFactors) -> str:
        if factors.serialization_style == "compact":
            return (
                f"{index}. {record.tag} | "
                f"L[{record.leader.entity}:{record.leader.color},{record.leader.shape},{record.leader.size}] -> "
                f"F[{record.follower.entity}:{record.follower.marker},{record.follower.pattern}]"
            )
        if factors.serialization_style == "tagged":
            return (
                f"{index}. {record.tag} | leader(entity={record.leader.entity}, color={record.leader.color}, "
                f"shape={record.leader.shape}, size={record.leader.size}) ; "
                f"follower(entity={record.follower.entity}, marker={record.follower.marker}, "
                f"pattern={record.follower.pattern})"
            )
        return (
            f"{index}. {record.tag} | block={{leader:{{entity:{record.leader.entity}, color:{record.leader.color}, "
            f"shape:{record.leader.shape}, size:{record.leader.size}}}, "
            f"follower:{{entity:{record.follower.entity}, marker:{record.follower.marker}, pattern:{record.follower.pattern}}}}}"
        )

    def _build_count_instruction(
        self,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> str:
        rule = self._rule_description(factors, target_definition)
        return (
            f"Count the lines matching this structured rule: {rule}.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n"
            '- "count" must be an integer\n'
            "- Use the structure inside each line, not cross-line evidence\n"
            "- Return only the JSON object"
        )

    def _build_filter_instruction(
        self,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> str:
        rule = self._rule_description(factors, target_definition)
        return (
            f"Return the line numbers (1-based) matching this structured rule: {rule}.\n"
            'Respond with a JSON object of the form {"lines": [<sorted unique integers>]}.\n'
            "Rules:\n"
            "- Use 1-based indexing\n"
            "- Sort ascending\n"
            "- Do not include duplicates\n"
            "- Use the leader/follower structure inside the same line only\n"
            "- Return only the JSON object"
        )

    def _rule_description(
        self,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> str:
        core = (
            f"leader.color={target_definition['leader_color']}, "
            f"leader.shape={target_definition['leader_shape']}, "
            f"follower.marker={target_definition['follower_marker']}"
        )
        if factors.structure_type == "grouping":
            return f"within one line, the leader must satisfy ({core}) and the follower must also have pattern={target_definition['local_pattern']}"
        if factors.structure_type == "relation":
            return f"the leader, not the follower, must satisfy leader.color={target_definition['leader_color']} and leader.shape={target_definition['leader_shape']}, while the follower must satisfy marker={target_definition['follower_marker']} and pattern={target_definition['local_pattern']}"
        if factors.structure_type == "scope":
            return f"inside the leader scope, the leader must satisfy ({core}) and leader.size={target_definition['scope_size']}; the follower must satisfy pattern={target_definition['local_pattern']}"
        return f"the local pair must satisfy ({core}), follower.pattern={target_definition['local_pattern']}, and the pair must preserve the same local block rather than mixing attributes across subparts"

    def _matches_target(
        self,
        record: SceneRecord,
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> bool:
        leader_match = (
            record.leader.color == target_definition["leader_color"]
            and record.leader.shape == target_definition["leader_shape"]
        )
        follower_match = (
            record.follower.marker == target_definition["follower_marker"]
            and record.follower.pattern == target_definition["local_pattern"]
        )

        if factors.structure_type == "grouping":
            return leader_match and follower_match
        if factors.structure_type == "relation":
            return leader_match and follower_match
        if factors.structure_type == "scope":
            return leader_match and follower_match and record.leader.size == target_definition["scope_size"]
        if factors.structure_type == "global_local":
            return leader_match and follower_match and record.leader.size == target_definition["scope_size"]
        raise ValueError(f"Unknown structure_type: {factors.structure_type}")

    def _matching_line_numbers(
        self,
        records: list[SceneRecord],
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> list[int]:
        return [i + 1 for i, record in enumerate(records) if self._matches_target(record, factors, target_definition)]

    def _passes_constraints(
        self,
        records: list[SceneRecord],
        factors: StructureTextFactors,
        target_definition: dict[str, str],
    ) -> bool:
        gold_lines = self._matching_line_numbers(records, factors, target_definition)
        if len(gold_lines) != factors.target_count:
            return False

        leader_only_count = sum(
            1
            for record in records
            if record.leader.color == target_definition["leader_color"]
            and record.leader.shape == target_definition["leader_shape"]
        )
        follower_only_count = sum(
            1
            for record in records
            if record.follower.marker == target_definition["follower_marker"]
            and record.follower.pattern == target_definition["local_pattern"]
        )

        if factors.target_count == 0:
            return leader_only_count > 0 and follower_only_count > 0

        return leader_only_count > factors.target_count and follower_only_count > factors.target_count


def scene_to_dataset_row(scene: StructureSceneSpec) -> dict[str, object]:
    factors = scene.factors
    return {
        "seed": scene.seed,
        "regime": scene.regime,
        "regime_level": scene.regime_level,
        "structure_type": scene.structure_type,
        "count_instruction": scene.count_instruction,
        "filter_instruction": scene.filter_instruction,
        "text_input": scene.text_input,
        "count_prompt": scene.count_prompt,
        "filter_prompt": scene.filter_prompt,
        "gold_count": scene.gold_count,
        "gold_lines": json.dumps(scene.gold_lines),
        "target_definition": json.dumps(scene.target_definition, sort_keys=True),
        "structure_depth": factors.structure_depth,
        "binding_distance": factors.binding_distance,
        "serialization_style": factors.serialization_style,
        "num_records": factors.num_records,
        "target_count": factors.target_count,
        "confound_count": factors.confound_count,
        "unrelated_count": factors.unrelated_count,
        "line_length_noise": factors.line_length_noise,
    }
