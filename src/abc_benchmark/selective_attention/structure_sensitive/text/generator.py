from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from typing import Literal

ColorValue = Literal["red", "blue", "green", "yellow"]
ShapeValue = Literal["circle", "square", "triangle"]
SizeValue = Literal["small", "medium", "large"]
PatternValue = Literal["solid", "striped", "dotted"]

FamilyName = Literal["selective_attention"]
AttentionalBasisName = Literal["structure_sensitive"]
ModalityName = Literal["text"]

PrincipleName = Literal[
    "paragraph_proximity",
    "section_common_region",
    "format_similarity",
    "scope_indentation",
    "continuation_chain",
]
DimensionName = Literal["baseline", "principle", "target_count", "combined"]
RenderStyle = Literal[
    "paragraph_blocks",
    "section_blocks",
    "format_runs",
    "indent_tree",
    "chain_links",
]
FormatGroupStyle = Literal["dash", "star", "numbered", "alpha"]
SectionStyle = Literal["header"]
ChainStyle = Literal["arrow", "continues_to"]

COLORS: tuple[ColorValue, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[ShapeValue, ...] = ("circle", "square", "triangle")
SIZES: tuple[SizeValue, ...] = ("small", "medium", "large")
PATTERNS: tuple[PatternValue, ...] = ("solid", "striped", "dotted")
FORMAT_GROUP_STYLES: tuple[FormatGroupStyle, ...] = ("dash", "star", "numbered", "alpha")
CHAIN_STYLES: tuple[ChainStyle, ...] = ("arrow", "continues_to")


@dataclass(frozen=True)
class TextItem:
    item_id: str
    color: ColorValue
    shape: ShapeValue
    size: SizeValue
    pattern: PatternValue
    group_id: str
    role: str = "member"
    format_style: FormatGroupStyle | None = None
    indent_level: int = 0
    next_id: str | None = None
    section_path: tuple[str, ...] | None = None
    is_anchor: bool = False


@dataclass(frozen=True)
class StructureSensitiveTextFactors:
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    principle: PrincipleName
    render_style: RenderStyle
    num_groups: int
    min_items_per_group: int
    max_items_per_group: int
    target_in_anchor_group: int
    target_outside_anchor_group: int
    non_target_in_anchor_group: int
    unrelated_count: int
    target_count_total: int
    format_styles: tuple[FormatGroupStyle, ...] | None = None
    section_style: SectionStyle | None = None
    top_level_section_count: int | None = None
    nested_section_depth_max: int | None = None
    nested_section_probability: float | None = None
    chain_style: ChainStyle | None = None
    chain_length_min: int | None = None
    chain_length_max: int | None = None


@dataclass
class StructureSensitiveTextScene:
    seed: int
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    principle: PrincipleName
    render_style: RenderStyle
    count_instruction: str
    filter_instruction: str
    text_input: str
    count_prompt: str
    filter_prompt: str
    gold_count: int
    gold_ids: list[str]
    target_definition: dict[str, str]
    anchor_item_id: str
    anchor_group_id: str
    factors: StructureSensitiveTextFactors
    items: list[TextItem]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["items"] = [asdict(item) for item in self.items]
        return payload


class GenerationError(RuntimeError):
    pass


class StructureSensitiveTextGenerator:
    def __init__(self, rng: random.Random | None = None, max_attempts: int = 1000) -> None:
        self.rng = rng or random.Random()
        self.max_attempts = max_attempts

    def generate(
        self,
        *,
        seed: int | None = None,
        factors: StructureSensitiveTextFactors | None = None,
        dimension: DimensionName = "combined",
        variant: str = "medium",
        target_count_override: int | None = None,
    ) -> StructureSensitiveTextScene:
        local_rng = random.Random(seed) if seed is not None else self.rng
        if factors is None:
            factors = self.sample_factors(rng=local_rng, dimension=dimension, variant=variant)
        if target_count_override is not None:
            factors = self._override_target_count(factors, target_count_override)

        for _ in range(self.max_attempts):
            target_definition = self._sample_target_definition(local_rng)
            items = self._build_items(local_rng, factors, target_definition)
            anchor_item_id, anchor_group_id = self._choose_anchor(local_rng, items, factors, target_definition)
            items = [replace(item, is_anchor=item.item_id == anchor_item_id) for item in items]
            text_input = self._render_items(items, factors)
            gold_ids = self._matching_ids(items, anchor_group_id, target_definition)
            if not self._passes_constraints(items, factors, target_definition, gold_ids, anchor_group_id):
                continue

            count_instruction = self._build_count_instruction(anchor_item_id, factors, target_definition)
            filter_instruction = self._build_filter_instruction(anchor_item_id, factors, target_definition)
            return StructureSensitiveTextScene(
                seed=seed if seed is not None else -1,
                family=factors.family,
                attentional_basis=factors.attentional_basis,
                modality=factors.modality,
                dimension=factors.dimension,
                variant=factors.variant,
                principle=factors.principle,
                render_style=factors.render_style,
                count_instruction=count_instruction,
                filter_instruction=filter_instruction,
                text_input=text_input,
                count_prompt=f"{count_instruction}\n\n{text_input}",
                filter_prompt=f"{filter_instruction}\n\n{text_input}",
                gold_count=len(gold_ids),
                gold_ids=gold_ids,
                target_definition=target_definition,
                anchor_item_id=anchor_item_id,
                anchor_group_id=anchor_group_id,
                factors=factors,
                items=items,
            )
        raise GenerationError("Failed to generate a valid structure-sensitive text scene within max_attempts")

    def generate_many(
        self,
        *,
        count: int,
        start_seed: int = 0,
        dimension: DimensionName = "combined",
        variant: str = "medium",
        target_count_override: int | None = None,
    ) -> list[StructureSensitiveTextScene]:
        return [
            self.generate(
                seed=start_seed + i,
                dimension=dimension,
                variant=variant,
                target_count_override=target_count_override,
            )
            for i in range(count)
        ]

    def sample_factors(
        self,
        *,
        rng: random.Random,
        dimension: DimensionName,
        variant: str,
    ) -> StructureSensitiveTextFactors:
        if dimension == "baseline":
            if variant != "simple":
                raise ValueError(f"Unknown baseline variant {variant!r}")
            return self._base_factors(
                dimension="baseline",
                variant="simple",
                principle="paragraph_proximity",
                render_style="paragraph_blocks",
                num_groups=3,
                min_items_per_group=3,
                max_items_per_group=4,
                target_in_anchor_group=1,
                target_outside_anchor_group=1,
                non_target_in_anchor_group=2,
                unrelated_count=0,
            )

        if dimension == "principle":
            mapping: dict[str, tuple[PrincipleName, RenderStyle, int, int, int, int, int, int]] = {
                "paragraph_proximity": ("paragraph_proximity", "paragraph_blocks", 3, 3, 5, 2, 2, 0),
                "section_common_region": ("section_common_region", "section_blocks", 4, 3, 5, 2, 2, 0),
                "format_similarity": ("format_similarity", "format_runs", 4, 3, 4, 2, 2, 0),
                "scope_indentation": ("scope_indentation", "indent_tree", 4, 3, 4, 2, 2, 0),
                "continuation_chain": ("continuation_chain", "chain_links", 3, 3, 4, 2, 2, 0),
            }
            if variant not in mapping:
                raise ValueError(f"Unknown principle variant {variant!r}")
            principle, render_style, num_groups, min_items, max_items, target_anchor, target_outside, unrelated = mapping[variant]
            return self._base_factors(
                dimension="principle",
                variant=variant,
                principle=principle,
                render_style=render_style,
                num_groups=num_groups,
                min_items_per_group=min_items,
                max_items_per_group=max_items,
                target_in_anchor_group=target_anchor,
                target_outside_anchor_group=target_outside,
                non_target_in_anchor_group=2,
                unrelated_count=unrelated,
                format_styles=self._sample_format_styles(rng) if principle == "format_similarity" else None,
                section_style="header" if principle == "section_common_region" else None,
                top_level_section_count=2 if principle == "section_common_region" else None,
                nested_section_depth_max=2 if principle == "section_common_region" else None,
                nested_section_probability=0.45 if principle == "section_common_region" else None,
                chain_style=rng.choice(CHAIN_STYLES) if principle == "continuation_chain" else None,
                chain_length_min=3 if principle == "continuation_chain" else None,
                chain_length_max=4 if principle == "continuation_chain" else None,
            )

        if dimension == "target_count":
            mapping = {"0": 0, "1": 1, "3": 3, "6": 6}
            if variant not in mapping:
                raise ValueError(f"Unknown target_count variant {variant!r}")
            return self._factors_for_target_count(rng, variant, mapping[variant])

        if dimension == "combined":
            mapping = {
                "easy": dict(num_groups=3, min_items_per_group=3, max_items_per_group=4, target_in_anchor_group=1, target_outside_anchor_group=1, non_target_in_anchor_group=2),
                "medium": dict(num_groups=4, min_items_per_group=3, max_items_per_group=5, target_in_anchor_group=2, target_outside_anchor_group=2, non_target_in_anchor_group=2),
                "hard": dict(num_groups=5, min_items_per_group=4, max_items_per_group=6, target_in_anchor_group=3, target_outside_anchor_group=3, non_target_in_anchor_group=3),
            }
            if variant not in mapping:
                raise ValueError(f"Unknown combined variant {variant!r}")
            principle = rng.choice((
                "paragraph_proximity",
                "section_common_region",
                "format_similarity",
                "scope_indentation",
                "continuation_chain",
            ))
            render_style: RenderStyle = {
                "paragraph_proximity": "paragraph_blocks",
                "section_common_region": "section_blocks",
                "format_similarity": "format_runs",
                "scope_indentation": "indent_tree",
                "continuation_chain": "chain_links",
            }[principle]
            cfg = mapping[variant]
            return self._base_factors(
                dimension="combined",
                variant=variant,
                principle=principle,
                render_style=render_style,
                unrelated_count=0,
                format_styles=self._sample_format_styles(rng) if principle == "format_similarity" else None,
                section_style="header" if principle == "section_common_region" else None,
                top_level_section_count=2 if principle == "section_common_region" else None,
                nested_section_depth_max=2 if principle == "section_common_region" else (None),
                nested_section_probability=0.5 if principle == "section_common_region" and variant != "easy" else (0.25 if principle == "section_common_region" else None),
                chain_style=rng.choice(CHAIN_STYLES) if principle == "continuation_chain" else None,
                chain_length_min=(3 if principle == "continuation_chain" else None),
                chain_length_max=({"easy": 4, "medium": 5, "hard": 6}[variant] if principle == "continuation_chain" else None),
                **cfg,
            )

        raise ValueError(f"Unknown dimension: {dimension}")

    def _base_factors(
        self,
        *,
        dimension: DimensionName,
        variant: str,
        principle: PrincipleName,
        render_style: RenderStyle,
        num_groups: int,
        min_items_per_group: int,
        max_items_per_group: int,
        target_in_anchor_group: int,
        target_outside_anchor_group: int,
        non_target_in_anchor_group: int,
        unrelated_count: int,
        format_styles: tuple[FormatGroupStyle, ...] | None = None,
        section_style: SectionStyle | None = None,
        top_level_section_count: int | None = None,
        nested_section_depth_max: int | None = None,
        nested_section_probability: float | None = None,
        chain_style: ChainStyle | None = None,
        chain_length_min: int | None = None,
        chain_length_max: int | None = None,
    ) -> StructureSensitiveTextFactors:
        return StructureSensitiveTextFactors(
            family="selective_attention",
            attentional_basis="structure_sensitive",
            modality="text",
            dimension=dimension,
            variant=variant,
            principle=principle,
            render_style=render_style,
            num_groups=num_groups,
            min_items_per_group=min_items_per_group,
            max_items_per_group=max_items_per_group,
            target_in_anchor_group=target_in_anchor_group,
            target_outside_anchor_group=target_outside_anchor_group,
            non_target_in_anchor_group=non_target_in_anchor_group,
            unrelated_count=unrelated_count,
            target_count_total=target_in_anchor_group,
            format_styles=format_styles,
            section_style=section_style,
            top_level_section_count=top_level_section_count,
            nested_section_depth_max=nested_section_depth_max,
            nested_section_probability=nested_section_probability,
            chain_style=chain_style,
            chain_length_min=chain_length_min,
            chain_length_max=chain_length_max,
        )

    def _factors_for_target_count(self, rng: random.Random, variant: str, total: int) -> StructureSensitiveTextFactors:
        principle = rng.choice((
            "paragraph_proximity",
            "section_common_region",
            "format_similarity",
            "scope_indentation",
            "continuation_chain",
        ))
        render_style: RenderStyle = {
            "paragraph_proximity": "paragraph_blocks",
            "section_common_region": "section_blocks",
            "format_similarity": "format_runs",
            "scope_indentation": "indent_tree",
            "continuation_chain": "chain_links",
        }[principle]
        if total == 0:
            return self._base_factors(
                dimension="target_count",
                variant=variant,
                principle=principle,
                render_style=render_style,
                num_groups=4,
                min_items_per_group=3,
                max_items_per_group=5,
                target_in_anchor_group=0,
                target_outside_anchor_group=2,
                non_target_in_anchor_group=3,
                unrelated_count=0,
                format_styles=self._sample_format_styles(rng) if principle == "format_similarity" else None,
                section_style="header" if principle == "section_common_region" else None,
                top_level_section_count=2 if principle == "section_common_region" else None,
                nested_section_depth_max=2 if principle == "section_common_region" else None,
                nested_section_probability=0.5 if principle == "section_common_region" else None,
                chain_style=rng.choice(CHAIN_STYLES) if principle == "continuation_chain" else None,
                chain_length_min=3 if principle == "continuation_chain" else None,
                chain_length_max=5 if principle == "continuation_chain" else None,
            )
        outside = max(1, total // 2)
        return self._base_factors(
            dimension="target_count",
            variant=variant,
            principle=principle,
            render_style=render_style,
            num_groups=max(4, 2 + outside),
            min_items_per_group=3,
            max_items_per_group=max(6, total + 2),
            target_in_anchor_group=total,
            target_outside_anchor_group=outside,
            non_target_in_anchor_group=max(2, total // 2),
            unrelated_count=0,
            format_styles=self._sample_format_styles(rng) if principle == "format_similarity" else None,
            section_style="header" if principle == "section_common_region" else None,
            top_level_section_count=2 if principle == "section_common_region" else None,
            nested_section_depth_max=2 if principle == "section_common_region" else None,
            nested_section_probability=0.5 if principle == "section_common_region" else None,
            chain_style=rng.choice(CHAIN_STYLES) if principle == "continuation_chain" else None,
            chain_length_min=3 if principle == "continuation_chain" else None,
            chain_length_max=max(4, min(7, total + 1)) if principle == "continuation_chain" else None,
        )

    def _override_target_count(self, factors: StructureSensitiveTextFactors, target_count_override: int) -> StructureSensitiveTextFactors:
        if target_count_override < 0:
            raise ValueError("target_count_override must be >= 0")
        return replace(
            factors,
            target_in_anchor_group=target_count_override,
            target_count_total=target_count_override,
            non_target_in_anchor_group=max(
                factors.non_target_in_anchor_group,
                2 if target_count_override == 0 else max(2, target_count_override // 2),
            ),
            max_items_per_group=max(
                factors.max_items_per_group,
                target_count_override + max(2, factors.non_target_in_anchor_group),
            ),
            chain_length_max=(
                max(factors.chain_length_max or 0, target_count_override + 1)
                if factors.principle == "continuation_chain"
                else factors.chain_length_max
            ),
        )

    def _sample_format_styles(self, rng: random.Random) -> tuple[FormatGroupStyle, ...]:
        styles = list(FORMAT_GROUP_STYLES)
        rng.shuffle(styles)
        return tuple(styles)

    def _sample_target_definition(self, rng: random.Random) -> dict[str, str]:
        attr = rng.choice(["color", "shape", "size", "pattern"])
        value = {
            "color": rng.choice(COLORS),
            "shape": rng.choice(SHAPES),
            "size": rng.choice(SIZES),
            "pattern": rng.choice(PATTERNS),
        }[attr]
        return {"target_attribute": attr, "target_value": value}

    def _build_items(
        self,
        rng: random.Random,
        factors: StructureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> list[TextItem]:
        sizes = self._sample_group_sizes(rng, factors)
        section_paths = self._sample_section_paths(rng, factors) if factors.principle == "section_common_region" else None
        items: list[TextItem] = []
        next_index = 1

        for group_index, group_size in enumerate(sizes, start=1):
            group_id = f"G{group_index}"
            style = (
                factors.format_styles[group_index - 1]
                if factors.format_styles and group_index - 1 < len(factors.format_styles)
                else None
            )
            chain_ids = (
                [self._make_item_id(next_index + i) for i in range(group_size)]
                if factors.principle == "continuation_chain"
                else None
            )
            section_path = section_paths[group_index - 1] if section_paths is not None else None

            for item_pos in range(group_size):
                item_id = chain_ids[item_pos] if chain_ids is not None else self._make_item_id(next_index)
                next_id = chain_ids[item_pos + 1] if chain_ids is not None and item_pos + 1 < len(chain_ids) else None
                role = "parent" if factors.principle == "scope_indentation" and item_pos == 0 else "member"
                indent_level = 0
                if factors.principle == "scope_indentation":
                    indent_level = 0 if item_pos == 0 else (1 if item_pos % 3 != 0 else 2)
                items.append(
                    TextItem(
                        item_id=item_id,
                        color=rng.choice(COLORS),
                        shape=rng.choice(SHAPES),
                        size=rng.choice(SIZES),
                        pattern=rng.choice(PATTERNS),
                        group_id=group_id,
                        role=role,
                        format_style=style,
                        indent_level=indent_level,
                        next_id=next_id,
                        section_path=section_path,
                    )
                )
                next_index += 1 if chain_ids is None else 0
            if chain_ids is not None:
                next_index += group_size

        anchor_group_id = "G1"
        items = self._assign_target_distribution(rng, items, factors, target_definition, anchor_group_id)
        if factors.principle == "continuation_chain":
            items = self._interleave_chain_groups(items)
        return items

    def _sample_group_sizes(self, rng: random.Random, factors: StructureSensitiveTextFactors) -> list[int]:
        if factors.principle == "continuation_chain":
            low = factors.chain_length_min or factors.min_items_per_group
            high = factors.chain_length_max or factors.max_items_per_group
            sizes = [rng.randint(low, high) for _ in range(factors.num_groups)]
        else:
            sizes = [rng.randint(factors.min_items_per_group, factors.max_items_per_group) for _ in range(factors.num_groups)]
        sizes[0] = max(
            sizes[0],
            factors.target_in_anchor_group + factors.non_target_in_anchor_group + (1 if factors.principle == "scope_indentation" else 0),
        )
        needed_outside = factors.target_outside_anchor_group
        if sum(sizes[1:]) < needed_outside and len(sizes) > 1:
            sizes[1] += needed_outside - sum(sizes[1:])
        return sizes

    def _sample_section_paths(
        self,
        rng: random.Random,
        factors: StructureSensitiveTextFactors,
    ) -> list[tuple[str, ...]]:
        num_groups = factors.num_groups
        top_level = min(factors.top_level_section_count or 2, num_groups)
        depth_max = max(1, factors.nested_section_depth_max or 1)
        nested_prob = max(0.0, min(1.0, factors.nested_section_probability or 0.0))

        paths: list[tuple[str, ...]] = []
        available_parents: list[tuple[str, ...]] = []
        next_top_level_idx = 0

        for _ in range(top_level):
            label = chr(ord("A") + next_top_level_idx)
            next_top_level_idx += 1
            path = (label,)
            paths.append(path)
            if len(path) < depth_max:
                available_parents.append(path)

        while len(paths) < num_groups:
            can_nest = bool(available_parents)
            should_nest = can_nest and (rng.random() < nested_prob or next_top_level_idx >= 26)
            if should_nest:
                parent = rng.choice(available_parents)
                siblings = [path for path in paths if path[:-1] == parent]
                child_index = len(siblings) + 1
                child_path = (*parent, str(child_index))
                paths.append(child_path)
                if len(child_path) < depth_max:
                    available_parents.append(child_path)
            else:
                label = chr(ord("A") + next_top_level_idx)
                next_top_level_idx += 1
                path = (label,)
                paths.append(path)
                if len(path) < depth_max:
                    available_parents.append(path)
        return paths[:num_groups]

    def _assign_target_distribution(
        self,
        rng: random.Random,
        items: list[TextItem],
        factors: StructureSensitiveTextFactors,
        target_definition: dict[str, str],
        anchor_group_id: str,
    ) -> list[TextItem]:
        attr = target_definition["target_attribute"]
        value = target_definition["target_value"]
        anchor_items = [
            item
            for item in items
            if item.group_id == anchor_group_id and (factors.principle != "scope_indentation" or item.role != "parent")
        ]
        other_items = [
            item
            for item in items
            if item.group_id != anchor_group_id and (factors.principle != "scope_indentation" or item.role != "parent")
        ]
        if len(anchor_items) < factors.target_in_anchor_group + factors.non_target_in_anchor_group:
            raise GenerationError("Anchor group too small")
        if len(other_items) < factors.target_outside_anchor_group:
            raise GenerationError("Outside groups too small")
        target_anchor_ids = {item.item_id for item in rng.sample(anchor_items, factors.target_in_anchor_group)} if factors.target_in_anchor_group else set()
        target_other_ids = {item.item_id for item in rng.sample(other_items, factors.target_outside_anchor_group)} if factors.target_outside_anchor_group else set()

        updated: list[TextItem] = []
        for item in items:
            if item.item_id in target_anchor_ids or item.item_id in target_other_ids:
                updated.append(replace(item, **{attr: value}))
            else:
                updated.append(self._make_non_target_item(rng, item, attr, value))
        return updated

    def _make_non_target_item(self, rng: random.Random, item: TextItem, attr: str, target_value: str) -> TextItem:
        kwargs: dict[str, str] = {}
        for name, values in (("color", COLORS), ("shape", SHAPES), ("size", SIZES), ("pattern", PATTERNS)):
            current = getattr(item, name)
            if name == attr:
                kwargs[name] = rng.choice([v for v in values if v != target_value])
            else:
                kwargs[name] = current
        return replace(item, **kwargs)

    def _interleave_chain_groups(self, items: list[TextItem]) -> list[TextItem]:
        grouped = {group_id: [item for item in items if item.group_id == group_id] for group_id in self._ordered_group_ids(items)}
        output: list[TextItem] = []
        group_order = list(grouped.keys())
        while any(grouped.values()):
            for group_id in group_order:
                if grouped[group_id]:
                    output.append(grouped[group_id].pop(0))
        return output

    def _choose_anchor(
        self,
        rng: random.Random,
        items: list[TextItem],
        factors: StructureSensitiveTextFactors,
        target_definition: dict[str, str],
    ) -> tuple[str, str]:
        target_attr = target_definition["target_attribute"]
        target_value = target_definition["target_value"]
        eligible_groups: list[str] = []
        for group_id in self._ordered_group_ids(items):
            group_items = [item for item in items if item.group_id == group_id]
            count = sum(1 for item in group_items if item.role != "parent" and getattr(item, target_attr) == target_value)
            if count == factors.target_in_anchor_group:
                eligible_groups.append(group_id)
        if not eligible_groups:
            raise GenerationError("No eligible anchor group")
        anchor_group_id = rng.choice(eligible_groups)
        anchor_candidates = [item for item in items if item.group_id == anchor_group_id]
        if factors.principle == "scope_indentation":
            parents = [item for item in anchor_candidates if item.role == "parent"]
            if parents:
                return parents[0].item_id, anchor_group_id
        return rng.choice(anchor_candidates).item_id, anchor_group_id

    def _render_items(self, items: list[TextItem], factors: StructureSensitiveTextFactors) -> str:
        if factors.principle == "paragraph_proximity":
            return self._render_paragraphs(items)
        if factors.principle == "section_common_region":
            return self._render_sections(items)
        if factors.principle == "format_similarity":
            return self._render_format_similarity(items)
        if factors.principle == "scope_indentation":
            return self._render_scope_indentation(items)
        if factors.principle == "continuation_chain":
            return self._render_continuation_chain(items, factors.chain_style or "arrow")
        raise ValueError(f"Unsupported principle: {factors.principle}")

    def _render_paragraphs(self, items: list[TextItem]) -> str:
        paragraphs = []
        for group_id in self._ordered_group_ids(items):
            group_items = [item for item in items if item.group_id == group_id]
            paragraphs.append("\n".join(self._render_plain_item(item) for item in group_items))
        return "\n\n".join(paragraphs)

    def _render_sections(self, items: list[TextItem]) -> str:
        blocks: list[str] = []
        for group_id in self._ordered_group_ids(items):
            group_items = [item for item in items if item.group_id == group_id]
            section_path = group_items[0].section_path
            if not section_path:
                raise GenerationError("Section items missing section_path")
            label = ".".join(section_path)
            body = "\n".join(self._render_plain_item(item) for item in group_items)
            blocks.append(f"[Section {label}]\n{body}")
        return "\n\n".join(blocks)

    def _render_format_similarity(self, items: list[TextItem]) -> str:
        lines: list[str] = []
        for group_id in self._ordered_group_ids(items):
            group_items = [item for item in items if item.group_id == group_id]
            for local_index, item in enumerate(group_items, start=1):
                lines.append(self._render_formatted_item(item, local_index))
        return "\n".join(lines)

    def _render_scope_indentation(self, items: list[TextItem]) -> str:
        lines: list[str] = []
        for group_id in self._ordered_group_ids(items):
            group_items = [item for item in items if item.group_id == group_id]
            parent = next((item for item in group_items if item.role == "parent"), None)
            children = [item for item in group_items if item.role != "parent"]
            if parent is not None:
                lines.append(self._render_plain_item(parent))
            for child in children:
                lines.append(f"{'  ' * child.indent_level}{self._render_plain_item(child)}")
        return "\n".join(lines)

    def _render_continuation_chain(self, items: list[TextItem], chain_style: ChainStyle) -> str:
        lines: list[str] = []
        for item in items:
            base = self._render_plain_item(item)
            if item.next_id is None:
                lines.append(base)
            elif chain_style == "arrow":
                lines.append(f"{base} => {item.next_id}")
            else:
                lines.append(f"{base} continues_to={item.next_id}")
        return "\n".join(lines)

    def _render_plain_item(self, item: TextItem) -> str:
        return f"id={item.item_id} color={item.color} shape={item.shape} size={item.size} pattern={item.pattern}"

    def _render_formatted_item(self, item: TextItem, local_index: int) -> str:
        prefix = {
            "dash": "-",
            "star": "*",
            "numbered": f"{local_index}.",
            "alpha": f"{self._alpha_label(local_index)})",
        }[item.format_style or "dash"]
        return f"{prefix} {self._render_plain_item(item)}"

    def _alpha_label(self, index: int) -> str:
        return chr(ord("a") + ((index - 1) % 26))

    def _matching_ids(self, items: list[TextItem], anchor_group_id: str, target_definition: dict[str, str]) -> list[str]:
        attr = target_definition["target_attribute"]
        value = target_definition["target_value"]
        return [
            item.item_id
            for item in items
            if item.group_id == anchor_group_id and item.role != "parent" and getattr(item, attr) == value
        ]

    def _passes_constraints(
        self,
        items: list[TextItem],
        factors: StructureSensitiveTextFactors,
        target_definition: dict[str, str],
        gold_ids: list[str],
        anchor_group_id: str,
    ) -> bool:
        attr = target_definition["target_attribute"]
        value = target_definition["target_value"]
        if len(gold_ids) != factors.target_in_anchor_group:
            return False
        outside_count = sum(
            1
            for item in items
            if item.group_id != anchor_group_id and item.role != "parent" and getattr(item, attr) == value
        )
        if outside_count != factors.target_outside_anchor_group:
            return False
        anchor_non_targets = sum(
            1
            for item in items
            if item.group_id == anchor_group_id and item.role != "parent" and getattr(item, attr) != value
        )
        if anchor_non_targets < factors.non_target_in_anchor_group:
            return False
        if len({item.item_id for item in items}) != len(items):
            return False
        if sum(1 for item in items if item.is_anchor) != 1:
            return False
        if factors.principle == "format_similarity":
            styles = [item.format_style for item in items if item.group_id == anchor_group_id]
            if len(set(styles)) != 1:
                return False
        if factors.principle == "scope_indentation":
            parents = [item for item in items if item.group_id == anchor_group_id and item.role == "parent"]
            if len(parents) != 1:
                return False
        if factors.principle == "section_common_region":
            paths = [item.section_path for item in items if item.group_id == anchor_group_id]
            if not paths or any(path != paths[0] for path in paths):
                return False
        if factors.principle == "continuation_chain":
            for group_id in self._ordered_group_ids(items):
                group_items = [item for item in items if item.group_id == group_id]
                pointers = sum(1 for item in group_items if item.next_id is not None)
                if len(group_items) >= 2 and pointers < len(group_items) - 1:
                    return False
        return True

    def _rule_description(self, factors: StructureSensitiveTextFactors, target_definition: dict[str, str]) -> str:
        cue = {
            "paragraph_proximity": "paragraph",
            "section_common_region": "section",
            "format_similarity": "formatting group",
            "scope_indentation": "indented scope",
            "continuation_chain": "continuation chain",
        }[factors.principle]
        return f"items where {target_definition['target_attribute']}={target_definition['target_value']} in the same {cue} as the marked item"

    def _build_count_instruction(self, anchor_item_id: str, factors: StructureSensitiveTextFactors, target_definition: dict[str, str]) -> str:
        rule = self._rule_description(factors, target_definition)
        return (
            f"Count the {rule}. The marked item has id={anchor_item_id}.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n"
            '- "count" must be an integer\n'
            "- Use only the structural organization shown in the text\n"
            "- Return only the JSON object"
        )

    def _build_filter_instruction(self, anchor_item_id: str, factors: StructureSensitiveTextFactors, target_definition: dict[str, str]) -> str:
        rule = self._rule_description(factors, target_definition)
        return (
            f"Return the ids of the {rule}. The marked item has id={anchor_item_id}.\n"
            'Respond with a JSON object of the form {"ids": [<sorted unique strings>]}.\n'
            "Rules:\n"
            "- ids must be strings exactly as shown in the text\n"
            "- Sort ascending lexicographically\n"
            "- Do not include duplicates\n"
            "- Use only the structural organization shown in the text\n"
            "- Return only the JSON object"
        )

    def _ordered_group_ids(self, items: list[TextItem]) -> list[str]:
        return list(dict.fromkeys(item.group_id for item in items))

    def _make_item_id(self, index: int) -> str:
        return f"T{index:03d}"


def scene_to_dataset_row(scene: StructureSensitiveTextScene) -> dict[str, object]:
    factors = scene.factors
    return {
        "seed": scene.seed,
        "family": scene.family,
        "attentional_basis": scene.attentional_basis,
        "modality": scene.modality,
        "dimension": scene.dimension,
        "variant": scene.variant,
        "principle": scene.principle,
        "render_style": scene.render_style,
        "count_instruction": scene.count_instruction,
        "filter_instruction": scene.filter_instruction,
        "text_input": scene.text_input,
        "count_prompt": scene.count_prompt,
        "filter_prompt": scene.filter_prompt,
        "gold_count": scene.gold_count,
        "gold_ids": json.dumps(sorted(scene.gold_ids)),
        "target_definition": json.dumps(scene.target_definition, sort_keys=True),
        "anchor_item_id": scene.anchor_item_id,
        "anchor_group_id": scene.anchor_group_id,
        "num_groups": factors.num_groups,
        "min_items_per_group": factors.min_items_per_group,
        "max_items_per_group": factors.max_items_per_group,
        "target_in_anchor_group": factors.target_in_anchor_group,
        "target_outside_anchor_group": factors.target_outside_anchor_group,
        "non_target_in_anchor_group": factors.non_target_in_anchor_group,
        "unrelated_count": factors.unrelated_count,
        "format_styles": json.dumps(list(factors.format_styles) if factors.format_styles else []),
        "section_style": factors.section_style,
        "top_level_section_count": factors.top_level_section_count,
        "nested_section_depth_max": factors.nested_section_depth_max,
        "nested_section_probability": factors.nested_section_probability,
        "chain_style": factors.chain_style,
        "chain_length_min": factors.chain_length_min,
        "chain_length_max": factors.chain_length_max,
        "items_json": json.dumps([asdict(item) for item in scene.items], sort_keys=True),
    }
