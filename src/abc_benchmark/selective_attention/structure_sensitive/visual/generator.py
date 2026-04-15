from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

ColorName = Literal["red", "blue", "green", "yellow"]
ShapeName = Literal["circle", "square", "triangle"]
SizeName = Literal["small", "large"]
GroupingPrinciple = Literal["proximity", "similarity", "common_region"]
StructureType = Literal["grouping", "relation", "scope", "global_local"]
RelationOperator = Literal[
    "inside",
    "outside",
    "left_of",
    "right_of",
    "above",
    "below",
    "same_group",
    "different_group",
]
CueConflict = Literal[
    "none",
    "proximity_vs_similarity",
    "proximity_vs_common_region",
    "similarity_vs_common_region",
]
BindingDistance = Literal["near", "far"]
StructureDepth = Literal["shallow", "nested"]
ConfoundLevel = Literal["low", "high"]
ConfoundType = Literal[
    "feature_only",
    "group_only",
    "cross_binding",
    "wrong_scope",
    "cue_conflict_trap",
    "boundary_ambiguous",
]
RenderStyle = Literal["plain", "boxed", "panelled"]
DimensionName = Literal[
    "baseline_structure",
    "grouping_principle",
    "cue_conflict",
    "structure_type",
    "relation_operator",
    "structure_depth",
    "binding_distance",
    "confound",
    "confound_type",
    "target_count_x_structure_depth",
    "render_style",
    "combined",
]
FamilyName = Literal["selective_attention"]
AttentionalBasisName = Literal["structure_sensitive"]
ModalityName = Literal["visual"]

COLORS: tuple[ColorName, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[ShapeName, ...] = ("circle", "square", "triangle")
SIZES: tuple[SizeName, ...] = ("small", "large")
COLOR_TO_RGB: dict[ColorName, tuple[int, int, int]] = {
    "red": (220, 60, 60),
    "blue": (70, 120, 220),
    "green": (70, 170, 90),
    "yellow": (220, 185, 55),
}
SIZE_TO_RADIUS: dict[SizeName, int] = {"small": 11, "large": 17}


class GenerationError(RuntimeError):
    """Raised when a valid structure-sensitive visual scene cannot be generated."""


@dataclass(frozen=True)
class VisualItemSpec:
    x: float
    y: float
    color: ColorName
    shape: ShapeName
    size: SizeName
    group_id: str
    similarity_key: str | None
    region_id: str | None
    panel_id: str | None
    is_target: bool
    is_anchor: bool
    role: str


@dataclass(frozen=True)
class VisualStructureSpec:
    type: str
    id: str
    payload: dict[str, object]


@dataclass(frozen=True)
class StructureSensitiveVisualFactors:
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    grouping_principle: GroupingPrinciple
    structure_type: StructureType
    relation_operator: RelationOperator
    cue_conflict: CueConflict
    num_items: int
    num_groups: int
    target_count: int
    binding_distance: BindingDistance
    structure_depth: StructureDepth
    confound_level: ConfoundLevel
    confound_type: ConfoundType
    render_style: RenderStyle
    width: int
    height: int
    margin: int
    item_size: SizeName


@dataclass(frozen=True)
class StructureSensitiveVisualScene:
    scene_id: str
    seed: int
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    grouping_principle: GroupingPrinciple
    structure_type: StructureType
    relation_operator: RelationOperator
    cue_conflict: CueConflict
    count_instruction: str
    filter_instruction: str
    count_prompt: str
    filter_prompt: str
    gold_count: int
    gold_indices: list[int]
    target_definition: dict[str, str]
    factors: StructureSensitiveVisualFactors
    items: list[VisualItemSpec]
    structures: list[VisualStructureSpec]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["items"] = [asdict(item) for item in self.items]
        payload["structures"] = [asdict(structure) for structure in self.structures]
        return payload


def make_scene_id(dimension: str, variant: str, seed: int) -> str:
    return f"structure_visual__{dimension}__{variant}__{seed}"


class StructureSensitiveVisualGenerator:
    def __init__(self, rng: random.Random | None = None, max_attempts: int = 1000) -> None:
        self.rng = rng or random.Random()
        self.max_attempts = max_attempts
        self._font = ImageFont.load_default()

    def generate(
        self,
        *,
        seed: int,
        factors: StructureSensitiveVisualFactors | None = None,
        dimension: DimensionName = "combined",
        variant: str = "medium",
    ) -> StructureSensitiveVisualScene:
        local_rng = random.Random(seed)
        if factors is None:
            factors = self.sample_factors(rng=local_rng, dimension=dimension, variant=variant)

        for _ in range(self.max_attempts):
            try:
                target_definition = self._sample_target_definition(local_rng, factors)
                items, structures = self._build_scene(local_rng, factors, target_definition)
                gold_indices = self._matching_item_indices(items, target_definition)
                if len(gold_indices) != factors.target_count:
                    continue
                if not self._validate_scene_geometry(items, structures, factors):
                    continue
                if not self._passes_anti_shortcut_constraints(items, factors, target_definition, gold_indices):
                    continue

                count_instruction = self._build_count_instruction(factors, target_definition)
                filter_instruction = self._build_filter_instruction(factors, target_definition)
                return StructureSensitiveVisualScene(
                    scene_id=make_scene_id(factors.dimension, factors.variant, seed),
                    seed=seed,
                    family=factors.family,
                    attentional_basis=factors.attentional_basis,
                    modality=factors.modality,
                    dimension=factors.dimension,
                    variant=factors.variant,
                    grouping_principle=factors.grouping_principle,
                    structure_type=factors.structure_type,
                    relation_operator=factors.relation_operator,
                    cue_conflict=factors.cue_conflict,
                    count_instruction=count_instruction,
                    filter_instruction=filter_instruction,
                    count_prompt=count_instruction,
                    filter_prompt=filter_instruction,
                    gold_count=len(gold_indices),
                    gold_indices=gold_indices,
                    target_definition=target_definition,
                    factors=factors,
                    items=items,
                    structures=structures,
                )
            except GenerationError:
                continue
        raise GenerationError("Failed to generate a valid structure-sensitive visual scene within max_attempts")

    def sample_factors(
        self,
        *,
        rng: random.Random,
        dimension: DimensionName,
        variant: str,
    ) -> StructureSensitiveVisualFactors:
        base = dict(
            family="selective_attention",
            attentional_basis="structure_sensitive",
            modality="visual",
            width=640,
            height=480,
            margin=36,
            item_size="large",
        )

        if dimension == "baseline_structure":
            return StructureSensitiveVisualFactors(
                **base,
                dimension="baseline_structure",
                variant="baseline",
                grouping_principle="common_region",
                structure_type="grouping",
                relation_operator="same_group",
                cue_conflict="none",
                num_items=10,
                num_groups=2,
                target_count=rng.randint(1, 2),
                binding_distance="near",
                structure_depth="shallow",
                confound_level="low",
                confound_type="feature_only",
                render_style="boxed",
            )

        if dimension == "grouping_principle":
            mapping: dict[str, tuple[GroupingPrinciple, StructureType, CueConflict, ConfoundType, RenderStyle, int, int, int]] = {
                "proximity": ("proximity", "grouping", "none", "feature_only", "plain", 12, 3, 2),
                "similarity": ("similarity", "grouping", "none", "feature_only", "plain", 12, 3, 2),
                "common_region": ("common_region", "grouping", "none", "feature_only", "boxed", 12, 3, 2),
            }
            gp, st, cc, ct, rs, ni, ng, tc = mapping[variant]
            return StructureSensitiveVisualFactors(
                **base, dimension="grouping_principle", variant=variant,
                grouping_principle=gp, structure_type=st, relation_operator="same_group",
                cue_conflict=cc, num_items=ni, num_groups=ng, target_count=tc,
                binding_distance="near", structure_depth="shallow",
                confound_level="low", confound_type=ct, render_style=rs,
            )

        if dimension == "cue_conflict":
            mapping = {
                "none": ("proximity", "grouping", "none", "feature_only", "plain", 12, 3, 2),
                "proximity_vs_similarity": ("proximity", "grouping", "proximity_vs_similarity", "cue_conflict_trap", "plain", 12, 3, 2),
                "proximity_vs_common_region": ("common_region", "grouping", "proximity_vs_common_region", "cue_conflict_trap", "boxed", 12, 3, 2),
            }
            gp, st, cc, ct, rs, ni, ng, tc = mapping[variant]
            return StructureSensitiveVisualFactors(
                **base, dimension="cue_conflict", variant=variant,
                grouping_principle=gp, structure_type=st, relation_operator="same_group",
                cue_conflict=cc, num_items=ni, num_groups=ng, target_count=tc,
                binding_distance="far", structure_depth="shallow",
                confound_level="high" if cc != "none" else "low", confound_type=ct, render_style=rs,
            )

        if dimension == "structure_type":
            mapping = {
                "grouping": ("common_region", "grouping", "same_group", "none", "feature_only", "boxed", 12, 3, 2, "shallow"),
                "relation": ("common_region", "relation", "above", "none", "feature_only", "boxed", 12, 2, 2, "shallow"),
                "scope": ("common_region", "scope", "inside", "none", "wrong_scope", "panelled", 12, 2, 2, "shallow"),
                "global_local": ("common_region", "global_local", "inside", "none", "feature_only", "boxed", 12, 3, 2, "nested"),
            }
            gp, st, op, cc, ct, rs, ni, ng, tc, depth = mapping[variant]
            return StructureSensitiveVisualFactors(
                **base, dimension="structure_type", variant=variant,
                grouping_principle=gp, structure_type=st, relation_operator=op, cue_conflict=cc,
                num_items=ni, num_groups=ng, target_count=tc,
                binding_distance="near", structure_depth=depth,
                confound_level="low" if ct != "wrong_scope" else "high", confound_type=ct, render_style=rs,
            )

        if dimension == "structure_depth":
            mapping = {
                "shallow": ("common_region", "relation", "inside", "none", "feature_only", "boxed", 12, 2, 2, "shallow"),
                "nested": ("common_region", "global_local", "inside", "none", "cross_binding", "boxed", 14, 3, 2, "nested"),
            }
            gp, st, op, cc, ct, rs, ni, ng, tc, depth = mapping[variant]
            return StructureSensitiveVisualFactors(
                **base, dimension="structure_depth", variant=variant,
                grouping_principle=gp, structure_type=st, relation_operator=op, cue_conflict=cc,
                num_items=ni, num_groups=ng, target_count=tc,
                binding_distance="far" if depth == "nested" else "near",
                structure_depth=depth, confound_level="high" if depth == "nested" else "low",
                confound_type=ct, render_style=rs,
            )

        if dimension == "confound_type":
            mapping = {
                "feature_only": ("common_region", "grouping", "same_group", "none", "feature_only", "boxed"),
                "cross_binding": ("common_region", "grouping", "same_group", "none", "cross_binding", "boxed"),
                "wrong_scope": ("common_region", "scope", "inside", "none", "wrong_scope", "panelled"),
                "cue_conflict_trap": ("common_region", "grouping", "same_group", "proximity_vs_common_region", "cue_conflict_trap", "boxed"),
            }
            gp, st, op, cc, ct, rs = mapping[variant]
            return StructureSensitiveVisualFactors(
                **base, dimension="confound_type", variant=variant,
                grouping_principle=gp, structure_type=st, relation_operator=op, cue_conflict=cc,
                num_items=12, num_groups=3, target_count=2,
                binding_distance="far" if ct in {"wrong_scope", "cue_conflict_trap"} else "near",
                structure_depth="nested" if ct == "wrong_scope" else "shallow",
                confound_level="high", confound_type=ct, render_style=rs,
            )

        if dimension == "combined":
            mapping = {
                "easy": ("common_region", "grouping", "same_group", "none", "feature_only", "boxed", 10, 2, 2, "shallow", "low", "near"),
                "medium": ("common_region", "scope", "inside", "none", "wrong_scope", "panelled", 12, 2, 3, "nested", "high", "far"),
                "hard": ("common_region", "global_local", "inside", "proximity_vs_common_region", "cue_conflict_trap", "boxed", 14, 3, 3, "nested", "high", "far"),
            }
            gp, st, op, cc, ct, rs, ni, ng, tc, depth, cl, bd = mapping[variant]
            return StructureSensitiveVisualFactors(
                **base, dimension="combined", variant=variant,
                grouping_principle=gp, structure_type=st, relation_operator=op, cue_conflict=cc,
                num_items=ni, num_groups=ng, target_count=tc,
                binding_distance=bd, structure_depth=depth, confound_level=cl, confound_type=ct, render_style=rs,
            )

        if dimension == "relation_operator":
            mapping = {
                "inside": "inside",
                "outside": "outside",
                "left_of": "left_of",
                "right_of": "right_of",
                "above": "above",
                "below": "below",
                "same_group": "same_group",
                "different_group": "different_group",
            }
            op = mapping[variant]
            st = "relation" if op in {"inside", "outside", "left_of", "right_of", "above", "below"} else "grouping"
            return StructureSensitiveVisualFactors(
                **base, dimension="relation_operator", variant=variant,
                grouping_principle="common_region", structure_type=st, relation_operator=op, cue_conflict="none",
                num_items=12, num_groups=2, target_count=2,
                binding_distance="near", structure_depth="shallow", confound_level="low", confound_type="feature_only",
                render_style="boxed",
            )

        if dimension == "binding_distance":
            return StructureSensitiveVisualFactors(
                **base, dimension="binding_distance", variant=variant,
                grouping_principle="proximity", structure_type="grouping", relation_operator="same_group", cue_conflict="proximity_vs_similarity" if variant == "far" else "none",
                num_items=12, num_groups=3, target_count=2,
                binding_distance="far" if variant == "far" else "near",
                structure_depth="shallow", confound_level="high" if variant == "far" else "low", confound_type="cue_conflict_trap" if variant == "far" else "feature_only",
                render_style="plain",
            )

        if dimension == "confound":
            return StructureSensitiveVisualFactors(
                **base, dimension="confound", variant=variant,
                grouping_principle="common_region", structure_type="grouping", relation_operator="same_group", cue_conflict="none",
                num_items=12 if variant == "low" else 14, num_groups=3, target_count=2,
                binding_distance="near", structure_depth="shallow", confound_level="high" if variant == "high" else "low", confound_type="cross_binding" if variant == "high" else "feature_only",
                render_style="boxed",
            )

        if dimension == "target_count_x_structure_depth":
            mapping = {
                "0_shallow": (0, "shallow", "relation", "inside"),
                "0_nested": (0, "nested", "global_local", "inside"),
                "3_shallow": (3, "shallow", "relation", "inside"),
                "3_nested": (3, "nested", "global_local", "inside"),
            }
            tc, depth, st, op = mapping[variant]
            return StructureSensitiveVisualFactors(
                **base, dimension="target_count_x_structure_depth", variant=variant,
                grouping_principle="common_region", structure_type=st, relation_operator=op, cue_conflict="none",
                num_items=14, num_groups=3, target_count=tc,
                binding_distance="far" if depth == "nested" else "near", structure_depth=depth,
                confound_level="high" if depth == "nested" else "low", confound_type="cross_binding" if depth == "nested" else "feature_only",
                render_style="boxed",
            )

        if dimension == "render_style":
            return StructureSensitiveVisualFactors(
                **base, dimension="render_style", variant=variant,
                grouping_principle="common_region", structure_type="scope" if variant == "panelled" else "grouping",
                relation_operator="inside", cue_conflict="none", num_items=12, num_groups=2, target_count=2,
                binding_distance="near", structure_depth="nested" if variant == "panelled" else "shallow",
                confound_level="low", confound_type="wrong_scope" if variant == "panelled" else "feature_only",
                render_style=variant,  # type: ignore[arg-type]
            )

        raise ValueError(f"Unknown dimension: {dimension}")

    def render(
        self,
        scene: StructureSensitiveVisualScene,
        output_path: str | Path | None = None,
        *,
        show_item_ids: bool = False,
    ) -> Image.Image:
        factors = scene.factors
        image = Image.new("RGB", (factors.width, factors.height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for structure in scene.structures:
            self._draw_structure(draw, structure)

        for idx, item in enumerate(scene.items, start=1):
            self._draw_item(draw, item)
            if item.is_anchor:
                self._draw_anchor(draw, item, factors)
            if show_item_ids:
                self._draw_item_id(draw, image, item, idx)

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        return image

    def _draw_structure(self, draw: ImageDraw.ImageDraw, structure: VisualStructureSpec) -> None:
        if structure.type == "box":
            x1 = float(structure.payload["x1"])
            y1 = float(structure.payload["y1"])
            x2 = float(structure.payload["x2"])
            y2 = float(structure.payload["y2"])
            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)
            draw.rectangle((left, top, right, bottom), outline=(120, 120, 120), width=2)
        elif structure.type == "panel":
            x1 = float(structure.payload["x1"])
            y1 = float(structure.payload["y1"])
            x2 = float(structure.payload["x2"])
            y2 = float(structure.payload["y2"])
            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)
            draw.rectangle((left, top, right, bottom), outline=(180, 180, 180), width=2)
        elif structure.type == "line":
            orientation = str(structure.payload["orientation"])
            if orientation == "horizontal":
                y = float(structure.payload["y"])
                x1 = float(structure.payload["x1"])
                x2 = float(structure.payload["x2"])
                draw.line((min(x1, x2), y, max(x1, x2), y), fill=(120, 120, 120), width=2)
            else:
                x = float(structure.payload["x"])
                y1 = float(structure.payload["y1"])
                y2 = float(structure.payload["y2"])
                draw.line((x, min(y1, y2), x, max(y1, y2)), fill=(120, 120, 120), width=2)

    def _draw_item(self, draw: ImageDraw.ImageDraw, item: VisualItemSpec) -> None:
        radius = SIZE_TO_RADIUS[item.size]
        color = COLOR_TO_RGB[item.color]
        bbox = (item.x - radius, item.y - radius, item.x + radius, item.y + radius)
        if item.shape == "circle":
            draw.ellipse(bbox, fill=color, outline=(30, 30, 30), width=2)
        elif item.shape == "square":
            draw.rectangle(bbox, fill=color, outline=(30, 30, 30), width=2)
        else:
            points = [
                (item.x, item.y - radius),
                (item.x - radius * 0.9, item.y + radius * 0.8),
                (item.x + radius * 0.9, item.y + radius * 0.8),
            ]
            draw.polygon(points, fill=color, outline=(30, 30, 30))

    def _draw_anchor(
        self,
        draw: ImageDraw.ImageDraw,
        item: VisualItemSpec,
        factors: StructureSensitiveVisualFactors,
    ) -> None:
        del factors
        item_radius = SIZE_TO_RADIUS[item.size]
        ring_radius = item_radius + 8
        dot_radius = 2
        dot_count = 12
        for i in range(dot_count):
            angle = -math.pi / 2 + i * (2 * math.pi / dot_count)
            cx = item.x + math.cos(angle) * ring_radius
            cy = item.y + math.sin(angle) * ring_radius
            draw.ellipse(
                (cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius),
                fill=(35, 35, 35),
            )

    def _draw_item_id(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        item: VisualItemSpec,
        item_index: int,
    ) -> None:
        radius = SIZE_TO_RADIUS[item.size]
        label = str(item_index)
        bbox = draw.textbbox((0, 0), label, font=self._font, stroke_width=2)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        offset = radius + 2
        candidates = [
            (item.x + offset, item.y - offset - text_h / 2),
            (item.x - offset - text_w, item.y - offset - text_h / 2),
            (item.x + offset, item.y + offset - text_h / 2),
            (item.x - offset - text_w, item.y + offset - text_h / 2),
        ]
        img_w, img_h = image.size
        pad = 2
        for tx, ty in candidates:
            if pad <= tx <= img_w - text_w - pad and pad <= ty <= img_h - text_h - pad:
                break
        else:
            tx, ty = candidates[0]
        tx = max(pad, min(tx, img_w - text_w - pad))
        ty = max(pad, min(ty, img_h - text_h - pad))
        draw.text((tx, ty), label, font=self._font, fill=(60, 60, 60), stroke_width=2, stroke_fill=(255, 255, 255))

    def _sample_target_definition(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> dict[str, str]:
        target_color = rng.choice(COLORS)
        target_shape = rng.choice(SHAPES)
        if factors.grouping_principle == "similarity":
            similarity_key = rng.choice(SHAPES)
            target_shape = similarity_key
            return {
                "rule_type": "same_similarity_group_as_anchor",
                "query_color": target_color,
                "query_shape": target_shape,
                "reference_similarity_key": similarity_key,
            }

        if factors.structure_type == "relation":
            rule_type = {
                "inside": "inside_region",
                "outside": "outside_region",
                "left_of": "left_of_line",
                "right_of": "right_of_line",
                "above": "above_line",
                "below": "below_line",
            }.get(factors.relation_operator, "inside_region")
        elif factors.structure_type == "scope":
            rule_type = "inside_region_in_panel"
        elif factors.structure_type == "global_local":
            rule_type = "inside_selected_region"
        else:
            rule_type = {
                "same_group": "same_group_as_anchor",
                "different_group": "different_group_from_anchor",
            }.get(factors.relation_operator, "same_group_as_anchor")

        return {
            "rule_type": rule_type,
            "query_color": target_color,
            "query_shape": target_shape,
        }

    def _build_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> tuple[list[VisualItemSpec], list[VisualStructureSpec]]:
        if factors.structure_type == "relation":
            return self._build_relation_scene(rng, factors, target_definition)
        if factors.structure_type == "scope":
            return self._build_scope_scene(rng, factors, target_definition)
        if factors.structure_type == "global_local":
            return self._build_global_local_scene(rng, factors, target_definition)
        return self._build_grouping_scene(rng, factors, target_definition)

    def _build_grouping_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> tuple[list[VisualItemSpec], list[VisualStructureSpec]]:
        if factors.grouping_principle == "common_region":
            return self._build_common_region_grouping_scene(rng, factors, target_definition)
        if factors.grouping_principle == "proximity":
            return self._build_proximity_grouping_scene(rng, factors, target_definition)
        return self._build_similarity_grouping_scene(rng, factors, target_definition)

    def _build_proximity_grouping_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> tuple[list[VisualItemSpec], list[VisualStructureSpec]]:
        query_color = target_definition["query_color"]
        query_shape = target_definition["query_shape"]
        size = factors.item_size
        radius = SIZE_TO_RADIUS[size]
        min_dist = 2 * radius + 6

        group_sizes = self._sample_group_sizes(rng, factors.num_items, factors.num_groups, min_size=3)
        cluster_radius = 52 if factors.binding_distance == "near" else 42
        min_sep = 170 if factors.binding_distance == "far" else 145
        centers = self._sample_cluster_centers(rng, factors, len(group_sizes), min_sep=min_sep)
        group_ids = [f"cluster_{i}" for i in range(len(group_sizes))]
        target_group_id = group_ids[0]
        anchor_shape = "triangle"
        anchor_color = "yellow"
        if factors.cue_conflict == "proximity_vs_similarity":
            anchor_shape = "square"
            anchor_color = "blue"

        items: list[VisualItemSpec] = []
        structures: list[VisualStructureSpec] = []

        for gid, center in zip(group_ids, centers):
            structures.append(VisualStructureSpec("cluster_hint", gid, {"cx": center[0], "cy": center[1]}))

        for gid, center, gsize in zip(group_ids, centers, group_sizes):
            positions = self._sample_points_in_circle(
                rng,
                center=center,
                count=gsize,
                radius=cluster_radius,
                bounds=(
                    factors.margin + radius + 2,
                    factors.width - factors.margin - radius - 2,
                    factors.margin + radius + 2,
                    factors.height - factors.margin - radius - 2,
                ),
                existing=[(it.x, it.y) for it in items],
                min_dist=min_dist,
            )

            if gid == target_group_id:
                anchor_x, anchor_y = positions[0]
                items.append(VisualItemSpec(anchor_x, anchor_y, anchor_color, anchor_shape, size, gid, anchor_shape, None, None, False, True, "anchor"))
                for x, y in positions[1 : 1 + factors.target_count]:
                    items.append(VisualItemSpec(x, y, query_color, query_shape, size, gid, query_shape, None, None, True, False, "target"))
                for x, y in positions[1 + factors.target_count :]:
                    items.append(VisualItemSpec(x, y, self._different_color(query_color), query_shape, size, gid, query_shape, None, None, False, False, "group_only"))
            else:
                trap_emitted = False
                for x, y in positions:
                    role = "distractor"
                    color = self._different_color(query_color)
                    shape = self._different_shape(query_shape)
                    similarity_key = shape
                    if not trap_emitted:
                        if factors.cue_conflict == "proximity_vs_similarity":
                            role = "cue_conflict_trap"
                            color = query_color
                            shape = anchor_shape
                            similarity_key = anchor_shape
                        else:
                            role = "feature_only"
                            color = query_color
                            shape = query_shape
                            similarity_key = query_shape
                        trap_emitted = True
                    items.append(VisualItemSpec(x, y, color, shape, size, gid, similarity_key, None, None, False, False, role))
        return items[: factors.num_items], structures

    def _build_similarity_grouping_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> tuple[list[VisualItemSpec], list[VisualStructureSpec]]:
        anchor_shape = target_definition["reference_similarity_key"]
        query_color = target_definition["query_color"]
        size = factors.item_size
        radius = SIZE_TO_RADIUS[size]
        min_dist = 2 * radius + 10

        left = factors.margin + radius + 6
        right = factors.width - factors.margin - radius - 6
        top = factors.margin + radius + 6
        bottom = factors.height - factors.margin - radius - 6

        # Spread positions across the whole canvas using a loose grid with jitter.
        cols = 4
        rows = 3
        cell_w = (right - left) / cols
        cell_h = (bottom - top) / rows

        candidate_positions: list[tuple[float, float]] = []
        for r in range(rows):
            for c in range(cols):
                cx1 = left + c * cell_w
                cy1 = top + r * cell_h
                cx2 = cx1 + cell_w
                cy2 = cy1 + cell_h

                x = rng.uniform(cx1 + 10, cx2 - 10)
                y = rng.uniform(cy1 + 10, cy2 - 10)
                candidate_positions.append((x, y))

        rng.shuffle(candidate_positions)

        # Light repair pass in case jitter created positions that are too close.
        positions: list[tuple[float, float]] = []
        for x, y in candidate_positions:
            if all(math.dist((x, y), p) >= min_dist for p in positions):
                positions.append((x, y))
            else:
                placed = False
                for _ in range(200):
                    nx = rng.uniform(left, right)
                    ny = rng.uniform(top, bottom)
                    if all(math.dist((nx, ny), p) >= min_dist for p in positions):
                        positions.append((nx, ny))
                        placed = True
                        break
                if not placed:
                    raise GenerationError(
                        "Failed to place similarity items without accidental clustering"
                    )

        items: list[VisualItemSpec] = []

        # Anchor
        ax, ay = positions[0]
        items.append(
            VisualItemSpec(
                ax,
                ay,
                "yellow",
                anchor_shape,
                size,
                f"shape_{anchor_shape}",
                anchor_shape,
                None,
                None,
                False,
                True,
                "anchor",
            )
        )

        cursor = 1

        # True targets: same shape-group as anchor, matching query color.
        for x, y in positions[cursor : cursor + factors.target_count]:
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    query_color,
                    anchor_shape,
                    size,
                    f"shape_{anchor_shape}",
                    anchor_shape,
                    None,
                    None,
                    True,
                    False,
                    "target",
                )
            )
        cursor += factors.target_count

        # Same-shape, wrong-color distractor.
        if cursor < len(positions):
            x, y = positions[cursor]
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    self._different_color(query_color),
                    anchor_shape,
                    size,
                    f"shape_{anchor_shape}",
                    anchor_shape,
                    None,
                    None,
                    False,
                    False,
                    "group_only",
                )
            )
            cursor += 1

        # Same-color, wrong-shape distractor.
        alt_shape = self._different_shape(anchor_shape)
        if cursor < len(positions):
            x, y = positions[cursor]
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    query_color,
                    alt_shape,
                    size,
                    f"shape_{alt_shape}",
                    alt_shape,
                    None,
                    None,
                    False,
                    False,
                    "partial_feature",
                )
            )
            cursor += 1

        # Remaining distractors: mixed, spread across the canvas.
        other_shapes = [shape for shape in SHAPES if shape != anchor_shape]
        other_colors = [color for color in COLORS if color != query_color]

        while cursor < len(positions):
            x, y = positions[cursor]
            shape = rng.choice(other_shapes)
            color = rng.choice(other_colors)
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    color,
                    shape,
                    size,
                    f"shape_{shape}",
                    shape,
                    None,
                    None,
                    False,
                    False,
                    "distractor",
                )
            )
            cursor += 1

        return items[: factors.num_items], []

    def _build_common_region_grouping_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> tuple[list[VisualItemSpec], list[VisualStructureSpec]]:
        query_color = target_definition["query_color"]
        query_shape = target_definition["query_shape"]
        size = factors.item_size
        radius = SIZE_TO_RADIUS[size]
        min_dist = 2 * radius + 5

        num_groups = factors.num_groups
        group_sizes = self._sample_group_sizes(rng, factors.num_items, num_groups, min_size=3 if num_groups == 2 else 2)
        layouts = self._sample_box_layouts(rng, factors, num_groups, group_sizes)
        target_group_idx = rng.randrange(num_groups)
        target_region_id = layouts[target_group_idx]["id"]
        structures = [
            VisualStructureSpec(
                "box",
                layout["id"],
                {"x1": layout["x1"], "y1": layout["y1"], "x2": layout["x2"], "y2": layout["y2"]},
            )
            for layout in layouts
        ]

        items: list[VisualItemSpec] = []
        used_positions: list[tuple[float, float]] = []

        positions_by_group: dict[str, list[tuple[float, float]]] = {}
        for layout, gsize in zip(layouts, group_sizes):
            positions = self._sample_points_in_box(
                rng,
                box=(layout["x1"], layout["y1"], layout["x2"], layout["y2"]),
                count=gsize,
                existing=used_positions,
                min_dist=min_dist,
                boundary_bias=(factors.cue_conflict == "proximity_vs_common_region"),
            )
            positions_by_group[layout["id"]] = positions
            used_positions.extend(positions)

        trap_assigned = False
        if factors.cue_conflict == "proximity_vs_common_region" and num_groups >= 2:
            neighbor_idx = 1 if target_group_idx == 0 else 0
            trap_group_id = layouts[neighbor_idx]["id"]
            tx1, ty1, tx2, ty2 = layouts[target_group_idx]["x1"], layouts[target_group_idx]["y1"], layouts[target_group_idx]["x2"], layouts[target_group_idx]["y2"]
            nx1, ny1, nx2, ny2 = layouts[neighbor_idx]["x1"], layouts[neighbor_idx]["y1"], layouts[neighbor_idx]["x2"], layouts[neighbor_idx]["y2"]
            vertical_gap = max(0.0, nx1 - tx2)
            if vertical_gap < 70:
                y = max(ty1 + radius + 6, min((ty1 + ty2) / 2, ty2 - radius - 6, ny2 - radius - 6))
                target_near = (tx2 - radius - 14, y)
                trap_near = (nx1 + radius + 14, y + rng.uniform(-8, 8))
                if self._pair_fits(target_near, trap_near, used_positions, min_dist):
                    positions_by_group[target_region_id][0] = target_near
                    positions_by_group[trap_group_id][0] = trap_near
                    trap_assigned = True

        for layout in layouts:
            gid = layout["id"]
            positions = positions_by_group[gid]
            if gid == target_region_id:
                anchor_idx = rng.randrange(len(positions))
                anchor_pos = positions[anchor_idx]
                positions = positions[:anchor_idx] + positions[anchor_idx + 1 :]
                items.append(
                    VisualItemSpec(anchor_pos[0], anchor_pos[1], "yellow", "triangle", size, gid, None, gid, None, False, True, "anchor")
                )
                for x, y in positions[: factors.target_count]:
                    items.append(
                        VisualItemSpec(x, y, query_color, query_shape, size, gid, None, gid, None, True, False, "target")
                    )
                remaining = positions[factors.target_count :]
                if remaining:
                    x, y = remaining[0]
                    items.append(
                        VisualItemSpec(x, y, self._different_color(query_color), query_shape, size, gid, None, gid, None, False, False, "group_only")
                    )
                for x, y in remaining[1:]:
                    color, shape = self._sample_non_target_feature(rng, query_color, query_shape, "group_only")
                    items.append(
                        VisualItemSpec(x, y, color, shape, size, gid, None, gid, None, False, False, "distractor")
                    )
            else:
                for idx, (x, y) in enumerate(positions):
                    if idx == 0:
                        role = "feature_only"
                        color = query_color
                        shape = query_shape
                        if factors.cue_conflict == "proximity_vs_common_region" and (trap_assigned or gid != target_region_id):
                            role = "cue_conflict_trap"
                        elif factors.confound_type == "cross_binding":
                            role = "cross_binding_feature"
                        items.append(
                            VisualItemSpec(x, y, color, shape, size, gid, None, gid, None, False, False, role)
                        )
                    elif idx == 1 and factors.confound_type == "cross_binding":
                        items.append(
                            VisualItemSpec(x, y, self._different_color(query_color), query_shape, size, gid, None, gid, None, False, False, "group_only")
                        )
                    else:
                        color, shape = self._sample_non_target_feature(rng, query_color, query_shape, "feature_only")
                        items.append(
                            VisualItemSpec(x, y, color, shape, size, gid, None, gid, None, False, False, "distractor")
                        )

        return items[: factors.num_items], structures

    def _build_relation_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> tuple[list[VisualItemSpec], list[VisualStructureSpec]]:
        query_color = target_definition["query_color"]
        query_shape = target_definition["query_shape"]
        size = factors.item_size
        radius = SIZE_TO_RADIUS[size]
        min_dist = 2 * radius + 8

        items: list[VisualItemSpec] = []
        structures: list[VisualStructureSpec] = []

        template = rng.choice(["single_box", "vertical_line", "horizontal_line"])

        used_positions: list[tuple[float, float]] = []

        def sample_points_in_rect(
            rect: tuple[float, float, float, float],
            count: int,
            *,
            pad: float = 8.0,
        ) -> list[tuple[float, float]]:
            x1, y1, x2, y2 = rect
            sx1 = x1 + radius + pad
            sy1 = y1 + radius + pad
            sx2 = x2 - radius - pad
            sy2 = y2 - radius - pad
            if sx2 <= sx1 or sy2 <= sy1:
                raise GenerationError("Rectangle too small in shallow scene")

            pts: list[tuple[float, float]] = []
            for _ in range(count):
                placed = False
                for _ in range(1200):
                    x = rng.uniform(sx1, sx2)
                    y = rng.uniform(sy1, sy2)
                    if any(math.dist((x, y), p) < min_dist for p in pts):
                        continue
                    if any(math.dist((x, y), p) < min_dist for p in used_positions):
                        continue
                    pts.append((x, y))
                    used_positions.append((x, y))
                    placed = True
                    break
                if not placed:
                    raise GenerationError("Failed to place points in shallow scene")
            return pts

        candidate_points: list[tuple[float, float, str | None]] = []

        if template == "single_box":
            box_w = rng.uniform(260, 340)
            box_h = rng.uniform(180, 240)
            x1 = rng.uniform(120, factors.width - 120 - box_w)
            y1 = rng.uniform(95, factors.height - 95 - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h

            structures.append(
                VisualStructureSpec("box", "box_main", {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            )

            inside_pts = sample_points_in_rect((x1, y1, x2, y2), max(5, factors.target_count + 2))

            outside_rects = [
                (factors.margin, factors.margin, x1 - 10, factors.height - factors.margin),
                (
                    x2 + 10,
                    factors.margin,
                    factors.width - factors.margin,
                    factors.height - factors.margin,
                ),
            ]
            outside_pts: list[tuple[float, float]] = []
            for rect in outside_rects:
                rx1, ry1, rx2, ry2 = rect
                if rx2 - rx1 > 2 * radius + 20:
                    try:
                        outside_pts.extend(sample_points_in_rect(rect, 3, pad=4.0))
                    except GenerationError:
                        pass

            for x, y in inside_pts:
                candidate_points.append((x, y, "box_main"))
            for x, y in outside_pts:
                candidate_points.append((x, y, None))

            def qualifies(x: float, y: float) -> bool:
                inside = x1 <= x <= x2 and y1 <= y <= y2
                if target_definition["rule_type"] == "inside_region":
                    return inside
                if target_definition["rule_type"] == "outside_region":
                    return not inside
                return inside

        elif template == "vertical_line":
            line_x = rng.uniform(260, 380)
            top = factors.margin + 20
            bottom = factors.height - factors.margin - 20
            structures.append(
                VisualStructureSpec(
                    "line",
                    "divider",
                    {"orientation": "vertical", "x": line_x, "y1": top, "y2": bottom},
                )
            )

            left_pts = sample_points_in_rect(
                (factors.margin, factors.margin, line_x - 10, factors.height - factors.margin),
                6,
                pad=4.0,
            )
            right_pts = sample_points_in_rect(
                (
                    line_x + 10,
                    factors.margin,
                    factors.width - factors.margin,
                    factors.height - factors.margin,
                ),
                6,
                pad=4.0,
            )

            for x, y in left_pts + right_pts:
                candidate_points.append((x, y, None))

            def qualifies(x: float, y: float) -> bool:
                if target_definition["rule_type"] == "left_of_line":
                    return x < line_x
                if target_definition["rule_type"] == "right_of_line":
                    return x > line_x
                # fallback so relation scenes still work even if factors mismatch
                return x < line_x

        else:
            line_y = rng.uniform(185, 295)
            left = factors.margin + 20
            right = factors.width - factors.margin - 20
            structures.append(
                VisualStructureSpec(
                    "line",
                    "divider",
                    {"orientation": "horizontal", "y": line_y, "x1": left, "x2": right},
                )
            )

            top_pts = sample_points_in_rect(
                (factors.margin, factors.margin, factors.width - factors.margin, line_y - 10),
                6,
                pad=4.0,
            )
            bottom_pts = sample_points_in_rect(
                (
                    factors.margin,
                    line_y + 10,
                    factors.width - factors.margin,
                    factors.height - factors.margin,
                ),
                6,
                pad=4.0,
            )

            for x, y in top_pts + bottom_pts:
                candidate_points.append((x, y, None))

            def qualifies(x: float, y: float) -> bool:
                if target_definition["rule_type"] == "above_line":
                    return y < line_y
                if target_definition["rule_type"] == "below_line":
                    return y > line_y
                return y < line_y

        rng.shuffle(candidate_points)

        qualifying = [(x, y, rid) for x, y, rid in candidate_points if qualifies(x, y)]
        nonqualifying = [(x, y, rid) for x, y, rid in candidate_points if not qualifies(x, y)]

        if len(qualifying) < max(1, factors.target_count):
            raise GenerationError("Not enough qualifying points in shallow scene")
        if len(nonqualifying) < 3:
            raise GenerationError("Not enough non-qualifying points in shallow scene")

        for x, y, region_id in qualifying[: factors.target_count]:
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    query_color,
                    query_shape,
                    size,
                    "target_zone",
                    None,
                    region_id,
                    None,
                    True,
                    False,
                    "target",
                )
            )

        remaining_q = qualifying[factors.target_count :]
        if remaining_q:
            x, y, region_id = remaining_q[0]
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    self._different_color(query_color),
                    query_shape,
                    size,
                    "target_zone",
                    None,
                    region_id,
                    None,
                    False,
                    False,
                    "group_only",
                )
            )

        x, y, region_id = nonqualifying[0]
        items.append(
            VisualItemSpec(
                x,
                y,
                query_color,
                query_shape,
                size,
                "other_zone",
                None,
                region_id,
                None,
                False,
                False,
                "feature_only",
            )
        )

        for x, y, region_id in nonqualifying[1:]:
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    self._different_color(query_color),
                    self._different_shape(query_shape),
                    size,
                    "other_zone",
                    None,
                    region_id,
                    None,
                    False,
                    False,
                    "distractor",
                )
            )

        if factors.target_count == 0:
            items = [
                VisualItemSpec(
                    item.x,
                    item.y,
                    self._different_color(query_color) if item.role == "target" else item.color,
                    item.shape,
                    item.size,
                    item.group_id,
                    item.similarity_key,
                    item.region_id,
                    item.panel_id,
                    False,
                    item.is_anchor,
                    "partial_feature" if item.role == "target" else item.role,
                )
                for item in items
            ]

        return items[: factors.num_items], structures

    def _build_scope_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> tuple[list[VisualItemSpec], list[VisualStructureSpec]]:
        query_color = target_definition["query_color"]
        query_shape = target_definition["query_shape"]
        size = factors.item_size
        radius = SIZE_TO_RADIUS[size]
        min_dist = 2 * radius + 8

        items: list[VisualItemSpec] = []
        structures: list[VisualStructureSpec] = []
        used_positions: list[tuple[float, float]] = []

        layout = rng.choice(["left_right", "top_bottom", "triple_box"])

        def sample_points_in_rect(
            rect: tuple[float, float, float, float],
            count: int,
            *,
            pad: float = 10.0,
        ) -> list[tuple[float, float]]:
            x1, y1, x2, y2 = rect
            sx1 = x1 + radius + pad
            sy1 = y1 + radius + pad
            sx2 = x2 - radius - pad
            sy2 = y2 - radius - pad
            if sx2 <= sx1 or sy2 <= sy1:
                raise GenerationError("Scope rectangle too small")

            pts: list[tuple[float, float]] = []
            for _ in range(count):
                placed = False
                for _ in range(1200):
                    x = rng.uniform(sx1, sx2)
                    y = rng.uniform(sy1, sy2)
                    if any(math.dist((x, y), p) < min_dist for p in pts):
                        continue
                    if any(math.dist((x, y), p) < min_dist for p in used_positions):
                        continue
                    pts.append((x, y))
                    used_positions.append((x, y))
                    placed = True
                    break
                if not placed:
                    raise GenerationError("Failed to place items in scope scene")
            return pts

        contexts: list[dict[str, object]] = []

        if layout == "left_right":
            gap = 26.0
            panel_w = 235.0
            panel_h = 270.0
            y1 = 90.0
            y2 = y1 + panel_h
            left_x1 = 50.0
            left_x2 = left_x1 + panel_w
            right_x1 = left_x2 + gap
            right_x2 = right_x1 + panel_w

            structures.extend(
                [
                    VisualStructureSpec(
                        "panel", "panel_left", {"x1": left_x1, "y1": y1, "x2": left_x2, "y2": y2}
                    ),
                    VisualStructureSpec(
                        "panel", "panel_right", {"x1": right_x1, "y1": y1, "x2": right_x2, "y2": y2}
                    ),
                ]
            )

            contexts = [
                {"context_id": "panel_left", "rect": (left_x1, y1, left_x2, y2), "selected": True},
                {
                    "context_id": "panel_right",
                    "rect": (right_x1, y1, right_x2, y2),
                    "selected": False,
                },
            ]
            target_definition["scope_selector"] = "left_panel"

        elif layout == "top_bottom":
            gap = 24.0
            panel_h = 125.0
            x1 = 85.0
            x2 = 555.0
            top_y1 = 70.0
            top_y2 = top_y1 + panel_h
            bottom_y1 = top_y2 + gap
            bottom_y2 = bottom_y1 + panel_h

            structures.extend(
                [
                    VisualStructureSpec(
                        "panel", "panel_top", {"x1": x1, "y1": top_y1, "x2": x2, "y2": top_y2}
                    ),
                    VisualStructureSpec(
                        "panel",
                        "panel_bottom",
                        {"x1": x1, "y1": bottom_y1, "x2": x2, "y2": bottom_y2},
                    ),
                ]
            )

            contexts = [
                {"context_id": "panel_top", "rect": (x1, top_y1, x2, top_y2), "selected": True},
                {
                    "context_id": "panel_bottom",
                    "rect": (x1, bottom_y1, x2, bottom_y2),
                    "selected": False,
                },
            ]
            target_definition["scope_selector"] = "top_panel"

        else:
            gap = 18.0
            box_w = 145.0
            box_h = 185.0
            y1 = 145.0
            y2 = y1 + box_h
            x_start = 62.0

            triple_boxes = []
            for i in range(3):
                bx1 = x_start + i * (box_w + gap)
                bx2 = bx1 + box_w
                triple_boxes.append((bx1, y1, bx2, y2))
                structures.append(
                    VisualStructureSpec(
                        "box", f"scope_box_{i}", {"x1": bx1, "y1": y1, "x2": bx2, "y2": y2}
                    )
                )

            contexts = [
                {"context_id": "scope_box_0", "rect": triple_boxes[0], "selected": True},
                {"context_id": "scope_box_1", "rect": triple_boxes[1], "selected": False},
                {"context_id": "scope_box_2", "rect": triple_boxes[2], "selected": False},
            ]
            target_definition["scope_selector"] = "leftmost_box"

        selected_context = next(ctx for ctx in contexts if bool(ctx["selected"]))
        selected_rect = selected_context["rect"]
        selected_context_id = str(selected_context["context_id"])

        selected_points = sample_points_in_rect(selected_rect, max(factors.target_count + 2, 5))
        other_context_points: list[tuple[float, float, str]] = []

        for ctx in contexts:
            if ctx is selected_context:
                continue
            pts = sample_points_in_rect(ctx["rect"], 4)
            for x, y in pts:
                other_context_points.append((x, y, str(ctx["context_id"])))

        # Anchor in selected context
        ax, ay = selected_points[0]
        items.append(
            VisualItemSpec(
                ax,
                ay,
                "yellow",
                "triangle",
                size,
                selected_context_id,
                None,
                None,
                selected_context_id if selected_context_id.startswith("panel_") else None,
                False,
                True,
                "anchor",
            )
        )

        # Targets in selected context
        for x, y in selected_points[1 : 1 + factors.target_count]:
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    query_color,
                    query_shape,
                    size,
                    selected_context_id,
                    None,
                    None,
                    selected_context_id if selected_context_id.startswith("panel_") else None,
                    True,
                    False,
                    "target",
                )
            )

        remaining_selected = selected_points[1 + factors.target_count :]
        if remaining_selected:
            x, y = remaining_selected[0]
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    self._different_color(query_color),
                    query_shape,
                    size,
                    selected_context_id,
                    None,
                    None,
                    selected_context_id if selected_context_id.startswith("panel_") else None,
                    False,
                    False,
                    "group_only",
                )
            )
            for x, y in remaining_selected[1:]:
                color, shape = self._sample_non_target_feature(
                    rng, query_color, query_shape, "group_only"
                )
                items.append(
                    VisualItemSpec(
                        x,
                        y,
                        color,
                        shape,
                        size,
                        selected_context_id,
                        None,
                        None,
                        selected_context_id if selected_context_id.startswith("panel_") else None,
                        False,
                        False,
                        "distractor",
                    )
                )

        # Wrong-scope exact match
        x, y, other_context_id = other_context_points[0]
        items.append(
            VisualItemSpec(
                x,
                y,
                query_color,
                query_shape,
                size,
                other_context_id,
                None,
                None,
                other_context_id if other_context_id.startswith("panel_") else None,
                False,
                False,
                "wrong_scope",
            )
        )

        # Same-shape wrong-color distractor
        if len(other_context_points) > 1:
            x, y, other_context_id = other_context_points[1]
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    self._different_color(query_color),
                    query_shape,
                    size,
                    other_context_id,
                    None,
                    None,
                    other_context_id if other_context_id.startswith("panel_") else None,
                    False,
                    False,
                    "distractor",
                )
            )

        # Same-color wrong-shape distractor
        if len(other_context_points) > 2:
            x, y, other_context_id = other_context_points[2]
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    query_color,
                    self._different_shape(query_shape),
                    size,
                    other_context_id,
                    None,
                    None,
                    other_context_id if other_context_id.startswith("panel_") else None,
                    False,
                    False,
                    "partial_feature",
                )
            )

        for x, y, other_context_id in other_context_points[3:]:
            color, shape = self._sample_non_target_feature(
                rng, query_color, query_shape, "feature_only"
            )
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    color,
                    shape,
                    size,
                    other_context_id,
                    None,
                    None,
                    other_context_id if other_context_id.startswith("panel_") else None,
                    False,
                    False,
                    "distractor",
                )
            )

        # Optional free distractor outside explicit scope regions
        free_rects = [
            (factors.margin, factors.margin, factors.width - factors.margin, 70.0),
            (
                factors.margin,
                factors.height - 70.0,
                factors.width - factors.margin,
                factors.height - factors.margin,
            ),
        ]
        for rect in free_rects:
            try:
                free_pts = sample_points_in_rect(rect, 1, pad=4.0)
                fx, fy = free_pts[0]
                color, shape = self._sample_non_target_feature(
                    rng, query_color, query_shape, "feature_only"
                )
                items.append(
                    VisualItemSpec(
                        fx,
                        fy,
                        color,
                        shape,
                        size,
                        "free",
                        None,
                        None,
                        None,
                        False,
                        False,
                        "distractor",
                    )
                )
                break
            except GenerationError:
                continue

        if factors.target_count == 0:
            items = [
                VisualItemSpec(
                    it.x,
                    it.y,
                    self._different_color(query_color) if it.role == "target" else it.color,
                    it.shape,
                    it.size,
                    it.group_id,
                    it.similarity_key,
                    it.region_id,
                    it.panel_id,
                    False,
                    it.is_anchor,
                    "partial_feature" if it.role == "target" else it.role,
                )
                for it in items
            ]

        return items[: factors.num_items], structures

    def _build_global_local_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> tuple[list[VisualItemSpec], list[VisualStructureSpec]]:
        query_color = target_definition["query_color"]
        query_shape = target_definition["query_shape"]
        size = factors.item_size
        radius = SIZE_TO_RADIUS[size]
        min_dist = 2 * radius + 8

        outer_x1 = 65.0
        outer_y1 = 85.0
        outer_x2 = 575.0
        outer_y2 = 375.0

        structures: list[VisualStructureSpec] = [
            VisualStructureSpec(
                "box",
                "outer_box",
                {"x1": outer_x1, "y1": outer_y1, "x2": outer_x2, "y2": outer_y2},
            )
        ]

        inner_box_count = rng.choice([2, 3])
        anchor_mode = rng.choice(["inner", "outer"])

        usable_left = outer_x1 + 18
        usable_top = outer_y1 + 18
        usable_right = outer_x2 - 18
        usable_bottom = outer_y2 - 18

        gap = 18.0
        total_w = usable_right - usable_left
        lane_w = (total_w - gap * (inner_box_count - 1)) / inner_box_count

        inner_boxes: list[dict[str, float | str]] = []
        for idx in range(inner_box_count):
            lane_x1 = usable_left + idx * (lane_w + gap)
            lane_x2 = lane_x1 + lane_w

            box_w = rng.uniform(max(118.0, lane_w - 32), lane_w - 4)
            box_h = rng.uniform(120.0, 205.0)
            x1 = rng.uniform(lane_x1 + 2, lane_x2 - box_w)
            y1 = rng.uniform(usable_top, usable_bottom - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h

            box_id = f"inner_{idx}"
            inner_boxes.append({"id": box_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            structures.append(
                VisualStructureSpec("box", box_id, {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            )

        inner_boxes_sorted = sorted(inner_boxes, key=lambda b: float(b["x1"]))
        target_region_id = str(inner_boxes_sorted[0]["id"])
        target_definition["selected_region_id"] = target_region_id
        target_definition["anchor_mode"] = anchor_mode

        used_positions: list[tuple[float, float]] = []
        items: list[VisualItemSpec] = []

        def sample_points_in_rect(
            rect: tuple[float, float, float, float],
            count: int,
            *,
            extra_pad: float = 10.0,
        ) -> list[tuple[float, float]]:
            x1, y1, x2, y2 = rect
            sx1 = x1 + radius + extra_pad
            sy1 = y1 + radius + extra_pad
            sx2 = x2 - radius - extra_pad
            sy2 = y2 - radius - extra_pad
            if sx2 <= sx1 or sy2 <= sy1:
                raise GenerationError("Sampling rectangle too small")

            pts: list[tuple[float, float]] = []
            for _ in range(count):
                placed = False
                for _ in range(1200):
                    x = rng.uniform(sx1, sx2)
                    y = rng.uniform(sy1, sy2)
                    if any(math.dist((x, y), p) < min_dist for p in pts):
                        continue
                    if any(math.dist((x, y), p) < min_dist for p in used_positions):
                        continue
                    pts.append((x, y))
                    used_positions.append((x, y))
                    placed = True
                    break
                if not placed:
                    raise GenerationError("Failed to place items in nested scene rectangle")
            return pts

        def box_capacity(box: dict[str, float | str]) -> int:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
            inner_w = (x2 - x1) - 2 * (radius + 10)
            inner_h = (y2 - y1) - 2 * (radius + 10)
            cols = max(1, int(inner_w // (2 * radius + 8)))
            rows = max(1, int(inner_h // (2 * radius + 8)))
            return max(1, cols * rows)

        counts_by_box: dict[str, int] = {}
        for box in inner_boxes:
            cap = box_capacity(box)
            box_id = str(box["id"])
            if box_id == target_region_id:
                # extra space if anchor sits inside the selected inner box
                desired = (
                    factors.target_count + (2 if anchor_mode == "inner" else 1) + rng.choice([0, 1])
                )
            else:
                desired = rng.choice([2, 3, 4])
            counts_by_box[box_id] = min(max(desired, 2), cap)

        positions_by_box: dict[str, list[tuple[float, float]]] = {}
        for box in inner_boxes:
            rect = (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
            positions_by_box[str(box["id"])] = sample_points_in_rect(
                rect, counts_by_box[str(box["id"])]
            )

        outer_candidates: list[tuple[float, float, float, float]] = []
        leftmost_x1 = min(float(b["x1"]) for b in inner_boxes)
        rightmost_x2 = max(float(b["x2"]) for b in inner_boxes)
        highest_y1 = min(float(b["y1"]) for b in inner_boxes)
        lowest_y2 = max(float(b["y2"]) for b in inner_boxes)

        if leftmost_x1 - outer_x1 > 55:
            outer_candidates.append((outer_x1, outer_y1, leftmost_x1 - 8, outer_y2))
        if outer_x2 - rightmost_x2 > 55:
            outer_candidates.append((rightmost_x2 + 8, outer_y1, outer_x2, outer_y2))
        if highest_y1 - outer_y1 > 45:
            outer_candidates.append((outer_x1 + 8, outer_y1, outer_x2 - 8, highest_y1 - 8))
        if outer_y2 - lowest_y2 > 45:
            outer_candidates.append((outer_x1 + 8, lowest_y2 + 8, outer_x2 - 8, outer_y2))

        outer_only_points: list[tuple[float, float]] = []
        outer_item_target = rng.choice([3, 4, 5])

        shuffled_candidates = outer_candidates[:]
        rng.shuffle(shuffled_candidates)
        for rect in shuffled_candidates:
            if len(outer_only_points) >= outer_item_target:
                break
            remaining = outer_item_target - len(outer_only_points)
            count_here = min(remaining, rng.choice([1, 2]))
            try:
                pts = sample_points_in_rect(rect, count_here, extra_pad=8.0)
                outer_only_points.extend(pts)
            except GenerationError:
                continue

        if len(outer_only_points) < 2:
            raise GenerationError("Failed to place enough outer-only items in nested scene")

        # Anchor placement mode
        if anchor_mode == "inner":
            target_positions = positions_by_box[target_region_id]
            anchor_x, anchor_y = target_positions[0]
            items.append(
                VisualItemSpec(
                    anchor_x,
                    anchor_y,
                    "yellow",
                    "triangle",
                    size,
                    target_region_id,
                    None,
                    target_region_id,
                    None,
                    False,
                    True,
                    "anchor",
                )
            )
            target_start_idx = 1
        else:
            anchor_x, anchor_y = outer_only_points[0]
            items.append(
                VisualItemSpec(
                    anchor_x,
                    anchor_y,
                    "yellow",
                    "triangle",
                    size,
                    "outer_only",
                    None,
                    "outer_box",
                    None,
                    False,
                    True,
                    "anchor",
                )
            )
            outer_only_points = outer_only_points[1:]
            target_start_idx = 0

        # Selected inner box targets
        target_positions = positions_by_box[target_region_id]
        for x, y in target_positions[target_start_idx : target_start_idx + factors.target_count]:
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    query_color,
                    query_shape,
                    size,
                    target_region_id,
                    None,
                    target_region_id,
                    None,
                    True,
                    False,
                    "target",
                )
            )

        remaining_target_positions = target_positions[target_start_idx + factors.target_count :]
        if remaining_target_positions:
            x, y = remaining_target_positions[0]
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    self._different_color(query_color),
                    query_shape,
                    size,
                    target_region_id,
                    None,
                    target_region_id,
                    None,
                    False,
                    False,
                    "group_only",
                )
            )
            for x, y in remaining_target_positions[1:]:
                color, shape = self._sample_non_target_feature(
                    rng, query_color, query_shape, "group_only"
                )
                items.append(
                    VisualItemSpec(
                        x,
                        y,
                        color,
                        shape,
                        size,
                        target_region_id,
                        None,
                        target_region_id,
                        None,
                        False,
                        False,
                        "distractor",
                    )
                )

        # Non-selected inner boxes
        for box in inner_boxes:
            box_id = str(box["id"])
            if box_id == target_region_id:
                continue

            pos_list = positions_by_box[box_id]
            if not pos_list:
                continue

            x, y = pos_list[0]
            role = (
                "cross_binding_feature"
                if factors.confound_type == "cross_binding"
                else "wrong_global_choice"
            )
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    query_color,
                    query_shape,
                    size,
                    box_id,
                    None,
                    box_id,
                    None,
                    False,
                    False,
                    role,
                )
            )

            cursor = 1
            if factors.confound_type == "cross_binding" and cursor < len(pos_list):
                x, y = pos_list[cursor]
                items.append(
                    VisualItemSpec(
                        x,
                        y,
                        self._different_color(query_color),
                        query_shape,
                        size,
                        box_id,
                        None,
                        box_id,
                        None,
                        False,
                        False,
                        "group_only",
                    )
                )
                cursor += 1

            for x, y in pos_list[cursor:]:
                color, shape = self._sample_non_target_feature(
                    rng, query_color, query_shape, factors.confound_type
                )
                items.append(
                    VisualItemSpec(
                        x,
                        y,
                        color,
                        shape,
                        size,
                        box_id,
                        None,
                        box_id,
                        None,
                        False,
                        False,
                        "distractor",
                    )
                )

        # Outer-only distractors and traps
        if not outer_only_points:
            raise GenerationError("No outer-only points left for nested scene")

        x, y = outer_only_points[0]
        outer_match_role = (
            "cue_conflict_trap"
            if factors.cue_conflict == "proximity_vs_common_region"
            else "wrong_scope"
        )
        items.append(
            VisualItemSpec(
                x,
                y,
                query_color,
                query_shape,
                size,
                "outer_only",
                None,
                "outer_box",
                None,
                False,
                False,
                outer_match_role,
            )
        )

        for x, y in outer_only_points[1:]:
            color, shape = self._sample_non_target_feature(
                rng, query_color, query_shape, "feature_only"
            )
            items.append(
                VisualItemSpec(
                    x,
                    y,
                    color,
                    shape,
                    size,
                    "outer_only",
                    None,
                    "outer_box",
                    None,
                    False,
                    False,
                    "distractor",
                )
            )

        # Additional near-boundary trap for hard cue-conflict cases
        if factors.cue_conflict == "proximity_vs_common_region":
            target_box = next(box for box in inner_boxes if str(box["id"]) == target_region_id)
            tx1 = float(target_box["x1"])
            ty1 = float(target_box["y1"])
            tx2 = float(target_box["x2"])
            ty2 = float(target_box["y2"])

            trap_candidates = [
                (tx1 - radius - 14, (ty1 + ty2) / 2),
                (tx2 + radius + 14, (ty1 + ty2) / 2),
                ((tx1 + tx2) / 2, ty1 - radius - 14),
                ((tx1 + tx2) / 2, ty2 + radius + 14),
            ]

            if not any(item.role == "cue_conflict_trap" for item in items):
                for trap_x, trap_y in trap_candidates:
                    inside_outer = (
                        outer_x1 + radius <= trap_x <= outer_x2 - radius
                        and outer_y1 + radius <= trap_y <= outer_y2 - radius
                    )
                    inside_any_inner = any(
                        float(box["x1"]) + radius <= trap_x <= float(box["x2"]) - radius
                        and float(box["y1"]) + radius <= trap_y <= float(box["y2"]) - radius
                        for box in inner_boxes
                    )
                    far_enough = all(
                        math.dist((trap_x, trap_y), p) >= min_dist for p in used_positions
                    )
                    if inside_outer and not inside_any_inner and far_enough:
                        items.append(
                            VisualItemSpec(
                                trap_x,
                                trap_y,
                                query_color,
                                query_shape,
                                size,
                                "outer_only",
                                None,
                                "outer_box",
                                None,
                                False,
                                False,
                                "cue_conflict_trap",
                            )
                        )
                        used_positions.append((trap_x, trap_y))
                        break
                else:
                    raise GenerationError("Failed to place cue-conflict trap in nested scene")

        if factors.target_count == 0:
            items = [
                VisualItemSpec(
                    it.x,
                    it.y,
                    self._different_color(query_color) if it.role == "target" else it.color,
                    it.shape,
                    it.size,
                    it.group_id,
                    it.similarity_key,
                    it.region_id,
                    it.panel_id,
                    False,
                    it.is_anchor,
                    "partial_feature" if it.role == "target" else it.role,
                )
                for it in items
            ]

        return items[: factors.num_items], structures

    def _sample_group_sizes(
        self,
        rng: random.Random,
        total: int,
        groups: int,
        *,
        min_size: int,
    ) -> list[int]:
        sizes = [min_size] * groups
        remainder = total - min_size * groups
        if remainder < 0:
            raise GenerationError("Impossible group size partition")
        for _ in range(remainder):
            sizes[rng.randrange(groups)] += 1
        if len(set(sizes)) == 1 and total - min_size * groups > 0:
            give = rng.randrange(groups)
            take = (give + 1) % groups
            if sizes[take] > min_size:
                sizes[give] += 1
                sizes[take] -= 1
        rng.shuffle(sizes)
        return sizes

    def _sample_cluster_centers(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        count: int,
        *,
        min_sep: float,
    ) -> list[tuple[float, float]]:
        centers: list[tuple[float, float]] = []
        edge_pad = 70
        for _ in range(1200):
            if len(centers) == count:
                break
            x = rng.uniform(factors.margin + edge_pad, factors.width - factors.margin - edge_pad)
            y = rng.uniform(factors.margin + edge_pad, factors.height - factors.margin - edge_pad)
            if all(math.dist((x, y), c) >= min_sep for c in centers):
                centers.append((x, y))
        if len(centers) != count:
            raise GenerationError("Failed to sample separated cluster centers")
        return centers

    def _sample_points_in_circle(
        self,
        rng: random.Random,
        *,
        center: tuple[float, float],
        count: int,
        radius: float,
        bounds: tuple[float, float, float, float],
        existing: list[tuple[float, float]],
        min_dist: float,
    ) -> list[tuple[float, float]]:
        left, right, top, bottom = bounds
        points: list[tuple[float, float]] = []
        for _ in range(count):
            placed = False
            for _ in range(1200):
                angle = rng.uniform(0, 2 * math.pi)
                rr = radius * math.sqrt(rng.uniform(0.03, 1.0))
                x = center[0] + math.cos(angle) * rr
                y = center[1] + math.sin(angle) * rr
                if not (left <= x <= right and top <= y <= bottom):
                    continue
                if any(math.dist((x, y), p) < min_dist for p in points):
                    continue
                if any(math.dist((x, y), p) < min_dist for p in existing):
                    continue
                points.append((x, y))
                placed = True
                break
            if not placed:
                raise GenerationError("Failed to place clustered items without overlap")
        return points

    def _box_can_fit_count(
        self,
        box: tuple[float, float, float, float],
        count: int,
        radius: int,
        min_gap: float,
    ) -> bool:
        x1, y1, x2, y2 = box
        inner_pad = radius + 12
        inner_w = (x2 - x1) - 2 * inner_pad
        inner_h = (y2 - y1) - 2 * inner_pad
        if inner_w <= 0 or inner_h <= 0:
            return False
        cols = max(1, int(inner_w // (2 * radius + min_gap)))
        rows = max(1, int(inner_h // (2 * radius + min_gap)))
        return cols * rows >= count

    def _boxes_too_close(
        self,
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
        *,
        margin: float,
    ) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 + margin < bx1 or bx2 + margin < ax1 or ay2 + margin < by1 or by2 + margin < ay1)

    def _sample_box_layouts(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        count: int,
        group_sizes: list[int],
    ) -> list[dict[str, float | str]]:
        radius = SIZE_TO_RADIUS[factors.item_size]
        usable_left = factors.margin + 2
        usable_top = factors.margin + 2
        usable_right = factors.width - factors.margin - 2
        usable_bottom = factors.height - factors.margin - 2

        layouts: list[dict[str, float | str]] = []
        min_sep = 8.0

        def candidate_size(group_size: int) -> tuple[float, float]:
            base_w = max(125.0, (2 * radius + 10) * min(group_size, 3) + 42)
            base_h = max(115.0, (2 * radius + 10) * max(2, math.ceil(group_size / 3)) + 38)

            profile = rng.choice(["wide", "tall", "balanced"])
            if profile == "wide":
                w = rng.uniform(base_w + 10, min(260.0, base_w + 70))
                h = rng.uniform(base_h, min(210.0, base_h + 30))
            elif profile == "tall":
                w = rng.uniform(base_w, min(210.0, base_w + 25))
                h = rng.uniform(base_h + 10, min(250.0, base_h + 60))
            else:
                w = rng.uniform(base_w, min(235.0, base_w + 45))
                h = rng.uniform(base_h, min(235.0, base_h + 45))
            return w, h

        for idx, group_size in enumerate(group_sizes):
            placed = False
            for _ in range(1500):
                box_w, box_h = candidate_size(group_size)

                x1 = rng.uniform(usable_left, usable_right - box_w)
                y1 = rng.uniform(usable_top, usable_bottom - box_h)
                x2 = x1 + box_w
                y2 = y1 + box_h
                box = (x1, y1, x2, y2)

                if not self._box_can_fit_count(box, group_size, radius, min_gap=6.0):
                    continue

                if any(
                    self._boxes_too_close(
                        box,
                        (float(l["x1"]), float(l["y1"]), float(l["x2"]), float(l["y2"])),
                        margin=min_sep,
                    )
                    for l in layouts
                ):
                    continue

                layouts.append({"id": f"box_{idx}", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                placed = True
                break

            if not placed:
                layouts = []
                break

        if len(layouts) == count:
            return layouts

        layouts = []
        if count == 2:
            gap = 24.0
            total_w = usable_right - usable_left
            lane_w = (total_w - gap) / 2.0
            xs = [usable_left, usable_left + lane_w + gap]

            for idx, group_size in enumerate(group_sizes):
                x1 = xs[idx] + rng.uniform(0, 12)
                x2 = xs[idx] + lane_w - rng.uniform(0, 12)

                min_h = max(130.0, 70.0 + group_size * 22.0)
                h = min(min_h + rng.uniform(10, 40), usable_bottom - usable_top - 20)
                y1 = rng.uniform(usable_top + 8, usable_bottom - h - 8)
                y2 = y1 + h

                box = (x1, y1, x2, y2)
                if not self._box_can_fit_count(box, group_size, radius, min_gap=6.0):
                    raise GenerationError("Fallback 2-box layout still cannot fit group items")

                layouts.append({"id": f"box_{idx}", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            return layouts

        if count == 3:
            gap = 20.0
            total_w = usable_right - usable_left
            lane_w = (total_w - 2 * gap) / 3.0
            xs = [
                usable_left,
                usable_left + lane_w + gap,
                usable_left + 2 * (lane_w + gap),
            ]

            for idx, group_size in enumerate(group_sizes):
                x1 = xs[idx] + rng.uniform(0, 10)
                x2 = xs[idx] + lane_w - rng.uniform(0, 10)

                min_h = max(120.0, 65.0 + group_size * 20.0)
                h = min(min_h + rng.uniform(8, 35), usable_bottom - usable_top - 16)
                y1 = rng.uniform(usable_top + 6, usable_bottom - h - 6)
                y2 = y1 + h

                box = (x1, y1, x2, y2)
                if not self._box_can_fit_count(box, group_size, radius, min_gap=6.0):
                    raise GenerationError("Fallback 3-box layout still cannot fit group items")

                layouts.append({"id": f"box_{idx}", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            return layouts

        raise GenerationError("Failed to sample non-overlapping common-region boxes")

    def _sample_points_in_box(
        self,
        rng: random.Random,
        *,
        box: tuple[float, float, float, float],
        count: int,
        existing: list[tuple[float, float]],
        min_dist: float,
        boundary_bias: bool,
    ) -> list[tuple[float, float]]:
        x1, y1, x2, y2 = box
        radius = 17.0
        pad = radius + 10.0
        inner_x1 = x1 + pad
        inner_y1 = y1 + pad
        inner_x2 = x2 - pad
        inner_y2 = y2 - pad

        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            raise GenerationError("Box interior too small for item placement")

        points: list[tuple[float, float]] = []
        for _ in range(count):
            placed = False
            for _ in range(1600):
                if boundary_bias and rng.random() < 0.28:
                    edge = rng.choice(["left", "right", "top", "bottom"])
                    edge_band = 12.0
                    if edge == "left":
                        x = rng.uniform(inner_x1, min(inner_x1 + edge_band, inner_x2))
                        y = rng.uniform(inner_y1, inner_y2)
                    elif edge == "right":
                        x = rng.uniform(max(inner_x2 - edge_band, inner_x1), inner_x2)
                        y = rng.uniform(inner_y1, inner_y2)
                    elif edge == "top":
                        x = rng.uniform(inner_x1, inner_x2)
                        y = rng.uniform(inner_y1, min(inner_y1 + edge_band, inner_y2))
                    else:
                        x = rng.uniform(inner_x1, inner_x2)
                        y = rng.uniform(max(inner_y2 - edge_band, inner_y1), inner_y2)
                else:
                    x = rng.uniform(inner_x1, inner_x2)
                    y = rng.uniform(inner_y1, inner_y2)

                if any(math.dist((x, y), p) < min_dist for p in points):
                    continue
                if any(math.dist((x, y), p) < min_dist for p in existing):
                    continue

                points.append((x, y))
                placed = True
                break
            if not placed:
                raise GenerationError("Failed to place points in box")
        return points

    def _pair_fits(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        existing: list[tuple[float, float]],
        min_dist: float,
    ) -> bool:
        if math.dist(a, b) < min_dist:
            return False
        for p in existing:
            if math.dist(a, p) < min_dist or math.dist(b, p) < min_dist:
                return False
        return True

    def _matching_item_indices(
        self,
        items: list[VisualItemSpec],
        target_definition: dict[str, str],
    ) -> list[int]:
        return [index + 1 for index, item in enumerate(items) if self._matches_target(item, target_definition)]

    def _matches_target(self, item: VisualItemSpec, target_definition: dict[str, str]) -> bool:
        if item.is_anchor:
            return False
        query_color = target_definition["query_color"]
        query_shape = target_definition["query_shape"]
        rule_type = target_definition["rule_type"]

        feature_match = item.color == query_color and item.shape == query_shape
        if rule_type == "same_group_as_anchor":
            return feature_match and item.role == "target"
        if rule_type == "same_similarity_group_as_anchor":
            return feature_match and item.role == "target"
        if rule_type in {"inside_region", "outside_region", "left_of_line", "right_of_line", "above_line", "below_line", "inside_region_in_panel", "inside_selected_region"}:
            return feature_match and item.role == "target"
        if rule_type == "different_group_from_anchor":
            return feature_match and item.role == "target"
        return feature_match and item.is_target

    def _validate_scene_geometry(
        self,
        items: list[VisualItemSpec],
        structures: list[VisualStructureSpec],
        factors: StructureSensitiveVisualFactors,
    ) -> bool:
        radius = SIZE_TO_RADIUS[factors.item_size]
        min_dist = 2 * radius + 6

        for item in items:
            if not (factors.margin + radius <= item.x <= factors.width - factors.margin - radius):
                return False
            if not (factors.margin + radius <= item.y <= factors.height - factors.margin - radius):
                return False

        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if math.dist((a.x, a.y), (b.x, b.y)) < min_dist:
                    return False

        box_map = {
            s.id: (
                min(float(s.payload["x1"]), float(s.payload["x2"])),
                min(float(s.payload["y1"]), float(s.payload["y2"])),
                max(float(s.payload["x1"]), float(s.payload["x2"])),
                max(float(s.payload["y1"]), float(s.payload["y2"])),
            )
            for s in structures
            if s.type == "box"
        }

        for item in items:
            if item.region_id and item.region_id in box_map:
                x1, y1, x2, y2 = box_map[item.region_id]
                if not (x1 + radius + 2 <= item.x <= x2 - radius - 2):
                    return False
                if not (y1 + radius + 2 <= item.y <= y2 - radius - 2):
                    return False

        if factors.grouping_principle == "proximity":
            centers: dict[str, tuple[float, float]] = {}
            for gid in {it.group_id for it in items}:
                grp = [(it.x, it.y) for it in items if it.group_id == gid]
                centers[gid] = (sum(x for x, _ in grp) / len(grp), sum(y for _, y in grp) / len(grp))
            for item in items:
                own = math.dist((item.x, item.y), centers[item.group_id])
                others = [math.dist((item.x, item.y), c) for gid, c in centers.items() if gid != item.group_id]
                if others and own >= min(others):
                    return False

        if factors.grouping_principle == "common_region":
            for s in structures:
                if s.type != "box":
                    continue
                x1 = min(float(s.payload["x1"]), float(s.payload["x2"]))
                y1 = min(float(s.payload["y1"]), float(s.payload["y2"]))
                x2 = max(float(s.payload["x1"]), float(s.payload["x2"]))
                y2 = max(float(s.payload["y1"]), float(s.payload["y2"]))
                contained = [it for it in items if it.region_id == s.id]
                if contained and not self._box_can_fit_count((x1, y1, x2, y2), len(contained), radius, 6.0):
                    return False

        return True

    def _passes_anti_shortcut_constraints(
        self,
        items: list[VisualItemSpec],
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
        gold_indices: list[int],
    ) -> bool:
        if len(gold_indices) != factors.target_count:
            return False

        query_color = target_definition["query_color"]
        query_shape = target_definition["query_shape"]
        global_feature_count = sum(1 for item in items if (not item.is_anchor) and item.color == query_color and item.shape == query_shape)

        if (
            factors.target_count > 0
            and global_feature_count == factors.target_count
            and target_definition["rule_type"] != "same_similarity_group_as_anchor"
        ):
            return False

        if factors.target_count == 0:
            partial_matches = sum(
                1 for item in items
                if not item.is_anchor and (
                    item.color == query_color or item.shape == query_shape or item.role in {"wrong_scope", "cue_conflict_trap", "cross_binding_feature", "group_only"}
                )
            )
            return partial_matches >= 2

        if factors.confound_type == "wrong_scope" and not any(item.role == "wrong_scope" for item in items):
            return False
        if factors.confound_type == "cross_binding":
            if not any(item.role in {"cross_binding_feature", "feature_only"} for item in items):
                return False
            if not any(item.role in {"group_only", "wrong_global_choice"} for item in items):
                return False
        if factors.confound_type == "cue_conflict_trap" and not any(item.role == "cue_conflict_trap" for item in items):
            return False

        if target_definition["rule_type"] == "same_similarity_group_as_anchor":
            same_shape_wrong_color = sum(
                1
                for item in items
                if not item.is_anchor
                and item.shape == target_definition["reference_similarity_key"]
                and item.color != query_color
            )
            same_color_wrong_shape = sum(
                1
                for item in items
                if not item.is_anchor
                and item.color == query_color
                and item.shape != query_shape
            )
            return same_shape_wrong_color >= 1 and same_color_wrong_shape >= 1

        matching_groups = {
            item.group_id
            for item in items
            if not item.is_anchor and item.color == query_color and item.shape == query_shape
        }
        if len(matching_groups) < 2:
            return False
        return True

    def _sample_non_target_feature(
        self,
        rng: random.Random,
        target_color: str,
        target_shape: str,
        confound_type: str,
    ) -> tuple[ColorName, ShapeName]:
        if confound_type == "group_only":
            return self._different_color(target_color), target_shape  # type: ignore[return-value]
        if confound_type == "feature_only":
            return target_color, self._different_shape(target_shape)  # type: ignore[return-value]
        return self._different_color(target_color), self._different_shape(target_shape)  # type: ignore[return-value]

    def _different_color(self, color: str) -> ColorName:
        return next(candidate for candidate in COLORS if candidate != color)

    def _different_shape(self, shape: str) -> ShapeName:
        return next(candidate for candidate in SHAPES if candidate != shape)

    def _third_shape(self, first: str, second: str) -> ShapeName:
        return next(candidate for candidate in SHAPES if candidate not in {first, second})

    def _build_count_instruction(
        self,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> str:
        description = self._rule_description(factors, target_definition)
        return (
            f"Count the items matching this structural rule: {description}.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n"
            '- "count" must be an integer\n'
            "- Apply the full structural rule exactly\n"
            "- Return only the JSON object"
        )

    def _build_filter_instruction(
        self,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> str:
        description = self._rule_description(factors, target_definition)
        return (
            f"Return the 1-based item indices matching this structural rule: {description}.\n"
            'Respond with a JSON object of the form {"indices": [<sorted unique integers>]}.\n'
            "Rules:\n"
            "- Use 1-based indexing\n"
            "- Sort ascending\n"
            "- Do not include duplicates\n"
            "- Apply the full structural rule exactly\n"
            "- Return only the JSON object"
        )

    def _rule_description(
        self,
        factors: StructureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> str:
        feature_desc = f"{target_definition['query_color']} {target_definition['query_shape']}s"
        if target_definition["rule_type"] == "same_group_as_anchor":
            return f"{feature_desc} in the same group as the marked item"
        if target_definition["rule_type"] == "same_similarity_group_as_anchor":
            return f"{feature_desc} in the same shape-based group as the marked item"
        if target_definition["rule_type"] == "inside_region":
            return f"{feature_desc} inside the box"
        if target_definition["rule_type"] == "outside_region":
            return f"{feature_desc} outside the box"
        if target_definition["rule_type"] == "left_of_line":
            return f"{feature_desc} left of the divider line"
        if target_definition["rule_type"] == "right_of_line":
            return f"{feature_desc} right of the divider line"
        if target_definition["rule_type"] == "above_line":
            return f"{feature_desc} above the divider line"
        if target_definition["rule_type"] == "below_line":
            return f"{feature_desc} below the divider line"
        if target_definition["rule_type"] == "inside_region_in_panel":
            return f"{feature_desc} inside the box in the left panel only"
        if target_definition["rule_type"] == "inside_selected_region":
            selected = target_definition.get("selected_region_id", "selected region")
            if selected == "box_left":
                return f"{feature_desc} inside the leftmost box"
            return f"{feature_desc} inside the selected box"
        if target_definition["rule_type"] == "different_group_from_anchor":
            return f"{feature_desc} in a different group from the marked item"
        if target_definition["rule_type"] == "inside_region_in_panel":
            selector = target_definition.get("scope_selector", "selected_scope")
            if selector == "left_panel":
                return f"{feature_desc} inside the left panel only"
            if selector == "top_panel":
                return f"{feature_desc} inside the top panel only"
            if selector == "leftmost_box":
                return f"{feature_desc} inside the leftmost box only"
            return f"{feature_desc} inside the selected scope only"
        return feature_desc


def scene_to_scene_row(scene: StructureSensitiveVisualScene) -> dict[str, object]:
    factors = scene.factors
    return {
        "scene_id": scene.scene_id,
        "seed": scene.seed,
        "family": scene.family,
        "attentional_basis": scene.attentional_basis,
        "modality": scene.modality,
        "dimension": scene.dimension,
        "variant": scene.variant,
        "grouping_principle": scene.grouping_principle,
        "structure_type": scene.structure_type,
        "relation_operator": scene.relation_operator,
        "cue_conflict": scene.cue_conflict,
        "count_instruction": scene.count_instruction,
        "filter_instruction": scene.filter_instruction,
        "count_prompt": scene.count_prompt,
        "filter_prompt": scene.filter_prompt,
        "gold_count": scene.gold_count,
        "gold_indices": json.dumps(scene.gold_indices),
        "target_definition": json.dumps(scene.target_definition, sort_keys=True),
        "items_json": json.dumps([asdict(item) for item in scene.items], sort_keys=True),
        "structures_json": json.dumps([asdict(structure) for structure in scene.structures], sort_keys=True),
        "num_items": len(scene.items),
        "num_groups": factors.num_groups,
        "target_count": factors.target_count,
        "binding_distance": factors.binding_distance,
        "structure_depth": factors.structure_depth,
        "confound_level": factors.confound_level,
        "confound_type": factors.confound_type,
        "render_style": factors.render_style,
        "width": factors.width,
        "height": factors.height,
        "margin": factors.margin,
    }
