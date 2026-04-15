from __future__ import annotations

import json
import math
import random
from itertools import combinations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

PrincipleName = Literal[
    "baseline",
    "proximity",
    "similarity",
    "continuity",
    "common_region",
]

ConditionTypeName = Literal[
    "simple",
    "random_clusters",
    "gap_grid",
    "color_only",
    "color_then_shape",
    "single_crossing",
    "uneven_boxes",
    "outside_items",
]

LayoutPatternName = Literal[
    "cluster",
    "free_clusters",
    "grid_with_gaps",
    "vertical_halves",
    "horizontal_halves",
    "vertical_stripes",
    "horizontal_stripes",
    "straight_vs_arc",
    "2_regions",
    "2_regions_plus_loose",
]

DifficultyName = Literal["easy", "medium", "hard"]
DimensionName = Literal["baseline", "principle", "combined"]
ShapeName = Literal["circle", "square", "triangle"]
ColorName = Literal["red", "blue", "green", "yellow"]
SizeName = Literal["small", "large"]
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
SIZE_TO_RADIUS: dict[SizeName, int] = {"small": 12, "large": 20}
PATH_STROKE_WIDTH = 4


class GenerationError(RuntimeError):
    """Raised when a valid scene cannot be generated within the retry budget."""


@dataclass(frozen=True)
class VisualItemSpec:
    item_id: str
    x: float
    y: float
    color: ColorName
    shape: ShapeName
    size: SizeName
    group_id: str
    region_id: str | None
    path_id: str | None
    is_loose_item: bool
    is_anchor: bool


@dataclass(frozen=True)
class VisualRegionSpec:
    region_id: str
    group_id: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class VisualPathSpec:
    path_id: str
    group_id: str
    path_type: str
    control_points: list[tuple[float, float]]


@dataclass(frozen=True)
class StructureSensitiveVisualFactors:
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    principle: PrincipleName
    condition_type: ConditionTypeName
    layout_pattern: LayoutPatternName
    difficulty: DifficultyName
    grouping_feature_primary: str
    grouping_feature_secondary: str | None
    queried_feature: str
    width: int
    height: int
    margin: int
    num_groups: int
    min_items_per_group: int
    max_items_per_group: int
    target_in_anchor_group: int
    target_outside_anchor_group: int
    non_target_in_anchor_group: int
    cluster_spread: float | None
    inter_group_margin: float | None
    stripe_count: int | None
    shape_bias_strength: float | None
    path_crossing_count: int | None
    region_count: int | None
    loose_item_count: int | None
    min_gap: int
    jitter: int


@dataclass(frozen=True)
class StructureSensitiveVisualScene:
    scene_id: str
    seed: int
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    principle: PrincipleName
    condition_type: ConditionTypeName
    layout_pattern: LayoutPatternName
    difficulty: DifficultyName
    generator_case: str
    count_instruction: str
    filter_instruction: str
    count_prompt: str
    filter_prompt: str
    anchor_item_id: str
    anchor_group_id: str
    grouping_feature_primary: str
    grouping_feature_secondary: str | None
    queried_feature: str
    target_value: str
    gold_count: int
    gold_indices: list[int]
    factors: StructureSensitiveVisualFactors
    items: list[VisualItemSpec]
    regions: list[VisualRegionSpec]
    paths: list[VisualPathSpec]
    metadata: dict[str, object]


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
                items, regions, paths, metadata = self._build_scene(local_rng, factors)
            except GenerationError:
                continue
            anchor_item_id, anchor_group_id = self._choose_anchor(local_rng, items, factors, paths)
            items = [
                replace(item, is_anchor=item.item_id == anchor_item_id)
                for item in items
            ]
            items, feature_metadata = self._assign_visual_features(
                local_rng,
                factors,
                items,
                anchor_group_id=anchor_group_id,
            )
            metadata = {**metadata, **feature_metadata}
            target_value = self._choose_target_value(local_rng, factors, items, anchor_group_id)
            if not self._passes_constraints(
                factors=factors,
                items=items,
                regions=regions,
                paths=paths,
                anchor_group_id=anchor_group_id,
                target_value=target_value,
            ):
                continue

            count_instruction = self._build_count_instruction(target_value)
            filter_instruction = self._build_filter_instruction(target_value)
            gold_indices = self._matching_item_indices(items, anchor_group_id, factors.queried_feature, target_value)
            anchor_item = next(item for item in items if item.item_id == anchor_item_id)
            if factors.principle == "continuity" and paths:
                crossing_point = self._continuity_crossing_point(paths)
                metadata = {
                    **metadata,
                    "anchor_crossing_distance": self._distance(
                        (anchor_item.x, anchor_item.y),
                        crossing_point,
                    ),
                }

            return StructureSensitiveVisualScene(
                scene_id=make_scene_id(factors.dimension, factors.variant, seed),
                seed=seed,
                family=factors.family,
                attentional_basis=factors.attentional_basis,
                modality=factors.modality,
                dimension=factors.dimension,
                variant=factors.variant,
                principle=factors.principle,
                condition_type=factors.condition_type,
                layout_pattern=factors.layout_pattern,
                difficulty=factors.difficulty,
                generator_case=self._generator_case_for_scene(factors, metadata),
                count_instruction=count_instruction,
                filter_instruction=filter_instruction,
                count_prompt=count_instruction,
                filter_prompt=filter_instruction,
                anchor_item_id=anchor_item_id,
                anchor_group_id=anchor_group_id,
                grouping_feature_primary=factors.grouping_feature_primary,
                grouping_feature_secondary=factors.grouping_feature_secondary,
                queried_feature=factors.queried_feature,
                target_value=target_value,
                gold_count=len(gold_indices),
                gold_indices=gold_indices,
                factors=factors,
                items=items,
                regions=regions,
                paths=paths,
                metadata={
                    **metadata,
                    "anchor_index": self._item_index(anchor_item.item_id),
                    "anchor_position": {"x": anchor_item.x, "y": anchor_item.y},
                },
            )

        raise GenerationError(
            "Failed to generate a valid structure-sensitive visual scene within max_attempts"
        )

    def sample_factors(
        self,
        *,
        rng: random.Random,
        dimension: DimensionName,
        variant: str,
    ) -> StructureSensitiveVisualFactors:
        if dimension == "baseline":
            if variant != "simple":
                raise ValueError(f"Unknown baseline variant: {variant}")
            return self._sample_baseline_factors(rng)
        if dimension == "principle":
            return self._sample_principle_factors(rng, principle_variant=variant)
        if dimension == "combined":
            return self._sample_combined_factors(rng, variant=variant)
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

        for region in scene.regions:
            self._draw_region(draw, region)
        for path in scene.paths:
            self._draw_path(draw, path)
        for index, item in enumerate(scene.items, start=1):
            self._draw_item(draw, item)
            if item.is_anchor:
                self._draw_anchor_marker(draw, item)
            if show_item_ids:
                self._draw_item_id(draw, item, index, factors, scene.items)

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        return image

    def _base_factors(
        self,
        *,
        dimension: DimensionName,
        variant: str,
        principle: PrincipleName,
        condition_type: ConditionTypeName,
        layout_pattern: LayoutPatternName,
        difficulty: DifficultyName,
        grouping_feature_primary: str,
        grouping_feature_secondary: str | None,
        queried_feature: str,
        num_groups: int,
        min_items_per_group: int,
        max_items_per_group: int,
        target_in_anchor_group: int,
        target_outside_anchor_group: int,
        non_target_in_anchor_group: int,
        cluster_spread: float | None,
        inter_group_margin: float | None,
        stripe_count: int | None,
        shape_bias_strength: float | None,
        path_crossing_count: int | None,
        region_count: int | None,
        loose_item_count: int | None,
        min_gap: int,
        jitter: int,
    ) -> StructureSensitiveVisualFactors:
        return StructureSensitiveVisualFactors(
            family="selective_attention",
            attentional_basis="structure_sensitive",
            modality="visual",
            dimension=dimension,
            variant=variant,
            principle=principle,
            condition_type=condition_type,
            layout_pattern=layout_pattern,
            difficulty=difficulty,
            grouping_feature_primary=grouping_feature_primary,
            grouping_feature_secondary=grouping_feature_secondary,
            queried_feature=queried_feature,
            width=640,
            height=480,
            margin=36,
            num_groups=num_groups,
            min_items_per_group=min_items_per_group,
            max_items_per_group=max_items_per_group,
            target_in_anchor_group=target_in_anchor_group,
            target_outside_anchor_group=target_outside_anchor_group,
            non_target_in_anchor_group=non_target_in_anchor_group,
            cluster_spread=cluster_spread,
            inter_group_margin=inter_group_margin,
            stripe_count=stripe_count,
            shape_bias_strength=shape_bias_strength,
            path_crossing_count=path_crossing_count,
            region_count=region_count,
            loose_item_count=loose_item_count,
            min_gap=min_gap,
            jitter=jitter,
        )

    def _sample_baseline_factors(self, rng: random.Random) -> StructureSensitiveVisualFactors:
        return self._base_factors(
            dimension="baseline",
            variant="simple",
            principle="baseline",
            condition_type="uneven_boxes",
            layout_pattern="2_regions",
            difficulty="easy",
            grouping_feature_primary="common_region",
            grouping_feature_secondary=None,
            queried_feature="shape",
            num_groups=2,
            min_items_per_group=3,
            max_items_per_group=3,
            target_in_anchor_group=1,
            target_outside_anchor_group=1,
            non_target_in_anchor_group=2,
            cluster_spread=None,
            inter_group_margin=None,
            stripe_count=0,
            shape_bias_strength=0.0,
            path_crossing_count=0,
            region_count=2,
            loose_item_count=0,
            min_gap=34,
            jitter=0,
        )

    def _sample_principle_factors(
        self,
        rng: random.Random,
        *,
        principle_variant: str,
        difficulty: DifficultyName = "medium",
        dimension: DimensionName = "principle",
        variant: str | None = None,
    ) -> StructureSensitiveVisualFactors:
        variant_name = variant or principle_variant
        effective_difficulty = difficulty
        if dimension == "principle":
            effective_difficulty = {
                "proximity": "medium",
                "similarity": "hard",
                "continuity": "hard",
                "common_region": "easy",
            }[principle_variant]
        if principle_variant == "proximity":
            condition_type, layout_pattern = rng.choice(
                [
                    ("random_clusters", "free_clusters"),
                    ("gap_grid", "grid_with_gaps"),
                ]
            )
            return self._factors_for_difficulty(
                dimension=dimension,
                variant=variant_name,
                principle="proximity",
                condition_type=condition_type,
                layout_pattern=layout_pattern,
                difficulty=effective_difficulty,
                grouping_feature_primary="proximity",
                grouping_feature_secondary=None,
                queried_feature="shape",
                num_groups=3 if effective_difficulty != "hard" else 4,
                cluster_spread=30.0 if condition_type == "random_clusters" else None,
                inter_group_margin=178.0 if condition_type == "random_clusters" else 84.0,
                stripe_count=0,
                shape_bias_strength=None,
                path_crossing_count=0,
                region_count=0,
                loose_item_count=0,
            )
        if principle_variant == "similarity":
            condition_type, layout_pattern = rng.choice(
                [
                    ("color_only", "vertical_halves"),
                    ("color_only", "horizontal_halves"),
                    ("color_only", "vertical_stripes"),
                    ("color_only", "horizontal_stripes"),
                ]
            )
            stripe_count = 4 if "stripes" in layout_pattern else 0
            return self._factors_for_difficulty(
                dimension=dimension,
                variant=variant_name,
                principle="similarity",
                condition_type=condition_type,
                layout_pattern=layout_pattern,
                difficulty=effective_difficulty,
                grouping_feature_primary="color",
                grouping_feature_secondary=None,
                queried_feature="shape",
                num_groups=2,
                cluster_spread=None,
                inter_group_margin=None,
                stripe_count=stripe_count,
                shape_bias_strength=0.0,
                path_crossing_count=0,
                region_count=0,
                loose_item_count=0,
            )
        if principle_variant == "continuity":
            return self._factors_for_difficulty(
                dimension=dimension,
                variant=variant_name,
                principle="continuity",
                condition_type="single_crossing",
                layout_pattern="straight_vs_arc",
                difficulty=effective_difficulty,
                grouping_feature_primary="path_continuity",
                grouping_feature_secondary=None,
                queried_feature="shape",
                num_groups=2,
                cluster_spread=None,
                inter_group_margin=None,
                stripe_count=0,
                shape_bias_strength=0.0,
                path_crossing_count=1,
                region_count=0,
                loose_item_count=0,
            )
        if principle_variant == "common_region":
            condition_type, layout_pattern = rng.choice(
                [
                    ("uneven_boxes", "2_regions"),
                    ("outside_items", "2_regions_plus_loose"),
                ]
            )
            return self._factors_for_difficulty(
                dimension=dimension,
                variant=variant_name,
                principle="common_region",
                condition_type=condition_type,
                layout_pattern=layout_pattern,
                difficulty=effective_difficulty,
                grouping_feature_primary="common_region",
                grouping_feature_secondary=None,
                queried_feature="shape",
                num_groups=2,
                cluster_spread=None,
                inter_group_margin=None,
                stripe_count=0,
                shape_bias_strength=0.0,
                path_crossing_count=0,
                region_count=2,
                loose_item_count=2 if condition_type == "outside_items" and effective_difficulty == "hard" else (1 if condition_type == "outside_items" else 0),
            )
        raise ValueError(f"Unknown principle variant: {principle_variant}")

    def _sample_combined_factors(
        self,
        rng: random.Random,
        *,
        variant: str,
    ) -> StructureSensitiveVisualFactors:
        if variant not in {"easy", "medium", "hard"}:
            raise ValueError(f"Unknown combined variant: {variant}")
        principle_variant = rng.choice(["proximity", "similarity", "continuity", "common_region"])
        return self._sample_principle_factors(
            rng,
            principle_variant=principle_variant,
            difficulty=variant,  # type: ignore[arg-type]
            dimension="combined",
            variant=variant,
        )

    def _factors_for_difficulty(
        self,
        *,
        dimension: DimensionName,
        variant: str,
        principle: PrincipleName,
        condition_type: ConditionTypeName,
        layout_pattern: LayoutPatternName,
        difficulty: DifficultyName,
        grouping_feature_primary: str,
        grouping_feature_secondary: str | None,
        queried_feature: str,
        num_groups: int,
        cluster_spread: float | None,
        inter_group_margin: float | None,
        stripe_count: int | None,
        shape_bias_strength: float | None,
        path_crossing_count: int | None,
        region_count: int | None,
        loose_item_count: int | None,
    ) -> StructureSensitiveVisualFactors:
        difficulty_map = {
            "easy": dict(
                min_items_per_group=4,
                max_items_per_group=5,
                target_in_anchor_group=1,
                target_outside_anchor_group=1,
                non_target_in_anchor_group=3,
                min_gap=30,
                jitter=6,
            ),
            "medium": dict(
                min_items_per_group=4,
                max_items_per_group=6,
                target_in_anchor_group=2,
                target_outside_anchor_group=2,
                non_target_in_anchor_group=3,
                min_gap=24,
                jitter=8,
            ),
            "hard": dict(
                min_items_per_group=5,
                max_items_per_group=7,
                target_in_anchor_group=3,
                target_outside_anchor_group=3,
                non_target_in_anchor_group=3,
                min_gap=20,
                jitter=10,
            ),
        }
        cfg = difficulty_map[difficulty]
        return self._base_factors(
            dimension=dimension,
            variant=variant,
            principle=principle,
            condition_type=condition_type,
            layout_pattern=layout_pattern,
            difficulty=difficulty,
            grouping_feature_primary=grouping_feature_primary,
            grouping_feature_secondary=grouping_feature_secondary,
            queried_feature=queried_feature,
            num_groups=num_groups,
            min_items_per_group=cfg["min_items_per_group"],
            max_items_per_group=cfg["max_items_per_group"],
            target_in_anchor_group=cfg["target_in_anchor_group"],
            target_outside_anchor_group=cfg["target_outside_anchor_group"],
            non_target_in_anchor_group=cfg["non_target_in_anchor_group"],
            cluster_spread=cluster_spread,
            inter_group_margin=inter_group_margin,
            stripe_count=stripe_count or 0,
            shape_bias_strength=shape_bias_strength,
            path_crossing_count=path_crossing_count or 0,
            region_count=region_count or 0,
            loose_item_count=loose_item_count or 0,
            min_gap=cfg["min_gap"],
            jitter=cfg["jitter"],
        )

    def _build_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> tuple[list[VisualItemSpec], list[VisualRegionSpec], list[VisualPathSpec], dict[str, object]]:
        if factors.principle == "baseline":
            return self._build_baseline_scene(rng, factors)
        if factors.principle == "proximity":
            return self._build_proximity_scene(rng, factors)
        if factors.principle == "similarity":
            return self._build_similarity_scene(rng, factors)
        if factors.principle == "continuity":
            return self._build_continuity_scene(rng, factors)
        if factors.principle == "common_region":
            return self._build_common_region_scene(rng, factors)
        raise ValueError(f"Unsupported principle: {factors.principle}")

    def _build_baseline_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> tuple[list[VisualItemSpec], list[VisualRegionSpec], list[VisualPathSpec], dict[str, object]]:
        items, regions, paths, metadata = self._build_common_region_scene(rng, factors)
        return items, regions, paths, {
            **metadata,
            "baseline_backing_principle": "common_region",
        }

    def _build_proximity_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> tuple[list[VisualItemSpec], list[VisualRegionSpec], list[VisualPathSpec], dict[str, object]]:
        group_sizes = self._sample_group_sizes(rng, factors)
        if factors.condition_type == "random_clusters":
            centers = self._sample_cluster_centers(rng, factors, group_sizes)
            items = self._items_from_clusters(rng, factors, group_sizes, centers)
            metadata = {
                "group_sizes": group_sizes,
                "cluster_centers": [{"x": x, "y": y} for x, y in centers],
            }
            return items, [], [], metadata
        if factors.condition_type == "gap_grid":
            items, gap_grid_metadata = self._items_from_gap_grid(rng, factors, group_sizes)
            metadata = {
                "group_sizes": gap_grid_metadata["group_sizes"],
                **gap_grid_metadata,
            }
            return items, [], [], metadata
        raise ValueError(f"Unsupported proximity condition_type: {factors.condition_type}")

    def _build_similarity_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> tuple[list[VisualItemSpec], list[VisualRegionSpec], list[VisualPathSpec], dict[str, object]]:
        group_sizes = self._sample_group_sizes(rng, factors)
        partitions = self._similarity_partitions(factors)
        items = self._items_from_partitions(rng, factors, group_sizes, partitions)
        band_to_group_map = self._similarity_band_to_group_map(factors)
        metadata = {
            "group_sizes": group_sizes,
            "partitions": partitions,
            "band_count": len(partitions),
            "band_to_group_map": band_to_group_map,
        }
        return items, [], [], metadata

    def _build_continuity_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> tuple[list[VisualItemSpec], list[VisualRegionSpec], list[VisualPathSpec], dict[str, object]]:
        group_sizes = self._sample_group_sizes(rng, factors)
        if len(group_sizes) != 2:
            group_sizes = [group_sizes[0], group_sizes[1] if len(group_sizes) > 1 else group_sizes[0]]
        path_specs, path_metadata = self._continuity_paths(rng, factors)
        intersections = self._continuity_intersections(path_specs)
        if not self._validate_single_crossing(intersections):
            raise GenerationError("Continuity paths must intersect exactly once")
        if not self._passes_continuity_geometry_constraints(factors, path_specs, intersections):
            raise GenerationError("Continuity geometry does not satisfy variability constraints")
        items = self._items_on_paths(rng, factors, group_sizes, path_specs)
        metadata = {
            "group_sizes": group_sizes,
            **path_metadata,
            "crossing_count": len(intersections),
            "crossing_points": [{"x": x, "y": y} for x, y in intersections],
            "crossing_clearance": self._continuity_crossing_clearance(),
        }
        return items, [], path_specs, metadata

    def _build_common_region_scene(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> tuple[list[VisualItemSpec], list[VisualRegionSpec], list[VisualPathSpec], dict[str, object]]:
        group_sizes = self._sample_group_sizes(rng, factors)
        regions = self._region_specs(factors)
        items = self._items_in_regions(rng, factors, group_sizes, regions)
        metadata = {
            "group_sizes": group_sizes,
            "loose_item_count": factors.loose_item_count or 0,
        }
        return items, regions, [], metadata

    def _sample_group_sizes(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> list[int]:
        if factors.principle == "baseline":
            return [3] * factors.num_groups

        if factors.principle == "similarity":
            anchor_floor = factors.target_in_anchor_group + factors.non_target_in_anchor_group
            even_size = anchor_floor if anchor_floor % 2 == 0 else anchor_floor + 1
            difficulty_floor = {
                "easy": 4,
                "medium": 6,
                "hard": 6,
            }[factors.difficulty]
            return [max(even_size, difficulty_floor)] * factors.num_groups

        if factors.principle == "proximity" and factors.condition_type == "gap_grid":
            anchor_floor = factors.target_in_anchor_group + factors.non_target_in_anchor_group
            size_bounds = {
                "easy": (4, 5),
                "medium": (4, 6),
                "hard": (5, 6),
            }
            low, high = size_bounds[factors.difficulty]
            sizes = [rng.randint(low, high) for _ in range(factors.num_groups)]
            sizes[0] = max(sizes[0], anchor_floor)
            return sizes

        sizes = [
            rng.randint(factors.min_items_per_group, factors.max_items_per_group)
            for _ in range(factors.num_groups)
        ]
        anchor_floor = factors.target_in_anchor_group + factors.non_target_in_anchor_group
        sizes[0] = max(sizes[0], anchor_floor)
        remaining_outside = sum(sizes[1:])
        min_outside = factors.target_outside_anchor_group
        if remaining_outside < min_outside and len(sizes) > 1:
            sizes[1] += min_outside - remaining_outside
        return sizes

    def _sample_cluster_centers(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
    ) -> list[tuple[float, float]]:
        if factors.principle == "baseline":
            return self._baseline_cluster_centers(factors)
        if factors.principle == "proximity" and factors.condition_type == "random_clusters":
            return self._proximity_cluster_centers(factors, group_sizes)

        centers: list[tuple[float, float]] = []
        radius = max(SIZE_TO_RADIUS.values())
        spread = factors.cluster_spread or 36.0
        left = factors.margin + radius + spread
        right = factors.width - factors.margin - radius - spread
        top = factors.margin + radius + spread
        bottom = factors.height - factors.margin - radius - spread
        required_distance = (factors.inter_group_margin or 100.0) + spread
        for _ in group_sizes:
            placed = False
            for _ in range(300):
                x = rng.uniform(left, right)
                y = rng.uniform(top, bottom)
                if any(self._distance((x, y), existing) < required_distance for existing in centers):
                    continue
                centers.append((x, y))
                placed = True
                break
            if not placed:
                raise GenerationError("Failed to place cluster centers")
        return centers

    def _baseline_cluster_centers(
        self,
        factors: StructureSensitiveVisualFactors,
    ) -> list[tuple[float, float]]:
        width = factors.width - 2 * factors.margin
        height = factors.height - 2 * factors.margin
        return [
            (factors.margin + width * 0.18, factors.margin + height * 0.26),
            (factors.margin + width * 0.82, factors.margin + height * 0.28),
            (factors.margin + width * 0.50, factors.margin + height * 0.78),
        ][: factors.num_groups]

    def _proximity_cluster_centers(
        self,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
    ) -> list[tuple[float, float]]:
        width = factors.width - 2 * factors.margin
        height = factors.height - 2 * factors.margin
        templates = {
            3: [
                (factors.margin + width * 0.16, factors.margin + height * 0.27),
                (factors.margin + width * 0.80, factors.margin + height * 0.25),
                (factors.margin + width * 0.50, factors.margin + height * 0.77),
            ],
            4: [
                (factors.margin + width * 0.18, factors.margin + height * 0.24),
                (factors.margin + width * 0.82, factors.margin + height * 0.24),
                (factors.margin + width * 0.18, factors.margin + height * 0.78),
                (factors.margin + width * 0.82, factors.margin + height * 0.78),
            ],
        }
        if factors.num_groups not in templates:
            raise GenerationError("Unsupported proximity cluster template")
        return templates[factors.num_groups][: len(group_sizes)]

    def _items_from_clusters(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
        centers: list[tuple[float, float]],
    ) -> list[VisualItemSpec]:
        items: list[VisualItemSpec] = []
        base_spread = factors.cluster_spread or 36.0
        max_radius = float(max(SIZE_TO_RADIUS.values()))
        required_clearance = 2 * max_radius + 6.0
        intra_group_min_distance = max(required_clearance, float(factors.min_gap) - 6.0)
        inter_group_min_distance = max(required_clearance, float(factors.min_gap))
        for group_index, (size, center) in enumerate(zip(group_sizes, centers), start=1):
            spread = max(base_spread, math.sqrt(size) * intra_group_min_distance * 0.72)
            group_positions: list[tuple[float, float]] = []
            for _ in range(size):
                placed = False
                for _ in range(600):
                    angle = rng.uniform(0.0, 2.0 * math.pi)
                    radius = rng.uniform(0.0, spread)
                    x = center[0] + math.cos(angle) * radius
                    y = center[1] + math.sin(angle) * radius
                    if not self._point_in_canvas(x, y, factors):
                        continue
                    if any(self._distance((x, y), pos) < intra_group_min_distance for pos in group_positions):
                        continue
                    if any(
                        item.group_id != f"G{group_index}"
                        and self._distance((x, y), (item.x, item.y)) < inter_group_min_distance
                        for item in items
                    ):
                        continue
                    group_positions.append((x, y))
                    items.append(self._make_placeholder_item(len(items) + 1, x, y, group_id=f"G{group_index}"))
                    placed = True
                    break
                if not placed:
                    # TODO: harden dense cluster packing with adaptive radius expansion if future slices increase crowding.
                    raise GenerationError("Failed to place clustered items")
        return items

    def _items_from_gap_grid(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
    ) -> tuple[list[VisualItemSpec], dict[str, object]]:
        max_radius = max(SIZE_TO_RADIUS.values())
        required_clearance = 2 * max_radius + 6
        usable_left = factors.margin + 28
        usable_right = factors.width - factors.margin - 28
        usable_top = factors.margin + 28
        usable_bottom = factors.height - factors.margin - 28
        layout = self._sample_gap_grid_layout(
            rng,
            factors,
            group_sizes,
            required_clearance=required_clearance,
            usable_left=usable_left,
            usable_right=usable_right,
            usable_top=usable_top,
            usable_bottom=usable_bottom,
        )

        items: list[VisualItemSpec] = []
        row_centers = layout["row_centers"]
        col_centers = layout["col_centers"]
        actual_group_sizes: list[int] = []
        for partition in layout["partitions"]:
            group_id = str(partition["group_id"])
            candidate_slots = [
                (row, col)
                for row in range(int(partition["row_start"]), int(partition["row_end"]))
                for col in range(int(partition["col_start"]), int(partition["col_end"]))
            ]
            if not candidate_slots:
                raise GenerationError("Gap grid partition must contain at least one slot")
            actual_group_sizes.append(len(candidate_slots))
            for row, col in candidate_slots:
                items.append(
                    self._make_placeholder_item(
                        len(items) + 1,
                        col_centers[col],
                        row_centers[row],
                        group_id=group_id,
                    )
                )
        metadata = {
            "group_sizes": actual_group_sizes,
            "grid_split_pattern": layout["template_name"],
            "grid_row_count": layout["row_count"],
            "grid_col_count": layout["col_count"],
            "gap_after_rows": layout["gap_after_rows"],
            "gap_after_cols": layout["gap_after_cols"],
            "group_partitions": layout["partitions"],
        }
        return items, metadata

    def _sample_gap_grid_layout(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
        *,
        required_clearance: float,
        usable_left: float,
        usable_right: float,
        usable_top: float,
        usable_bottom: float,
    ) -> dict[str, object]:
        if factors.num_groups == 3:
            template_names = ["vertical_bands", "horizontal_bands", "l_split"]
        elif factors.num_groups == 4:
            template_names = ["quadrants"]
        else:
            raise GenerationError("Unsupported gap grid group count")
        rng.shuffle(template_names)
        for template_name in template_names:
            attempts = 18 if template_name != "quadrants" else 24
            for _ in range(attempts):
                layout = self._build_gap_grid_template(template_name, rng, group_sizes)
                row_centers = self._gap_grid_axis_centers(
                    count=int(layout["row_count"]),
                    span_start=usable_top,
                    span_end=usable_bottom,
                    preferred_step=max(float(factors.min_gap) - 2.0, required_clearance + 4.0),
                    min_step=required_clearance + 2.0,
                    gap_after=layout["gap_after_rows"],
                    gap_multiplier=2.15,
                )
                if row_centers is None:
                    continue
                col_centers = self._gap_grid_axis_centers(
                    count=int(layout["col_count"]),
                    span_start=usable_left,
                    span_end=usable_right,
                    preferred_step=max(float(factors.min_gap) + 8.0, required_clearance + 12.0),
                    min_step=required_clearance + 8.0,
                    gap_after=layout["gap_after_cols"],
                    gap_multiplier=2.25,
                )
                if col_centers is None:
                    continue
                return {
                    **layout,
                    "row_centers": row_centers,
                    "col_centers": col_centers,
                }
        raise GenerationError("Gap grid cells too small")

    def _build_gap_grid_template(
        self,
        template_name: str,
        rng: random.Random,
        group_sizes: list[int],
    ) -> dict[str, object]:
        if template_name == "vertical_bands":
            rows = rng.choice([2, 3])
            col_widths = [max(2, math.ceil(size / rows)) for size in group_sizes]
            partitions: list[dict[str, int | str]] = []
            cursor_col = 0
            for group_index, col_width in enumerate(col_widths, start=1):
                partitions.append(
                    {
                        "group_id": f"G{group_index}",
                        "row_start": 0,
                        "row_end": rows,
                        "col_start": cursor_col,
                        "col_end": cursor_col + col_width,
                    }
                )
                cursor_col += col_width
            return {
                "template_name": template_name,
                "row_count": rows,
                "col_count": cursor_col,
                "gap_after_rows": [],
                "gap_after_cols": self._gap_boundaries(col_widths),
                "partitions": partitions,
            }
        if template_name == "horizontal_bands":
            cols = rng.choice([4, 5, 6])
            row_heights = [max(1, math.ceil(size / cols)) for size in group_sizes]
            partitions = []
            cursor_row = 0
            for group_index, row_height in enumerate(row_heights, start=1):
                partitions.append(
                    {
                        "group_id": f"G{group_index}",
                        "row_start": cursor_row,
                        "row_end": cursor_row + row_height,
                        "col_start": 0,
                        "col_end": cols,
                    }
                )
                cursor_row += row_height
            return {
                "template_name": template_name,
                "row_count": cursor_row,
                "col_count": cols,
                "gap_after_rows": self._gap_boundaries(row_heights),
                "gap_after_cols": [],
                "partitions": partitions,
            }
        if template_name == "l_split":
            return self._build_gap_grid_l_split(rng, group_sizes)
        if template_name == "quadrants":
            return self._build_gap_grid_quadrants(rng, group_sizes)
        raise GenerationError(f"Unsupported gap grid template: {template_name}")

    def _build_gap_grid_l_split(
        self,
        rng: random.Random,
        group_sizes: list[int],
    ) -> dict[str, object]:
        orientation = rng.choice(["left_full", "right_full", "top_full", "bottom_full"])
        primary_size, secondary_a, secondary_b = group_sizes
        if orientation in {"left_full", "right_full"}:
            right_cols = rng.choice([2, 3])
            top_rows = rng.choice([1, 2])
            bottom_rows = rng.choice([1, 2])
            total_rows = top_rows + bottom_rows
            primary_cols = max(2, math.ceil(primary_size / total_rows))
            right_cols = max(
                right_cols,
                math.ceil(secondary_a / top_rows),
                math.ceil(secondary_b / bottom_rows),
            )
            if orientation == "left_full":
                partitions = [
                    {"group_id": "G1", "row_start": 0, "row_end": total_rows, "col_start": 0, "col_end": primary_cols},
                    {"group_id": "G2", "row_start": 0, "row_end": top_rows, "col_start": primary_cols, "col_end": primary_cols + right_cols},
                    {"group_id": "G3", "row_start": top_rows, "row_end": total_rows, "col_start": primary_cols, "col_end": primary_cols + right_cols},
                ]
            else:
                partitions = [
                    {"group_id": "G1", "row_start": 0, "row_end": total_rows, "col_start": right_cols, "col_end": right_cols + primary_cols},
                    {"group_id": "G2", "row_start": 0, "row_end": top_rows, "col_start": 0, "col_end": right_cols},
                    {"group_id": "G3", "row_start": top_rows, "row_end": total_rows, "col_start": 0, "col_end": right_cols},
                ]
            return {
                "template_name": f"l_split_{orientation}",
                "row_count": total_rows,
                "col_count": primary_cols + right_cols,
                "gap_after_rows": [top_rows],
                "gap_after_cols": [primary_cols] if orientation == "left_full" else [right_cols],
                "partitions": partitions,
            }
        secondary_rows = rng.choice([1, 2])
        left_cols = max(2, math.ceil(secondary_a / secondary_rows))
        right_cols = max(2, math.ceil(secondary_b / secondary_rows))
        total_cols = left_cols + right_cols
        primary_rows = max(1, math.ceil(primary_size / total_cols))
        if orientation == "top_full":
            partitions = [
                {"group_id": "G1", "row_start": 0, "row_end": primary_rows, "col_start": 0, "col_end": total_cols},
                {"group_id": "G2", "row_start": primary_rows, "row_end": primary_rows + secondary_rows, "col_start": 0, "col_end": left_cols},
                {"group_id": "G3", "row_start": primary_rows, "row_end": primary_rows + secondary_rows, "col_start": left_cols, "col_end": total_cols},
            ]
        else:
            partitions = [
                {"group_id": "G1", "row_start": secondary_rows, "row_end": secondary_rows + primary_rows, "col_start": 0, "col_end": total_cols},
                {"group_id": "G2", "row_start": 0, "row_end": secondary_rows, "col_start": 0, "col_end": left_cols},
                {"group_id": "G3", "row_start": 0, "row_end": secondary_rows, "col_start": left_cols, "col_end": total_cols},
            ]
        return {
            "template_name": f"l_split_{orientation}",
            "row_count": max(partition["row_end"] for partition in partitions),
            "col_count": total_cols,
            "gap_after_rows": [primary_rows] if orientation == "top_full" else [secondary_rows],
            "gap_after_cols": [left_cols],
            "partitions": partitions,
        }

    def _build_gap_grid_quadrants(
        self,
        rng: random.Random,
        group_sizes: list[int],
    ) -> dict[str, object]:
        top_rows = rng.choice([1, 2])
        bottom_rows = rng.choice([1, 2])
        left_cols = max(2, math.ceil(max(group_sizes[0], group_sizes[2]) / max(top_rows, bottom_rows)))
        right_cols = max(2, math.ceil(max(group_sizes[1], group_sizes[3]) / max(top_rows, bottom_rows)))
        partitions = [
            {"group_id": "G1", "row_start": 0, "row_end": top_rows, "col_start": 0, "col_end": left_cols},
            {"group_id": "G2", "row_start": 0, "row_end": top_rows, "col_start": left_cols, "col_end": left_cols + right_cols},
            {"group_id": "G3", "row_start": top_rows, "row_end": top_rows + bottom_rows, "col_start": 0, "col_end": left_cols},
            {"group_id": "G4", "row_start": top_rows, "row_end": top_rows + bottom_rows, "col_start": left_cols, "col_end": left_cols + right_cols},
        ]
        return {
            "template_name": "quadrants",
            "row_count": top_rows + bottom_rows,
            "col_count": left_cols + right_cols,
            "gap_after_rows": [top_rows],
            "gap_after_cols": [left_cols],
            "partitions": partitions,
        }

    def _gap_boundaries(self, segment_lengths: list[int]) -> list[int]:
        boundaries: list[int] = []
        cursor = 0
        for segment_length in segment_lengths[:-1]:
            cursor += segment_length
            boundaries.append(cursor)
        return boundaries

    def _gap_grid_axis_centers(
        self,
        *,
        count: int,
        span_start: float,
        span_end: float,
        preferred_step: float,
        min_step: float,
        gap_after: list[int],
        gap_multiplier: float,
    ) -> list[float] | None:
        if count <= 0:
            return None
        if count == 1:
            return [(span_start + span_end) / 2.0]
        gap_boundaries = set(gap_after)
        units = (count - 1) + len(gap_boundaries) * (gap_multiplier - 1.0)
        available = span_end - span_start
        step = min(preferred_step, available / units)
        if step < min_step:
            return None
        current = span_start + (available - units * step) / 2.0
        centers = [current]
        for axis_index in range(1, count):
            previous_index = axis_index - 1
            delta = step * (gap_multiplier if axis_index in gap_boundaries else 1.0)
            current += delta
            centers.append(current)
        return centers

    def _similarity_partitions(
        self,
        factors: StructureSensitiveVisualFactors,
    ) -> list[dict[str, float | str]]:
        left = float(factors.margin + 24)
        right = float(factors.width - factors.margin - 24)
        top = float(factors.margin + 24)
        bottom = float(factors.height - factors.margin - 24)
        width = right - left
        height = bottom - top

        if factors.layout_pattern == "vertical_halves":
            return [
                {"group_id": "G1", "x": left, "y": top, "width": width / 2, "height": height},
                {"group_id": "G2", "x": left + width / 2, "y": top, "width": width / 2, "height": height},
            ]
        if factors.layout_pattern == "horizontal_halves":
            return [
                {"group_id": "G1", "x": left, "y": top, "width": width, "height": height / 2},
                {"group_id": "G2", "x": left, "y": top + height / 2, "width": width, "height": height / 2},
            ]
        if factors.layout_pattern == "vertical_stripes":
            stripe_count = factors.stripe_count or 4
            stripe_width = width / stripe_count
            return [
                {
                    "group_id": f"G{(index % 2) + 1}",
                    "x": left + index * stripe_width,
                    "y": top,
                    "width": stripe_width,
                    "height": height,
                }
                for index in range(stripe_count)
            ]
        if factors.layout_pattern == "horizontal_stripes":
            stripe_count = factors.stripe_count or 4
            stripe_height = height / stripe_count
            return [
                {
                    "group_id": f"G{(index % 2) + 1}",
                    "x": left,
                    "y": top + index * stripe_height,
                    "width": width,
                    "height": stripe_height,
                }
                for index in range(stripe_count)
            ]
        raise ValueError(f"Unsupported similarity layout_pattern: {factors.layout_pattern}")

    def _items_from_partitions(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
        partitions: list[dict[str, float | str]],
    ) -> list[VisualItemSpec]:
        items: list[VisualItemSpec] = []
        grouped_partitions: dict[str, list[dict[str, float | str]]] = {}
        for partition in partitions:
            grouped_partitions.setdefault(str(partition["group_id"]), []).append(partition)

        for group_index, group_size in enumerate(group_sizes, start=1):
            group_id = f"G{group_index}"
            group_partitions = grouped_partitions[group_id]
            slots: list[tuple[float, float]] = []
            for partition in group_partitions:
                slots.extend(self._partition_slots(partition, factors))
            if len(slots) < group_size:
                raise GenerationError("Not enough partition slots for similarity group")
            selected_slots = slots[:group_size]
            for px, py in selected_slots:
                items.append(self._make_placeholder_item(len(items) + 1, px, py, group_id=group_id))
        return items

    def _partition_slots(
        self,
        partition: dict[str, float | str],
        factors: StructureSensitiveVisualFactors,
    ) -> list[tuple[float, float]]:
        x = float(partition["x"])
        y = float(partition["y"])
        width = float(partition["width"])
        height = float(partition["height"])
        max_radius = max(SIZE_TO_RADIUS.values())
        required_clearance = 2 * max_radius + 10
        cols = max(1, int(width // required_clearance))
        rows = max(1, int(height // required_clearance))
        cell_w = width / cols
        cell_h = height / rows
        return [
            (x + (col + 0.5) * cell_w, y + (row + 0.5) * cell_h)
            for row in range(rows)
            for col in range(cols)
        ]

    def _sample_continuity_template(self, rng: random.Random) -> str:
        return rng.choice(
            [
                "diag_desc_arc_up",
                "diag_asc_arc_down",
                "horizontal_arch",
                "vertical_side_arc",
            ]
        )

    def _continuity_paths(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> tuple[list[VisualPathSpec], dict[str, object]]:
        template_name = self._sample_continuity_template(rng)
        path_specs = self._build_continuity_paths_from_template(rng, factors, template_name)
        return path_specs, {"continuity_template": template_name}

    def _build_continuity_paths_from_template(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        template_name: str,
    ) -> list[VisualPathSpec]:
        if template_name == "diag_desc_arc_up":
            return self._continuity_template_diag_desc_arc_up(rng, factors)
        if template_name == "diag_asc_arc_down":
            return self._continuity_template_diag_asc_arc_down(rng, factors)
        if template_name == "horizontal_arch":
            return self._continuity_template_horizontal_arch(rng, factors)
        if template_name == "vertical_side_arc":
            return self._continuity_template_vertical_side_arc(rng, factors)
        raise ValueError(f"Unsupported continuity template: {template_name}")

    def _continuity_template_diag_desc_arc_up(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> list[VisualPathSpec]:
        left = factors.margin + 42
        right = factors.width - factors.margin - 42
        top = factors.margin + 54
        bottom = factors.height - factors.margin - 54
        line_start = (left, top + rng.uniform(0, 40))
        line_end = (right, bottom - rng.uniform(0, 40))
        arc_start = (left, bottom - rng.uniform(0, 36))
        arc_end = (right, top + rng.uniform(0, 36))
        control = (
            factors.width * rng.uniform(0.42, 0.58),
            factors.height * rng.uniform(0.12, 0.22),
        )
        return [
            VisualPathSpec("P1", "G1", "line", [line_start, line_end]),
            VisualPathSpec("P2", "G2", "arc", [arc_start, control, arc_end]),
        ]

    def _continuity_template_diag_asc_arc_down(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> list[VisualPathSpec]:
        left = factors.margin + 42
        right = factors.width - factors.margin - 42
        top = factors.margin + 54
        bottom = factors.height - factors.margin - 54
        line_start = (left, bottom - rng.uniform(0, 40))
        line_end = (right, top + rng.uniform(0, 40))
        arc_start = (left, top + rng.uniform(0, 36))
        arc_end = (right, bottom - rng.uniform(0, 36))
        control = (
            factors.width * rng.uniform(0.42, 0.58),
            factors.height * rng.uniform(0.78, 0.88),
        )
        return [
            VisualPathSpec("P1", "G1", "line", [line_start, line_end]),
            VisualPathSpec("P2", "G2", "arc", [arc_start, control, arc_end]),
        ]

    def _continuity_template_horizontal_arch(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> list[VisualPathSpec]:
        left = factors.margin + 44
        right = factors.width - factors.margin - 44
        center_y = factors.height * rng.uniform(0.42, 0.58)
        line_start = (left, center_y + rng.uniform(-14, 14))
        line_end = (right, center_y + rng.uniform(-14, 14))
        cross_x = factors.width * rng.uniform(0.35, 0.65)
        arc_start = (left + rng.uniform(0, 22), factors.margin + rng.uniform(56, 96))
        arc_end = (right - rng.uniform(0, 22), factors.height - factors.margin - rng.uniform(56, 96))
        control = (
            cross_x + rng.uniform(-26, 26),
            factors.height * rng.uniform(0.82, 0.92),
        )
        return [
            VisualPathSpec("P1", "G1", "line", [line_start, line_end]),
            VisualPathSpec("P2", "G2", "arc", [arc_start, control, arc_end]),
        ]

    def _continuity_template_vertical_side_arc(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
    ) -> list[VisualPathSpec]:
        top = factors.margin + 44
        bottom = factors.height - factors.margin - 44
        center_x = factors.width * rng.uniform(0.40, 0.60)
        line_start = (center_x + rng.uniform(-14, 14), top)
        line_end = (center_x + rng.uniform(-14, 14), bottom)
        cross_y = factors.height * rng.uniform(0.35, 0.65)
        arc_start = (factors.margin + rng.uniform(56, 96), top + rng.uniform(0, 20))
        arc_end = (factors.width - factors.margin - rng.uniform(56, 96), bottom - rng.uniform(0, 20))
        control = (
            factors.width * rng.uniform(0.08, 0.20),
            cross_y + rng.uniform(-26, 26),
        )
        if rng.random() < 0.5:
            arc_start, arc_end = (
                (factors.width - factors.margin - rng.uniform(56, 96), top + rng.uniform(0, 20)),
                (factors.margin + rng.uniform(56, 96), bottom - rng.uniform(0, 20)),
            )
            control = (
                factors.width * rng.uniform(0.80, 0.92),
                cross_y + rng.uniform(-26, 26),
            )
        return [
            VisualPathSpec("P1", "G1", "line", [line_start, line_end]),
            VisualPathSpec("P2", "G2", "arc", [arc_start, control, arc_end]),
        ]

    def _items_on_paths(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
        path_specs: list[VisualPathSpec],
    ) -> list[VisualItemSpec]:
        items: list[VisualItemSpec] = []
        if factors.principle == "continuity":
            return self._items_on_continuity_paths(factors, group_sizes, path_specs)
        for size, path_spec in zip(group_sizes, path_specs):
            control_points = path_spec.control_points
            path_parameters = self._path_parameters(size)
            for index, t in enumerate(path_parameters):
                x, y = self._point_on_path(control_points, t)
                if 0 < index < size - 1 and factors.jitter:
                    x += rng.uniform(-factors.jitter, factors.jitter)
                    y += rng.uniform(-factors.jitter, factors.jitter)
                items.append(
                    self._make_placeholder_item(
                        len(items) + 1,
                        x,
                        y,
                        group_id=path_spec.group_id,
                        path_id=path_spec.path_id,
                    )
                )
        if not self._passes_no_overlap(items):
            raise GenerationError("Path items overlap")
        return items

    def _items_on_continuity_paths(
        self,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
        path_specs: list[VisualPathSpec],
    ) -> list[VisualItemSpec]:
        min_distance = float((2 * max(SIZE_TO_RADIUS.values())) + 6)
        intersections = self._continuity_intersections(path_specs)
        if not self._validate_single_crossing(intersections):
            raise GenerationError("Continuity scene does not have exactly one crossing")
        path_candidates = [
            self._continuity_path_candidates(path_spec, intersections)
            for path_spec in path_specs
        ]
        path_combinations = [
            self._continuity_candidate_combinations(candidates, count=size, min_distance=min_distance)
            for candidates, size in zip(path_candidates, group_sizes)
        ]
        if any(not combos for combos in path_combinations):
            raise GenerationError("Not enough continuity path combinations")

        items: list[VisualItemSpec] = []
        item_index = 1
        selected_points = self._select_non_overlapping_path_combinations(
            path_combinations,
            min_distance=min_distance,
        )
        if selected_points is None:
            raise GenerationError("Failed to place non-overlapping continuity path items")
        for path_spec, points in zip(path_specs, selected_points):
            for x, y in points:
                items.append(
                    self._make_placeholder_item(
                        item_index,
                        x,
                        y,
                        group_id=path_spec.group_id,
                        path_id=path_spec.path_id,
                    )
                )
                item_index += 1

        if not self._passes_no_overlap(items):
            raise GenerationError("Path items overlap")
        return items

    def _region_specs(
        self,
        factors: StructureSensitiveVisualFactors,
    ) -> list[VisualRegionSpec]:
        left = factors.margin + 18
        right = factors.width - factors.margin - 18
        top = factors.margin + 32
        bottom = factors.height - factors.margin - 32
        width = right - left
        height = bottom - top
        region_gap = 28
        if factors.condition_type == "uneven_boxes":
            region1 = VisualRegionSpec(
                region_id="R1",
                group_id="G1",
                x=left,
                y=top + 10,
                width=width * 0.42,
                height=height * 0.68,
            )
            region2 = VisualRegionSpec(
                region_id="R2",
                group_id="G2",
                x=left + width * 0.50,
                y=top + height * 0.18,
                width=width * 0.40,
                height=height * 0.56,
            )
            return [region1, region2]
        half_width = (width - region_gap) / 2
        return [
            VisualRegionSpec(
                region_id="R1",
                group_id="G1",
                x=left,
                y=top + 12,
                width=half_width,
                height=height * 0.74,
            ),
            VisualRegionSpec(
                region_id="R2",
                group_id="G2",
                x=left + half_width + region_gap,
                y=top + 12,
                width=half_width,
                height=height * 0.74,
            ),
        ]

    def _items_in_regions(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        group_sizes: list[int],
        regions: list[VisualRegionSpec],
    ) -> list[VisualItemSpec]:
        items: list[VisualItemSpec] = []
        inner_padding = max(SIZE_TO_RADIUS.values()) + 14
        for region, group_size in zip(regions, group_sizes):
            positions = self._sample_positions_in_box(
                rng,
                region.x + inner_padding,
                region.y + inner_padding,
                region.x + region.width - inner_padding,
                region.y + region.height - inner_padding,
                group_size,
                min_distance=max(2 * max(SIZE_TO_RADIUS.values()) + 6, float(factors.min_gap)),
                existing_items=items,
            )
            for x, y in positions:
                items.append(
                    self._make_placeholder_item(
                        len(items) + 1,
                        x,
                        y,
                        group_id=region.group_id,
                        region_id=region.region_id,
                    )
                )

        loose_count = factors.loose_item_count or 0
        if loose_count:
            positions = self._sample_loose_positions(rng, factors, regions, loose_count, items)
            for x, y in positions:
                items.append(
                    self._make_placeholder_item(
                        len(items) + 1,
                        x,
                        y,
                        group_id="LOOSE",
                        region_id=None,
                        is_loose_item=True,
                    )
                )
        return items

    def _sample_positions_in_box(
        self,
        rng: random.Random,
        left: float,
        top: float,
        right: float,
        bottom: float,
        count: int,
        *,
        min_distance: float,
        existing_items: list[VisualItemSpec],
    ) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        for _ in range(count):
            placed = False
            for _ in range(600):
                x = rng.uniform(left, right)
                y = rng.uniform(top, bottom)
                if any(self._distance((x, y), pos) < min_distance for pos in positions):
                    continue
                if any(self._distance((x, y), (item.x, item.y)) < min_distance for item in existing_items):
                    continue
                positions.append((x, y))
                placed = True
                break
            if not placed:
                raise GenerationError("Failed to sample positions in box")
        return positions

    def _sample_loose_positions(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        regions: list[VisualRegionSpec],
        count: int,
        items: list[VisualItemSpec],
    ) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        region_rects = [
            (region.x, region.y, region.x + region.width, region.y + region.height)
            for region in regions
        ]
        item_radius = max(SIZE_TO_RADIUS.values())
        boundary_candidates = self._boundary_loose_candidates(regions, factors, radius=item_radius + 10)
        for index in range(count):
            placed = False
            candidate_pool = boundary_candidates[index:] + boundary_candidates[:index]
            for base_x, base_y in candidate_pool:
                x = base_x + rng.uniform(-8, 8)
                y = base_y + rng.uniform(-8, 8)
                if any(self._point_too_close_to_region(x, y, rect, radius=item_radius + 4) for rect in region_rects):
                    continue
                if any(self._distance((x, y), pos) < factors.min_gap for pos in positions):
                    continue
                if any(self._distance((x, y), (item.x, item.y)) < factors.min_gap for item in items):
                    continue
                positions.append((x, y))
                placed = True
                break
            if not placed:
                for _ in range(700):
                    x = rng.uniform(factors.margin + item_radius + 8, factors.width - factors.margin - item_radius - 8)
                    y = rng.uniform(factors.margin + item_radius + 8, factors.height - factors.margin - item_radius - 8)
                    if any(self._point_too_close_to_region(x, y, rect, radius=item_radius + 4) for rect in region_rects):
                        continue
                    if any(self._distance((x, y), pos) < factors.min_gap for pos in positions):
                        continue
                    if any(self._distance((x, y), (item.x, item.y)) < factors.min_gap for item in items):
                        continue
                    positions.append((x, y))
                    placed = True
                    break
            if not placed:
                raise GenerationError("Failed to place loose items")
        return positions

    def _boundary_loose_candidates(
        self,
        regions: list[VisualRegionSpec],
        factors: StructureSensitiveVisualFactors,
        *,
        radius: float,
    ) -> list[tuple[float, float]]:
        candidates: list[tuple[float, float]] = []
        scene_center_x = factors.width / 2
        for region in regions:
            left = region.x
            right = region.x + region.width
            top = region.y
            bottom = region.y + region.height
            if (left + right) / 2 < scene_center_x:
                candidates.append((right + radius, region.y + region.height * 0.52))
                candidates.append((left - radius, region.y + region.height * 0.36))
            else:
                candidates.append((left - radius, region.y + region.height * 0.48))
                candidates.append((right + radius, region.y + region.height * 0.64))
            candidates.append((region.x + region.width * 0.50, top - radius))
            candidates.append((region.x + region.width * 0.50, bottom + radius))
        return [
            (
                min(max(x, factors.margin + 18), factors.width - factors.margin - 18),
                min(max(y, factors.margin + 18), factors.height - factors.margin - 18),
            )
            for x, y in candidates
        ]

    def _make_placeholder_item(
        self,
        index: int,
        x: float,
        y: float,
        *,
        group_id: str,
        region_id: str | None = None,
        path_id: str | None = None,
        is_loose_item: bool = False,
    ) -> VisualItemSpec:
        return VisualItemSpec(
            item_id=f"item_{index:02d}",
            x=x,
            y=y,
            color="red",
            shape="circle",
            size="large",
            group_id=group_id,
            region_id=region_id,
            path_id=path_id,
            is_loose_item=is_loose_item,
            is_anchor=False,
        )

    def _choose_anchor(
        self,
        rng: random.Random,
        items: list[VisualItemSpec],
        factors: StructureSensitiveVisualFactors,
        paths: list[VisualPathSpec],
    ) -> tuple[str, str]:
        grouped = self._group_items(items)
        eligible_group_ids = [
            group_id
            for group_id, group_items in grouped.items()
            if group_id != "LOOSE"
            and len(group_items) >= factors.target_in_anchor_group + factors.non_target_in_anchor_group
        ]
        if not eligible_group_ids:
            raise GenerationError("No eligible anchor group found")
        anchor_group_id = rng.choice(eligible_group_ids)
        valid_items = [item for item in grouped[anchor_group_id]]
        if factors.principle == "continuity" and paths:
            crossing_point = self._continuity_crossing_point(paths)
            far_items = [
                item
                for item in valid_items
                if self._distance((item.x, item.y), crossing_point) >= self._continuity_anchor_crossing_clearance()
            ]
            if far_items:
                valid_items = far_items
        anchor = rng.choice(valid_items)
        return anchor.item_id, anchor.group_id

    def _assign_visual_features(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        *,
        anchor_group_id: str,
    ) -> tuple[list[VisualItemSpec], dict[str, object]]:
        target_value = self._sample_target_domain_value(rng, factors.queried_feature)
        anchor_items = [item for item in items if item.group_id == anchor_group_id]
        other_items = [item for item in items if item.group_id != anchor_group_id]
        if len(anchor_items) < factors.target_in_anchor_group + factors.non_target_in_anchor_group:
            raise GenerationError("Anchor group is too small for requested target/non-target counts")
        if len(other_items) < factors.target_outside_anchor_group:
            raise GenerationError("Non-anchor groups are too small for requested outside targets")

        target_anchor_ids = {
            item.item_id
            for item in rng.sample(anchor_items, factors.target_in_anchor_group)
        }
        target_other_ids = {
            item.item_id
            for item in rng.sample(other_items, factors.target_outside_anchor_group)
        }

        group_color_map: dict[str, ColorName] = {}
        if factors.principle == "similarity":
            available_colors = list(COLORS)
            rng.shuffle(available_colors)
            groups = [group_id for group_id in self._ordered_group_ids(items) if group_id != "LOOSE"]
            if len(groups) > len(available_colors):
                raise GenerationError("Not enough distinct colors for similarity grouping")
            group_color_map = {
                group_id: available_colors[index]
                for index, group_id in enumerate(groups)
            }

        updated_items: list[VisualItemSpec] = []
        target_shape_map: dict[str, ShapeName] = {}
        for group_id in self._ordered_group_ids(items):
            if group_id == "LOOSE":
                continue
            non_target_shapes = [shape for shape in SHAPES if shape != target_value]
            target_shape_map[group_id] = rng.choice(non_target_shapes) if factors.queried_feature == "shape" else rng.choice(SHAPES)

        for item in items:
            color = rng.choice(COLORS)
            shape = rng.choice(SHAPES)
            size = rng.choice(SIZES)

            if factors.principle == "similarity":
                color = group_color_map[item.group_id] if item.group_id in group_color_map else rng.choice(COLORS)
                target_shape = str(target_value)
                non_target_shape = target_shape_map.get(item.group_id, rng.choice([shape for shape in SHAPES if shape != target_shape]))
                if item.item_id in target_anchor_ids or item.item_id in target_other_ids:
                    shape = target_shape  # type: ignore[assignment]
                elif factors.condition_type == "color_then_shape" and item.group_id != "LOOSE":
                    if rng.random() < (factors.shape_bias_strength or 0.0):
                        shape = non_target_shape
                    else:
                        shape = rng.choice([candidate for candidate in SHAPES if candidate != target_shape])
                else:
                    shape = rng.choice([candidate for candidate in SHAPES if candidate != target_shape])
            else:
                if item.item_id in target_anchor_ids or item.item_id in target_other_ids:
                    if factors.queried_feature == "color":
                        color = target_value  # type: ignore[assignment]
                    elif factors.queried_feature == "shape":
                        shape = target_value  # type: ignore[assignment]
                    else:
                        size = target_value  # type: ignore[assignment]
                else:
                    color, shape, size = self._non_target_features(rng, factors.queried_feature, str(target_value))

            if factors.queried_feature != "size":
                size = "large"

            updated_items.append(replace(item, color=color, shape=shape, size=size))

        return updated_items, {
            "assigned_target_value": target_value,
            "group_color_map": group_color_map,
            "secondary_shape_bias_enabled": factors.principle == "similarity" and factors.condition_type == "color_then_shape",
        }

    def _similarity_band_to_group_map(
        self,
        factors: StructureSensitiveVisualFactors,
    ) -> dict[str, str]:
        if "stripes" not in factors.layout_pattern:
            return {}
        band_count = factors.stripe_count or 0
        return {f"band_{index}": f"G{(index % 2) + 1}" for index in range(band_count)}

    def _generator_case_for_scene(
        self,
        factors: StructureSensitiveVisualFactors,
        metadata: dict[str, object],
    ) -> str:
        if factors.principle == "continuity":
            template_name = str(metadata.get("continuity_template", "unknown"))
            return (
                f"{factors.principle}_{factors.condition_type}_{factors.layout_pattern}_"
                f"{template_name}_v1"
            )
        return f"{factors.principle}_{factors.condition_type}_{factors.layout_pattern}_v1"

    def _actual_scene_counts(
        self,
        scene: StructureSensitiveVisualScene,
    ) -> dict[str, int]:
        anchor_group_id = scene.anchor_group_id
        target_value = scene.target_value
        queried_feature = scene.queried_feature
        target_in_anchor = 0
        target_outside_anchor = 0
        non_target_in_anchor = 0
        for item in scene.items:
            is_target = self._matches_target_value(item, queried_feature, target_value)
            if item.group_id == anchor_group_id:
                if is_target:
                    target_in_anchor += 1
                else:
                    non_target_in_anchor += 1
            elif is_target:
                target_outside_anchor += 1
        return {
            "actual_target_in_anchor_group": target_in_anchor,
            "actual_target_outside_anchor_group": target_outside_anchor,
            "actual_non_target_in_anchor_group": non_target_in_anchor,
        }

    def _sample_target_domain_value(self, rng: random.Random, queried_feature: str) -> str:
        if queried_feature == "color":
            return rng.choice(COLORS)
        if queried_feature == "shape":
            return rng.choice(SHAPES)
        if queried_feature == "size":
            return rng.choice(SIZES)
        raise ValueError(f"Unsupported queried_feature: {queried_feature}")

    def _non_target_features(
        self,
        rng: random.Random,
        queried_feature: str,
        target_value: str,
    ) -> tuple[ColorName, ShapeName, SizeName]:
        while True:
            color = rng.choice(COLORS)
            shape = rng.choice(SHAPES)
            size = rng.choice(SIZES)
            if queried_feature == "color" and color == target_value:
                continue
            if queried_feature == "shape" and shape == target_value:
                continue
            if queried_feature == "size" and size == target_value:
                continue
            return color, shape, size

    def _choose_target_value(
        self,
        rng: random.Random,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        anchor_group_id: str,
    ) -> str:
        anchor_items = self._items_in_anchor_group(items, anchor_group_id)
        candidates = []
        domain = self._feature_domain(factors.queried_feature)
        for value in domain:
            anchor_count = sum(
                1 for item in anchor_items if self._matches_target_value(item, factors.queried_feature, value)
            )
            total_count = sum(
                1 for item in items if self._matches_target_value(item, factors.queried_feature, value)
            )
            if anchor_count == factors.target_in_anchor_group and total_count > anchor_count:
                candidates.append(value)
        if not candidates:
            raise GenerationError("No valid target value found for scene")
        return rng.choice(candidates)

    def _items_in_anchor_group(
        self,
        items: list[VisualItemSpec],
        anchor_group_id: str,
    ) -> list[VisualItemSpec]:
        return [item for item in items if item.group_id == anchor_group_id]

    def _matching_item_indices(
        self,
        items: list[VisualItemSpec],
        anchor_group_id: str,
        queried_feature: str,
        target_value: str,
    ) -> list[int]:
        return [
            index + 1
            for index, item in enumerate(items)
            if item.group_id == anchor_group_id and self._matches_target_value(item, queried_feature, target_value)
        ]

    def _matches_target_value(
        self,
        item: VisualItemSpec,
        queried_feature: str,
        target_value: str,
    ) -> bool:
        return getattr(item, queried_feature) == target_value

    def _passes_constraints(
        self,
        *,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        regions: list[VisualRegionSpec],
        paths: list[VisualPathSpec],
        anchor_group_id: str,
        target_value: str,
    ) -> bool:
        if not self._passes_universal_constraints(
            factors=factors,
            items=items,
            anchor_group_id=anchor_group_id,
            target_value=target_value,
        ):
            return False
        if factors.principle == "baseline":
            return self._passes_baseline_constraints(factors, items, regions, paths)
        if factors.principle == "proximity":
            return self._passes_proximity_constraints(factors, items, regions, paths)
        if factors.principle == "similarity":
            return self._passes_similarity_constraints(factors, items, regions, paths)
        if factors.principle == "continuity":
            return self._passes_continuity_constraints(factors, items, regions, paths)
        if factors.principle == "common_region":
            return self._passes_common_region_constraints(factors, items, regions, paths)
        return False

    def _passes_universal_constraints(
        self,
        *,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        anchor_group_id: str,
        target_value: str,
    ) -> bool:
        anchor_items = self._items_in_anchor_group(items, anchor_group_id)
        gold_indices = self._matching_item_indices(items, anchor_group_id, factors.queried_feature, target_value)
        if len(gold_indices) != factors.target_in_anchor_group:
            return False
        outside_target_count = sum(
            1
            for item in items
            if item.group_id != anchor_group_id
            and self._matches_target_value(item, factors.queried_feature, target_value)
        )
        if outside_target_count != factors.target_outside_anchor_group:
            return False
        if len(anchor_items) < factors.target_in_anchor_group + factors.non_target_in_anchor_group:
            return False
        if sum(1 for item in anchor_items if not self._matches_target_value(item, factors.queried_feature, target_value)) < factors.non_target_in_anchor_group:
            return False
        if sum(1 for item in items if item.is_anchor) != 1:
            return False
        if not self._passes_no_overlap(items):
            return False
        return True

    def _passes_no_overlap(self, items: list[VisualItemSpec]) -> bool:
        for index, item_a in enumerate(items):
            radius_a = SIZE_TO_RADIUS[item_a.size]
            for item_b in items[index + 1 :]:
                radius_b = SIZE_TO_RADIUS[item_b.size]
                min_distance = radius_a + radius_b + 6
                if self._distance((item_a.x, item_a.y), (item_b.x, item_b.y)) < min_distance:
                    return False
        return True

    def _passes_baseline_constraints(
        self,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        regions: list[VisualRegionSpec],
        paths: list[VisualPathSpec],
    ) -> bool:
        return self._passes_common_region_constraints(factors, items, regions, paths)

    def _passes_proximity_constraints(
        self,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        regions: list[VisualRegionSpec],
        paths: list[VisualPathSpec],
    ) -> bool:
        if regions or paths:
            return False
        grouped = self._group_items(items)
        for group_items in grouped.values():
            if len({item.shape for item in group_items}) < 2:
                return False
            if len({item.color for item in group_items}) < 2:
                return False
        if factors.condition_type == "random_clusters":
            return self._cluster_separation_is_readable(items, threshold=factors.inter_group_margin or 72.0)
        return True

    def _passes_similarity_constraints(
        self,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        regions: list[VisualRegionSpec],
        paths: list[VisualPathSpec],
    ) -> bool:
        if regions or paths:
            return False
        grouped = self._group_items(items)
        if len(grouped) != 2:
            return False
        dominant_colors = {next(iter({item.color for item in group_items})) for group_items in grouped.values() if group_items}
        if len(dominant_colors) != 2:
            return False
        for group_items in grouped.values():
            colors = {item.color for item in group_items}
            if len(colors) != 1:
                return False
        if factors.condition_type == "color_then_shape":
            for group_items in grouped.values():
                shapes = {item.shape for item in group_items}
                if len(shapes) < 2:
                    return False
        return True

    def _passes_continuity_constraints(
        self,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        regions: list[VisualRegionSpec],
        paths: list[VisualPathSpec],
    ) -> bool:
        if regions or len(paths) != 2:
            return False
        if factors.path_crossing_count != 1:
            return False
        intersections = self._continuity_intersections(paths)
        if not self._validate_single_crossing(intersections):
            return False
        if not self._passes_continuity_geometry_constraints(factors, paths, intersections):
            return False
        grouped = self._group_items(items)
        if not all(len(group_items) >= factors.min_items_per_group for group_items in grouped.values()):
            return False
        if any(
            self._item_overlaps_any_crossing(
                item,
                intersections,
                extra_padding=(PATH_STROKE_WIDTH / 2) + 10.0,
            )
            for item in items
        ):
            return False
        anchor_item = next((item for item in items if item.is_anchor), None)
        if anchor_item is None:
            return False
        if self._distance((anchor_item.x, anchor_item.y), intersections[0]) < self._continuity_anchor_crossing_clearance():
            return False
        return True

    def _passes_common_region_constraints(
        self,
        factors: StructureSensitiveVisualFactors,
        items: list[VisualItemSpec],
        regions: list[VisualRegionSpec],
        paths: list[VisualPathSpec],
    ) -> bool:
        if paths or len(regions) != (factors.region_count or 0):
            return False
        region_map = {region.region_id: region for region in regions}
        for region in regions:
            region_items = [item for item in items if item.region_id == region.region_id]
            if not region_items:
                return False
            if any(not self._item_fully_inside_region(item, region) for item in region_items):
                return False
        loose_items = [item for item in items if item.group_id == "LOOSE"]
        if any(
            any(self._item_touches_region(item, region) for region in regions)
            for item in loose_items
        ):
            return False
        for item in items:
            if item.region_id is None:
                continue
            region = region_map.get(item.region_id)
            if region is None or not self._item_fully_inside_region(item, region):
                return False
        return True

    def _cluster_separation_is_readable(
        self,
        items: list[VisualItemSpec],
        *,
        threshold: float,
    ) -> bool:
        centers = {
            group_id: self._group_center(group_items)
            for group_id, group_items in self._group_items(items).items()
            if group_id != "LOOSE"
        }
        center_list = list(centers.values())
        for index, center_a in enumerate(center_list):
            for center_b in center_list[index + 1 :]:
                if self._distance(center_a, center_b) < threshold:
                    return False
        return True

    def _draw_item(self, draw: ImageDraw.ImageDraw, item: VisualItemSpec) -> None:
        radius = SIZE_TO_RADIUS[item.size]
        color = COLOR_TO_RGB[item.color]
        bbox = [item.x - radius, item.y - radius, item.x + radius, item.y + radius]
        if item.shape == "circle":
            draw.ellipse(bbox, fill=color, outline=(35, 35, 35), width=2)
            return
        if item.shape == "square":
            draw.rectangle(bbox, fill=color, outline=(35, 35, 35), width=2)
            return
        if item.shape == "triangle":
            points = [
                (item.x, item.y - radius),
                (item.x - radius * 0.9, item.y + radius * 0.8),
                (item.x + radius * 0.9, item.y + radius * 0.8),
            ]
            draw.polygon(points, fill=color, outline=(35, 35, 35))
            return
        raise ValueError(f"Unsupported shape: {item.shape}")

    def _draw_anchor_marker(self, draw: ImageDraw.ImageDraw, item: VisualItemSpec) -> None:
        radius = SIZE_TO_RADIUS[item.size] + 8
        dot_radius = 2.2
        dot_count = 18
        for index in range(dot_count):
            angle = (2.0 * math.pi * index) / dot_count
            cx = item.x + math.cos(angle) * radius
            cy = item.y + math.sin(angle) * radius
            draw.ellipse(
                [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
                fill=(20, 20, 20),
            )

    def _draw_region(self, draw: ImageDraw.ImageDraw, region: VisualRegionSpec) -> None:
        bbox = [region.x, region.y, region.x + region.width, region.y + region.height]
        fill = (245, 245, 245)
        outline = (180, 180, 180)
        draw.rounded_rectangle(bbox, radius=16, fill=fill, outline=outline, width=2)

    def _draw_path(self, draw: ImageDraw.ImageDraw, path: VisualPathSpec) -> None:
        points = self._path_polyline(path.control_points, samples=48)
        draw.line(points, fill=(185, 185, 185), width=4)

    def _draw_item_id(
        self,
        draw: ImageDraw.ImageDraw,
        item: VisualItemSpec,
        item_index: int,
        factors: StructureSensitiveVisualFactors,
        all_items: list[VisualItemSpec],
    ) -> None:
        radius = SIZE_TO_RADIUS[item.size]
        label = str(item_index)
        stroke_width = 2
        text_bbox = draw.textbbox((0, 0), label, font=self._font, stroke_width=stroke_width)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad = 4
        gap = 3
        side_offset = radius + gap
        if item.shape == "triangle":
            side_offset = max(radius * 0.72, radius - 4)
        if factors.condition_type == "gap_grid":
            candidates = self._gap_grid_id_candidates(
                item,
                text_w=text_w,
                text_h=text_h,
                radius=radius,
                side_offset=side_offset,
                gap=gap,
                factors=factors,
            )
            use_collision_guard = True
        else:
            candidates = [
                (item.x + side_offset, item.y - text_h / 2),
                (item.x - side_offset - text_w, item.y - text_h / 2),
                (item.x + side_offset, item.y - text_h - gap),
                (item.x - side_offset - text_w, item.y - text_h - gap),
            ]
            use_collision_guard = False

        def fits(x: float, y: float) -> bool:
            if not (
                pad <= x <= factors.width - text_w - pad
                and pad <= y <= factors.height - text_h - pad
            ):
                return False
            if not use_collision_guard:
                return True
            label_box = (x, y, x + text_w, y + text_h)
            return not self._label_box_hits_other_items(label_box, item, all_items)

        x, y = next(((cx, cy) for cx, cy in candidates if fits(cx, cy)), candidates[0])
        x = max(pad, min(x, factors.width - text_w - pad))
        y = max(pad, min(y, factors.height - text_h - pad))
        draw.text(
            (x, y),
            label,
            fill=(60, 60, 60),
            font=self._font,
            stroke_width=stroke_width,
            stroke_fill=(255, 255, 255),
        )

    def _gap_grid_id_candidates(
        self,
        item: VisualItemSpec,
        *,
        text_w: float,
        text_h: float,
        radius: float,
        side_offset: float,
        gap: float,
        factors: StructureSensitiveVisualFactors,
    ) -> list[tuple[float, float]]:
        usable_top = factors.margin + 28
        estimated_row_band = max(1, int((radius * 2) + 18))
        row_index = max(0, int((item.y - usable_top) // estimated_row_band))
        vertical_offset = radius + 1
        above = (item.x - text_w / 2, item.y - vertical_offset - text_h)
        below = (item.x - text_w / 2, item.y + vertical_offset - 1)
        right = (item.x + side_offset - 2, item.y - text_h / 2)
        left = (item.x - side_offset - text_w + 2, item.y - text_h / 2)
        if row_index % 2 == 0:
            return [above, below, right, left]
        return [below, above, right, left]

    def _label_box_hits_other_items(
        self,
        label_box: tuple[float, float, float, float],
        current_item: VisualItemSpec,
        all_items: list[VisualItemSpec],
    ) -> bool:
        for other_item in all_items:
            if other_item.item_id == current_item.item_id:
                continue
            if self._label_box_overlaps_item(label_box, other_item):
                return True
            if other_item.is_anchor and self._label_box_overlaps_anchor_marker(label_box, other_item):
                return True
        return False

    def _label_box_overlaps_item(
        self,
        label_box: tuple[float, float, float, float],
        item: VisualItemSpec,
    ) -> bool:
        radius = self._shape_cover_radius(item)
        item_box = (
            item.x - radius,
            item.y - radius,
            item.x + radius,
            item.y + radius,
        )
        return self._rects_overlap(label_box, item_box)

    def _label_box_overlaps_anchor_marker(
        self,
        label_box: tuple[float, float, float, float],
        item: VisualItemSpec,
    ) -> bool:
        marker_radius = SIZE_TO_RADIUS[item.size] + 10
        marker_box = (
            item.x - marker_radius,
            item.y - marker_radius,
            item.x + marker_radius,
            item.y + marker_radius,
        )
        return self._rects_overlap(label_box, marker_box)

    def _target_description(self, target_value: str) -> str:
        return target_value

    def _build_count_instruction(self, target_value: str) -> str:
        description = self._target_description(target_value)
        return (
            f"Count the {description} items in the same group as the marked item.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n"
            '- "count" must be an integer\n'
            "- Apply the full rule exactly\n"
            "- Return only the JSON object"
        )

    def _build_filter_instruction(self, target_value: str) -> str:
        description = self._target_description(target_value)
        return (
            f"Return the 1-based item indices of the {description} items in the same group as the marked item.\n"
            'Respond with a JSON object of the form {"indices": [<sorted unique integers>]}.\n'
            "Rules:\n"
            "- Use 1-based indexing\n"
            "- Sort ascending\n"
            "- Do not include duplicates\n"
            "- Apply the full rule exactly\n"
            "- Return only the JSON object"
        )

    def _group_items(self, items: list[VisualItemSpec]) -> dict[str, list[VisualItemSpec]]:
        grouped: dict[str, list[VisualItemSpec]] = {}
        for item in items:
            grouped.setdefault(item.group_id, []).append(item)
        return grouped

    def _ordered_group_ids(self, items: list[VisualItemSpec]) -> list[str]:
        return list(dict.fromkeys(item.group_id for item in items))

    def _group_center(self, items: list[VisualItemSpec]) -> tuple[float, float]:
        return (
            sum(item.x for item in items) / len(items),
            sum(item.y for item in items) / len(items),
        )

    def _path_parameters(self, count: int) -> list[float]:
        if count <= 1:
            return [0.18]
        candidates = [0.06, 0.16, 0.26, 0.34, 0.70, 0.80, 0.90]
        if count > len(candidates):
            raise GenerationError("Not enough non-crossing path slots for requested count")
        if count == len(candidates):
            return candidates
        step = (len(candidates) - 1) / (count - 1)
        return [candidates[round(index * step)] for index in range(count)]

    def _continuity_path_candidates(
        self,
        path: VisualPathSpec,
        intersections: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        exclusion_radius = self._continuity_crossing_clearance()
        if path.path_type == "line":
            sampled_points = self._sample_line_points(
                path.control_points[0],
                path.control_points[1],
                samples=48,
            )
        else:
            sampled_points = self._sample_quadratic_bezier_points(
                path.control_points[0],
                path.control_points[1],
                path.control_points[2],
                samples=64,
            )
        filtered_points = [
            point
            for index, point in enumerate(sampled_points)
            if 3 <= index <= len(sampled_points) - 4
            and not self._point_near_any_crossing(point, intersections, exclusion_radius)
        ]
        max_candidates = 12
        if len(filtered_points) <= max_candidates:
            return filtered_points
        step = (len(filtered_points) - 1) / (max_candidates - 1)
        return [filtered_points[round(index * step)] for index in range(max_candidates)]

    def _continuity_candidate_combinations(
        self,
        candidates: list[tuple[float, float]],
        *,
        count: int,
        min_distance: float,
    ) -> list[list[tuple[float, float]]]:
        combos = [
            list(combo)
            for combo in combinations(candidates, count)
            if self._points_respect_min_distance(combo, min_distance=min_distance)
        ]
        combos.sort(
            key=lambda combo: min(
                self._distance(point_a, point_b)
                for index, point_a in enumerate(combo)
                for point_b in combo[index + 1 :]
            ),
            reverse=True,
        )
        return combos

    def _select_non_overlapping_path_combinations(
        self,
        path_combinations: list[list[list[tuple[float, float]]]],
        *,
        min_distance: float,
    ) -> list[list[tuple[float, float]]] | None:
        def backtrack(
            path_index: int,
            selected: list[list[tuple[float, float]]],
        ) -> list[list[tuple[float, float]]] | None:
            if path_index >= len(path_combinations):
                return selected
            for combo in path_combinations[path_index]:
                if all(
                    self._distance(point, other_point) >= min_distance
                    for group_points in selected
                    for point in combo
                    for other_point in group_points
                ):
                    result = backtrack(path_index + 1, [*selected, combo])
                    if result is not None:
                        return result
            return None

        return backtrack(0, [])

    def _points_respect_min_distance(
        self,
        points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
        *,
        min_distance: float,
    ) -> bool:
        for index, point_a in enumerate(points):
            for point_b in points[index + 1 :]:
                if self._distance(point_a, point_b) < min_distance:
                    return False
        return True

    def _path_polyline(
        self,
        control_points: list[tuple[float, float]],
        *,
        samples: int,
    ) -> list[tuple[float, float]]:
        if len(control_points) == 2:
            return self._sample_line_points(control_points[0], control_points[1], samples=samples)
        if len(control_points) == 3:
            return self._sample_quadratic_bezier_points(
                control_points[0],
                control_points[1],
                control_points[2],
                samples=samples,
            )
        return [self._point_on_path(control_points, step / (samples - 1)) for step in range(samples)]

    def _point_on_path(
        self,
        control_points: list[tuple[float, float]],
        t: float,
    ) -> tuple[float, float]:
        if len(control_points) == 2:
            return (
                control_points[0][0] + (control_points[1][0] - control_points[0][0]) * t,
                control_points[0][1] + (control_points[1][1] - control_points[0][1]) * t,
            )
        if len(control_points) == 3:
            return self._quadratic_bezier(control_points[0], control_points[1], control_points[2], t)
        if len(control_points) == 5:
            if t <= 0.5:
                local_t = t / 0.5
                return self._quadratic_bezier(control_points[0], control_points[1], control_points[2], local_t)
            local_t = (t - 0.5) / 0.5
            return self._quadratic_bezier(control_points[2], control_points[3], control_points[4], local_t)
        raise ValueError("Unsupported path control point count")

    def _continuity_intersections(
        self,
        paths: list[VisualPathSpec],
    ) -> list[tuple[float, float]]:
        if len(paths) != 2:
            return []
        return self._find_all_path_intersections(paths[0], paths[1])

    def _validate_single_crossing(
        self,
        intersections: list[tuple[float, float]],
    ) -> bool:
        return len(intersections) == 1

    def _passes_continuity_geometry_constraints(
        self,
        factors: StructureSensitiveVisualFactors,
        paths: list[VisualPathSpec],
        intersections: list[tuple[float, float]],
    ) -> bool:
        if len(intersections) != 1:
            return False
        crossing = intersections[0]
        crossing_margin = factors.margin + 80
        if not (
            crossing_margin <= crossing[0] <= factors.width - crossing_margin
            and crossing_margin <= crossing[1] <= factors.height - crossing_margin
        ):
            return False
        minimum_visible_length = 110.0
        return all(
            self._path_lengths_around_crossing(path, crossing)[0] >= minimum_visible_length
            and self._path_lengths_around_crossing(path, crossing)[1] >= minimum_visible_length
            for path in paths
        )

    def _continuity_crossing_point(
        self,
        paths: list[VisualPathSpec],
    ) -> tuple[float, float]:
        intersections = self._continuity_intersections(paths)
        if len(intersections) != 1:
            raise GenerationError("Continuity scene does not have exactly one crossing")
        return intersections[0]

    def _path_lengths_around_crossing(
        self,
        path: VisualPathSpec,
        crossing: tuple[float, float],
    ) -> tuple[float, float]:
        polyline = self._path_polyline(path.control_points, samples=160)
        nearest_index = min(
            range(len(polyline)),
            key=lambda index: self._distance(polyline[index], crossing),
        )
        before_length = sum(
            self._distance(point_a, point_b)
            for point_a, point_b in zip(polyline[:nearest_index], polyline[1 : nearest_index + 1])
        )
        after_length = sum(
            self._distance(point_a, point_b)
            for point_a, point_b in zip(polyline[nearest_index:], polyline[nearest_index + 1 :])
        )
        return before_length, after_length

    def _sample_line_points(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        samples: int,
    ) -> list[tuple[float, float]]:
        return [
            (
                p0[0] + (p1[0] - p0[0]) * (index / (samples - 1)),
                p0[1] + (p1[1] - p0[1]) * (index / (samples - 1)),
            )
            for index in range(samples)
        ]

    def _sample_quadratic_bezier_points(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        samples: int,
    ) -> list[tuple[float, float]]:
        return [
            self._quadratic_bezier(p0, p1, p2, index / (samples - 1))
            for index in range(samples)
        ]

    def _segment_intersection(
        self,
        a0: tuple[float, float],
        a1: tuple[float, float],
        b0: tuple[float, float],
        b1: tuple[float, float],
    ) -> tuple[float, float] | None:
        x1, y1 = a0
        x2, y2 = a1
        x3, y3 = b0
        x4, y4 = b1
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-6:
            return None
        det_a = x1 * y2 - y1 * x2
        det_b = x3 * y4 - y3 * x4
        px = (det_a * (x3 - x4) - (x1 - x2) * det_b) / denominator
        py = (det_a * (y3 - y4) - (y1 - y2) * det_b) / denominator
        tolerance = 1e-6
        if (
            min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance
            and min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance
            and min(x3, x4) - tolerance <= px <= max(x3, x4) + tolerance
            and min(y3, y4) - tolerance <= py <= max(y3, y4) + tolerance
        ):
            return (px, py)
        return None

    def _merge_nearby_points(
        self,
        points: list[tuple[float, float]],
        tolerance: float,
    ) -> list[tuple[float, float]]:
        merged: list[list[tuple[float, float]]] = []
        for point in points:
            for cluster in merged:
                if any(self._distance(point, existing) <= tolerance for existing in cluster):
                    cluster.append(point)
                    break
            else:
                merged.append([point])
        return [
            (
                sum(point[0] for point in cluster) / len(cluster),
                sum(point[1] for point in cluster) / len(cluster),
            )
            for cluster in merged
        ]

    def _find_all_path_intersections(
        self,
        path_a: VisualPathSpec,
        path_b: VisualPathSpec,
    ) -> list[tuple[float, float]]:
        polyline_a = self._path_polyline(path_a.control_points, samples=160)
        polyline_b = self._path_polyline(path_b.control_points, samples=160)
        intersections: list[tuple[float, float]] = []
        for a0, a1 in zip(polyline_a, polyline_a[1:]):
            for b0, b1 in zip(polyline_b, polyline_b[1:]):
                intersection = self._segment_intersection(a0, a1, b0, b1)
                if intersection is not None:
                    intersections.append(intersection)
        return self._merge_nearby_points(intersections, tolerance=8.0)

    def _quadratic_bezier(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        t: float,
    ) -> tuple[float, float]:
        one_minus_t = 1.0 - t
        x = (one_minus_t ** 2) * p0[0] + 2 * one_minus_t * t * p1[0] + (t ** 2) * p2[0]
        y = (one_minus_t ** 2) * p0[1] + 2 * one_minus_t * t * p1[1] + (t ** 2) * p2[1]
        return x, y

    def _point_in_canvas(self, x: float, y: float, factors: StructureSensitiveVisualFactors) -> bool:
        radius = max(SIZE_TO_RADIUS.values())
        return (
            factors.margin + radius <= x <= factors.width - factors.margin - radius
            and factors.margin + radius <= y <= factors.height - factors.margin - radius
        )

    def _point_in_rect(
        self,
        x: float,
        y: float,
        rect: tuple[float, float, float, float],
    ) -> bool:
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    def _rects_overlap(
        self,
        rect_a: tuple[float, float, float, float],
        rect_b: tuple[float, float, float, float],
    ) -> bool:
        left_a, top_a, right_a, bottom_a = rect_a
        left_b, top_b, right_b, bottom_b = rect_b
        return not (
            right_a <= left_b
            or right_b <= left_a
            or bottom_a <= top_b
            or bottom_b <= top_a
        )

    def _point_too_close_to_region(
        self,
        x: float,
        y: float,
        rect: tuple[float, float, float, float],
        *,
        radius: float,
    ) -> bool:
        left, top, right, bottom = rect
        return (left - radius) <= x <= (right + radius) and (top - radius) <= y <= (bottom + radius)

    def _item_fully_inside_region(
        self,
        item: VisualItemSpec,
        region: VisualRegionSpec,
    ) -> bool:
        radius = SIZE_TO_RADIUS[item.size]
        frame_padding = 6
        left = region.x + radius + frame_padding
        top = region.y + radius + frame_padding
        right = region.x + region.width - radius - frame_padding
        bottom = region.y + region.height - radius - frame_padding
        return left <= item.x <= right and top <= item.y <= bottom

    def _item_touches_region(
        self,
        item: VisualItemSpec,
        region: VisualRegionSpec,
    ) -> bool:
        radius = SIZE_TO_RADIUS[item.size] + 4
        rect = (region.x, region.y, region.x + region.width, region.y + region.height)
        return self._point_too_close_to_region(item.x, item.y, rect, radius=radius)

    def _point_near_any_crossing(
        self,
        point: tuple[float, float],
        intersections: list[tuple[float, float]],
        clearance: float,
    ) -> bool:
        return any(self._distance(point, intersection) < clearance for intersection in intersections)

    def _item_covers_point(
        self,
        item: VisualItemSpec,
        point: tuple[float, float],
        *,
        extra_padding: float = 0.0,
    ) -> bool:
        effective_radius = self._shape_cover_radius(item) + extra_padding
        return self._distance((item.x, item.y), point) <= effective_radius

    def _item_overlaps_any_crossing(
        self,
        item: VisualItemSpec,
        intersections: list[tuple[float, float]],
        extra_padding: float = 0.0,
    ) -> bool:
        return any(
            self._item_covers_point(item, intersection, extra_padding=extra_padding)
            for intersection in intersections
        )

    def _shape_cover_radius(self, item: VisualItemSpec) -> float:
        radius = float(SIZE_TO_RADIUS[item.size])
        if item.shape == "square":
            return radius * math.sqrt(2.0)
        if item.shape == "triangle":
            return radius * 1.15
        return radius

    def _continuity_crossing_clearance(self) -> float:
        max_cover_radius = max(float(radius) * math.sqrt(2.0) for radius in SIZE_TO_RADIUS.values())
        return max_cover_radius + (PATH_STROKE_WIDTH / 2) + 10.0

    def _continuity_anchor_crossing_clearance(self) -> float:
        return self._continuity_crossing_clearance() + 20.0

    def _feature_domain(self, queried_feature: str) -> tuple[str, ...]:
        if queried_feature == "color":
            return COLORS
        if queried_feature == "shape":
            return SHAPES
        if queried_feature == "size":
            return SIZES
        raise ValueError(f"Unsupported queried_feature: {queried_feature}")

    def _item_index(self, item_id: str) -> int:
        return int(item_id.split("_")[-1])

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.dist(a, b)


def scene_to_scene_row(scene: StructureSensitiveVisualScene) -> dict[str, object]:
    factors = scene.factors
    generator = StructureSensitiveVisualGenerator()
    actual_counts = generator._actual_scene_counts(scene)
    return {
        "scene_id": scene.scene_id,
        "seed": scene.seed,
        "family": scene.family,
        "attentional_basis": scene.attentional_basis,
        "modality": scene.modality,
        "dimension": scene.dimension,
        "variant": scene.variant,
        "principle": scene.principle,
        "condition_type": scene.condition_type,
        "layout_pattern": scene.layout_pattern,
        "difficulty": scene.difficulty,
        "generator_case": scene.generator_case,
        "count_instruction": scene.count_instruction,
        "filter_instruction": scene.filter_instruction,
        "count_prompt": scene.count_prompt,
        "filter_prompt": scene.filter_prompt,
        "gold_count": scene.gold_count,
        "gold_indices": json.dumps(scene.gold_indices),
        "anchor_item_id": scene.anchor_item_id,
        "anchor_group_id": scene.anchor_group_id,
        "grouping_feature_primary": scene.grouping_feature_primary,
        "grouping_feature_secondary": scene.grouping_feature_secondary,
        "queried_feature": scene.queried_feature,
        "target_value": scene.target_value,
        "num_groups": factors.num_groups,
        "items_json": json.dumps([asdict(item) for item in scene.items], sort_keys=True),
        "regions_json": json.dumps([asdict(region) for region in scene.regions], sort_keys=True),
        "paths_json": json.dumps([asdict(path) for path in scene.paths], sort_keys=True),
        "metadata_json": json.dumps(scene.metadata, sort_keys=True),
        "planned_target_in_anchor_group": factors.target_in_anchor_group,
        "planned_target_outside_anchor_group": factors.target_outside_anchor_group,
        "planned_non_target_in_anchor_group": factors.non_target_in_anchor_group,
        **actual_counts,
        "min_items_per_group": factors.min_items_per_group,
        "max_items_per_group": factors.max_items_per_group,
        "width": factors.width,
        "height": factors.height,
        "margin": factors.margin,
        "min_gap": factors.min_gap,
        "jitter": factors.jitter,
        "cluster_spread": factors.cluster_spread,
        "inter_group_margin": factors.inter_group_margin,
        "stripe_count": factors.stripe_count,
        "shape_bias_strength": factors.shape_bias_strength,
        "path_crossing_count": factors.path_crossing_count,
        "region_count": factors.region_count,
        "loose_item_count": factors.loose_item_count,
    }
