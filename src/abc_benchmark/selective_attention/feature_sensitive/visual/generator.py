from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

ColorName = Literal["red", "blue", "green", "yellow"]
ShapeName = Literal["circle", "square", "triangle"]
SizeName = Literal["small", "large"]
RuleVariant = Literal[
    "color_only",
    "shape_only",
    "color_shape",
    "color_shape_size",
]
LayoutRegularity = Literal["random", "grid", "clustered"]
SpatialDensityName = Literal["sparse", "medium", "dense", "crowded"]
DimensionName = Literal[
    "baseline",
    "set_size",
    "rule_arity",
    "confound",
    "spatial_density",
    "target_count",
    "layout_regularity",
    "target_count_x_confound",
    "combined",
]
FamilyName = Literal["selective_attention"]
AttentionalBasisName = Literal["feature_sensitive"]
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
DENSITY_TO_GAP: dict[SpatialDensityName, int] = {
    "sparse": 56,
    "medium": 36,
    "dense": 22,
    "crowded": 10,
}
DENSITY_TO_ACTIVE_AREA_SCALE: dict[SpatialDensityName, float] = {
    "sparse": 1.00,
    "medium": 0.80,
    "dense": 0.65,
    "crowded": 0.50,
}


class GenerationError(RuntimeError):
    """Raised when a valid scene cannot be generated within the retry budget."""


@dataclass(frozen=True)
class VisualItemSpec:
    x: float
    y: float
    color: ColorName
    shape: ShapeName
    size: SizeName
    is_target: bool
    role: str


@dataclass(frozen=True)
class FeatureSensitiveVisualFactors:
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    num_items: int
    rule_variant: RuleVariant
    target_count: int
    same_color_wrong_shape_count: int
    same_shape_wrong_color_count: int
    same_color_shape_wrong_size_count: int
    unrelated_count: int
    spatial_density: SpatialDensityName
    layout_regularity: LayoutRegularity
    fixed_size: SizeName | None
    active_area_scale: float
    width: int
    height: int
    margin: int
    min_gap: int
    jitter: int


@dataclass(frozen=True)
class FeatureSensitiveVisualScene:
    scene_id: str
    seed: int
    family: FamilyName
    attentional_basis: AttentionalBasisName
    modality: ModalityName
    dimension: DimensionName
    variant: str
    count_instruction: str
    filter_instruction: str
    count_prompt: str
    filter_prompt: str
    gold_count: int
    gold_indices: list[int]
    target_definition: dict[str, str]
    factors: FeatureSensitiveVisualFactors
    items: list[VisualItemSpec]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["items"] = [asdict(item) for item in self.items]
        return payload


def make_scene_id(dimension: str, variant: str, seed: int) -> str:
    return f"feature_visual__{dimension}__{variant}__{seed}"


class FeatureSensitiveVisualGenerator:
    def __init__(self, rng: random.Random | None = None, max_attempts: int = 1000) -> None:
        self.rng = rng or random.Random()
        self.max_attempts = max_attempts
        self._font = ImageFont.load_default()

    def generate(
        self,
        *,
        seed: int,
        factors: FeatureSensitiveVisualFactors | None = None,
        dimension: DimensionName = "combined",
        variant: str = "medium",
        target_count_override: int | None = None,
    ) -> FeatureSensitiveVisualScene:
        local_rng = random.Random(seed)
        if factors is None:
            factors = self.sample_factors(rng=local_rng, dimension=dimension, variant=variant)
        if target_count_override is not None:
            factors = self._override_target_count(factors, target_count_override)

        for _ in range(self.max_attempts):
            target_definition = self._sample_target_definition(local_rng, factors.rule_variant)
            positions = self._sample_positions(local_rng, factors)
            if positions is None:
                continue

            items = self._build_items(local_rng, factors, positions, target_definition)
            if not self._passes_constraints(items, factors, target_definition):
                continue

            count_instruction = self._build_count_instruction(factors.rule_variant, target_definition)
            filter_instruction = self._build_filter_instruction(factors.rule_variant, target_definition)
            gold_indices = self._matching_item_indices(items, target_definition, factors.rule_variant)

            return FeatureSensitiveVisualScene(
                scene_id=make_scene_id(factors.dimension, factors.variant, seed),
                seed=seed,
                family=factors.family,
                attentional_basis=factors.attentional_basis,
                modality=factors.modality,
                dimension=factors.dimension,
                variant=factors.variant,
                count_instruction=count_instruction,
                filter_instruction=filter_instruction,
                count_prompt=count_instruction,
                filter_prompt=filter_instruction,
                gold_count=len(gold_indices),
                gold_indices=gold_indices,
                target_definition=target_definition,
                factors=factors,
                items=items,
            )

        raise GenerationError(
            "Failed to generate a valid feature-sensitive visual scene within max_attempts"
        )

    def sample_factors(
        self,
        *,
        rng: random.Random,
        dimension: DimensionName,
        variant: str,
    ) -> FeatureSensitiveVisualFactors:
        if dimension == "baseline":
            num_items = 6
            target_count = rng.randint(1, 2)
            same_color_wrong_shape_count = 1
            same_shape_wrong_color_count = 1
            unrelated_count = num_items - target_count - same_color_wrong_shape_count - same_shape_wrong_color_count
            return self._base_factors(
                dimension="baseline",
                variant="baseline",
                num_items=num_items,
                rule_variant="color_shape",
                target_count=target_count,
                same_color_wrong_shape_count=same_color_wrong_shape_count,
                same_shape_wrong_color_count=same_shape_wrong_color_count,
                same_color_shape_wrong_size_count=0,
                unrelated_count=unrelated_count,
                spatial_density="sparse",
                layout_regularity="random",
                fixed_size="large",
                active_area_scale=1.0,
                min_gap=56,
                jitter=10,
            )

        if dimension == "set_size":
            mapping = {
                "xs": (5, 1),
                "s": (6, 1),
                "m": (10, 2),
                "l": (16, 3),
            }
            if variant not in mapping:
                raise ValueError(f"Unknown set_size variant: {variant}")
            num_items, target_count = mapping[variant]
            same_color_wrong_shape_count = 1
            same_shape_wrong_color_count = 1
            unrelated_count = num_items - target_count - same_color_wrong_shape_count - same_shape_wrong_color_count
            if unrelated_count < 2:
                raise GenerationError("set_size configuration leaves too few unrelated items")
            return self._base_factors(
                dimension="set_size",
                variant=variant,
                num_items=num_items,
                rule_variant="color_shape",
                target_count=target_count,
                same_color_wrong_shape_count=same_color_wrong_shape_count,
                same_shape_wrong_color_count=same_shape_wrong_color_count,
                same_color_shape_wrong_size_count=0,
                unrelated_count=unrelated_count,
                spatial_density="medium",
                layout_regularity="random",
                fixed_size="large",
                active_area_scale=1.0,
                min_gap=36,
                jitter=12,
            )

        if dimension == "rule_arity":
            valid = {"color_only", "shape_only", "color_shape", "color_shape_size"}
            if variant not in valid:
                raise ValueError(f"Unknown rule_arity variant: {variant}")
            same_color_wrong_shape_count = 2 if variant in {"color_shape", "color_shape_size"} else 0
            same_shape_wrong_color_count = 2 if variant in {"color_shape", "color_shape_size"} else 0
            same_color_shape_wrong_size_count = 2 if variant == "color_shape_size" else 0
            num_items = 12
            unrelated_count = (
                num_items
                - 3
                - same_color_wrong_shape_count
                - same_shape_wrong_color_count
                - same_color_shape_wrong_size_count
            )
            return self._base_factors(
                dimension="rule_arity",
                variant=variant,
                num_items=num_items,
                rule_variant=variant,
                target_count=3,
                same_color_wrong_shape_count=same_color_wrong_shape_count,
                same_shape_wrong_color_count=same_shape_wrong_color_count,
                same_color_shape_wrong_size_count=same_color_shape_wrong_size_count,
                unrelated_count=unrelated_count,
                spatial_density="medium",
                layout_regularity="random",
                fixed_size=None,
                active_area_scale=1.0,
                min_gap=36,
                jitter=12,
            )

        if dimension == "confound":
            mapping = {
                "low": (12, 1, 1, 1, 6),
                "medium": (14, 2, 2, 1, 6),
                "high": (18, 4, 4, 2, 5),
                "extreme": (22, 5, 5, 3, 6),
            }
            if variant not in mapping:
                raise ValueError(f"Unknown confound variant: {variant}")
            num_items, scws, sswc, scsws, unrelated = mapping[variant]
            return self._base_factors(
                dimension="confound",
                variant=variant,
                num_items=num_items,
                rule_variant="color_shape_size",
                target_count=3,
                same_color_wrong_shape_count=scws,
                same_shape_wrong_color_count=sswc,
                same_color_shape_wrong_size_count=scsws,
                unrelated_count=unrelated,
                spatial_density="medium",
                layout_regularity="random",
                fixed_size=None,
                active_area_scale=1.0,
                min_gap=36,
                jitter=12,
            )

        if dimension == "spatial_density":
            if variant not in DENSITY_TO_GAP:
                raise ValueError(f"Unknown spatial_density variant: {variant}")
            return self._base_factors(
                dimension="spatial_density",
                variant=variant,
                num_items=12,
                rule_variant="color_shape",
                target_count=3,
                same_color_wrong_shape_count=2,
                same_shape_wrong_color_count=2,
                same_color_shape_wrong_size_count=0,
                unrelated_count=5,
                spatial_density=variant,
                layout_regularity="random",
                fixed_size="large",
                active_area_scale=DENSITY_TO_ACTIVE_AREA_SCALE[variant],
                min_gap=DENSITY_TO_GAP[variant],
                jitter=12,
            )

        if dimension == "target_count":
            mapping = {"0": 0, "1": 1, "3": 3, "5": 5}
            if variant not in mapping:
                raise ValueError(f"Unknown target_count variant: {variant}")
            target_count = mapping[variant]
            num_items = 14
            same_color_wrong_shape_count = 3
            same_shape_wrong_color_count = 3
            unrelated_count = num_items - target_count - same_color_wrong_shape_count - same_shape_wrong_color_count
            if unrelated_count < 2:
                raise GenerationError("target_count configuration leaves too few unrelated items")
            return self._base_factors(
                dimension="target_count",
                variant=variant,
                num_items=num_items,
                rule_variant="color_shape",
                target_count=target_count,
                same_color_wrong_shape_count=same_color_wrong_shape_count,
                same_shape_wrong_color_count=same_shape_wrong_color_count,
                same_color_shape_wrong_size_count=0,
                unrelated_count=unrelated_count,
                spatial_density="medium",
                layout_regularity="random",
                fixed_size="large",
                active_area_scale=1.0,
                min_gap=36,
                jitter=12,
            )

        if dimension == "layout_regularity":
            valid = {"random", "grid", "clustered"}
            if variant not in valid:
                raise ValueError(f"Unknown layout_regularity variant: {variant}")
            if variant == "clustered":
                return self._base_factors(
                    dimension="layout_regularity",
                    variant="clustered",
                    num_items=12,
                    rule_variant="color_shape",
                    target_count=3,
                    same_color_wrong_shape_count=2,
                    same_shape_wrong_color_count=2,
                    same_color_shape_wrong_size_count=0,
                    unrelated_count=5,
                    spatial_density="medium",
                    layout_regularity="clustered",
                    fixed_size="large",
                    active_area_scale=1.0,
                    min_gap=20,
                    jitter=4,
                )
            return self._base_factors(
                dimension="layout_regularity",
                variant=variant,
                num_items=12,
                rule_variant="color_shape",
                target_count=3,
                same_color_wrong_shape_count=2,
                same_shape_wrong_color_count=2,
                same_color_shape_wrong_size_count=0,
                unrelated_count=5,
                spatial_density="medium",
                layout_regularity=variant,
                fixed_size="large",
                active_area_scale=1.0,
                min_gap=36,
                jitter=0 if variant == "grid" else 10,
            )

        if dimension == "target_count_x_confound":
            mapping = {
                "0_low": (0, 1, 1, 0, 10),
                "0_medium": (0, 2, 2, 1, 9),
                "0_extreme": (0, 5, 5, 3, 7),
                "3_low": (3, 1, 1, 1, 6),
                "3_medium": (3, 2, 2, 1, 6),
                "3_extreme": (3, 5, 5, 3, 6),
            }
            if variant not in mapping:
                raise ValueError(f"Unknown target_count_x_confound variant: {variant}")
            target_count, scws, sswc, scsws, unrelated = mapping[variant]
            return self._base_factors(
                dimension="target_count_x_confound",
                variant=variant,
                num_items=target_count + scws + sswc + scsws + unrelated,
                rule_variant="color_shape_size",
                target_count=target_count,
                same_color_wrong_shape_count=scws,
                same_shape_wrong_color_count=sswc,
                same_color_shape_wrong_size_count=scsws,
                unrelated_count=unrelated,
                spatial_density="medium",
                layout_regularity="random",
                fixed_size=None,
                active_area_scale=1.0,
                min_gap=36,
                jitter=12,
            )

        if dimension == "combined":
            mapping = {
                "easy": dict(
                    num_items=10,
                    rule_variant="color_shape",
                    target_count=2,
                    scws=1,
                    sswc=1,
                    scsws=0,
                    unrelated=6,
                    density="sparse",
                    layout="random",
                    fixed_size="large",
                    active_area_scale=1.0,
                    gap=42,
                ),
                "medium": dict(
                    num_items=14,
                    rule_variant="color_shape_size",
                    target_count=3,
                    scws=2,
                    sswc=2,
                    scsws=1,
                    unrelated=6,
                    density="medium",
                    layout="random",
                    fixed_size=None,
                    active_area_scale=1.0,
                    gap=36,
                ),
                "hard": dict(
                    num_items=18,
                    rule_variant="color_shape_size",
                    target_count=5,
                    scws=4,
                    sswc=4,
                    scsws=2,
                    unrelated=3,
                    density="dense",
                    layout="random",
                    fixed_size=None,
                    active_area_scale=0.85,
                    gap=22,
                ),
            }
            if variant not in mapping:
                raise ValueError(f"Unknown combined variant: {variant}")
            cfg = mapping[variant]
            return self._base_factors(
                dimension="combined",
                variant=variant,
                num_items=cfg["num_items"],
                rule_variant=cfg["rule_variant"],
                target_count=cfg["target_count"],
                same_color_wrong_shape_count=cfg["scws"],
                same_shape_wrong_color_count=cfg["sswc"],
                same_color_shape_wrong_size_count=cfg["scsws"],
                unrelated_count=cfg["unrelated"],
                spatial_density=cfg["density"],
                layout_regularity=cfg["layout"],
                fixed_size=cfg["fixed_size"],
                active_area_scale=cfg["active_area_scale"],
                min_gap=cfg["gap"],
                jitter=10,
            )

        raise ValueError(f"Unknown dimension: {dimension}")

    def render(
        self,
        scene: FeatureSensitiveVisualScene,
        output_path: str | Path | None = None,
        *,
        show_item_ids: bool = False,
    ) -> Image.Image:
        factors = scene.factors
        image = Image.new("RGB", (factors.width, factors.height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for index, item in enumerate(scene.items, start=1):
            self._draw_item(draw, item)
            if show_item_ids:
                self._draw_item_id(draw, item, index, factors)

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        return image

    def _draw_item(self, draw: ImageDraw.ImageDraw, item: VisualItemSpec) -> None:
        radius = SIZE_TO_RADIUS[item.size]
        color = COLOR_TO_RGB[item.color]
        bbox = [item.x - radius, item.y - radius, item.x + radius, item.y + radius]

        if item.shape == "circle":
            draw.ellipse(bbox, fill=color, outline=(30, 30, 30), width=2)
            return
        if item.shape == "square":
            draw.rectangle(bbox, fill=color, outline=(30, 30, 30), width=2)
            return
        if item.shape == "triangle":
            points = [
                (item.x, item.y - radius),
                (item.x - radius * 0.9, item.y + radius * 0.8),
                (item.x + radius * 0.9, item.y + radius * 0.8),
            ]
            draw.polygon(points, fill=color, outline=(30, 30, 30))
            return
        raise ValueError(f"Unsupported shape: {item.shape}")

    def _draw_item_id(
        self,
        draw: ImageDraw.ImageDraw,
        item: VisualItemSpec,
        item_index: int,
        factors: FeatureSensitiveVisualFactors,
    ) -> None:
        radius = SIZE_TO_RADIUS[item.size]
        label = str(item_index)
        stroke_width = 2
        text_bbox = draw.textbbox((0, 0), label, font=self._font, stroke_width=stroke_width)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        pad = 4
        gap = 2
        side_offset = radius + gap
        vertical_nudge = text_h * 0.35
        candidates = [
            (item.x + side_offset, item.y - vertical_nudge),
            (item.x - side_offset - text_w, item.y - vertical_nudge),
            (item.x + side_offset, item.y - text_h + gap),
            (item.x - side_offset - text_w, item.y - text_h + gap),
            (item.x + side_offset, item.y - gap),
            (item.x - side_offset - text_w, item.y - gap),
        ]

        def fits(x: float, y: float) -> bool:
            return (
                pad <= x <= factors.width - text_w - pad
                and pad <= y <= factors.height - text_h - pad
            )

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

    def _base_factors(
        self,
        *,
        dimension: DimensionName,
        variant: str,
        num_items: int,
        rule_variant: RuleVariant,
        target_count: int,
        same_color_wrong_shape_count: int,
        same_shape_wrong_color_count: int,
        same_color_shape_wrong_size_count: int,
        unrelated_count: int,
        spatial_density: SpatialDensityName,
        layout_regularity: LayoutRegularity,
        fixed_size: SizeName | None,
        active_area_scale: float,
        min_gap: int,
        jitter: int,
    ) -> FeatureSensitiveVisualFactors:
        return FeatureSensitiveVisualFactors(
            family="selective_attention",
            attentional_basis="feature_sensitive",
            modality="visual",
            dimension=dimension,
            variant=variant,
            num_items=num_items,
            rule_variant=rule_variant,
            target_count=target_count,
            same_color_wrong_shape_count=same_color_wrong_shape_count,
            same_shape_wrong_color_count=same_shape_wrong_color_count,
            same_color_shape_wrong_size_count=same_color_shape_wrong_size_count,
            unrelated_count=unrelated_count,
            spatial_density=spatial_density,
            layout_regularity=layout_regularity,
            fixed_size=fixed_size,
            active_area_scale=active_area_scale,
            width=640,
            height=480,
            margin=36,
            min_gap=min_gap,
            jitter=jitter,
        )

    def _override_target_count(
        self,
        factors: FeatureSensitiveVisualFactors,
        target_count_override: int,
    ) -> FeatureSensitiveVisualFactors:
        if target_count_override < 0:
            raise ValueError("target_count_override must be >= 0")
        delta = target_count_override - factors.target_count
        return replace(
            factors,
            target_count=target_count_override,
            num_items=factors.num_items + max(0, delta),
            unrelated_count=factors.unrelated_count + max(0, -delta),
        )

    def _sample_target_definition(
        self,
        rng: random.Random,
        rule_variant: RuleVariant,
    ) -> dict[str, str]:
        target = {
            "color": rng.choice(COLORS),
            "shape": rng.choice(SHAPES),
            "size": rng.choice(SIZES),
        }
        if rule_variant == "color_only":
            return {"color": target["color"]}
        if rule_variant == "shape_only":
            return {"shape": target["shape"]}
        if rule_variant == "color_shape":
            return {"color": target["color"], "shape": target["shape"]}
        return target

    def _sampling_bounds(
        self,
        factors: FeatureSensitiveVisualFactors,
    ) -> tuple[float, float, float, float]:
        radius = max(SIZE_TO_RADIUS.values())
        usable_left = factors.margin + radius
        usable_right = factors.width - factors.margin - radius
        usable_top = factors.margin + radius
        usable_bottom = factors.height - factors.margin - radius

        usable_w = usable_right - usable_left
        usable_h = usable_bottom - usable_top
        scale = max(0.30, min(1.0, factors.active_area_scale))
        active_w = usable_w * scale
        active_h = usable_h * scale

        left = usable_left + (usable_w - active_w) / 2
        right = usable_right - (usable_w - active_w) / 2
        top = usable_top + (usable_h - active_h) / 2
        bottom = usable_bottom - (usable_h - active_h) / 2
        return left, right, top, bottom

    def _sample_positions(
        self,
        rng: random.Random,
        factors: FeatureSensitiveVisualFactors,
    ) -> list[tuple[float, float]] | None:
        if factors.layout_regularity == "grid":
            return self._grid_positions(rng, factors)
        if factors.layout_regularity == "clustered":
            return self._clustered_positions(rng, factors)
        return self._random_positions(rng, factors)

    def _random_positions(
        self,
        rng: random.Random,
        factors: FeatureSensitiveVisualFactors,
    ) -> list[tuple[float, float]] | None:
        positions: list[tuple[float, float]] = []
        radius = max(SIZE_TO_RADIUS.values())
        left, right, top, bottom = self._sampling_bounds(factors)

        for _ in range(factors.num_items):
            placed = False
            for _ in range(500):
                x = rng.uniform(left, right)
                y = rng.uniform(top, bottom)
                if any(self._distance((x, y), p) < factors.min_gap + 2 * radius for p in positions):
                    continue
                positions.append((x, y))
                placed = True
                break
            if not placed:
                return None
        return positions

    def _grid_positions(
        self,
        rng: random.Random,
        factors: FeatureSensitiveVisualFactors,
    ) -> list[tuple[float, float]] | None:
        cols = math.ceil(math.sqrt(factors.num_items))
        rows = math.ceil(factors.num_items / cols)
        left, right, top, bottom = self._sampling_bounds(factors)
        usable_w = right - left
        usable_h = bottom - top
        cell_w = usable_w / cols
        cell_h = usable_h / rows
        max_radius = max(SIZE_TO_RADIUS.values())
        if min(cell_w, cell_h) < 2 * max_radius + factors.min_gap:
            return None

        positions: list[tuple[float, float]] = []
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= factors.num_items:
                    break
                x = left + cell_w * (col + 0.5)
                y = top + cell_h * (row + 0.5)
                if factors.jitter:
                    x += rng.uniform(-min(factors.jitter, cell_w / 5), min(factors.jitter, cell_w / 5))
                    y += rng.uniform(-min(factors.jitter, cell_h / 5), min(factors.jitter, cell_h / 5))
                positions.append((x, y))
        return positions

    def _clustered_positions(
        self,
        rng: random.Random,
        factors: FeatureSensitiveVisualFactors,
    ) -> list[tuple[float, float]] | None:
        left, right, top, bottom = self._sampling_bounds(factors)
        width = right - left
        height = bottom - top
        radius = max(SIZE_TO_RADIUS.values())

        cluster_boxes = [
            (left + 0.06 * width, left + 0.34 * width, top + 0.06 * height, top + 0.34 * height),
            (left + 0.66 * width, left + 0.94 * width, top + 0.06 * height, top + 0.34 * height),
            (left + 0.06 * width, left + 0.34 * width, top + 0.66 * height, top + 0.94 * height),
            (left + 0.66 * width, left + 0.94 * width, top + 0.66 * height, top + 0.94 * height),
        ]

        base = factors.num_items // 4
        remainder = factors.num_items % 4
        cluster_counts = [base] * 4
        for cluster_index in range(remainder):
            cluster_counts[cluster_index] += 1

        positions: list[tuple[float, float]] = []
        local_jitter = min(factors.jitter, 4)

        for (box_left, box_right, box_top, box_bottom), count in zip(cluster_boxes, cluster_counts):
            inner_left = box_left + radius
            inner_right = box_right - radius
            inner_top = box_top + radius
            inner_bottom = box_bottom - radius
            if inner_left >= inner_right or inner_top >= inner_bottom:
                return None

            for _ in range(count):
                placed = False
                for _ in range(800):
                    x = rng.uniform(inner_left, inner_right)
                    y = rng.uniform(inner_top, inner_bottom)
                    if local_jitter:
                        x += rng.uniform(-local_jitter, local_jitter)
                        y += rng.uniform(-local_jitter, local_jitter)
                        x = min(max(x, inner_left), inner_right)
                        y = min(max(y, inner_top), inner_bottom)
                    if any(self._distance((x, y), p) < factors.min_gap + 2 * radius for p in positions):
                        continue
                    positions.append((x, y))
                    placed = True
                    break
                if not placed:
                    return None
        return positions

    def _build_items(
        self,
        rng: random.Random,
        factors: FeatureSensitiveVisualFactors,
        positions: list[tuple[float, float]],
        target_definition: dict[str, str],
    ) -> list[VisualItemSpec]:
        if len(positions) != factors.num_items:
            raise GenerationError("Position count does not match num_items")

        indices = list(range(factors.num_items))
        rng.shuffle(indices)
        items: list[VisualItemSpec | None] = [None] * factors.num_items
        fixed_size = factors.fixed_size

        cursor = 0
        target_indices = indices[cursor : cursor + factors.target_count]
        cursor += factors.target_count
        same_color_wrong_shape_indices = indices[cursor : cursor + factors.same_color_wrong_shape_count]
        cursor += factors.same_color_wrong_shape_count
        same_shape_wrong_color_indices = indices[cursor : cursor + factors.same_shape_wrong_color_count]
        cursor += factors.same_shape_wrong_color_count
        same_color_shape_wrong_size_indices = indices[cursor : cursor + factors.same_color_shape_wrong_size_count]
        cursor += factors.same_color_shape_wrong_size_count
        filler_indices = indices[cursor:]

        for idx in target_indices:
            x, y = positions[idx]
            items[idx] = VisualItemSpec(
                x=x,
                y=y,
                color=target_definition.get("color", rng.choice(COLORS)),
                shape=target_definition.get("shape", rng.choice(SHAPES)),
                size=target_definition.get("size", fixed_size or rng.choice(SIZES)),
                is_target=True,
                role="target",
            )

        for idx in same_color_wrong_shape_indices:
            x, y = positions[idx]
            items[idx] = VisualItemSpec(
                x=x,
                y=y,
                color=target_definition.get("color", rng.choice(COLORS)),
                shape=self._different_choice(rng, SHAPES, target_definition.get("shape")),
                size=target_definition.get("size", fixed_size or rng.choice(SIZES)),
                is_target=False,
                role="same_color_wrong_shape",
            )

        for idx in same_shape_wrong_color_indices:
            x, y = positions[idx]
            items[idx] = VisualItemSpec(
                x=x,
                y=y,
                color=self._different_choice(rng, COLORS, target_definition.get("color")),
                shape=target_definition.get("shape", rng.choice(SHAPES)),
                size=target_definition.get("size", fixed_size or rng.choice(SIZES)),
                is_target=False,
                role="same_shape_wrong_color",
            )

        for idx in same_color_shape_wrong_size_indices:
            x, y = positions[idx]
            items[idx] = VisualItemSpec(
                x=x,
                y=y,
                color=target_definition["color"],
                shape=target_definition["shape"],
                size=self._different_choice(rng, SIZES, target_definition.get("size")),
                is_target=False,
                role="same_color_shape_wrong_size",
            )

        for idx in filler_indices:
            x, y = positions[idx]
            items[idx] = self._sample_unrelated_item(
                rng,
                x=x,
                y=y,
                target_definition=target_definition,
                fixed_size=fixed_size,
            )

        return [item for item in items if item is not None]

    def _sample_unrelated_item(
        self,
        rng: random.Random,
        *,
        x: float,
        y: float,
        target_definition: dict[str, str],
        fixed_size: SizeName | None,
    ) -> VisualItemSpec:
        for _ in range(100):
            candidate = {
                "color": rng.choice(COLORS),
                "shape": rng.choice(SHAPES),
                "size": fixed_size or rng.choice(SIZES),
            }
            if all(candidate[field] != value for field, value in target_definition.items()):
                return VisualItemSpec(
                    x=x,
                    y=y,
                    color=candidate["color"],
                    shape=candidate["shape"],
                    size=candidate["size"],
                    is_target=False,
                    role="unrelated",
                )
        raise GenerationError("Failed to sample unrelated filler item")

    def _different_choice(
        self,
        rng: random.Random,
        values: tuple[str, ...],
        current: str | None,
    ) -> str:
        if current is None:
            return rng.choice(values)
        return rng.choice([value for value in values if value != current])

    def _matching_item_indices(
        self,
        items: list[VisualItemSpec],
        target_definition: dict[str, str],
        rule_variant: RuleVariant,
    ) -> list[int]:
        return [
            index + 1
            for index, item in enumerate(items)
            if self._matches_target(item, target_definition, rule_variant)
        ]

    def _matches_target(
        self,
        item: VisualItemSpec,
        target_definition: dict[str, str],
        rule_variant: RuleVariant,
    ) -> bool:
        if rule_variant == "color_only":
            return item.color == target_definition["color"]
        if rule_variant == "shape_only":
            return item.shape == target_definition["shape"]
        if rule_variant == "color_shape":
            return item.color == target_definition["color"] and item.shape == target_definition["shape"]
        return (
            item.color == target_definition["color"]
            and item.shape == target_definition["shape"]
            and item.size == target_definition["size"]
        )

    def _passes_constraints(
        self,
        items: list[VisualItemSpec],
        factors: FeatureSensitiveVisualFactors,
        target_definition: dict[str, str],
    ) -> bool:
        gold_indices = self._matching_item_indices(items, target_definition, factors.rule_variant)
        if len(gold_indices) != factors.target_count:
            return False

        if factors.rule_variant in {"color_shape", "color_shape_size"}:
            if self._count_same_color_wrong_shape(items, target_definition) < factors.same_color_wrong_shape_count:
                return False
            if self._count_same_shape_wrong_color(items, target_definition) < factors.same_shape_wrong_color_count:
                return False
        if factors.rule_variant == "color_shape_size":
            if (
                self._count_same_color_shape_wrong_size(items, target_definition)
                < factors.same_color_shape_wrong_size_count
            ):
                return False
        if self._count_unrelated(items, target_definition) < factors.unrelated_count:
            return False

        for field in target_definition:
            simplified_count = self._simplified_rule_count(items, target_definition, drop_field=field)
            if factors.target_count == 0:
                if simplified_count <= 0:
                    return False
            elif simplified_count == factors.target_count:
                return False
        return True

    def _simplified_rule_count(
        self,
        items: list[VisualItemSpec],
        target_definition: dict[str, str],
        *,
        drop_field: str,
    ) -> int:
        count = 0
        for item in items:
            if all(
                getattr(item, field) == value
                for field, value in target_definition.items()
                if field != drop_field
            ):
                count += 1
        return count

    def _count_same_color_wrong_shape(
        self,
        items: list[VisualItemSpec],
        target_definition: dict[str, str],
    ) -> int:
        if "color" not in target_definition or "shape" not in target_definition:
            return 0
        count = 0
        for item in items:
            if item.color != target_definition["color"] or item.shape == target_definition["shape"]:
                continue
            if "size" in target_definition and item.size != target_definition["size"]:
                continue
            count += 1
        return count

    def _count_same_shape_wrong_color(
        self,
        items: list[VisualItemSpec],
        target_definition: dict[str, str],
    ) -> int:
        if "color" not in target_definition or "shape" not in target_definition:
            return 0
        count = 0
        for item in items:
            if item.shape != target_definition["shape"] or item.color == target_definition["color"]:
                continue
            if "size" in target_definition and item.size != target_definition["size"]:
                continue
            count += 1
        return count

    def _count_same_color_shape_wrong_size(
        self,
        items: list[VisualItemSpec],
        target_definition: dict[str, str],
    ) -> int:
        if "size" not in target_definition:
            return 0
        return sum(
            1
            for item in items
            if item.color == target_definition["color"]
            and item.shape == target_definition["shape"]
            and item.size != target_definition["size"]
        )

    def _count_unrelated(
        self,
        items: list[VisualItemSpec],
        target_definition: dict[str, str],
    ) -> int:
        return sum(
            1
            for item in items
            if all(getattr(item, field) != value for field, value in target_definition.items())
        )

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.dist(a, b)

    def _target_description(
        self,
        rule_variant: RuleVariant,
        target_definition: dict[str, str],
    ) -> str:
        ordered_parts: list[str] = []
        if "color" in target_definition:
            ordered_parts.append(target_definition["color"])
        if "size" in target_definition:
            ordered_parts.append(target_definition["size"])
        if "shape" in target_definition:
            ordered_parts.append(target_definition["shape"])
        if rule_variant == "shape_only":
            return f"{target_definition['shape']}s"
        return " ".join(ordered_parts + ["objects"])

    def _build_count_instruction(
        self,
        rule_variant: RuleVariant,
        target_definition: dict[str, str],
    ) -> str:
        description = self._target_description(rule_variant, target_definition)
        return (
            f"Count the {description} in the image.\n"
            'Respond with a JSON object of the form {"count": <integer>}.\n'
            "Rules:\n"
            '- "count" must be an integer\n'
            "- Apply the full rule exactly\n"
            "- Return only the JSON object"
        )

    def _build_filter_instruction(
        self,
        rule_variant: RuleVariant,
        target_definition: dict[str, str],
    ) -> str:
        description = self._target_description(rule_variant, target_definition)
        return (
            f"Return the 1-based item indices of the {description} in the image.\n"
            'Respond with a JSON object of the form {"indices": [<sorted unique integers>]}.\n'
            "Rules:\n"
            "- Use 1-based indexing\n"
            "- Sort ascending\n"
            "- Do not include duplicates\n"
            "- Apply the full rule exactly\n"
            "- Return only the JSON object"
        )


def scene_to_scene_row(scene: FeatureSensitiveVisualScene) -> dict[str, object]:
    factors = scene.factors
    return {
        "scene_id": scene.scene_id,
        "seed": scene.seed,
        "family": scene.family,
        "attentional_basis": scene.attentional_basis,
        "modality": scene.modality,
        "dimension": scene.dimension,
        "variant": scene.variant,
        "count_instruction": scene.count_instruction,
        "filter_instruction": scene.filter_instruction,
        "count_prompt": scene.count_prompt,
        "filter_prompt": scene.filter_prompt,
        "gold_count": scene.gold_count,
        "gold_indices": json.dumps(scene.gold_indices),
        "target_definition": json.dumps(scene.target_definition, sort_keys=True),
        "items_json": json.dumps([asdict(item) for item in scene.items], sort_keys=True),
        "num_items": factors.num_items,
        "rule_variant": factors.rule_variant,
        "target_count": factors.target_count,
        "same_color_wrong_shape_count": factors.same_color_wrong_shape_count,
        "same_shape_wrong_color_count": factors.same_shape_wrong_color_count,
        "same_color_shape_wrong_size_count": factors.same_color_shape_wrong_size_count,
        "unrelated_count": factors.unrelated_count,
        "spatial_density": factors.spatial_density,
        "layout_regularity": factors.layout_regularity,
        "fixed_size": factors.fixed_size,
        "active_area_scale": factors.active_area_scale,
        "width": factors.width,
        "height": factors.height,
        "margin": factors.margin,
        "min_gap": factors.min_gap,
        "jitter": factors.jitter,
    }
