from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image, ImageDraw, ImageFont

Shape = Literal["circle", "square", "triangle"]
ColorName = Literal["red", "blue", "green", "yellow"]

PALETTE: dict[ColorName, tuple[int, int, int]] = {
    "red": (220, 60, 60),
    "blue": (70, 120, 230),
    "green": (70, 170, 90),
    "yellow": (220, 180, 60),
}

SHAPES: tuple[Shape, ...] = ("circle", "square", "triangle")
COLORS: tuple[ColorName, ...] = tuple(PALETTE.keys())


@dataclass(frozen=True)
class DifficultyConfig:
    width: int = 640
    height: int = 480
    margin: int = 36
    num_clusters_min: int = 3
    num_clusters_max: int = 5
    cluster_size_min: int = 4
    cluster_size_max: int = 8
    cluster_radius: float = 40.0
    min_cluster_separation: float = 140.0
    object_radius: int = 10
    min_object_gap: float = 4.0
    max_attempts: int = 1000
    target_count_min: int = 0
    target_count_max: int = 5
    noise_items: int = 0


EASY = DifficultyConfig(
    num_clusters_min=3,
    num_clusters_max=4,
    cluster_size_min=4,
    cluster_size_max=6,
    cluster_radius=52.0,
    min_cluster_separation=160.0,
    noise_items=0,
)

MEDIUM = DifficultyConfig(
    num_clusters_min=4,
    num_clusters_max=5,
    cluster_size_min=5,
    cluster_size_max=7,
    cluster_radius=56.0,
    min_cluster_separation=145.0,
    noise_items=1,
)

HARD = DifficultyConfig(
    num_clusters_min=4,
    num_clusters_max=5,
    cluster_size_min=6,
    cluster_size_max=8,
    cluster_radius=60.0,
    min_cluster_separation=135.0,
    noise_items=1,
)


@dataclass
class ItemSpec:
    x: float
    y: float
    shape: Shape
    color: ColorName
    cluster_id: int
    is_anchor: bool = False


@dataclass
class SceneSpec:
    seed: int
    difficulty: str
    width: int
    height: int
    target_shape: Shape
    target_color: ColorName
    target_cluster_id: int
    gold_label: int
    items: list[ItemSpec]
    prompt: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["items"] = [asdict(i) for i in self.items]
        return payload


class GenerationError(RuntimeError):
    pass


class ClusterConstrainedCountingGenerator:
    def __init__(self, config: DifficultyConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()

    def generate(self, seed: int | None = None, difficulty_name: str = "custom") -> SceneSpec:
        local_rng = random.Random(seed) if seed is not None else self.rng

        for _ in range(self.config.max_attempts):
            centers = self._sample_cluster_centers(local_rng)
            if centers is None:
                continue

            items = self._sample_cluster_items(centers, local_rng)
            if items is None:
                continue

            target_cluster_id = local_rng.randrange(len(centers))
            target_shape = local_rng.choice(SHAPES)
            target_color = local_rng.choice(COLORS)

            items = self._assign_anchor(items, target_cluster_id, local_rng)
            items = self._enforce_target_count_constraints(
                items, target_cluster_id, target_shape, target_color, local_rng
            )
            items = self._add_noise_items(items, centers, local_rng)

            if not self._passes_geometry_checks(items):
                continue
            if not self._passes_anti_shortcut_constraints(
                items, target_cluster_id, target_shape, target_color
            ):
                continue

            gold = self._count_targets(items, target_cluster_id, target_shape, target_color)
            prompt = (
                f"Count the {target_color} {target_shape}s in the same cluster as the starred item. "
                "Ignore all other items. Respond with a number only."
            )

            return SceneSpec(
                seed=seed if seed is not None else -1,
                difficulty=difficulty_name,
                width=self.config.width,
                height=self.config.height,
                target_shape=target_shape,
                target_color=target_color,
                target_cluster_id=target_cluster_id,
                gold_label=gold,
                items=items,
                prompt=prompt,
            )

        raise GenerationError("Failed to generate a valid scene within max_attempts")

    def render(self, scene: SceneSpec, output_path: str | Path | None = None) -> Image.Image:
        img = Image.new("RGB", (scene.width, scene.height), (250, 250, 248))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for item in scene.items:
            self._draw_item(draw, item)

        for item in scene.items:
            if item.is_anchor:
                self._draw_anchor_marker(draw, item)
                break

        prompt_y = scene.height - 28
        draw.rectangle([(0, prompt_y - 6), (scene.width, scene.height)], fill=(245, 245, 240))
        draw.text((12, prompt_y), scene.prompt, fill=(30, 30, 30), font=font)

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
        return img

    def generate_many(self, count: int, difficulty_name: str = "custom", start_seed: int = 0) -> list[SceneSpec]:
        return [
            self.generate(seed=start_seed + i, difficulty_name=difficulty_name)
            for i in range(count)
        ]

    def _sample_cluster_centers(self, rng: random.Random) -> list[tuple[float, float]] | None:
        n = rng.randint(self.config.num_clusters_min, self.config.num_clusters_max)
        centers: list[tuple[float, float]] = []
        for _ in range(n):
            placed = False
            for _ in range(200):
                x = rng.uniform(
                    self.config.margin + self.config.cluster_radius,
                    self.config.width - self.config.margin - self.config.cluster_radius,
                )
                y = rng.uniform(
                    self.config.margin + self.config.cluster_radius,
                    self.config.height - self.config.margin - self.config.cluster_radius - 36,
                )
                if all(self._distance((x, y), c) >= self.config.min_cluster_separation for c in centers):
                    centers.append((x, y))
                    placed = True
                    break
            if not placed:
                return None
        return centers

    def _sample_cluster_items(self, centers: list[tuple[float, float]], rng: random.Random) -> list[ItemSpec] | None:
        items: list[ItemSpec] = []
        for cluster_id, center in enumerate(centers):
            cluster_size = rng.randint(self.config.cluster_size_min, self.config.cluster_size_max)
            cluster_points: list[tuple[float, float]] = []
            for _ in range(cluster_size):
                placed = False
                for _ in range(200):
                    angle = rng.uniform(0, 2 * math.pi)
                    radius = rng.uniform(0, self.config.cluster_radius)
                    x = center[0] + math.cos(angle) * radius
                    y = center[1] + math.sin(angle) * radius
                    if not self._within_bounds((x, y)):
                        continue
                    if any(
                        self._distance((x, y), p) < (2 * self.config.object_radius + self.config.min_object_gap)
                        for p in cluster_points
                    ):
                        continue
                    if any(
                        self._distance((x, y), (it.x, it.y)) < (2 * self.config.object_radius + self.config.min_object_gap)
                        for it in items
                    ):
                        continue
                    cluster_points.append((x, y))
                    items.append(
                        ItemSpec(
                            x=x,
                            y=y,
                            shape=rng.choice(SHAPES),
                            color=rng.choice(COLORS),
                            cluster_id=cluster_id,
                        )
                    )
                    placed = True
                    break
                if not placed:
                    return None
        return items

    def _assign_anchor(self, items: list[ItemSpec], target_cluster_id: int, rng: random.Random) -> list[ItemSpec]:
        eligible = [i for i, item in enumerate(items) if item.cluster_id == target_cluster_id]
        anchor_idx = rng.choice(eligible)
        updated = list(items)
        updated[anchor_idx] = ItemSpec(**{**asdict(updated[anchor_idx]), "is_anchor": True})
        return updated

    def _enforce_target_count_constraints(
        self,
        items: list[ItemSpec],
        target_cluster_id: int,
        target_shape: Shape,
        target_color: ColorName,
        rng: random.Random,
    ) -> list[ItemSpec]:
        updated = list(items)

        in_cluster = [i for i, it in enumerate(updated) if it.cluster_id == target_cluster_id]
        anchor_idx = next(
            i for i, it in enumerate(updated) if it.cluster_id == target_cluster_id and it.is_anchor
        )
        non_anchor_in_cluster = [i for i in in_cluster if i != anchor_idx]

        desired_total = rng.randint(self.config.target_count_min, self.config.target_count_max)
        desired_total = min(desired_total, len(in_cluster))

        include_anchor = rng.choice([True, False]) if desired_total > 0 else False
        if include_anchor:
            desired_non_anchor = max(0, desired_total - 1)
            updated[anchor_idx] = ItemSpec(
                updated[anchor_idx].x,
                updated[anchor_idx].y,
                target_shape,
                target_color,
                updated[anchor_idx].cluster_id,
                True,
            )
        else:
            desired_non_anchor = desired_total
            new_shape, new_color = self._sample_non_target_feature(target_shape, target_color, rng)
            updated[anchor_idx] = ItemSpec(
                updated[anchor_idx].x,
                updated[anchor_idx].y,
                new_shape,
                new_color,
                updated[anchor_idx].cluster_id,
                True,
            )

        desired_non_anchor = min(desired_non_anchor, len(non_anchor_in_cluster))

        rng.shuffle(non_anchor_in_cluster)
        chosen = set(non_anchor_in_cluster[:desired_non_anchor])

        for idx in non_anchor_in_cluster:
            item = updated[idx]
            if idx in chosen:
                updated[idx] = ItemSpec(
                    item.x, item.y, target_shape, target_color, item.cluster_id, item.is_anchor
                )
            else:
                new_shape, new_color = self._sample_non_target_feature(
                    target_shape, target_color, rng
                )
                updated[idx] = ItemSpec(
                    item.x, item.y, new_shape, new_color, item.cluster_id, item.is_anchor
                )

        outside = [i for i, it in enumerate(updated) if it.cluster_id != target_cluster_id]
        rng.shuffle(outside)
        if outside:
            idx = outside[0]
            item = updated[idx]
            updated[idx] = ItemSpec(
                item.x, item.y, target_shape, target_color, item.cluster_id, item.is_anchor
            )

        return updated

    def _add_noise_items(
        self,
        items: list[ItemSpec],
        centers: list[tuple[float, float]],
        rng: random.Random,
    ) -> list[ItemSpec]:
        updated = list(items)
        for _ in range(self.config.noise_items):
            placed = False
            for _ in range(200):
                x = rng.uniform(self.config.margin, self.config.width - self.config.margin)
                y = rng.uniform(self.config.margin, self.config.height - self.config.margin - 36)
                if any(
                    self._distance((x, y), (it.x, it.y)) < (2 * self.config.object_radius + self.config.min_object_gap)
                    for it in updated
                ):
                    continue
                if min(self._distance((x, y), c) for c in centers) < self.config.cluster_radius + 18:
                    continue
                updated.append(
                    ItemSpec(
                        x=x,
                        y=y,
                        shape=rng.choice(SHAPES),
                        color=rng.choice(COLORS),
                        cluster_id=-1,
                    )
                )
                placed = True
                break
            if not placed:
                break
        return updated

    def _passes_geometry_checks(self, items: list[ItemSpec]) -> bool:
        anchors = [it for it in items if it.is_anchor]
        if len(anchors) != 1:
            return False
        for i, a in enumerate(items):
            if not self._within_bounds((a.x, a.y)):
                return False
            for b in items[i + 1 :]:
                if self._distance((a.x, a.y), (b.x, b.y)) < (2 * self.config.object_radius + self.config.min_object_gap):
                    return False
        return True

    def _passes_anti_shortcut_constraints(
        self,
        items: list[ItemSpec],
        target_cluster_id: int,
        target_shape: Shape,
        target_color: ColorName,
    ) -> bool:
        correct = self._count_targets(items, target_cluster_id, target_shape, target_color)
        global_count = sum(1 for it in items if it.shape == target_shape and it.color == target_color)
        outside_count = sum(
            1 for it in items if it.cluster_id != target_cluster_id and it.shape == target_shape and it.color == target_color
        )
        inside_nontarget = sum(
            1
            for it in items
            if it.cluster_id == target_cluster_id and not it.is_anchor and not (it.shape == target_shape and it.color == target_color)
        )
        cluster_size = sum(1 for it in items if it.cluster_id == target_cluster_id and not it.is_anchor)

        if global_count == correct:
            return False
        if outside_count < 1:
            return False
        if inside_nontarget < 1:
            return False
        if cluster_size == correct:
            return False
        if correct < self.config.target_count_min or correct > self.config.target_count_max:
            return False
        return True

    def _count_targets(self, items: Iterable[ItemSpec], target_cluster_id: int, target_shape: Shape, target_color: ColorName) -> int:
        return sum(
            1
            for it in items
            if it.cluster_id == target_cluster_id
            and it.shape == target_shape
            and it.color == target_color
        )
    def _sample_non_target_feature(self, target_shape: Shape, target_color: ColorName, rng: random.Random) -> tuple[Shape, ColorName]:
        candidates = [(s, c) for s in SHAPES for c in COLORS if not (s == target_shape and c == target_color)]
        return rng.choice(candidates)

    def _draw_item(self, draw: ImageDraw.ImageDraw, item: ItemSpec) -> None:
        x, y = item.x, item.y
        r = self.config.object_radius
        fill = PALETTE[item.color]
        outline = (30, 30, 30)

        if item.shape == "circle":
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=fill, outline=outline, width=2)
        elif item.shape == "square":
            draw.rectangle([(x - r, y - r), (x + r, y + r)], fill=fill, outline=outline, width=2)
        elif item.shape == "triangle":
            pts = [(x, y - r), (x - 0.9 * r, y + 0.8 * r), (x + 0.9 * r, y + 0.8 * r)]
            draw.polygon(pts, fill=fill, outline=outline)
        else:
            raise ValueError(f"Unknown shape: {item.shape}")

    def _draw_anchor_marker(self, draw: ImageDraw.ImageDraw, item: ItemSpec) -> None:
        cx, cy = item.x, item.y
        outer = self.config.object_radius + 10
        inner = self.config.object_radius + 4
        points: list[tuple[float, float]] = []
        for i in range(10):
            angle = -math.pi / 2 + i * math.pi / 5
            radius = outer if i % 2 == 0 else inner
            points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        draw.polygon(points, outline=(20, 20, 20), width=2)

    def _within_bounds(self, point: tuple[float, float]) -> bool:
        x, y = point
        r = self.config.object_radius
        return (
            self.config.margin + r <= x <= self.config.width - self.config.margin - r
            and self.config.margin + r <= y <= self.config.height - self.config.margin - r - 36
        )

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.dist(a, b)


def make_generator(difficulty: str) -> ClusterConstrainedCountingGenerator:
    configs = {"easy": EASY, "medium": MEDIUM, "hard": HARD}
    try:
        return ClusterConstrainedCountingGenerator(configs[difficulty.lower()])
    except KeyError as exc:
        raise ValueError(f"Unknown difficulty: {difficulty}") from exc


def generate_sample_set(output_dir: str | Path, count_per_difficulty: int = 5, start_seed: int = 0) -> None:
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