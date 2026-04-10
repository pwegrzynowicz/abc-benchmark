from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

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
    target_cluster_id: int | None
    gold_label: int
    items: list[ItemSpec]
    prompt: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["items"] = [asdict(i) for i in self.items]
        return payload


class GenerationError(RuntimeError):
    pass


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def draw_item(
    draw: ImageDraw.ImageDraw,
    item: ItemSpec,
    object_radius: int,
) -> None:
    x, y = item.x, item.y
    r = object_radius
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


def draw_anchor_marker(
    draw: ImageDraw.ImageDraw,
    item: ItemSpec,
    object_radius: int,
) -> None:
    cx, cy = item.x, item.y
    outer = object_radius + 10
    inner = object_radius + 4
    points: list[tuple[float, float]] = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        radius = outer if i % 2 == 0 else inner
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    draw.polygon(points, outline=(20, 20, 20), width=2)


def render_scene(
    scene: SceneSpec,
    object_radius: int,
    output_path: str | Path | None = None,
) -> Image.Image:
    img = Image.new("RGB", (scene.width, scene.height), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for item in scene.items:
        draw_item(draw, item, object_radius=object_radius)

    for item in scene.items:
        if item.is_anchor:
            draw_anchor_marker(draw, item, object_radius=object_radius)
            break

    prompt_y = scene.height - 28
    draw.rectangle([(0, prompt_y - 6), (scene.width, scene.height)], fill=(245, 245, 240))
    draw.text((12, prompt_y), scene.prompt, fill=(30, 30, 30), font=font)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    return img