#!/usr/bin/env python3
"""
Generate a benchmark structure diagram for the ABC Selective Attention Benchmark.

Outputs:
- abc_benchmark_structure.png
- abc_benchmark_structure.svg

The diagram highlights:
- the full benchmark design space
- the currently implemented slice used in the submission

Usage:
    python generate_abc_benchmark_structure_diagram.py -o out_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_box(ax, x, y, w, h, title, lines, highlighted=False, fontsize=11):
    rect = Rectangle(
        (x, y), w, h,
        fill=False,
        linewidth=2.2 if highlighted else 1.4,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=fontsize, fontweight="bold")
    body = "\n".join(lines)
    ax.text(x + 0.03, y + h - 0.12, body, ha="left", va="top", fontsize=fontsize - 1)


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )


def make_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.96,
        "ABC benchmark structure",
        ha="center", va="top", fontsize=18, fontweight="bold"
    )
    ax.text(
        0.5, 0.92,
        "Structured design space and implemented submission slice",
        ha="center", va="top", fontsize=11
    )

    # Left: design space
    draw_box(
        ax, 0.05, 0.58, 0.25, 0.26,
        "1. Attention family",
        [
            "• selective",
            "• sustained",
            "• shifting",
            "• divided",
        ],
        highlighted=True,
    )

    draw_box(
        ax, 0.37, 0.58, 0.25, 0.26,
        "2. Attentional basis",
        [
            "• feature-sensitive",
            "• structure-sensitive",
        ],
        highlighted=True,
    )

    draw_box(
        ax, 0.69, 0.58, 0.25, 0.26,
        "3. Modality",
        [
            "• text",
            "• visual",
            "• mixed (future)",
        ],
        highlighted=True,
    )

    draw_box(
        ax, 0.37, 0.24, 0.25, 0.20,
        "4. Task type",
        [
            "• filtering",
            "• counting",
        ],
        highlighted=True,
    )

    # Connecting arrows
    draw_arrow(ax, 0.175, 0.58, 0.175, 0.46)
    draw_arrow(ax, 0.495, 0.58, 0.495, 0.46)
    draw_arrow(ax, 0.815, 0.58, 0.815, 0.46)

    # Mid strip: implemented slice
    draw_box(
        ax, 0.13, 0.06, 0.74, 0.12,
        "Implemented in this submission",
        [
            "Selective attention × {feature-sensitive, structure-sensitive} × {text, visual} × {filtering, counting}",
        ],
        highlighted=True,
        fontsize=12,
    )

    draw_arrow(ax, 0.50, 0.24, 0.50, 0.18)

    # Right side note: task groups
    draw_box(
        ax, 0.69, 0.24, 0.25, 0.20,
        "Task groups",
        [
            "Each basis × modality × task type",
            "combination defines a task group.",
            "",
            "Each group contains:",
            "• dimensions",
            "• variants",
            "• baseline",
        ],
        highlighted=False,
        fontsize=10,
    )

    # Bottom left note: per-example structure
    draw_box(
        ax, 0.05, 0.24, 0.25, 0.20,
        "Per-example structure",
        [
            "input",
            "→ selection rule",
            "→ output",
            "",
            "output = filtering or counting",
        ],
        highlighted=False,
        fontsize=10,
    )

    fig.tight_layout()

    png_path = out_dir / "abc_benchmark_structure.png"
    svg_path = out_dir / "abc_benchmark_structure.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("abc_structure_diagram"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_diagram(args.out_dir)
    print(f"Wrote diagram to: {args.out_dir}")


if __name__ == "__main__":
    main()
