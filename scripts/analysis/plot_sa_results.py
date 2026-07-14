#!/usr/bin/env python3
"""
Generate charts for the ABC Selective Attention benchmark from extracted CSVs.

Inputs:
- runs_overall.csv
- dimension_variant_stats.csv

Usage:
    python plot_sa_results.py stats_dir -o charts_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_ORDER = [
    "selective attention - feature sensitive text filtering",
    "selective attention - feature sensitive text counting",
    "selective attention - feature sensitive visual filtering",
    "selective attention - feature sensitive visual counting",
    "selective attention - structure sensitive text filtering",
    "selective attention - structure sensitive text counting",
    "selective attention - structure sensitive visual filtering",
    "selective attention - structure sensitive visual counting",
]

TASK_SHORT = {
    "selective attention - feature sensitive text filtering": "FT\nfilter",
    "selective attention - feature sensitive text counting": "FT\ncount",
    "selective attention - feature sensitive visual filtering": "FV\nfilter",
    "selective attention - feature sensitive visual counting": "FV\ncount",
    "selective attention - structure sensitive text filtering": "ST\nfilter",
    "selective attention - structure sensitive text counting": "ST\ncount",
    "selective attention - structure sensitive visual filtering": "SV\nfilter",
    "selective attention - structure sensitive visual counting": "SV\ncount",
}

TOP_GROUPS = [
    ("feature-sensitive text", 0, 1),
    ("feature-sensitive visual", 2, 3),
    ("structure-sensitive text", 4, 5),
    ("structure-sensitive visual", 6, 7),
]


def shorten_model_label(label: str) -> str:
    s = str(label).lower()
    explicit = [
        ("gpt-5.4-mini", "GPT-5.4 mini"),
        ("gpt-5.4", "GPT-5.4"),
        ("gemini-3.1-pro-preview", "Gemini 3.1 Pro"),
        ("gemini-3-flash-preview", "Gemini 3 Flash"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
        ("claude-opus-4-6", "Claude Opus 4.6"),
        ("claude-opus-4-5", "Claude Opus 4.5"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
        ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
        ("glm-5", "GLM-5"),
        ("qwen3-235b-a22b", "Qwen3 235B"),
        ("qwen3-next-80b-a3b", "Qwen3 Next 80B"),
        ("deepseek-v3.2", "DeepSeek V3.2"),
    ]
    for needle, out in explicit:
        if needle in s:
            return out
    return str(label)


def make_heatmap(runs_csv: Path, out_path: Path, annotate: bool = True) -> None:
    df = pd.read_csv(runs_csv)
    pivot = df.pivot_table(index="model", columns="task_name", values="accuracy_pct", aggfunc="mean")
    pivot = pivot.reindex(columns=[c for c in TASK_ORDER if c in pivot.columns])
    pivot["__avg__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__avg__", ascending=False).drop(columns="__avg__")

    row_labels = [shorten_model_label(x) for x in pivot.index]
    col_labels = [TASK_SHORT.get(c, c) for c in pivot.columns]

    dupes = pd.Series(row_labels).duplicated(keep=False)
    if dupes.any():
        print("\nWARNING: duplicate heatmap labels detected:")
        for raw, short in zip(pivot.index, row_labels):
            print(f"  {raw} -> {short}")

    fig_h = max(6.5, 0.55 * len(row_labels))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_title("ABC benchmark results by model and task", fontsize=16, pad=30)

    for x in [1.5, 3.5, 5.5]:
        ax.axvline(x, color="white", linewidth=2.0)

    for label, start, end in TOP_GROUPS:
        center = (start + end) / 2
        ax.text(center, 1.04, label, ha="center", va="bottom", fontsize=10, transform=ax.get_xaxis_transform())

    if annotate:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=12)

    fig.subplots_adjust(left=0.28, right=0.96, top=0.86, bottom=0.10)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_gap_chart(runs_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(runs_csv)
    grouped = df.groupby(["basis_folder", "modality"])["accuracy_pct"].mean().reset_index()

    wanted = [
        ("feature_sensitive", "text", "feature\ntext"),
        ("feature_sensitive", "visual", "feature\nvisual"),
        ("structure_sensitive", "text", "structure\ntext"),
        ("structure_sensitive", "visual", "structure\nvisual"),
    ]

    labels, values = [], []
    for basis, modality, label in wanted:
        sub = grouped[(grouped["basis_folder"] == basis) & (grouped["modality"] == modality)]
        if not sub.empty:
            labels.append(label)
            values.append(float(sub.iloc[0]["accuracy_pct"]))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Average accuracy (%)")
    ax.set_title("Average accuracy by attentional basis and modality")
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def shorten_task_name(task_name: str) -> str:
    s = task_name.replace("selective attention - ", "")
    s = s.replace("feature sensitive", "feature")
    s = s.replace("structure sensitive", "structure")
    return s


def make_hardest_dimensions_chart(dv_csv: Path, out_path: Path, top_n: int = 8) -> None:
    df = pd.read_csv(dv_csv)
    grouped = (
        df.groupby(["task_name", "dimension", "variant"])["accuracy"]
        .mean()
        .reset_index()
        .sort_values("accuracy", ascending=True)
        .head(top_n)
    )

    labels = [f"{shorten_task_name(r['task_name'])}\n{r['dimension']} / {r['variant']}" for _, r in grouped.iterrows()]
    values = list(grouped["accuracy"] * 100.0)

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average accuracy (%)")
    ax.set_title("Hardest dimensions / variants")
    ax.set_xlim(0, max(100, max(values) + 5 if values else 100))

    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_hardest_by_group_chart(dv_csv: Path, out_path: Path, top_n: int = 3) -> None:
    df = pd.read_csv(dv_csv).copy()
    df["group"] = df["basis_folder"].str.replace("_", "-", regex=False) + " " + df["modality"]

    grouped = (
        df.groupby(["group", "dimension", "variant"])["accuracy"]
        .mean()
        .reset_index()
    )

    group_order = [
        "feature-sensitive text",
        "feature-sensitive visual",
        "structure-sensitive text",
        "structure-sensitive visual",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, group in zip(axes, group_order):
        sub = grouped[grouped["group"] == group].sort_values("accuracy", ascending=True).head(top_n).copy()
        labels = [f"{d} / {v}" for d, v in zip(sub["dimension"], sub["variant"])]
        values = list(sub["accuracy"] * 100.0)

        ax.barh(range(len(labels)), values)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(group)
        ax.set_xlabel("Avg accuracy (%)")

        for i, v in enumerate(values):
            ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)

    fig.suptitle("Most pressuring dimensions by task group", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_dir", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("abc_charts"))
    parser.add_argument("--no-annotate-heatmap", action="store_true")
    args = parser.parse_args()

    stats_dir = args.stats_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    make_heatmap(stats_dir / "runs_overall.csv", out_dir / "heatmap_model_by_task.png", annotate=not args.no_annotate_heatmap)
    make_gap_chart(stats_dir / "runs_overall.csv", out_dir / "aggregate_gap_chart.png")
    make_hardest_dimensions_chart(stats_dir / "dimension_variant_stats.csv", out_dir / "hardest_dimensions_chart.png")
    make_hardest_by_group_chart(stats_dir / "dimension_variant_stats.csv", out_dir / "hardest_dimensions_by_group.png", top_n=3)

    print(f"Wrote charts to: {out_dir}")


if __name__ == "__main__":
    main()
