#!/usr/bin/env python3
"""
Generate model ranking charts for the ABC Selective Attention benchmark.

Creates sorted horizontal bar charts for:
- overall
- text
- visual
- feature-sensitive
- structure-sensitive

Input:
- runs_overall.csv (from extract_sa_results.py)

Usage:
    python plot_sa_model_rankings.py stats_dir -o charts_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
        ("deepseek-v3.2", "DeepSeek V3.2"),
    ]
    for needle, out in explicit:
        if needle in s:
            return out
    return str(label)


def make_ranking_chart(df: pd.DataFrame, title: str, out_path: Path) -> None:
    grouped = df.groupby("model", as_index=False)["accuracy_pct"].mean()
    grouped = grouped.sort_values("accuracy_pct", ascending=False).reset_index(drop=True)
    grouped["label"] = grouped["model"].map(shorten_model_label)

    print(f"\n=== DEBUG ranking table: {title} ===")
    print(grouped[["model", "label", "accuracy_pct"]].to_string(index=False))

    dups = grouped[grouped["label"].duplicated(keep=False)].sort_values("label")
    if not dups.empty:
        print("\nWARNING: duplicate display labels detected:")
        print(dups[["model", "label", "accuracy_pct"]].to_string(index=False))

    fig_h = max(6, 0.45 * len(grouped))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    y = list(range(len(grouped)))
    ax.barh(y, grouped["accuracy_pct"])
    ax.set_yticks(y)
    ax.set_yticklabels(grouped["label"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Average accuracy (%)")
    ax.set_title(title, fontsize=16, pad=12)

    for i, val in enumerate(grouped["accuracy_pct"]):
        ax.text(val + 1.0, i, f"{val:.1f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_dir", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("abc_rankings"))
    args = parser.parse_args()

    stats_dir = args.stats_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(stats_dir / "runs_overall.csv")

    make_ranking_chart(df, "Overall model accuracy", out_dir / "ranking_overall.png")
    make_ranking_chart(df[df["modality"] == "text"], "Model accuracy — text tasks", out_dir / "ranking_text.png")
    make_ranking_chart(df[df["modality"] == "visual"], "Model accuracy — visual tasks", out_dir / "ranking_visual.png")
    make_ranking_chart(df[df["basis_folder"] == "feature_sensitive"], "Model accuracy — feature-sensitive tasks", out_dir / "ranking_feature_sensitive.png")
    make_ranking_chart(df[df["basis_folder"] == "structure_sensitive"], "Model accuracy — structure-sensitive tasks", out_dir / "ranking_structure_sensitive.png")

    print(f"\nWrote ranking charts to: {out_dir}")


if __name__ == "__main__":
    main()
