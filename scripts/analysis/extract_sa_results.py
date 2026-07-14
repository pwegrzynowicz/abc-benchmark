#!/usr/bin/env python3
"""
Extract ABC Selective Attention benchmark stats from a zip bundle of logs.

Outputs:
- runs_overall.csv
- dimension_variant_stats.csv
- baseline_stats.csv
- task_group_summary.csv

Usage:
    python extract_sa_results.py /path/to/selective_attention_20260416.zip -o out_dir
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


HEADING_RE = re.compile(r"===\s+(?P<task>.*?):\s+(?P<section>info|by .+?)\s+===")
TIMED_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+)?s\s+\d+\s+(.*)$")
INFO_KV_RE = re.compile(r"^(task|model|timestamp|run id|errors|overall|overall accuracy):\s+(.*)$")
ERRORS_RE = re.compile(r"(?P<count>\d+)\s+\(rate\s+(?P<rate>[^)]+)\)")
OVERALL_RE = re.compile(
    r"(?P<passed>\d+)/(?P<total>\d+)\s+=\s+(?P<acc>[0-9.]+)%\s+\(std\s+(?P<std>[0-9.]+)%\)"
)


def normalize_line(raw: str) -> str:
    m = TIMED_LINE_RE.match(raw.rstrip("\n"))
    return m.group(1) if m else raw.rstrip("\n")


def path_metadata(zip_name: str) -> Dict[str, str]:
    p = Path(zip_name)
    parts = p.parts
    if len(parts) < 5:
        raise ValueError(f"Unexpected log path structure: {zip_name}")
    return {
        "basis_folder": parts[-4],
        "modality": parts[-3],
        "task_type": parts[-2],
        "log_filename": parts[-1],
    }


def parse_fixed_width_table(lines: List[str]) -> List[Dict[str, str]]:
    if not lines:
        return []
    header = re.split(r"\s{2,}|\t+|\s+", lines[0].strip())
    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        vals = re.split(r"\s{2,}|\t+|\s+", stripped)
        # Heuristic: use last 3 numeric-ish fields as passed/total/accuracy if overflow occurs
        if len(vals) > len(header) and len(header) >= 4:
            tail_n = 3
            vals = vals[: len(header) - tail_n] + vals[-tail_n:]
        if len(vals) != len(header):
            continue
        rows.append(dict(zip(header, vals)))
    return rows


def read_log_from_zip(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="ignore")


def parse_log(name: str, text: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    meta = path_metadata(name)
    overall: Dict[str, str] = {
        "zip_path": name,
        "basis_folder": meta["basis_folder"],
        "modality": meta["modality"],
        "task_type": meta["task_type"],
        "log_filename": meta["log_filename"],
    }
    section_rows: List[Dict[str, str]] = []

    lines = [normalize_line(line) for line in text.splitlines()]
    i = 0
    current_task = None
    while i < len(lines):
        line = lines[i]
        h = HEADING_RE.search(line)
        if not h:
            i += 1
            continue

        current_task = h.group("task").strip()
        section = h.group("section").strip()

        j = i + 1
        block: List[str] = []
        while j < len(lines):
            nxt = lines[j]
            if nxt.startswith("==="):
                break
            block.append(nxt)
            j += 1

        if section == "info":
            overall["task_name"] = current_task
            for b in block:
                m = INFO_KV_RE.match(b.strip())
                if not m:
                    continue
                key = m.group(1).lower()
                value = m.group(2).strip()
                if key == "task":
                    overall["task"] = value
                elif key == "model":
                    overall["model"] = value
                elif key == "timestamp":
                    overall["timestamp"] = value
                elif key == "run id":
                    overall["run_id"] = value
                elif key == "errors":
                    em = ERRORS_RE.search(value)
                    if em:
                        overall["errors"] = em.group("count")
                        overall["error_rate"] = em.group("rate")
                elif key in {"overall", "overall accuracy"}:
                    om = OVERALL_RE.search(value)
                    if om:
                        overall["passed"] = om.group("passed")
                        overall["total"] = om.group("total")
                        overall["accuracy_pct"] = om.group("acc")
                        overall["std_pct"] = om.group("std")
        elif section.startswith("by "):
            table_lines = [b for b in block if b.strip()]
            rows = parse_fixed_width_table(table_lines)
            for row in rows:
                item = {
                    "zip_path": name,
                    "task_name": current_task or "",
                    "section": section,
                    "basis_folder": meta["basis_folder"],
                    "modality": meta["modality"],
                    "task_type": meta["task_type"],
                }
                item.update(row)
                section_rows.append(item)
        i = j

    return overall, section_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("abc_stats"))
    args = parser.parse_args()

    overall_rows: List[Dict[str, object]] = []
    section_rows: List[Dict[str, str]] = []

    with zipfile.ZipFile(args.zip_path) as zf:
        log_names = [n for n in zf.namelist() if n.endswith(".log") and not n.startswith("__MACOSX/")]
        for name in sorted(log_names):
            overall, sections = parse_log(name, read_log_from_zip(zf, name))
            if overall.get("model") and overall.get("accuracy_pct") is not None:
                overall_rows.append(overall)
            section_rows.extend(sections)

    for row in overall_rows:
        for key in ("errors", "passed", "total"):
            row[key] = int(row.get(key, 0))
        for key in ("error_rate", "accuracy_pct", "std_pct"):
            row[key] = float(row.get(key, 0.0))

    # Debug: show duplicate model/task rows if any
    seen = defaultdict(int)
    for row in overall_rows:
        seen[(str(row["model"]), str(row["task_name"]))] += 1
    dupes = [(k, v) for k, v in seen.items() if v > 1]
    if dupes:
        print("\nWARNING: duplicate model/task rows detected:")
        for (model, task), count in dupes:
            print(f"  {model} | {task} -> {count} rows")

    dv_rows: List[Dict[str, object]] = []
    baseline_rows: List[Dict[str, object]] = []
    for row in section_rows:
        if row["section"] != "by dimension, variant":
            continue
        out = {
            "zip_path": row["zip_path"],
            "task_name": row["task_name"],
            "basis_folder": row["basis_folder"],
            "modality": row["modality"],
            "task_type": row["task_type"],
            "dimension": row.get("dimension", ""),
            "variant": row.get("variant", ""),
            "passed": int(float(row.get("passed", 0))),
            "total": int(float(row.get("total", 0))),
            "accuracy": float(row.get("accuracy", 0.0)),
        }
        dv_rows.append(out)
        if str(out["dimension"]).lower() == "baseline":
            baseline_rows.append(out.copy())

    grouped = defaultdict(list)
    for row in overall_rows:
        key = (row["basis_folder"], row["modality"], row["task_type"], row["task_name"])
        grouped[key].append(float(row["accuracy_pct"]))

    task_group_summary: List[Dict[str, object]] = []
    for (basis_folder, modality, task_type, task_name), vals in sorted(grouped.items()):
        task_group_summary.append({
            "basis_folder": basis_folder,
            "modality": modality,
            "task_type": task_type,
            "task_name": task_name,
            "num_models": len(vals),
            "avg_accuracy_pct": round(sum(vals) / len(vals), 2),
        })

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_fields = [
        "zip_path", "task_name", "task", "model", "timestamp", "run_id",
        "basis_folder", "modality", "task_type",
        "errors", "error_rate", "passed", "total", "accuracy_pct", "std_pct",
        "log_filename",
    ]
    write_csv(out_dir / "runs_overall.csv", overall_rows, overall_fields)

    dv_fields = [
        "zip_path", "task_name", "basis_folder", "modality", "task_type",
        "dimension", "variant", "passed", "total", "accuracy",
    ]
    write_csv(out_dir / "dimension_variant_stats.csv", dv_rows, dv_fields)
    write_csv(out_dir / "baseline_stats.csv", baseline_rows, dv_fields)

    task_group_fields = [
        "basis_folder", "modality", "task_type", "task_name", "num_models", "avg_accuracy_pct"
    ]
    write_csv(out_dir / "task_group_summary.csv", task_group_summary, task_group_fields)

    print(f"Wrote CSV files to: {out_dir}")


if __name__ == "__main__":
    main()
