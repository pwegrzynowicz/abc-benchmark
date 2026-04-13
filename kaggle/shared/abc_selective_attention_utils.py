import ast
import json
import re
from pathlib import Path
from typing import Any
from datetime import datetime

import pandas as pd

DEFAULT_DATASET_ROOT = Path(
    "/kaggle/input/datasets/patrycjawegrzynowicz/abc-factorized-attention-benchmark"
)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def normalize_result_runs(runs: Any) -> pd.DataFrame:
    results_df = runs.as_dataframe().copy()
    if "result" not in results_df.columns:
        raise ValueError("Expected a 'result' column in runs.as_dataframe() output.")
    flat = pd.json_normalize(results_df["result"]).copy()
    for column in ("model", "model_name", "llm"):
        if column in results_df.columns and column not in flat.columns:
            flat[column] = results_df[column].values
    return flat


def normalize_count_result(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        payload = json.loads(value) if value.strip().startswith("{") else {"count": value}
        count_value = payload["count"]
        if isinstance(count_value, bool):
            raise ValueError("count must be an integer")
        return int(count_value)
    if isinstance(value, dict):
        return int(value["count"])
    raise ValueError(f"Unsupported count result: {value!r}")


def normalize_lines_result(value: object) -> list[int]:
    if isinstance(value, dict):
        raw_lines = value.get("lines", [])
    elif isinstance(value, str):
        raw_lines = json.loads(value).get("lines", [])
    else:
        raise ValueError(f"Unsupported lines result: {value!r}")

    if not isinstance(raw_lines, list):
        raise ValueError("lines must be a list")

    normalized = [int(line) for line in raw_lines]
    if normalized != sorted(set(normalized)):
        raise ValueError("lines must be sorted unique integers")
    return normalized


def parse_gold_lines(gold_lines: object) -> list[int]:
    if isinstance(gold_lines, list):
        return [int(item) for item in gold_lines]
    if isinstance(gold_lines, str):
        text = gold_lines.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError("gold_lines must decode to a JSON list")
        return [int(item) for item in parsed]
    raise ValueError(f"Unsupported gold_lines value: {gold_lines!r}")


def classify_prompt_error(exc: Exception) -> tuple[str, str]:
    error = str(exc)
    message = error.lower()

    if "validation error" in message or "pydantic" in message or "list_type" in message or "schema" in message:
        return "parse_or_schema_error", error
    if "timeout" in message:
        return "timeout_error", error
    if "connection" in message or "connectivity" in message or "api connection" in message:
        return "connection_error", error
    if "rate limit" in message or "429" in message:
        return "rate_limit_error", error
    return "other_error", error


def print_overall_report(summary: dict[str, int | float]) -> None:
    passed = summary["passed"]
    total = summary["total"]
    accuracy = summary["accuracy"]
    std = summary["std"]
    print(f"overall: {passed}/{total} = {accuracy:.2%} (std {std:.2%})")


def summarize_accuracy(result_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        result_df.groupby(group_cols)["is_correct"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "passed", "count": "total", "mean": "accuracy"})
        .reset_index()
    )


def summarize_errors(result_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if "has_error" not in result_df.columns:
        return pd.DataFrame()
    return (
        result_df.groupby(group_cols)["has_error"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "errors", "count": "total", "mean": "error_rate"})
        .reset_index()
    )


def summarize_error_types(result_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if "error_type" not in result_df.columns:
        return pd.DataFrame()
    errored = result_df[result_df["error_type"].fillna("") != ""]
    if errored.empty:
        return pd.DataFrame()
    return errored.groupby(group_cols + ["error_type"]).size().reset_index(name="count")


def print_group_reports(result_df: pd.DataFrame, *, task_name: str, groupings: list[list[str]]) -> None:
    for columns in groupings:
        if any(column not in result_df.columns for column in columns):
            continue
        print(f"=== {task_name}: by {', '.join(columns)} ===")
        print(summarize_accuracy(result_df, columns).to_string(index=False))


def print_error_reports(result_df: pd.DataFrame, *, task_name: str, groupings: list[list[str]]) -> None:
    if "has_error" not in result_df.columns:
        return

    if int(result_df["has_error"].fillna(False).astype(int).sum()) == 0:
        return

    base_columns = [["model"]] if "model" in result_df.columns else []
    for columns in base_columns + groupings:
        if any(column not in result_df.columns for column in columns):
            continue

        error_summary = summarize_errors(result_df, columns)
        if not error_summary.empty and int(error_summary["errors"].sum()) > 0:
            print(f"=== {task_name}: errors by {', '.join(columns)} ===")
            print(error_summary.to_string(index=False))

        error_type_summary = summarize_error_types(result_df, columns)
        if not error_type_summary.empty:
            print(f"=== {task_name}: error types by {', '.join(columns)} ===")
            print(error_type_summary.to_string(index=False))

def print_failures(test: str, result_df: pd.DataFrame, columns: list[str]) -> None:
    existing_columns = available_columns(result_df, columns)
    failures = result_df.loc[~result_df["is_correct"].astype(bool)].copy()

    print(f"=== failures: {test} {len(failures)} ===")
    if failures.empty:
        return

    if existing_columns:
        print(failures[existing_columns].to_string(index=False))
    else:
        print("(no requested failure columns available)")


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def build_text_groupings(df: pd.DataFrame) -> list[list[str]]:
    groupings: list[list[str]] = [
        ["dimension"],
        ["dimension", "variant"],
        ["structure_type"] if "structure_type" in df.columns else ["target_feature_count"],
        ["num_records"],
    ]

    if "family" in df.columns:
        groupings.insert(0, ["family"])
    if "attentional_basis" in df.columns:
        groupings.insert(1, ["family", "attentional_basis"] if "family" in df.columns else ["attentional_basis"])
    if "modality" in df.columns and "family" in df.columns and "attentional_basis" in df.columns:
        groupings.insert(2, ["family", "attentional_basis", "modality"])

    if "structure_depth" in df.columns:
        groupings.append(["structure_depth"])
        if "structure_type" in df.columns:
            groupings.append(["structure_type", "structure_depth"])
    if "binding_distance" in df.columns:
        groupings.append(["binding_distance"])
    if "serialization_style" in df.columns:
        groupings.append(["serialization_style"])
    if "position_mode" in df.columns:
        groupings.append(["position_mode"])
    if "target_count" in df.columns:
        groupings.append(["target_count"])
        groupings.append(["dimension", "target_count"])
    if "confound_count" in df.columns:
        groupings.append(["confound_count"])
    if "confound_type" in df.columns:
        groupings.append(["confound_type"])
        if "structure_type" in df.columns:
            groupings.append(["structure_type", "confound_type"])
    if "line_length_noise" in df.columns:
        groupings.append(["line_length_noise"])

    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for columns in groupings:
        key = tuple(columns)
        if key not in seen and all(column in df.columns for column in columns):
            seen.add(key)
            deduped.append(columns)
    return deduped


def calculate_summary(result_df: pd.DataFrame) -> dict[str, int | float]:
    is_correct = result_df["is_correct"].astype(int)

    total = int(is_correct.count())
    passed = int(is_correct.sum())

    if total == 0:
        accuracy = 0.0
        std = 0.0
    elif total == 1:
        accuracy = float(is_correct.mean())
        std = 0.0
    else:
        accuracy = float(is_correct.mean())
        std = float(is_correct.std(ddof=1))

    return {
        "passed": passed,
        "total": total,
        "accuracy": accuracy,
        "std": std,
    }


def log_info(
        task_name: str,
        model_name: str,
        summary: dict[str, int | float],
        result: pd.DataFrame,
        groupings: list[list[str]],
        failure_cols: list[str],
) -> None:
    print(f"=== {task_name} - {model_name}: info ===")
    print_overall_report(summary)
    if groupings:
        print_group_reports(result, task_name=task_name, groupings=groupings)
        print_error_reports(result, task_name=task_name, groupings=groupings)
    if failure_cols:
        print_failures(task_name, result, failure_cols)


def log_file(
        task_name: str,
        model_name: str,
        result: pd.DataFrame,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = re.sub(r"[/\\ \-]", "_", f"results_{model_name}_{timestamp}.csv")
    result.to_csv(out_file, index=False)


def summarize_and_log_runs(
        task_name: str,
        model_name: str,
        runs: Any,
        groupings: list[list[str]],
        failure_cols: list[str]
) -> tuple[float, float]:
    result_df = normalize_result_runs(runs)
    summary = calculate_summary(result_df)
    log_info(task_name, model_name, summary, result_df, groupings, failure_cols)
    log_file(task_name, model_name, result_df)
    return summary['accuracy'], summary['std']