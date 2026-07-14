# Selective Attention Analysis Tools

These scripts support post-hoc analysis and visualization for the ABC Selective Attention Benchmark. They are intended for result extraction, figure generation, and submission materials, not for dataset generation itself.

## Folder layout

```text
scripts/
  analysis/
    extract_sa_results.py
    plot_sa_results.py
    draw_sa_structure_diagram.py
    README.md

artifacts/
  analysis/
    selective_attention/
      stats/
      charts/
      diagrams/
```

## Scripts

- `extract_sa_results.py`  
  Parse benchmark log bundles and export structured CSV summaries.

- `plot_sa_results.py`  
  Generate charts from extracted CSVs, including:
  - model × task heatmap
  - aggregate gap chart
  - hardest dimensions / variants chart

- `draw_sa_structure_diagram.py`  
  Generate the conceptual benchmark-structure diagram used in the writeup.

## Inputs and outputs

### Input
The extractor expects a zip bundle containing benchmark logs in the current ABC selective-attention results layout.

### Output
Generated files should be written under:

```text
artifacts/analysis/selective_attention/
  stats/
  charts/
  diagrams/
```

## Recommended workflow

### 1. Extract stats from logs

```bash
python scripts/analysis/extract_sa_results.py \
  artifacts/results/selective_attention_20260416.zip \
  -o artifacts/analysis/selective_attention/stats
```

### 2. Generate charts

```bash
python scripts/analysis/plot_sa_results.py \
  artifacts/analysis/selective_attention/stats \
  -o artifacts/analysis/selective_attention/charts
```

### 3. Generate benchmark structure diagram

```bash
python scripts/analysis/draw_sa_structure_diagram.py \
  -o artifacts/analysis/selective_attention/diagrams
```

### 4. Generate ranking charts

```bash
python scripts/analysis/plot_sa_model_rankings.py \
  artifacts/analysis/selective_attention/stats \
  -o artifacts/analysis/selective_attention/charts
```
## Generated files

### Stats
- `runs_overall.csv`
- `dimension_variant_stats.csv`
- `baseline_stats.csv`
- `task_group_summary.csv`

### Charts
- `heatmap_model_by_task.png`
- `aggregate_gap_chart.png`
- `hardest_dimensions_chart.png`

### Diagrams
- `abc_benchmark_structure.png`
- `abc_benchmark_structure.svg`

## Notes

- These scripts are analysis utilities and are separate from the core benchmark generation code.
- The extractor is tailored to the current ABC log format and may need adjustment if log formatting changes.
- The structure diagram is conceptual and is not derived from the result logs.
