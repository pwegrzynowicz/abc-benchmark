# ABC Benchmark

Minimal repo skeleton for the ABC-mini attention benchmark.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_sa_feature_text_dataset.py
python scripts/generate_sa_feature_visual_dataset.py
python scripts/generate_sa_structure_text_dataset.py
python scripts/generate_sa_structure_visualv_dataset.py
```

```bash
python scripts/dataset_viewer.py artifacts/datasets/  
```
