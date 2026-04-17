#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

try:
    import pandas as pd
except Exception:
    print("This viewer requires pandas. Install it with: pip install pandas pyarrow", file=sys.stderr)
    raise

TABULAR_EXTENSIONS = {".csv", ".jsonl", ".parquet"}
GROUP_COLUMNS = ["family", "attentional_basis", "modality", "dimension", "variant"]
ID_COLUMNS = ["scene_id", "sample_id", "id"]
TEXT_COLUMNS = ["text_input", "text", "input_text"]
IMAGE_COLUMNS = [
    "image_path",
    "image_file",
    "image_filename",
    "render_path",
    "png_path",
    "jpg_path",
    "jpeg_path",
]
IMAGE_WITH_IDS_COLUMNS = [
    "image_with_ids_path",
    "image_ids_path",
    "image_with_item_ids_path",
    "render_with_ids_path",
    "png_with_ids_path",
]


def discover_table_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_EXTENSIONS)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def maybe_parse_json(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if text[0] in "[{":
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def first_present(record: dict, columns: list[str]):
    for col in columns:
        value = record.get(col)
        if value is not None and value != "":
            return value
    return None


class DatasetIndex:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root.resolve()
        self.rows: list[dict] = []
        self.groups: dict[str, list[int]] = {}
        self._load()

    def _load(self) -> None:
        table_files = discover_table_files(self.dataset_root)
        if not table_files:
            raise FileNotFoundError(f"No CSV/JSONL/Parquet files found under {self.dataset_root}")

        for table_file in table_files:
            try:
                df = read_table(table_file)
            except Exception:
                continue

            if not set(["family", "attentional_basis", "modality"]).issubset(df.columns):
                continue

            for _, row in df.iterrows():
                record = {col: normalize_value(row[col]) for col in df.columns}
                for key in ["gold_indices", "gold_lines", "gold_ids"]:
                    if key in record:
                        record[key] = maybe_parse_json(record[key])

                record["__source_file"] = str(table_file.relative_to(self.dataset_root))
                record["__group_id"] = " ".join(
                    str(record.get(col, "")).strip()
                    for col in GROUP_COLUMNS
                    if record.get(col) not in (None, "")
                ).strip()
                record["__image_relpath"] = self._find_image_path(
                    record,
                    table_file.parent,
                    IMAGE_COLUMNS,
                    image_family="plain",
                )
                record["__image_ids_relpath"] = self._find_image_path(
                    record,
                    table_file.parent,
                    IMAGE_WITH_IDS_COLUMNS,
                    image_family="ids",
                )

                idx = len(self.rows)
                self.rows.append(record)
                self.groups.setdefault(record["__group_id"], []).append(idx)

        if not self.rows:
            raise FileNotFoundError("Found files, but none matched the expected dataset structure.")

    def _find_image_path(
        self,
        record: dict,
        table_dir: Path,
        columns: list[str],
        *,
        image_family: str,
    ) -> str | None:
        for col in columns:
            value = record.get(col)
            if not value:
                continue
            candidate = Path(str(value))
            if candidate.is_absolute() and candidate.exists():
                try:
                    return str(candidate.relative_to(self.dataset_root))
                except Exception:
                    return str(candidate)
            resolved = (table_dir / candidate).resolve()
            if resolved.exists():
                try:
                    return str(resolved.relative_to(self.dataset_root))
                except Exception:
                    return str(resolved)

        scene_id = record.get("scene_id")
        if scene_id:
            dimension = record.get("dimension")
            variant = record.get("variant")
            if dimension and variant:
                preferred = self.dataset_root / "images" / image_family / str(dimension) / str(variant) / f"{scene_id}.png"
                if preferred.exists():
                    try:
                        return str(preferred.relative_to(self.dataset_root))
                    except Exception:
                        return str(preferred)
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                matches = [
                    path
                    for path in self.dataset_root.rglob(f"{scene_id}{ext}")
                    if f"{Path('images') / image_family}" in str(path.relative_to(self.dataset_root))
                ]
                if matches:
                    try:
                        return str(matches[0].relative_to(self.dataset_root))
                    except Exception:
                        return str(matches[0])
        return None

    def list_groups(self) -> list[dict]:
        return [
            {"group_id": group_id, "count": len(indexes)}
            for group_id, indexes in sorted(self.groups.items())
        ]

    def get_sample(self, group_id: str, position: int) -> dict:
        if group_id not in self.groups:
            raise KeyError(f"Unknown group: {group_id}")

        indexes = self.groups[group_id]
        position = max(0, min(position, len(indexes) - 1))
        row = self.rows[indexes[position]]

        sample_id = first_present(row, ID_COLUMNS)
        text_value = first_present(row, TEXT_COLUMNS)
        filter_gold = row.get("gold_indices")
        if filter_gold is None:
            filter_gold = row.get("gold_lines")
        if filter_gold is None:
            filter_gold = row.get("gold_ids")

        return {
            "id": sample_id,
            "family": row.get("family"),
            "attentional_basis": row.get("attentional_basis"),
            "modality": row.get("modality"),
            "dimension": row.get("dimension"),
            "variant": row.get("variant"),
            "group_id": row["__group_id"],
            "position": position,
            "size": len(indexes),
            "source_file": row["__source_file"],
            "image_url": f"/image?path={quote(row['__image_relpath'])}" if row.get("__image_relpath") else None,
            "image_ids_url": f"/image?path={quote(row['__image_ids_relpath'])}" if row.get("__image_ids_relpath") else None,
            "text_value": text_value,
            "count_prompt": row.get("count_prompt"),
            "count_gold": row.get("gold_count"),
            "filter_prompt": row.get("filter_prompt"),
            "filter_gold": filter_gold,
        }


HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Dataset Viewer</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      margin: 0;
      padding: 0;
      background: #f6f7f9;
      color: #111827;
    }
    .wrap {
      max-width: 1560px;
      margin: 0 auto;
      padding: 20px;
    }
    .bar, .panel {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    }
    .panel { padding: 16px; }
    .bar {
      display: grid;
      grid-template-columns: 1.7fr auto auto auto auto;
      gap: 12px;
      padding: 14px;
      align-items: center;
      margin-bottom: 16px;
    }
    .title-panel {
      margin-bottom: 16px;
      padding: 18px 20px;
    }
    .title-meta {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
    }
    .sample-title {
      font-size: 30px;
      font-weight: 800;
      line-height: 1.2;
    }
    .content {
      display: block;
    }
    .content.text-mode {
      max-width: 768px;
    }
    h1 {
      margin: 0 0 16px 0;
      font-size: 24px;
    }
    h2 {
      margin: 0 0 10px 0;
      font-size: 18px;
    }
    h3 {
      margin: 0 0 8px 0;
      font-size: 15px;
    }
    select, button, input {
      border: 1px solid #d1d5db;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      background: white;
    }
    button {
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { background: #f3f4f6; }
    img {
      max-width: 100%;
      max-height: 72vh;
      object-fit: contain;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
      background: #fff;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #f9fafb;
      border: 1px solid #eef0f3;
      padding: 12px;
      border-radius: 12px;
      margin: 0;
      font-size: 13px;
    }
    .content.text-mode #countPrompt,
    .content.text-mode #filterPrompt {
      white-space: pre-wrap;
      word-break: normal;
      overflow-wrap: anywhere;
    }
    .content.text-mode .panel,
    .content.text-mode .prompt-media,
    .content.text-mode .prompt-media > div {
      min-width: 0;
    }
    .stack { display: grid; gap: 14px; }
    .muted { color: #6b7280; font-size: 13px; }
    .headerline {
      display:flex;
      justify-content:space-between;
      gap: 12px;
      align-items:center;
    }
    .pill {
      display:inline-block;
      padding: 5px 10px;
      border-radius:999px;
      background:#eef2ff;
      color:#3730a3;
      font-size:12px;
      font-weight:600;
    }
    .gold {
      font-weight: 700;
      font-size: 16px;
      color: #111827;
      background: #eefbf3;
      border: 1px solid #cdeed8;
      border-radius: 12px;
      padding: 10px 12px;
    }
    .image-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .image-card {
      display: grid;
      gap: 8px;
    }
    .image-label {
      font-size: 13px;
      font-weight: 700;
      color: #374151;
    }
    .prompt-media {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 0.9fr);
      gap: 14px;
      align-items: start;
    }
    .content.text-mode .prompt-media {
      grid-template-columns: 1fr;
    }
    .prompt-media .image-card img {
      max-height: 42vh;
    }
    @media (max-width: 980px) {
      .content.text-mode {
        max-width: none;
      }
      .prompt-media {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
<div class=\"wrap\">
  <h1>Dataset Viewer</h1>

  <div class=\"panel title-panel\">
    <div id=\"sampleTitle\" class=\"sample-title\"></div>
    <div class=\"title-meta\">
      <span id=\"positionBadge\" class=\"pill\"></span>
      <div id=\"sourceFile\" class=\"muted\"></div>
    </div>
  </div>

  <div class=\"bar\">
    <select id=\"groupSelect\"></select>
    <button id=\"prevBtn\">Prev</button>
    <button id=\"nextBtn\">Next</button>
    <input id=\"jumpInput\" type=\"number\" min=\"1\" step=\"1\" placeholder=\"Go to #\">
    <button id=\"jumpBtn\">Go</button>
  </div>

  <div class=\"content\">
    <div class=\"stack\">
      <div class=\"panel stack\">
        <h2>Count</h2>
        <div class=\"prompt-media\">
          <div>
            <h3>Prompt</h3>
            <pre id=\"countPrompt\"></pre>
          </div>
          <div id=\"countImageView\"></div>
        </div>
        <div>
          <h3>Gold</h3>
          <div id=\"countGold\" class=\"gold\"></div>
        </div>
      </div>

      <div class=\"panel stack\">
        <h2>Filter</h2>
        <div class=\"prompt-media\">
          <div>
            <h3>Prompt</h3>
            <pre id=\"filterPrompt\"></pre>
          </div>
          <div id=\"filterImageView\"></div>
        </div>
        <div>
          <h3>Gold</h3>
          <pre id=\"filterGold\"></pre>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  let groups = [];
  let currentGroup = null;
  let currentPosition = 0;

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  function renderSample(sample) {
    const content = document.querySelector('.content');
    content.classList.toggle('text-mode', sample.modality === 'text');
    renderPromptImage(
      document.getElementById('countImageView'),
      sample.image_url,
      'Plain',
      (sample.id || 'sample') + ' plain',
      'No plain image found.'
    );
    renderPromptImage(
      document.getElementById('filterImageView'),
      sample.image_ids_url,
      'With IDs',
      (sample.id || 'sample') + ' with ids',
      'No image-with-ids found.'
    );

    if (sample.modality !== 'visual' || (!sample.image_url && !sample.image_ids_url)) {
      document.getElementById('countImageView').innerHTML = '';
      document.getElementById('filterImageView').innerHTML = '';
    }
  }

  function renderPromptImage(container, imageUrl, labelText, altText, emptyText) {
    container.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'image-card';
    const label = document.createElement('div');
    label.className = 'image-label';
    label.textContent = labelText;
    card.appendChild(label);
    if (imageUrl) {
      const img = document.createElement('img');
      img.src = imageUrl;
      img.alt = altText;
      card.appendChild(img);
    } else {
      const empty = document.createElement('pre');
      empty.textContent = emptyText;
      card.appendChild(empty);
    }
    container.appendChild(card);
  }

  async function loadGroups() {
    const data = await getJSON('/api/groups');
    groups = data.groups;
    const select = document.getElementById('groupSelect');
    select.innerHTML = '';
    groups.forEach(g => {
      const option = document.createElement('option');
      option.value = g.group_id;
      option.textContent = `${g.group_id} (${g.count})`;
      select.appendChild(option);
    });
    if (groups.length > 0) {
      currentGroup = groups[0].group_id;
      select.value = currentGroup;
      await loadSample(0);
    }
  }

  async function loadSample(position) {
    if (!currentGroup) return;
    const data = await getJSON(`/api/sample?group_id=${encodeURIComponent(currentGroup)}&position=${position}`);
    currentPosition = data.position;

    document.getElementById('positionBadge').textContent =
      `${data.position + 1} / ${data.size}` + (data.id ? ` • ${data.id}` : '');
    document.getElementById('sourceFile').textContent = data.source_file || '';
    document.getElementById('sampleTitle').textContent = data.group_id || [
      data.family,
      data.attentional_basis,
      data.modality,
      data.dimension,
      data.variant,
    ].filter(Boolean).join(' ');

    renderSample(data);

    document.getElementById('countPrompt').textContent = data.count_prompt || '';
    document.getElementById('countGold').textContent = data.count_gold == null ? '' : String(data.count_gold);
    document.getElementById('filterPrompt').textContent = data.filter_prompt || '';
    document.getElementById('filterGold').textContent =
      data.filter_gold == null ? '' : (typeof data.filter_gold === 'string' ? data.filter_gold : JSON.stringify(data.filter_gold, null, 2));

    document.getElementById('jumpInput').value = String(data.position + 1);
  }

  document.getElementById('groupSelect').addEventListener('change', async (e) => {
    currentGroup = e.target.value;
    await loadSample(0);
  });

  document.getElementById('prevBtn').addEventListener('click', async () => {
    await loadSample(Math.max(0, currentPosition - 1));
  });

  document.getElementById('nextBtn').addEventListener('click', async () => {
    await loadSample(currentPosition + 1);
  });

  document.getElementById('jumpBtn').addEventListener('click', async () => {
    const raw = parseInt(document.getElementById('jumpInput').value, 10);
    if (!Number.isFinite(raw)) return;
    await loadSample(Math.max(0, raw - 1));
  });

  loadGroups().catch(err => {
    document.body.innerHTML = `<pre style=\"padding:20px;\">Failed to load viewer:

${err.stack || err}</pre>`;
  });
</script>
</body>
</html>"""


class ViewerHandler(BaseHTTPRequestHandler):
    dataset_index: DatasetIndex | None = None

    def _send_bytes(self, data: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(payload, content_type="application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_bytes(HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/groups":
                assert self.dataset_index is not None
                self._send_json({"groups": self.dataset_index.list_groups()})
                return
            if parsed.path == "/api/sample":
                assert self.dataset_index is not None
                group_id = params.get("group_id", [""])[0]
                position = int(params.get("position", ["0"])[0])
                self._send_json(self.dataset_index.get_sample(group_id, position))
                return
            if parsed.path == "/image":
                assert self.dataset_index is not None
                rel = unquote(params.get("path", [""])[0])
                if not rel:
                    self.send_error(404, "Missing image path")
                    return
                path = (self.dataset_index.dataset_root / rel).resolve()
                if not str(path).startswith(str(self.dataset_index.dataset_root)):
                    self.send_error(403, "Path escapes dataset root")
                    return
                if not path.exists() or not path.is_file():
                    self.send_error(404, "Image not found")
                    return
                mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                self._send_bytes(path.read_bytes(), content_type=mime)
                return
            self.send_error(404, "Not found")
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple local viewer for benchmark datasets.")
    parser.add_argument("dataset_root", help="Root directory containing CSV/JSONL/Parquet dataset rows and images.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind. Default: 8000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    ViewerHandler.dataset_index = DatasetIndex(dataset_root)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Viewer running at http://{args.host}:{args.port}")
    print(f"Dataset root: {dataset_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
