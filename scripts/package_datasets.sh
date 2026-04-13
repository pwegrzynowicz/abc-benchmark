#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

datasets_root="${repo_root}/artifacts/datasets"
archive_path="${datasets_root}/abc-factorized-attention-benchmark.zip"
staging_dir="${datasets_root}/_zip_staging"
archive_root_name="abc-factorized-attention-benchmark"

if [[ ! -d "${datasets_root}" ]]; then
  echo "Missing datasets directory: ${datasets_root}" >&2
  exit 1
fi

rm -rf "${staging_dir}"
mkdir -p "${staging_dir}/${archive_root_name}"

find "${datasets_root}" -mindepth 1 -maxdepth 1 -type d ! -name "_zip_staging" | while read -r dataset_dir; do
  dataset_name="$(basename "${dataset_dir}")"
  cp -R "${dataset_dir}" "${staging_dir}/${archive_root_name}/${dataset_name}"
done

find "${staging_dir}" \
  \( -name ".DS_Store" -o -name "__pycache__" -o -name ".pytest_cache" -o -name "_smoke" \) \
  -exec rm -rf {} +

rm -f "${archive_path}"

(
  cd "${staging_dir}"
  zip -rq "${archive_path}" "${archive_root_name}"
)

rm -rf "${staging_dir}"

echo "Created ${archive_path}"
