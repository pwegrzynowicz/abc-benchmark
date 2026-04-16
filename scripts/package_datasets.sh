#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

datasets_root="${repo_root}/artifacts/datasets"
archive_path="${datasets_root}/abc_selective_attention_dataset.zip"
staging_dir="${datasets_root}/_zip_staging"
source_root_name="selective_attention"
source_root_path="${datasets_root}/${source_root_name}"

if [[ ! -d "${source_root_path}" ]]; then
  echo "Missing dataset root: ${source_root_path}" >&2
  exit 1
fi

rm -rf "${staging_dir}"
mkdir -p "${staging_dir}"
cp -R "${source_root_path}" "${staging_dir}/${source_root_name}"

find "${staging_dir}" \
  \( -name ".DS_Store" -o -name "__pycache__" -o -name ".pytest_cache" -o -name "_smoke" \) \
  -exec rm -rf {} +

rm -f "${archive_path}"

(
  cd "${staging_dir}"
  zip -rq "${archive_path}" "${source_root_name}"
)

rm -rf "${staging_dir}"

echo "Created ${archive_path}"
