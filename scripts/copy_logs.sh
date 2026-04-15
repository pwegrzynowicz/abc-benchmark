#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./copy_logs.sh [SOURCE_DIR] [DESTINATION_ROOT]
#
# Example:
#   ./copy_logs.sh "/Users/you/Desktop/logi" "/Users/you/code/abc-benchmark/artifacts/results"

SOURCE_DIR="${1:-/path/to/logs}"
DESTINATION_ROOT="${2:-/path/to/abc-benchmark/artifacts/results}"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

shopt -s nullglob

copied_count=0
skipped_count=0

for file in "$SOURCE_DIR"/*.log; do
  [[ -f "$file" ]] || continue

  task=""
  run_id=""

  while IFS= read -r line; do
    if [[ -z "$task" && "$line" =~ ===[[:space:]](.*)[[:space:]]:[[:space:]]info[[:space:]]=== ]]; then
      task="${BASH_REMATCH[1]}"
    fi

    if [[ -z "$run_id" && "$line" =~ run[[:space:]]id:[[:space:]]*(.+)$ ]]; then
      run_id="${BASH_REMATCH[1]}"
    fi

    [[ -n "$task" && -n "$run_id" ]] && break
  done < "$file"

  if [[ -z "$task" ]]; then
    echo "Skipped $(basename "$file"): task line not found"
    ((skipped_count+=1))
    continue
  fi

  if [[ -z "$run_id" ]]; then
    echo "Skipped $(basename "$file"): run id not found"
    ((skipped_count+=1))
    continue
  fi

  if [[ "$task" != *" - "* ]]; then
    echo "Skipped $(basename "$file"): task does not contain expected separator ' - '"
    ((skipped_count+=1))
    continue
  fi

  first_segment="${task%% - *}"
  second_segment_full="${task#* - }"

  if [[ "$first_segment" == "$task" || -z "$second_segment_full" ]]; then
    echo "Skipped $(basename "$file"): could not split task into expected parts"
    ((skipped_count+=1))
    continue
  fi

  first_part="${first_segment// /_}"

  read -r -a words <<< "$second_segment_full"
  if (( ${#words[@]} < 3 )); then
    echo "Skipped $(basename "$file"): second task segment has fewer than 3 words"
    ((skipped_count+=1))
    continue
  fi

  second_part="${words[0]}_${words[1]}"
  folder_path="$first_part/$second_part"

  for ((i = 2; i < ${#words[@]}; i++)); do
    folder_path="$folder_path/${words[i]}"
  done

  final_dir="$DESTINATION_ROOT/$folder_path"
  mkdir -p "$final_dir"

  destination_path="$final_dir/$run_id.log"
  cp -f "$file" "$destination_path"

  echo "Copied: $(basename "$file") -> $destination_path"
  ((copied_count+=1))
done

echo
echo "Done. Copied: $copied_count, skipped: $skipped_count"
