#!/usr/bin/env bash
set -euo pipefail

root="baseline_artifacts/checkpoints/pusht_paper_protocol_on_off"

for condition in off on; do
  directory="$root/$condition"
  destination="$directory/model_latest.pth"
  checksum_file="$directory/model_latest.pth.sha256"
  parts=("$directory"/model_latest.pth.part-*)

  if [[ ! -e "${parts[0]}" ]]; then
    echo "Missing checkpoint parts under $directory" >&2
    exit 1
  fi

  cat "${parts[@]}" > "$destination.tmp"
  mv "$destination.tmp" "$destination"
  (
    cd "$directory"
    sha256sum --check "$(basename "$checksum_file")"
  )
done
