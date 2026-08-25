#!/usr/bin/env bash
set -euo pipefail

root="baseline_artifacts/checkpoints/umaze_paper_protocol_on_off"

for condition in off on; do
  directory="$root/$condition"
  destination="$directory/model_20.pth"
  cat "$directory"/model_20.pth.part-* > "$destination"
  (cd "$directory" && shasum -a 256 -c model_20.pth.sha256)
done

