#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/temporal-straightening}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_DIR/baseline_artifacts/checkpoints/umaze_physics_layer_ablations}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO_DIR/baseline_artifacts/analysis/umaze_physics_layer_ablations}"
LOG_DIR="$REPO_DIR/baseline_artifacts/logs"
STATUS="$LOG_DIR/umaze_motion_geometry_watcher.status"
TARGET_EPOCHS="${TARGET_EPOCHS:-20}"
PROBE_GPU="${PROBE_GPU:-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
status() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

has_checkpoint() {
  local root="$1"
  [[ -s "$root/checkpoints/model_${TARGET_EPOCHS}.pth" || -s "$root/model_${TARGET_EPOCHS}.pth" ]]
}

status "WATCH_START target_epoch=$TARGET_EPOCHS probe_gpu=$PROBE_GPU"
while ! has_checkpoint "$CHECKPOINT_ROOT/calibrated_speed" \
  || ! has_checkpoint "$CHECKPOINT_ROOT/factorized" \
  || ! has_checkpoint "$CHECKPOINT_ROOT/layer_aware_factorized" \
  || [[ ! -s "$ANALYSIS_ROOT/layer_probes/selected_layers.json" ]]; do
  sleep "$POLL_SECONDS"
done
status "TRAINING_COMPLETE waiting_for_gpu=$PROBE_GPU"

while [[ -n "$(nvidia-smi --id="$PROBE_GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  sleep "$POLL_SECONDS"
done

status "GPU_AVAILABLE gpu=$PROBE_GPU"
CUDA_VISIBLE_DEVICES="$PROBE_GPU" bash scripts/run_umaze_motion_geometry_probe.sh
status "ALL_COMPLETE"
