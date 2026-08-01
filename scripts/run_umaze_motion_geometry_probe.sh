#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/temporal-straightening}"
DATA_DIR="${DATA_DIR:-$HOME/data/point_maze}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_DIR/baseline_artifacts/checkpoints/umaze_physics_layer_ablations}"
Q1_ROOT="${Q1_ROOT:-$REPO_DIR/baseline_artifacts/checkpoints/umaze_q1_retrain}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO_DIR/baseline_artifacts/analysis/umaze_physics_layer_ablations}"
OUTPUT_DIR="${OUTPUT_DIR:-$ANALYSIS_ROOT/motion_geometry_probes}"
SELECTED_LAYERS="${SELECTED_LAYERS:-$ANALYSIS_ROOT/layer_probes/selected_layers.json}"
LOG_DIR="$REPO_DIR/baseline_artifacts/logs"
STATUS="$LOG_DIR/umaze_motion_geometry_probe.status"
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/ts310}"
[[ -x "$ENV_PREFIX/bin/python" ]] || ENV_PREFIX="$HOME/miniconda3/envs/ts310"
PYTHON="${PYTHON:-$ENV_PREFIX/bin/python}"
TARGET_EPOCHS="${TARGET_EPOCHS:-20}"
R2_PREFIX="${R2_PREFIX:-s3://temporal-straightening/umaze_physics_layer_ablations/analysis/motion_geometry_probes}"
R2_ENDPOINT="${R2_ENDPOINT:-https://2914c19ff6db6db0ee4a54ff30e02f9c.r2.cloudflarestorage.com}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

status() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

checkpoint_file() {
  local root="$1"
  if [[ -s "$root/checkpoints/model_${TARGET_EPOCHS}.pth" ]]; then
    echo "$root/checkpoints/model_${TARGET_EPOCHS}.pth"
  elif [[ -s "$root/model_${TARGET_EPOCHS}.pth" ]]; then
    echo "$root/model_${TARGET_EPOCHS}.pth"
  else
    return 1
  fi
}

conditions=(
  "r0=$Q1_ROOT/r0_direction_only"
  "r2=$Q1_ROOT/r2_full_matched"
  "calibrated_speed=$CHECKPOINT_ROOT/calibrated_speed"
  "factorized=$CHECKPOINT_ROOT/factorized"
  "layer_aware_factorized=$CHECKPOINT_ROOT/layer_aware_factorized"
)
checkpoint_args=()
for entry in "${conditions[@]}"; do
  label="${entry%%=*}"
  root="${entry#*=}"
  file="$(checkpoint_file "$root")" || {
    status "MISSING_CHECKPOINT condition=$label path=$root"
    exit 1
  }
  checkpoint_args+=(--checkpoint "$label=$file")
done

if [[ -s "$OUTPUT_DIR/metrics.json" ]]; then
  status "PROBE_SKIP existing=$OUTPUT_DIR/metrics.json"
  exit 0
fi

status "PROBE_START device=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" scripts/probe_umaze_motion_geometry.py \
  "${checkpoint_args[@]}" \
  --data-dir "$DATA_DIR" \
  --selected-layers "$SELECTED_LAYERS" \
  --output-dir "$OUTPUT_DIR" \
  --max-windows "${PROBE_WINDOWS:-512}" \
  --batch-size "${PROBE_BATCH_SIZE:-8}" \
  --device cuda:0 \
  > "$OUTPUT_DIR/probe.log" 2>&1
status "PROBE_COMPLETE output=$OUTPUT_DIR"

if [[ -x "$HOME/.local/bin/aws" ]]; then
  "$HOME/.local/bin/aws" --profile r2 --endpoint-url "$R2_ENDPOINT" s3 sync \
    "$OUTPUT_DIR" "$R2_PREFIX" --only-show-errors || status "R2_SYNC_WARNING"
  status "R2_SYNC_COMPLETE destination=$R2_PREFIX"
fi
