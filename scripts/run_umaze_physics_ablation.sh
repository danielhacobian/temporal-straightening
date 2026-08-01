#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/temporal-straightening}"
DATA_DIR="${DATA_DIR:-$HOME/data/point_maze}"
R0_DIR="${R0_DIR:-$REPO_DIR/baseline_artifacts/checkpoints/umaze_q1_retrain/r0_direction_only}"
R2_DIR="${R2_DIR:-$REPO_DIR/baseline_artifacts/checkpoints/umaze_q1_retrain/r2_full_matched}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/baseline_artifacts/checkpoints/umaze_physics_layer_ablations}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO_DIR/baseline_artifacts/analysis/umaze_physics_layer_ablations}"
LOG_DIR="$REPO_DIR/baseline_artifacts/logs"
STATUS="$LOG_DIR/umaze_physics_ablation.status"
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/ts310}"
[[ -x "$ENV_PREFIX/bin/python" ]] || ENV_PREFIX="$HOME/miniconda3/envs/ts310"
PYTHON="${PYTHON:-$ENV_PREFIX/bin/python}"
ACCELERATE="${ACCELERATE:-$ENV_PREFIX/bin/accelerate}"
TARGET_EPOCHS="${TARGET_EPOCHS:-20}"
R2_PREFIX="${R2_PREFIX:-s3://temporal-straightening/umaze_physics_layer_ablations}"
R2_ENDPOINT="${R2_ENDPOINT:-https://2914c19ff6db6db0ee4a54ff30e02f9c.r2.cloudflarestorage.com}"

mkdir -p "$OUT_ROOT" "$ANALYSIS_ROOT" "$LOG_DIR"
cd "$REPO_DIR"
export DATASET_DIR="${DATASET_DIR:-$HOME/data}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

status() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

sync_r2() {
  local source="$1" destination="$2"
  if [[ -x "$HOME/.local/bin/aws" ]]; then
    "$HOME/.local/bin/aws" --profile r2 --endpoint-url "$R2_ENDPOINT" s3 sync \
      "$source" "$R2_PREFIX/$destination" --only-show-errors || \
      status "R2_SYNC_WARNING source=$source"
  fi
}

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

for required in "$R0_DIR" "$R2_DIR"; do
  checkpoint_file "$required" >/dev/null || {
    status "MISSING_EXISTING_CHECKPOINT path=$required"
    exit 1
  }
done

PROBE_DIR="$ANALYSIS_ROOT/layer_probes"
if [[ ! -s "$PROBE_DIR/selected_layers.json" ]]; then
  status "LAYER_PROBES_START gpu=7"
  CUDA_VISIBLE_DEVICES=7 "$PYTHON" scripts/probe_umaze_layers.py \
    --checkpoint "$R0_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$PROBE_DIR" \
    --max-windows "${PROBE_WINDOWS:-512}" \
    --batch-size "${PROBE_BATCH_SIZE:-8}" \
    --device cuda:0 \
    > "$PROBE_DIR.log" 2>&1
  status "LAYER_PROBES_COMPLETE"
  sync_r2 "$PROBE_DIR" "analysis/layer_probes"
else
  status "LAYER_PROBES_SKIP existing=true"
fi

read -r DINO_LAYER PREDICTOR_LAYER < <(
  "$PYTHON" - "$PROBE_DIR/selected_layers.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(data["dino"]["layer"], data["predictor"]["layer"])
PY
)
status "SELECTED_LAYERS dino=$DINO_LAYER predictor=$PREDICTOR_LAYER"

USE_FRAME_FILES=false
[[ -s "$DATA_DIR/obses/episode_000_frame_000.pth" ]] && USE_FRAME_FILES=true

run_condition() {
  local name="$1" gpus="$2" port="$3" mode="$4" extra="$5"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  if checkpoint_file "$out" >/dev/null 2>&1; then
    status "TRAIN_SKIP condition=$name epoch=$TARGET_EPOCHS"
    return
  fi
  status "TRAIN_START condition=$name mode=$mode gpus=$gpus"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpus" "$ACCELERATE" launch \
    --multi_gpu --num_processes 2 --main_process_port "$port" \
    train.py --config-name umaze_ablation_base \
    "training.epochs=$TARGET_EPOCHS" \
    "training.straighten=false" \
    "training.motion_regularizer=$mode" \
    "env.dataset.use_frame_files=$USE_FRAME_FILES" \
    "ckpt_base_path=$out" \
    "hydra.run.dir=$out" \
    $extra \
    > "$out/launcher.log" 2>&1
  checkpoint_file "$out" >/dev/null
  sha256sum "$(checkpoint_file "$out")" > "$(checkpoint_file "$out").sha256"
  status "TRAIN_COMPLETE condition=$name"
  sync_r2 "$out" "checkpoints/$name"
}

run_condition calibrated_speed "1,2" 29731 calibrated_speed "" & p1=$!
run_condition factorized "3,4" 29732 factorized \
  "predictor.direction_projection_dim=4 predictor.speed_projection_dim=2" & p2=$!
run_condition layer_aware_factorized "5,6" 29733 layer_aware_factorized \
  "encoder.feature_layer=$DINO_LAYER training.regularizer_predictor_layer=$PREDICTOR_LAYER predictor.direction_projection_dim=4 predictor.speed_projection_dim=2" & p3=$!

failed=0
wait "$p1" || failed=1
wait "$p2" || failed=1
wait "$p3" || failed=1
if (( failed )); then
  status "TRAINING_FAILED inspect=$OUT_ROOT"
  exit 1
fi
status "ALL_TRAINING_COMPLETE"

for condition in calibrated_speed factorized layer_aware_factorized; do
  pair_dir="$ANALYSIS_ROOT/geodesic_r0_vs_$condition"
  mkdir -p "$pair_dir"
  status "GEODESIC_START condition=$condition gpu=7"
  CUDA_VISIBLE_DEVICES=7 "$PYTHON" scripts/evaluate_umaze_latent_geodesic.py \
    --r0-checkpoint "$R0_DIR" \
    --r2-checkpoint "$OUT_ROOT/$condition" \
    --output-dir "$pair_dir" \
    --device cuda:0 \
    > "$pair_dir/analysis.log" 2>&1
  status "GEODESIC_COMPLETE condition=$condition"
done

sync_r2 "$ANALYSIS_ROOT" "analysis"
status "ANALYSIS_COMPLETE"
status "READY_FOR_PLANNING"
