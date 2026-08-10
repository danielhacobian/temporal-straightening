#!/usr/bin/env bash
set -euo pipefail

NVME_ROOT="${NVME_ROOT:-/opt/dlami/nvme/deuk-d011}"
REPO_DIR="${REPO_DIR:-$NVME_ROOT/repo}"
VENV="${VENV:-$NVME_ROOT/venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
DATASET_DIR="${DATASET_DIR:-$NVME_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$NVME_ROOT/checkpoints/umaze_calibrated_beta_sweep}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$NVME_ROOT/analysis/umaze_calibrated_beta_sweep}"
LOG_ROOT="${LOG_ROOT:-$NVME_ROOT/logs}"
STATUS="$LOG_ROOT/umaze_calibrated_beta_sweep.status"
R0_DIR="${R0_DIR:-$NVME_ROOT/reference/r0_direction_only}"
BETA1_DIR="${BETA1_DIR:-$NVME_ROOT/reference/calibrated_speed}"
TARGET_EPOCHS="${TARGET_EPOCHS:-20}"
R2_ENDPOINT="${R2_ENDPOINT:-https://2914c19ff6db6db0ee4a54ff30e02f9c.r2.cloudflarestorage.com}"
R2_PREFIX="${R2_PREFIX:-s3://temporal-straightening/umaze_calibrated_beta_sweep_20260809}"

mkdir -p "$OUT_ROOT" "$ANALYSIS_ROOT" "$LOG_ROOT" "$NVME_ROOT/tmp" \
  "$NVME_ROOT/torch_cache" "$NVME_ROOT/hf_cache"

status() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

export TMPDIR="$NVME_ROOT/tmp"
export TORCH_HOME="$NVME_ROOT/torch_cache"
export HF_HOME="$NVME_ROOT/hf_cache"
export DATASET_DIR
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=8

cd "$REPO_DIR"

if [[ -s "$LOG_ROOT/venv_install.pid" ]]; then
  installer_pid="$(cat "$LOG_ROOT/venv_install.pid")"
  while kill -0 "$installer_pid" 2>/dev/null; do
    status "WAIT_RUNTIME installer_pid=$installer_pid"
    sleep 30
  done
fi

if ! "$PYTHON" -c 'import torch, hydra, accelerate, wandb, einops, decord, omegaconf' \
  > "$LOG_ROOT/runtime_smoke.log" 2>&1; then
  status "SETUP_FAILED log=$LOG_ROOT/runtime_smoke.log"
  exit 1
fi
status "RUNTIME_READY python=$PYTHON"

for required in \
  "$DATASET_DIR/point_maze/obses/episode_000_frame_000.pth" \
  "$R0_DIR/checkpoints/model_20.pth" \
  "$BETA1_DIR/checkpoints/model_20.pth"; do
  [[ -s "$required" ]] || { status "MISSING_REQUIRED path=$required"; exit 1; }
done

free_gpu_ids() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '$2 + 0 < 100 {gsub(/[[:space:]]/, "", $1); print $1}'
}

while true; do
  mapfile -t FREE_GPUS < <(free_gpu_ids)
  if (( ${#FREE_GPUS[@]} >= 2 )); then
    break
  fi
  status "WAIT_GPUS need=2 free=${FREE_GPUS[*]:-none}"
  sleep 60
done

GPU_WEAK="${FREE_GPUS[0]}"
GPU_STRONG="${FREE_GPUS[1]}"
status "GPU_ASSIGN weak=$GPU_WEAK strong=$GPU_STRONG"

run_condition() {
  local name="$1" speed_weight="$2" beta="$3" gpu="$4" port="$5"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  if [[ -s "$out/checkpoints/model_${TARGET_EPOCHS}.pth" ]]; then
    status "TRAIN_SKIP condition=$name epoch=$TARGET_EPOCHS"
    return
  fi
  status "TRAIN_START condition=$name beta=$beta direction_weight=0.1 speed_weight=$speed_weight gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m accelerate.commands.launch \
    --num_processes 1 --main_process_port "$port" \
    train.py --config-name umaze_ablation_base \
    "training.epochs=$TARGET_EPOCHS" \
    "training.straighten=false" \
    "training.motion_regularizer=calibrated_speed" \
    "training.motion_regularizer_scale=0.1" \
    "training.speed_calibration_scale=$speed_weight" \
    "env.dataset.use_frame_files=true" \
    "env.num_workers=8" \
    "ckpt_base_path=$out" \
    "hydra.run.dir=$out" \
    > "$out/launcher.log" 2>&1
  [[ -s "$out/checkpoints/model_${TARGET_EPOCHS}.pth" ]]
  sha256sum "$out/checkpoints/model_${TARGET_EPOCHS}.pth" \
    > "$out/checkpoints/model_${TARGET_EPOCHS}.pth.sha256"
  status "TRAIN_COMPLETE condition=$name epoch=$TARGET_EPOCHS"
}

# beta = lambda_speed / lambda_direction. R0 direction stays fixed at 0.1.
# The existing calibrated condition is beta=1 (0.1 / 0.1).
run_condition beta_0p1_weak 0.01 0.1 "$GPU_WEAK" 29851 & weak_pid=$!
run_condition beta_10_strong 1.0 10 "$GPU_STRONG" 29852 & strong_pid=$!

failed=0
wait "$weak_pid" || failed=1
wait "$strong_pid" || failed=1
if (( failed )); then
  status "TRAINING_FAILED inspect=$OUT_ROOT"
  exit 1
fi
status "ALL_TRAINING_COMPLETE"

run_geodesic() {
  local name="$1" checkpoint="$2" gpu="$3"
  local out="$ANALYSIS_ROOT/geodesic_r0_vs_$name"
  mkdir -p "$out"
  status "GEODESIC_START condition=$name gpu=$gpu"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/evaluate_umaze_latent_geodesic.py \
    --r0-checkpoint "$R0_DIR" \
    --r2-checkpoint "$checkpoint" \
    --output-dir "$out" \
    --device cuda:0 \
    > "$out/analysis.log" 2>&1; then
    status "GEODESIC_COMPLETE condition=$name"
  else
    status "GEODESIC_FAILED condition=$name log=$out/analysis.log"
  fi
}

run_geodesic beta_0p1_weak "$OUT_ROOT/beta_0p1_weak" "$GPU_WEAK"
run_geodesic beta_1_reference "$BETA1_DIR" "$GPU_WEAK"
run_geodesic beta_10_strong "$OUT_ROOT/beta_10_strong" "$GPU_WEAK"

cat > "$ANALYSIS_ROOT/README.md" <<'EOF'
# UMaze calibrated-speed beta sweep

This sweep holds the existing R0 direction regularizer fixed at 0.1 and varies
only the true-displacement calibrated speed term.

| Condition | Direction weight | Speed weight | beta = speed / direction |
|---|---:|---:|---:|
| Weak | 0.1 | 0.01 | 0.1 |
| Existing calibrated reference | 0.1 | 0.1 | 1 |
| Strong / deliberately overpowered | 0.1 | 1.0 | 10 |

The speed loss uses the existing implementation: for each adjacent frame,
`latent_speed = ||z[t+1]-z[t]||`, `physical_speed = ||s[t+1,:2]-s[t,:2]||`,
and `r[t] = log(latent_speed) - log(physical_speed)`. The penalty is smooth-L1
on `r[t] - stop_gradient(mean(r))`, so it calibrates relative latent pace to
true displacement without fixing an arbitrary global latent scale.

Training uses the same UMaze dataset, seed, architecture, 20 epochs, batch size
32, direction weight, and optimizer settings as the existing calibrated-speed
experiment. Only the speed weight changes.
EOF

if [[ -x "$HOME/.local/bin/aws" ]]; then
  "$HOME/.local/bin/aws" --profile r2 --endpoint-url "$R2_ENDPOINT" s3 sync \
    "$ANALYSIS_ROOT" "$R2_PREFIX/analysis" --only-show-errors || \
    status "R2_SYNC_WARNING source=analysis"
  for name in beta_0p1_weak beta_10_strong; do
    "$HOME/.local/bin/aws" --profile r2 --endpoint-url "$R2_ENDPOINT" s3 cp \
      "$OUT_ROOT/$name/checkpoints/model_${TARGET_EPOCHS}.pth" \
      "$R2_PREFIX/checkpoints/$name/model_${TARGET_EPOCHS}.pth" \
      --only-show-errors || status "R2_SYNC_WARNING condition=$name"
    "$HOME/.local/bin/aws" --profile r2 --endpoint-url "$R2_ENDPOINT" s3 cp \
      "$OUT_ROOT/$name/launcher.log" "$R2_PREFIX/checkpoints/$name/launcher.log" \
      --only-show-errors || true
  done
fi

status "ALL_COMPLETE r2=$R2_PREFIX"
