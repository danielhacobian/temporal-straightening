#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/temporal-straightening}"
DATASET_DIR="${DATASET_DIR:-$HOME/data}"
OLD_ROOT="$REPO_DIR/baseline_artifacts/checkpoints/umaze_q1_retrain"
NEW_ROOT="$REPO_DIR/baseline_artifacts/checkpoints/umaze_physics_layer_ablations"
OUTPUT_ROOT="$REPO_DIR/baseline_artifacts/plans/umaze_physics_layer_ablations"
STATUS="$REPO_DIR/baseline_artifacts/logs/umaze_physics_planning.status"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/ts310/bin/python}"
R2_PREFIX="${R2_PREFIX:-s3://temporal-straightening/umaze_physics_layer_ablations}"
R2_ENDPOINT="${R2_ENDPOINT:-https://2914c19ff6db6db0ee4a54ff30e02f9c.r2.cloudflarestorage.com}"
GPU_IDS="${PLANNING_GPU_IDS:-1,2,3,4,5,6,7}"
IFS=',' read -r -a gpus <<< "$GPU_IDS"
conditions=(r0 r2 calibrated_speed factorized layer_aware_factorized)
seeds=(100 200 300)
offsets=(0 10 20 30 40)

mkdir -p "$OUTPUT_ROOT" "$(dirname "$STATUS")"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled MUJOCO_GL=osmesa MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/ts310/lib:${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

status() { echo "$(date -Is) $*" | tee -a "$STATUS"; }
condition_dir() {
  case "$1" in
    r0) echo "$OLD_ROOT/r0_direction_only" ;;
    r2) echo "$OLD_ROOT/r2_full_matched" ;;
    *) echo "$NEW_ROOT/$1" ;;
  esac
}

run_job() {
  local gpu="$1" condition="$2" seed="$3" offset="$4"
  local out="$OUTPUT_ROOT/$condition/seed_$seed/chunk_$offset"
  mkdir -p "$out"
  if [[ -s "$out/logs.json" ]] && grep -q 'final_eval/success_rate' "$out/logs.json"; then
    status "PLAN_SKIP condition=$condition seed=$seed offset=$offset"
    return
  fi
  status "PLAN_START gpu=$gpu condition=$condition seed=$seed offset=$offset"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" plan.py --config-name plan_gd.yaml \
    "ckpt_base_path=$(condition_dir "$condition")" \
    "model_name=$condition" model_epoch=20 n_evals=10 \
    +wandb_logging=false "seed=$seed" "+eval_start_index=$offset" \
    "hydra.run.dir=$out" > "$out/runner.log" 2>&1
  status "PLAN_END gpu=$gpu condition=$condition seed=$seed offset=$offset"
}

run_worker() {
  local worker_index="$1" gpu="$2" index=0
  for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do
      for offset in "${offsets[@]}"; do
        if (( index % ${#gpus[@]} == worker_index )); then
          run_job "$gpu" "$condition" "$seed" "$offset"
        fi
        index=$((index + 1))
      done
    done
  done
}

pids=()
for ((worker=0; worker<${#gpus[@]}; worker++)); do
  run_worker "$worker" "${gpus[$worker]}" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for condition in "${conditions[@]}"; do
  for seed in "${seeds[@]}"; do
    seed_dir="$OUTPUT_ROOT/$condition/seed_$seed"
    chunks=()
    for offset in "${offsets[@]}"; do chunks+=(--chunk "$offset:10:$seed_dir/chunk_$offset"); done
    "$PYTHON" aggregate_plan_chunks.py "${chunks[@]}" --seed "$seed" \
      --expected-evals 50 --output "$seed_dir/aggregate.json" \
      > "$seed_dir/aggregate.stdout"
  done
done

args=()
for condition in "${conditions[@]}"; do args+=(--condition "$condition=$OUTPUT_ROOT/$condition"); done
"$PYTHON" aggregate_condition_seeds.py "${args[@]}" --baseline r0 \
  --output "$OUTPUT_ROOT/comparison.json" > "$OUTPUT_ROOT/comparison.stdout"

if [[ -x "$HOME/.local/bin/aws" ]]; then
  "$HOME/.local/bin/aws" --profile r2 --endpoint-url "$R2_ENDPOINT" s3 sync \
    "$OUTPUT_ROOT" "$R2_PREFIX/plans" --only-show-errors
fi
status "ALL_PLANNING_COMPLETE comparison=$OUTPUT_ROOT/comparison.json"
