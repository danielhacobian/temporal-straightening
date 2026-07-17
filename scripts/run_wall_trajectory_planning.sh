#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-$HOME/temporal-straightening}"

checkpoint_root="$PWD/baseline_artifacts/checkpoints/wall_speed_ablations"
output_root="$PWD/baseline_artifacts/plans/wall_trajectory_penalty_ablations"
status_log="$PWD/logs/wall_trajectory_planning.status"
gpu_count="${PLANNING_GPUS:-7}"

conditions=(r3_beta1 r1_speed_only r2_full_matched)
seeds=(100 200 300)
offsets=(0 10 20 30 40)

mkdir -p "$output_root" "$PWD/logs"

for condition in "${conditions[@]}"; do
  checkpoint="$checkpoint_root/$condition/checkpoints/model_20.pth"
  while [[ ! -s "$checkpoint" ]]; do
    echo "$(date -Is) WAITING condition=$condition checkpoint=$checkpoint" >> "$status_log"
    sleep 60
  done
done

echo "$(date -Is) ALL_CHECKPOINTS_READY" >> "$status_log"

run_worker() {
  local slot="$1"
  local index=0
  local condition seed offset out rc

  for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do
      for offset in "${offsets[@]}"; do
        if (( index % gpu_count == slot )); then
          out="$output_root/$condition/seed_$seed/chunk_$offset"
          mkdir -p "$out"
          if [[ -s "$out/logs.json" ]] && grep -q 'final_eval/success_rate' "$out/logs.json"; then
            echo "$(date -Is) SKIP condition=$condition seed=$seed offset=$offset" >> "$status_log"
          else
            echo "$(date -Is) START gpu=$slot condition=$condition seed=$seed offset=$offset" >> "$status_log"
            set +e
            env \
              CUDA_VISIBLE_DEVICES="$slot" \
              CPATH="$HOME/.conda/envs/ts/include" \
              LIBRARY_PATH="$HOME/.conda/envs/ts/lib" \
              LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:$HOME/.conda/envs/ts/lib:/usr/lib/nvidia" \
              MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210" \
              DATASET_DIR="$HOME/ts_data/data" \
              WANDB_MODE=disabled \
              HYDRA_FULL_ERROR=1 \
              "$HOME/.conda/envs/ts/bin/python" plan.py \
                --config-name plan_gd.yaml \
                "ckpt_base_path=$checkpoint_root/$condition" \
                "model_name=$condition" \
                model_epoch=20 \
                n_evals=10 \
                +wandb_logging=false \
                "seed=$seed" \
                "+eval_start_index=$offset" \
                "hydra.run.dir=$out" \
                > "$out/runner.log" 2>&1
            rc=$?
            set -e
            echo "$(date -Is) END rc=$rc gpu=$slot condition=$condition seed=$seed offset=$offset" >> "$status_log"
          fi
        fi
        index=$((index + 1))
      done
    done
  done
}

for ((slot = 0; slot < gpu_count; slot++)); do
  run_worker "$slot" &
done
wait

echo "$(date -Is) PLANNING_FINISHED" >> "$status_log"
