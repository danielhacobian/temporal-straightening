#!/usr/bin/env bash
# run_pusht_a100.sh — train the six DINOv2 PushT rows of Table 1 on one box.
#
#   Wang et al., "Temporal Straightening for Latent Planning", ICML 2026.
#   PushT column, DINOv2 encoder rows (the ResNet rows are a separate script).
#
# THE FOUR CONDITIONS (key -> encoder / straightening / paper open-loop target):
#   cls          dino_cls      OFF          1x384    19.33 +/- 8.22
#   patch        dino          OFF        14x14x384  56.00 +/- 4.32
#   channel_off  dino_channel  OFF        14x14x8    70.00 +/- 1.63
#   channel_on   dino_channel  aggcos1e-1 14x14x8    77.33 +/- 6.18   <- best (shaded)
#
# NO SMOKE STAGE — by request. Every shape that a smoke test would confirm is
# logged in the FIRST SECONDS of the real run and stays in the log:
#   - predictor pos_embedding shape       (train.py:218  "... pos_embedding shape=(1, N, D)")
#   - "Unknown projector 'none' ..."      (models/dino.py:145, patch-no-projector rows)
# So you verify dimensions DURING the run:  bash run_pusht_a100.sh dims
#
#   Expected pos_embedding = (1, tokens, emb):
#     emb (last dim):  404 = 384 DINO features + 10 proprio + 10 action
#                       28 =   8 channel-projector features + 10 + 10
#     tokens (middle): 3          -> pooled to 1 CLS token/frame  (cls)
#                      3 x 196=588 -> channel proj, 14x14 grid    (channel_*)
#                      3 x (grid)  -> raw patch, hundreds of tokens (patch)
#   Quick read: last dim 28 => channel; ==3 middle => cls; big middle => patch.
#
# PushT vs the PointMaze scripts:
#   - Dataset is VIDEO (decord), not per-frame files. There is NO fast-loader /
#     preprocess step here — that whole dance was PointMaze-only.
#   - Data lives at $DATASET_DIR/pusht_noise (conf/env/pusht.yaml).
#   - Training needs no MuJoCo; the pusht gym env only loads at PLANNING time.
#
# encoder_lr: Table 3's footnote ("lr=1e-6 for no straightening") governs the
#   PROJECTOR lr, so channel_off uses 1e-6 and the straightening row (channel_on)
#   uses the 1e-5 default. cls and patch have NO trainable encoder params, so lr
#   is a no-op there — but the run-dir template omits feature_key, so cls and
#   patch would collide on 'pusht_False_..._projnone_dim384_hw14_...'. We split
#   them by lr (cls=1e-5, patch=1e-6), the same trick run_row6_a100.sh uses.
#
# STAGES (JupyterHub terminal):
#   export DATASET_DIR=/home/jupyter-deuk-c4e4/data     # must contain pusht_noise/
#   bash run_pusht_a100.sh train                 # all 6, one per free GPU, nohup'd
#   bash run_pusht_a100.sh train channel_on cls  # OR just the ones you name
#   bash run_pusht_a100.sh dims                  # verify shapes on the LIVE runs
#   bash run_pusht_a100.sh status                # epochs done + checkpoints
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ts_ckpts_pusht}"
DATA="${DATA:-${DATASET_DIR:-$HOME/data}/pusht_noise}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
EPOCHS="${EPOCHS:-20}"
FREE_MIB="${FREE_MIB:-2000}"   # a GPU with less than this used (MiB) counts as free
STAGE="${1:-}"; shift || true

ALL_CONDS="cls patch channel_off channel_on"

mkdir -p "$LOG_DIR"
usage() { echo "usage: bash run_pusht_a100.sh {train|dims|status} [cond ...]"; echo "  conds: $ALL_CONDS"; exit 1; }
[ -n "$STAGE" ] || usage
[ -f "$REPO/train.py" ] || { echo "!! run from the repo root (train.py not found)"; exit 1; }

# torch.hub.load() needs the cached dinov2 repo importable.
if [ -z "${PYTHONPATH:-}" ] || ! echo "${PYTHONPATH:-}" | grep -q facebookresearch_dinov2; then
  HUB_DINO="$(ls -d "$HOME/.cache/torch/hub/facebookresearch_dinov2"* 2>/dev/null | head -1)"
  [ -n "$HUB_DINO" ] && export PYTHONPATH="$HUB_DINO:${PYTHONPATH:-}"
fi

# cond -> "encoder=<..> training.straighten=<..> training.encoder_lr=<..>"
cond_flags() {
  case "$1" in
    cls)         echo "encoder=dino_cls     training.straighten=False      training.encoder_lr=1e-5" ;;
    patch)       echo "encoder=dino         training.straighten=False      training.encoder_lr=1e-6" ;;
    channel_off) echo "encoder=dino_channel training.straighten=False      training.encoder_lr=1e-6" ;;
    channel_on)  echo "encoder=dino_channel training.straighten=aggcos1e-1 training.encoder_lr=1e-5" ;;
    *) echo "!! unknown cond '$1' (want: $ALL_CONDS)" >&2; return 1 ;;
  esac
}

# cond -> glob matching its run dir under $CKPT_ROOT/test (proj+dim+hw+lr are unique)
find_run_dir() {
  local g
  case "$1" in
    cls)         g='pusht_False_*projnone_dim384_hw14*lr1e-05' ;;
    patch)       g='pusht_False_*projnone_dim384_hw14*lr1e-06' ;;
    channel_off) g='pusht_*projchannel_dim8_hw14*lr1e-06' ;;
    channel_on)  g='pusht_*projchannel_dim8_hw14*lr1e-05' ;;
  esac
  find "$CKPT_ROOT/test" -maxdepth 1 -type d -name "$g" 2>/dev/null | head -1
}

CONDS="${*:-$ALL_CONDS}"
for c in $CONDS; do cond_flags "$c" >/dev/null || usage; done

case "$STAGE" in

train)
  [ -d "$DATA" ] || { echo "!! dataset dir missing: $DATA"; echo "   set DATASET_DIR so \$DATASET_DIR/pusht_noise exists (see README > Datasets)."; exit 1; }

  # Pick free GPUs unless GPUS is given explicitly (comma list, e.g. GPUS=2,3,4).
  if [ -n "${GPUS:-}" ]; then
    read -r -a GPU_ARR <<< "$(echo "$GPUS" | tr ',' ' ')"
  else
    mapfile -t GPU_ARR < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v t="$FREE_MIB" '($2+0)<t {gsub(/ /,"",$1); print $1}')
  fi
  N_GPU=${#GPU_ARR[@]}
  [ "$N_GPU" -gt 0 ] || { echo "!! no free GPU (used < ${FREE_MIB} MiB). Set GPUS=... to force."; exit 1; }
  echo "[gpu] free/assigned GPUs: ${GPU_ARR[*]}"

  i=0
  for c in $CONDS; do
    if [ "$i" -ge "$N_GPU" ]; then
      echo "!! only $N_GPU GPU(s) — skipping '$c' and any after it. Re-run 'train $c ...' when a GPU frees."
      break
    fi
    g="${GPU_ARR[$i]}"; i=$((i+1))
    log="$LOG_DIR/pusht_${c}.log"
    read -r -a F <<< "$(cond_flags "$c")"
    echo "----- launch '$c' on GPU $g -> $log -----"
    # setsid + nohup: new session AND SIGHUP ignored. Fully detaches from this
    # terminal, so closing the browser / logging out of JupyterLab does NOT stop
    # it. (A JupyterHub *idle cull* kills the whole cgroup regardless — see below.)
    CUDA_VISIBLE_DEVICES="$g" setsid nohup python train.py --config-name train.yaml \
      env=pusht "${F[@]}" training.epochs="$EPOCHS" ckpt_base_path="$CKPT_ROOT" \
      > "$log" 2>&1 &
    echo "  pid $!   flags: ${F[*]}"
  done
  echo ""
  echo "All launched detached (setsid+nohup): closing the browser / logging out"
  echo "of JupyterLab does NOT stop them. The ONE thing that does is a JupyterHub"
  echo "idle cull, which kills the whole cgroup — but files persist and train.py"
  echo "resumes from model_latest.pth, so just re-run the same 'train <conds>'."
  echo "Verify shapes on the live runs:  bash run_pusht_a100.sh dims"
  ;;

dims)
  # Read the shape/projector lines straight out of the running logs.
  for c in $CONDS; do
    log="$LOG_DIR/pusht_${c}.log"
    echo "===== $c ====="
    if [ ! -f "$log" ]; then echo "  (no log — not launched)"; continue; fi
    grep -m1 -i 'pos_embedding[^=]*shape' "$log" 2>/dev/null | sed 's/^/  /' \
      || echo "  (pos_embedding not logged yet — give it a few seconds)"
    grep -m1 -i "Unknown projector" "$log" 2>/dev/null | sed 's/^/  /' || true
    grep -m1 -iE 'Traceback|Error|CUDA out of memory|FileNotFound' "$log" 2>/dev/null | sed 's/^/  !! /' || true
  done
  echo ""
  echo "expect: cls -> (1, 3, 404)   channel_* -> (1, 588, 28)   patch -> (1, <big>, 404)"
  ;;

status)
  for c in $CONDS; do
    log="$LOG_DIR/pusht_${c}.log"; run="$(find_run_dir "$c")"
    ep=$(grep -c "Training loss" "$log" 2>/dev/null || echo 0)
    echo "===== $c   (epochs w/ loss logged: $ep / $EPOCHS) ====="
    [ -f "$log" ] && tail -2 "$log" | sed 's/^/  /' || echo "  (no log yet)"
    [ -n "$run" ] && ls -1 "$run/checkpoints/" 2>/dev/null | sed 's/^/  ckpt: /' || echo "  (no run dir yet)"
  done
  ;;

*) usage ;;
esac
