#!/usr/bin/env bash
# Ablation: 4 diffusion_slot_mode variants, round-robin across GPUs.
#
# Usage:
#   ./run_ablation.sh [--gpus 0 1] [--epochs 5000] [--extra "..."] [--dry-run]
#
# Extra flags are appended to every training command (e.g. train_batch_size).
set -euo pipefail

GPUS=(0 1)             # default: two GPUs
EPOCHS=5000
EXTRA=""
DRY_RUN=0

# --- minimal flag parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)  shift; GPUS=("$@"); while [[ $# -gt 0 && "$1" != --* ]]; do shift; done ;;
    --epochs) shift; EPOCHS="$1"; shift ;;
    --extra)  shift; EXTRA="$1";   shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

MODES=(normal shuffled stopgrad noisy_slots)

CMD=()
gi=0
for mode in "${MODES[@]}"; do
  gpu="${GPUS[$((gi % ${#GPUS[@]}))]}"
  ((gi++)) || true
  log="ablation_${mode}"

  one=( python train.py \
        --model_type SlotDiffModel \
        --enable_slot_diffusion \
        --diffusion_slot_mode "$mode" \
        --gpu_id "$gpu" \
        --epochs "$EPOCHS" \
        --log_dir "./results/$log" )
  # shellcheck disable=SC2086
  if [[ -n "$EXTRA" ]]; then one+=( $EXTRA ); fi

  CMD+=( "echo \"[ablation] $mode on GPU $gpu (log: $log)\"" )
  CMD+=( "${one[@]}" ">" "./results/$log.log" "2>&1" )

  echo ">> $mode -> GPU $gpu"
done

echo
echo "Modes: ${MODES[*]}"
echo "Epochs: $EPOCHS"
echo "Launcher (bash run_ablation.sh to execute): "
for c in "${CMD[@]}"; do echo "  $c"; done

if [[ $DRY_RUN -eq 1 ]]; then exit 0; fi
# actually run them sequentially (each is a long training job)
for ((i=0; i<${#CMD[@]}; i+=2)); do
  echo "=============================="
  eval "${CMD[$i]}"
  eval "${CMD[$((i+1))]}"
done
