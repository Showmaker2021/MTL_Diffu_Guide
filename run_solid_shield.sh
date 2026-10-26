#!/usr/bin/env bash
# =========================================================================
# Train SOLID-MVMoE (SlotDiffMOEModel) under SHIELD's TRAINING PROTOCOL
# =========================================================================
# SHIELD (ICML 2025) training budget reproduced:
#   - Adam, lr 1e-4, weight_decay 1e-6
#   - batch size 128
#   - 20,000 episodes/epoch, 1,000 epochs          (SHIELD used 1k, NOT 5k)
#   - Multi-task: CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW   (Train_ALL)
#   - POMO-style REINFORCE with shared baseline    (already in Trainer.py)
#
# SOLID-MVMoE's OWN hyperparameters are KEPT (not retuned on this benchmark):
#   - slot_num K = 32, slot_iter 3
#   - slot diffusion objective (lambda 0.1) + contrastive (0.01)
#   - dual-pointer gate ceiling gamma = 0.3         (README.md §1.2)
#
# NOTE: this uses each env's U[0,1]^2 random sampling (your existing pipeline),
# NOT SHIELD's country-map sampler. Training *steps/budget* match SHIELD; the
# input distribution does not, so absolute gaps won't match SHIELD's numbers.
# =========================================================================
set -euo pipefail

GPU_ID=${1:-0}
EPOCHS=${2:-1000}          # SHIELD: 1000 epochs
EPISODES=${3:-20000}       # SHIELD: 20000 episodes per epoch
BATCH=${4:-128}            # SHIELD: batch size 128

echo "=========================================================="
echo "SOLID-MVMoE under SHIELD training protocol (GPU $GPU_ID)"
echo "Epochs=$EPOCHS  Episodes/epoch=$EPISODES  Batch=$BATCH"
echo "=========================================================="

python train.py \
    --model_type SlotDiffMOEModel \
    --problem Train_ALL \
    --problem_size 50 \
    --pomo_size 50 \
    --epochs "$EPOCHS" \
    --train_episodes "$EPISODES" \
    --train_batch_size "$BATCH" \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    \
    --embedding_dim 128 \
    --encoder_layer_num 6 \
    --qkv_dim 16 \
    --head_num 8 \
    --ff_hidden_dim 512 \
    --num_experts 4 \
    --topk 2 \
    --expert_loc Enc0 Enc1 Enc2 Enc3 Enc4 Enc5 Dec \
    --routing_level node \
    --routing_method input_choice \
    \
    --slot_num 32 \
    --slot_iter_num 3 \
    --enable_slot_diffusion \
    --max_timesteps 1000 \
    --denoiser_heads 4 \
    --denoiser_layers 2 \
    --diffusion_slot_mode noisy_slots \
    --lambda_diffusion 0.1 \
    --lambda_contrastive 0.01 \
    \
    --seed 2023 \
    --gpu_id "$GPU_ID"
