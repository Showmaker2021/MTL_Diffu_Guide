#!/usr/bin/env bash
# =========================================================================
# Train SOLID-MVMoE (SlotDiffMOEModel) under URS's TRAINING PROTOCOL
# =========================================================================
# The URS baseline's generalist training default (baselines/urs/README.md)
# applied to SOLID-MVMoE -- budget + optimizer + scheduler only:
#   - 500 epochs of 20,000 instances (10M instances)      [URS 500 / 10000*2]
#   - AdamW, lr 1e-4, weight_decay 1e-6
#   - MultiStepLR milestone 451, gamma 0.1
#   - Multi-task Train_ALL (CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW)
#
# SOLID-MVMoE keeps its OWN architecture (NOT URS's 12-layer attention-free
# net / logit-clipping 50 / demand_max1): MoE encoder + K=32 slots + dual
# pointer gate gamma=0.3 + slot-diffusion aux objective. Only the *training
# protocol* is borrowed. Diffusion/contrastive aux losses are kept because
# the diffusion branch is what makes it SOLID; flip the --lambda_* to 0 to
# make it a plain MVMoE comparison.
#
# NOTE: coordinate sampling is still each env's U[0,1]^2 (shared envs), NOT
# URS's or SHIELD's original distribution. So steps/budget match URS; absolute
# gaps will not match any published table.
# =========================================================================
set -euo pipefail

GPU_ID=${1:-0}
EPOCHS=${2:-500}           # URS default: 500
EPISODES=${3:-20000}       # URS default: 20,000 instances / epoch
BATCH=${4:-128}            # URS default: batch 128

echo "=========================================================="
echo "SOLID-MVMoE under URS training protocol (GPU $GPU_ID)"
echo "Epochs=$EPOCHS  Episodes/epoch=$EPISODES  Batch=$BATCH"
echo "Schedule: AdamW lr 1e-4 wd 1e-6, milestone 451, gamma 0.1"
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
    --optimizer_type AdamW \
    --milestones 451 \
    --gamma 0.1 \
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
