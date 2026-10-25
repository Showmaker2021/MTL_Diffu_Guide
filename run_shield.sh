#!/usr/bin/env bash
# Script huấn luyện Full SHIELD 5000 Epochs trên server
set -euo pipefail

GPU_ID=${1:-0}
EPOCHS=${2:-5000}
BATCH_SIZE=${3:-128}

echo "=========================================================="
echo "Starting Full SHIELD Training on GPU $GPU_ID"
echo "Epochs: $EPOCHS, Batch size: $BATCH_SIZE"
echo "=========================================================="

python train.py \
    --model_type SHIELD \
    --problem Train_ALL \
    --problem_size 50 \
    --pomo_size 50 \
    --train_batch_size "$BATCH_SIZE" \
    --train_episodes 20000 \
    --epochs "$EPOCHS" \
    --embedding_dim 128 \
    --encoder_layer_num 6 \
    --decoder_layers 3 \
    --use_mod \
    --mod_beta 0.10 \
    --use_cluster \
    --n_clusters 5 \
    --cluster_iters 5 \
    --use_task_prompt \
    --task_dim 5 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --gpu_id "$GPU_ID"
