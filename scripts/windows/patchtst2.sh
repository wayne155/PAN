#!/bin/bash
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/shortterm/PatchTST.py \
    config_wandb --project=ForecastBase \
    --dataset_type="PEMS08" \
    --device="cuda:7" \
    --patch_len=64 \
    --stride=48 \
    --batch_size=32 \
    --horizon=1 \
    --pred_len="12" \
    --windows=720 \
    runs --seeds='[1,2,3,4,5]'


CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/shortterm/PatchTST.py \
    config_wandb --project=ForecastBase \
    --dataset_type="PEMS08" \
    --device="cuda:7" \
    --patch_len=64 \
    --stride=48 \
    --batch_size=32 \
    --horizon=1 \
    --pred_len="12" \
    --windows=1024 \
    runs --seeds='[1,2,3,4,5]'

CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/shortterm/PatchTST.py \
    config_wandb --project=ForecastBase \
    --dataset_type="PEMS08" \
    --device="cuda:7" \
    --patch_len=64 \
    --stride=48 \
    --batch_size=32 \
    --horizon=1 \
    --pred_len="12" \
    --windows=1536 \
    runs --seeds='[1,2,3,4,5]'

CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/shortterm/PatchTST.py \
    config_wandb --project=ForecastBase \
    --dataset_type="PEMS08" \
    --device="cuda:7" \
    --patch_len=64 \
    --stride=48 \
    --batch_size=32 \
    --horizon=1 \
    --pred_len="12" \
    --windows=2048 \
    runs --seeds='[1,2,3,4,5]'


