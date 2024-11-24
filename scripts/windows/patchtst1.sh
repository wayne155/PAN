#!/bin/bash
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/shortterm/PatchTST.py \
    config_wandb --project=ForecastBase \
    --dataset_type="PEMS08" \
    --device="cuda:4" \
    --patch_len=64 \
    --stride=48 \
    --batch_size=32 \
    --horizon=1 \
    --pred_len="12" \
    --windows=168 \
    runs --seeds='[1,2,3,4,5]'


CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/shortterm/PatchTST.py \
    config_wandb --project=ForecastBase \
    --dataset_type="PEMS08" \
    --device="cuda:4" \
    --patch_len=64 \
    --stride=48 \
    --batch_size=32 \
    --horizon=1 \
    --pred_len="12" \
    --windows=336 \
    runs --seeds='[1,2,3,4,5]'


CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/shortterm/PatchTST.py \
    config_wandb --project=ForecastBase \
    --dataset_type="PEMS08" \
    --device="cuda:4" \
    --patch_len=64 \
    --stride=48 \
    --batch_size=32 \
    --horizon=1 \
    --pred_len="12" \
    --windows=512 \
    runs --seeds='[1,2,3,4,5]'


