#!/bin/bash
export PYTHONPATH=./
device="cuda:2"
./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 168 24  0.001 $device
./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 168 48  0.001 $device
./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 168 64  0.001 $device


# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 96 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 512 128 512 96 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 1024 128 512 96 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 2048 128 512 96 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 1024 128 512 288 144  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 2048 128 512 288 144  0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 96 32  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 256 512 96 64  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 256 512 96 32  0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 192 96   0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 192 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 256 96  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 256 128  0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 96 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 512 512 192 96  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 256 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 512 512 96 48  0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 512 1024 256 128  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 512 1024 256 192 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 256 1024 256 128 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 256 1024 256 192 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 1024 256 128 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 1024 256 128 0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 32 128 96 96 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 32 128 128 128  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 64 192 96 96 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 64 192 128 128 0.001 $device


# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 64 256 32 24 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 64 256 64 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 64 256 128 96  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 64 256 192 128 0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 32 24 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 64 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 128 96  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 512 192 128 0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 128 32 24 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 128 64 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 128 128 96  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh2 "96 168 336 720" 336 128 128 192 128 0.001 $device
