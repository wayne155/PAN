#!/bin/bash
export PYTHONPATH=./

device="cuda:1"
./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 168 24  0.001 $device
./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 168 48  0.001 $device
./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 168 64  0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 96 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 512 512 192 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 256 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 512 512 256 48  0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 192 96  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 192 48  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 256 96  0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 256 128 0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 192 128   0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 192 192 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 1024 128 128 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 1024 192 192 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 1024 168 128 0.001 $device
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 1024 168 128 0.001 $device

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 64 256 32 24 0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 64 256 64 48  0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 64 256 128 96  0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 64 256 192 128 0.001 "cuda:5" 

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 32 24 0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 64 48  0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 128 96  0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 512 192 128 0.001 "cuda:5" 

# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 128 32 24 0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 128 64 48  0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 128 128 96  0.001 "cuda:5" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" ETTh1 "96 168 336 720" 336 128 128 192 128 0.001 "cuda:5" 
