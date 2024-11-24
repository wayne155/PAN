#!/bin/bash
export PYTHONPATH=./
device="cuda:0"

./scripts/run_wandb_CIDPG.sh "CiDPGA" ETTh2 "96 168 336 720" 336 128 512 192 48  0.001 $device
./scripts/run_wandb_CIDPG.sh "CiDPGA" ETTh2 "96 168 336 720" 336 128 512 96 48  0.001 $device

# ./scripts/run_wandb_CiDPGA.sh "CiDPGAA" ETTh2 "96 168 336 720" 1024 128 512 96 48  0.001 $device
# ./scripts/run_wandb_CiDPGA.sh "CiDPGAA" ETTh2 "96 168 336 720" 1024 128 512 288 144  0.001 $device
# ./scripts/run_wandb_CiDPGA.sh "CiDPGAA" ETTh2 "96 168 336 720" 2048 128 512 96 48  0.001 $device
# ./scripts/run_wandb_CiDPGA.sh "CiDPGAA" ETTh2 "96 168 336 720" 2048 128 512 288 144  0.001 $device
