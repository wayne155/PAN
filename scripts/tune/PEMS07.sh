#!/bin/bash
export PYTHONPATH=./

device="cuda:0"

# 96, 24

./scripts/traffics/run_wandb_CIDPG.sh "CiDPG" PEMS07 "12" 96 128 512 48 12  0.001 $device
./scripts/traffics/run_wandb_CIDPG.sh "CiDPG" PEMS07 "12" 96 128 512 96 12  0.001 $device
# ./scripts/traffics/run_wandb_CIDPG.sh "CiDPG" PEMS07 "12" 96 128 512 16 8   0.001 $device
# ./scripts/traffics/run_wandb_CIDPG.sh "CiDPG" PEMS07 "12" 96 128 512 24 12   0.001 $device
# ./scripts/traffics/run_wandb_CIDPG.sh "CiDPG" PEMS07 "12" 96 128 512 48 24  0.001 $device
 