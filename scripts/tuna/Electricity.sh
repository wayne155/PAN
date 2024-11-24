#!/bin/bash
export PYTHONPATH=./

device="cuda:0"
./scripts/run_wandb_CIDPG.sh "CiDPGA" Electricity "96 168 336 720" 336 128 512 192 48  0.001 $device
./scripts/run_wandb_CIDPG.sh "CiDPGA" Electricity "96 168 336 720" 336 128 512 96 48  0.001 $device
./scripts/run_wandb_CIDPG.sh "CiDPGA" Electricity "96 168 336 720" 336 128 512 192 128  0.001 $device
