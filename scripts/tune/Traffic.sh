#!/bin/bash
export PYTHONPATH=./
# $device="cuda:0"
./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 192 48  0.001 "cuda:0" 
./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 192 96  0.001 "cuda:0" 


# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 96 48  0.001 "cuda:0" 

# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 168 24  0.001 "cuda:0" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 168 48  0.001 "cuda:0" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 168 64  0.001 "cuda:0" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 168 12  0.001 "cuda:0" 





# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 64 256 32 24 0.001 "cuda:0" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 64 256 64 48  0.001 "cuda:0" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 64 256 128 96  0.001 "cuda:0" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 64 256 192 128 0.001 "cuda:0" 

# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 32 24 0.001 "cuda:1" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 64 48  0.001 "cuda:1" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 128 96  0.001 "cuda:1" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 512 192 128 0.001 "cuda:1" 

# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 128 32 24 0.001 "cuda:2" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 128 64 48  0.001 "cuda:2" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 128 128 96  0.001 "cuda:2" 
# ./scripts/run_wandb_CIDPG.sh "CiDPG" Traffic "96 168 336 720" 336 128 128 192 128 0.001 "cuda:2" 
