#!/bin/bash

# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 64 256 32 24 0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 64 256 64 48  0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 64 256 128 96  0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 64 256 192 128 0.001 "cuda:0" 

# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 128 512 32 24 0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 128 512 64 48  0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 128 512 128 96  0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 128 512 192 128 0.001 "cuda:0" 

# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 128 128 32 24 0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 128 128 64 48  0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 128 128 128 96  0.001 "cuda:0" 
# ./scripts/run_wandb.sh "CiDPG" ExchangeRate "96 168 336 720" 336 128 128 192 128 0.001 "cuda:0" 


model=$1
datasets=($2)  
pred_lens=($3)      
windows=$4
start_d_model=($5)
end_d_models=($6)
patch_len=$7
stride=$8
lr=$9
device=${10}

for dataset in "${datasets[@]}"
do
    for pred_len in "${pred_lens[@]}"
    do
        for end_d_model in "${end_d_models[@]}"
        do
            echo $device

            CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/shortterm/$model.py \
            --dataset_type="$dataset" \
            --device="$device" \
            --batch_size=32 \
            --horizon=1 \
            --patch_len=$patch_len \
            --stride=$stride \
            --lr=$lr \
            --end_d_model=$end_d_model \
            --start_d_model=$start_d_model \
            --pred_len="$pred_len" \
            --windows=$windows \
            runs --seeds='[1,2,3,4,5]'
        done
    done
done

echo "All runs completed."
