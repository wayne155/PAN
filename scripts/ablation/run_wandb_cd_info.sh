#!/bin/bash
# ./scripts/run_wandb.sh "CATS" ExchangeRate "96 168 336 720" "cuda:0" 
model=$1
datasets=($2)  
pred_lens=($3)      
windows=$4
device=$5

for dataset in "${datasets[@]}"
do
    for pred_len in "${pred_lens[@]}"
    do
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/$model.py \
         config_wandb --project=ForecastBase \
          --dataset_type="$dataset" \
          --device="$device" \
          --cd_info=False \
          --batch_size=32 \
          --horizon=1 \
          --pred_len="$pred_len" \
          --windows=$windows \
          runs --seeds='[1,2,3,4,5]'
    done
done

echo "All runs completed."
