export PYTHONPATH=./
python3 ./src/experiments/PAN.py \
    --dataset_type="ETTm1" \
    --device="cuda:0" \
    --windows=96 \
    --pred_len=96 \
    --raw_info=False \
    --flatten=False \
    runs  --seeds='[61,2,3,4,5]'

    # config_wandb --project=ForecastBase \

