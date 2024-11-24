export PYTHONPATH=./
python3 ./src/experiments/PFN.py \
    --dataset_type="ExchangeRate" \
    --device="cuda:0" \
    --windows=96 \
    --pred_len=96 \
    --flatten=False \
    runs  --seeds='[61,2,3,4,5]'

    # config_wandb --project=ForecastBase \

