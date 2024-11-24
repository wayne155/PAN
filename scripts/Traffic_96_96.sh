export PYTHONPATH=./
python3 ./src/experiments/CATS.py \
    --dataset_type="ETTh1" \
    --device="cuda:0" \
    --windows=336 \
    --pred_len=168 \
    runs  --seeds='[1,2,3,4,5]'

    # config_wandb --project=ForecastBase \

