export PYTHONPATH=./
python3 ./src/experiments/CiDPF.py \
    --dataset_type="ExchangeRate" \
    --device="cuda:0" \
    --windows=96 \
    --pred_len=96 \
    runs  --seeds='[1,2,3,4,5]'

    # config_wandb --project=ForecastBase \

