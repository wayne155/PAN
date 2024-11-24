export PYTHONPATH=./
python3 ./src/experiments/CiDP.py \
    --dataset_type="Traffic" \
    --device="cuda:0" \
    --windows=96 \
    --pred_len=720 \
    runs  --seeds='[1,2,3,4,5]'

    # config_wandb --project=ForecastBase \

