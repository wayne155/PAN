export PYTHONPATH=./
python3 ./src/experiments/CiD.py \
    --dataset_type="Traffic" \
    --device="cuda:0" \
    --mask=False \
    --windows=96 \
    --pred_len=96 \
    runs  --seeds='[1,2,3,4,5]'

    # config_wandb --project=ForecastBase \

