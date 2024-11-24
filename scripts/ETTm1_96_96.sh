export PYTHONPATH=./
python3 ./src/experiments/CiDPG.py \
    --dataset_type="Traffic" \
    --device="cuda:0" \
    --windows=336 \
    --start_d_model=64 \
    --end_d_model=64 \
    --patch_len=192 \
    --patience=5 \
    --stride=192 \
    --flatten=False \
    runs  --seeds='[611,2,3,4,5]'

    # config_wandb --project=ForecastBase \


