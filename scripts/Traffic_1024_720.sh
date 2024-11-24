export PYTHONPATH=./
python3 ./src/experiments/CiDPG.py \
    --dataset_type="Traffic" \
    --device="cuda:0" \
    --windows=1024 \
    --start_d_model=96 \
    --end_d_model=512 \
    --patch_len=720 \
    --stride=92 \
    --flatten=False \
    runs  --seeds='[61,2,3,4,5]'

    # config_wandb --project=ForecastBase \


