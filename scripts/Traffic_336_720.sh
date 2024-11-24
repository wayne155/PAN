export PYTHONPATH=./
python3 ./src/experiments/CiDPG.py \
    --dataset_type="Traffic" \
    --device="cuda:0" \
    --windows=336 \
    --start_d_model=128 \
    --end_d_model=512 \
    --patch_len=192 \
    --patience=5\
    --stride=128 \
    --lr=0.0005 \
    --flatten=False \
    runs  --seeds='[612,2,3,4,5]'

    # config_wandb --project=ForecastBase \


