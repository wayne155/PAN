export PYTHONPATH=./
python3 ./src/experiments/CiDPG.py \
    --dataset_type="ETTm2" \
    --device="cuda:0" \
    --windows=336 \
    --start_d_model=128 \
    --end_d_model=256 \
    --pred_len=96 \
    --patch_len=96 \
    --patience=8 \
    --stride=64 \
    --flatten=False \
    runs  --seeds='[61,2,3,4,5]'

    # config_wandb --project=ForecastBase \


