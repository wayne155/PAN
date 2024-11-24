export PYTHONPATH=./
python3 ./src/experiments/CDPatchTST.py \
    --dataset_type="Weather" \
    --device="cuda:0" \
    --windows=96 \
    --pred_len=96 \
    runs  --seeds='[1,2,3,4,5]'

    # config_wandb --project=ForecastBase \

