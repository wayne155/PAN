export PYTHONPATH=./

python3 ./src/experiments/PAN_PA.py \
    --dataset_type="Traffic" \
    --device="cuda:1" \
    --d_c=128 \
    --d_patch=128 \
    --start_d_model=128 \
    --end_d_model=128 \
    --patch_len=192 \
    --raw_info=False \
    --stride=128 \
    --shared=True \
    --windows=336 \
    --pred_len=720 \
    runs  --seeds='[1,2,3,4,5]'
