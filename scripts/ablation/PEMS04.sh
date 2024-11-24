export PYTHONPATH=./

# python3 ./src/experiments/shortterm/PAN_PA.py \
#     --dataset_type="PEMS04" \
#     --device="cuda:0" \
#     --d_c=256 \
#     --d_patch=128 \
#     --patch_len=48 \
#     --start_d_model=128 \
#     --end_d_model=128 \
#     --raw_info=False \
#     --flatten=True \
#     --shared=true \
#     --stride=24 \
#     --windows=96 \
#     --pred_len=12 \
#     runs  --seeds='[2,3,4,5]'
python3 ./src/experiments/shortterm/PAN_PA.py \
    --dataset_type="PEMS08" \
    --device="cuda:0" \
    --d_c=256 \
    --d_patch=128 \
    --patch_len=48 \
    --start_d_model=128 \
    --end_d_model=128 \
    --raw_info=False \
    --flatten=True \
    --shared=true \
    --stride=24 \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[2,3,4,5]'
