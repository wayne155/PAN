export PYTHONPATH=./
python3 ./src/experiments/PAN.py \
    --dataset_type="ETTh2" \
    --device="cuda:0" \
    --d_c=16 \
    --d_patch=128 \
    --d_model_start=128 \
    --d_model_end=512 \
    --patch_len=192 \
    --stride=48 \
    --windows=336 \
    --pred_len=96 \
    runs  --seeds='[1,2,3,4,5]'

# python3 ./src/experiments/PAN.py \
#     --dataset_type="ETTh2" \
#     --device="cuda:0" \
#     --d_c=8 \
#     --d_patch=128 \
#     --d_model_start=128 \
#     --d_model_end=512 \
#     --patch_len=192 \
#     --stride=48 \
#     --windows=336 \
#     --pred_len=168 \
#     runs  --seeds='[1,2,3,4,5]'

# python3 ./src/experiments/PAN.py \
#     --dataset_type="ETTh2" \
#     --device="cuda:0" \
#     --d_c=8 \
#     --d_patch=128 \
#     --d_model_start=128 \
#     --d_model_end=512 \
#     --patch_len=96 \
#     --stride=48 \
#     --windows=336 \
#     --pred_len=336 \
#     runs  --seeds='[1,2,3,4,5]'
python3 ./src/experiments/PAN.py \
    --dataset_type="ETTh2" \
    --device="cuda:0" \
    --d_c=7 \
    --d_patch=192 \
    --d_model_start=128 \
    --d_model_end=1024 \
    --patch_len=192 \
    --stride=48 \
    --windows=336 \
    --pred_len=720 \
    runs  --seeds='[1,2,3,4,5]'

# python3 ./src/experiments/PAN.py \
#     --dataset_type="ETTh2" \
#     --device="cuda:0" \
#     --d_c=64 \
#     --d_patch=256 \
#     --d_model_start=128 \
#     --d_model_end=1024 \
#     --patch_len=192 \
#     --stride=48 \
#     --windows=336 \
#     --pred_len=720 \
#     runs  --seeds='[1,2,3,4,5]'
