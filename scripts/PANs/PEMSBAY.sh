export PYTHONPATH=./
# python3 ./src/experiments/shortterm/PAN_s.py \
#     --dataset_type="PEMS_BAY" \
#     --device="cuda:0" \
#     --d_c=128 \
#     --d_patch=128 \
#     --patch_len=48 \
#     --stride=24 \
#     --windows=96 \
#     --pred_len=3 \
#     runs  --seeds='[1,2,3,4,5]'

# python3 ./src/experiments/shortterm/PAN_s.py \
#     --dataset_type="PEMS_BAY" \
#     --device="cuda:0" \
#     --d_c=128 \
#     --d_patch=128 \
#     --patch_len=48 \
#     --stride=24 \
#     --windows=96 \
#     --pred_len=6 \
#     runs  --seeds='[1,2,3,4,5]'


python3 ./src/experiments/shortterm/PAN_s.py \
    --dataset_type="PEMS_BAY" \
    --device="cuda:0" \
    --d_c=128 \
    --d_patch=128 \
    --start_d_model=512 \
    --sh_pa=True \
    --end_d_model=512 \
    --patch_len=48 \
    --stride=24 \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[1,2,3,4,5]'
