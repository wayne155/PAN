export PYTHONPATH=./


python3 ./src/experiments/shortterm/PAN_s.py \
    --dataset_type="PEMS07" \
    --device="cuda:0" \
    --d_c=256 \
    --d_patch=128 \
    --sh_ch=True \
    --start_d_model=512 \
    --end_d_model=512 \
    --patch_len=48 \
    --stride=24 \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[1,2,3,4,5]'

