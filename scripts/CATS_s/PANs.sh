
export PYTHONPATH=./
python3 ./src/experiments/shortterm_s/PAN_s.py \
    --dataset_type="PEMS04" \
    --device="cuda:1" \
    --d_c=128 \
    --d_patch=128 \
    --start_d_model=512 \
    --end_d_model=512 \
    --sh_pa=True \
    --sh_ch=True \
    --patch_len=48 \
    --stride=24 \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[4123,5]'
