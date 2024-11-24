
export PYTHONPATH=./
python3 ./src/experiments/shortterm_s/DLinear.py \
    --dataset_type="PEMS04" \
    --device="cuda:0" \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[2,3,4,5]'
