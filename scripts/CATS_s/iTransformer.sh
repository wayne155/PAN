
export PYTHONPATH=./
python3 ./src/experiments/shortterm_s/iTransformer.py \
    --dataset_type="PEMS04" \
    --device="cuda:1" \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[1,2,3,4,5]'
