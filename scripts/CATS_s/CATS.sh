
export PYTHONPATH=./
python3 ./src/experiments/shortterm_s/CATS.py \
    --dataset_type="PEMS08" \
    --device="cuda:4" \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[1,2,3,4,5]'
