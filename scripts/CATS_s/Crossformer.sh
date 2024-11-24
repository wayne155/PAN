
export PYTHONPATH=./
python3 ./src/experiments/shortterm_s/Crossformer.py \
    --dataset_type="PEMS08" \
    --device="cuda:0" \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[2,3,4,5]'
