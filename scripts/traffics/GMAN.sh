export PYTHONPATH=./
python3 ./src/experiments/shortterm/GMAN.py \
    --dataset_type="METR_LA" \
    --device="cuda:0" \
    --windows=96 \
    --pred_len=12 \
    runs  --seeds='[1,2,3,4,5]'


