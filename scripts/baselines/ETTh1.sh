CUDA_DEVICE_ORDER=PCI_BUS_ID pytexp \
--device="cuda:0" \
--model PatchTST \
--dataset_type=ETTh1 \
--task Forecast \
--windows=96 \
--pred_len=96 \
--e_layers=2 \
runs --seeds='[1,2,3,4,5]'