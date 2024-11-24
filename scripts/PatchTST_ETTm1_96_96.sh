CUDA_DEVICE_ORDER=PCI_BUS_ID pytexp \
--device="cuda:0" \
--model PatchTST \
--dataset_type=Traffic \
--task Forecast \
--windows=96 \
--pred_len=96 \
--e_layers=7 \
runs --seeds='[1,2,3,4,5]'