# attention heatmap visualization for qwen3.5-9b

CUDA_VISIBLE_DEVICES=0,1 python pred.py --model qwen3.5-9b --attn_heatmap_mode --num_samples 2 --attn_max_prefill_tokens 5000

python attn_viewer/server.py --root results/attn_heatmaps
