# attention heatmap visualization for qwen3.5-9b

CUDA_VISIBLE_DEVICES=0,1 python pred.py --model qwen3.5-9b --attn_heatmap_mode --num_samples 2 --attn_max_prefill_tokens 5000

python attn_viewer/server.py --root results/attn_heatmaps

# efficiency benchmark for qwen3.5-9b

CUDA_VISIBLE_DEVICES=0,1 python efficiency_benchmark.py --model_path Qwen/Qwen3.5-9B --batch_size 1 --input_len 4096 --output_len 128 --num_warmups 1 --num_runs 3