#!/bin/bash
# RULER benchmark on Llama-3.1-8B-Instruct
# Tests across context lengths 4K-128K, budget=1024L

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"  # e.g., meta-llama/Llama-3.1-8B-Instruct
SAVE_DIR="output_dir/results_ruler"
ATTN="flash_attention_2"
MODEL_MAXLEN=120000

EXPERIMENTS=(
    "snapkv 1024"
)

# Each entry is one visible GPU group for a single process.
# device_map="auto" in run_ruler.py will shard the model over the GPUs in that group.
CUDA_DEVICE_GROUPS=("0,1,2")
# For concurrent multi-GPU jobs, split the groups, e.g.:
# CUDA_DEVICE_GROUPS=("0,1" "2,3")
NUM_GPU_GROUPS=${#CUDA_DEVICE_GROUPS[@]}
IDX=0
PIDS=()

cleanup() {
    trap - INT TERM
    if ((${#PIDS[@]} > 0)); then
        echo
        echo "Interrupted. Stopping running RULER Llama-3.1-8B-Instruct jobs..."
        kill -TERM "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
    exit 130
}

wait_for_batch() {
    if ((${#PIDS[@]} == 0)); then
        return
    fi
    wait "${PIDS[@]}"
    PIDS=()
}

trap cleanup INT TERM

for exp in "${EXPERIMENTS[@]}"; do
    read -r method capacity <<< "$exp"
    gpu_group=${CUDA_DEVICE_GROUPS[$((IDX % NUM_GPU_GROUPS))]}

    compression_args=()
    if [[ "${method}" != "FullKV" ]]; then
        compression_args=(
            --compression
            --compression_mode "${method}"
            --compression_budget "${capacity}"
        )
    fi
    CUDA_VISIBLE_DEVICES="${gpu_group}" python3 -u run_ruler.py \
        --model_path "${MODEL_PATH}" \
        --attn_implementation "${ATTN}" \
        --save_dir "${SAVE_DIR}" \
        --model_maxlen "${MODEL_MAXLEN}" \
        "${compression_args[@]}" &
    PIDS+=("$!")

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPU_GROUPS == 0 )); then
        wait_for_batch
    fi
done
wait_for_batch
echo "All RULER Llama-3.1-8B-Instruct experiments completed."
