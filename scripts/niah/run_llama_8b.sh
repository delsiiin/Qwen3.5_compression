#!/bin/bash
# Needle-in-a-Haystack on Llama-3.1-8B-Instruct
# Tests at budget=128L and budget=1024L

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"  # e.g., meta-llama/Llama-3.1-8B-Instruct
ATTN="flash_attention_2"

EXPERIMENTS=(
    "FullKV 1024 Llama"
    "snapkv 1024 Llama"
    "criticalkv 1024 Llama"
    "defensivekv 1024 Llama"
    "snapkv_hidden_mix 1024 Llama"
    "snapkv_hidden_mix_layer 1024 Llama"
)

# Each entry is one visible GPU group for a single process.
# device_map="auto" in run_niah.py will shard the model over the GPUs in that group.
CUDA_DEVICE_GROUPS=("0,1,2,3")
# For concurrent multi-GPU jobs, split the groups, e.g.:
# CUDA_DEVICE_GROUPS=("0,1" "2,3")
NUM_GPU_GROUPS=${#CUDA_DEVICE_GROUPS[@]}
IDX=0
PIDS=()

cleanup() {
    trap - INT TERM
    if ((${#PIDS[@]} > 0)); then
        echo
        echo "Interrupted. Stopping running Needle Llama-3.1-8B-Instruct jobs..."
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
    read -r method capacity provider <<< "$exp"
    gpu_group=${CUDA_DEVICE_GROUPS[$((IDX % NUM_GPU_GROUPS))]}

    compression_args=()
    version_args=(--model_version "Llama-3.1-8B-Instruct_FullKV_${capacity}")
    if [[ "${method}" != "FullKV" ]]; then
        compression_args=(
            --compression
            --compression_mode "${method}"
            --compression_budget "${capacity}"
            --hidden_mix_profile_path "/home/yangx/new_compression/hidden_mix_profile/hidden_mix_profile_Llama-3.1-8B-Instruct.json"
        )
        version_args=(--model_version "Llama-3.1-8B-Instruct")
    fi
    CUDA_VISIBLE_DEVICES="${gpu_group}" python -u run_niah.py \
        --s_len 5000 --e_len 50001 \
        --model_provider "${provider}" \
        --model_name "${MODEL_PATH}" \
        --attn_implementation "${ATTN}" \
        --step 5000 \
        "${version_args[@]}" \
        "${compression_args[@]}" &
    PIDS+=("$!")

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPU_GROUPS == 0 )); then
        wait_for_batch
    fi
done
wait_for_batch
echo "All Needle Llama-3.1-8B-Instruct experiments completed."
