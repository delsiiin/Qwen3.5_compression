###### attention heatmap visualization for qwen3.5-9b

# CUDA_VISIBLE_DEVICES=0,1 python run_longbench.py --model qwen3.5-9b --attn_heatmap_mode --num_samples 3 --attn_max_prefill_tokens 2500 --domain "single-document QA" --model_maxlen 2000

# python attn_viewer/server.py --root results/attn_heatmaps

###### observation for qwen3-8b
# python run_longbench.py \
#   --model Qwen2.5-7B-Instruct \
#   --query_window_similarity_mode \
#   --query_window_similarity_submode hidden_states \
#   --query_window_size 8 \
#   --num_samples 3 \
#   --model_maxlen 10000 \
#   --domain "single-document QA" 

# python run_longbench.py \
#   --model Llama-3.1-8B-Instruct \
#   --snapkv_observation_mode \
#   --snapkv_observation_budget 1024 \
#   --num_samples 3 \
#   --model_maxlen 10000 \
#   --domain "single-document QA"

# python run_longbench.py \
#   --model Qwen2.5-7B-Instruct \
#   --attn_output_ratio_mode \
#   --attn_output_ratio_layers all \
#   --num_samples 3 \
#   --model_maxlen 10000 \
#   --domain "multi-document QA"

# python run_longbench.py \
#   --model Qwen2.5-7B-Instruct \
#   --hidden_state_pca_mode \
#   --hidden_state_pca_layers all \
#   --num_samples 3 \
#   --model_maxlen 10000 \
#   --domain "Code Repository Understanding"

# python build_hidden_mix_profile.py \
#   --similarity_npz output_dir/results_longbench/query_window_similarity/qwen3-4b_domain_single-document_QA_20260606_055353/samples/sample_0000_66f36490821e116aacb2cc22/prefill_000_hidden_states_query_window_similarity.npz \
#   --attn_output_ratio_npz output_dir/results_longbench/attn_output_ratios/qwen3-4b_domain_multi-document_QA_20260606_055443/samples/sample_0000_66f94f9ebb02136c067c4fde/prefill_000_attn_output_hidden_l2_ratio.npz \
#   --hidden_state_pca_npz output_dir/results_longbench/hidden_state_pca/qwen3-4b_domain_Code_Repository_Understanding_20260606_055715/samples/sample_0000_66fa208bbb02136c067c5fc1/prefill_000_key_value_state_pca.npz \
#   --group_scheme similarity \
#   --output hidden_mix_profile/hidden_mix_profile_qwen3_4b.json \
#   --key_pca_angle_threshold_mode mean_without_outliers\
#   --key_pca_angle_threshold 10 \
#   --group_threshold 0.9 \
#   --max_group_size 5

# python build_hidden_mix_profile.py \
#   --similarity_npz output_dir/results_longbench/query_window_similarity/Llama-3.1-8B-Instruct_domain_multi-document_QA_20260527_082733/samples/sample_0001_66ec0c4c821e116aacb1994a/prefill_000_query_window_similarity.npz \
#   --attn_output_ratio_npz output_dir/results_longbench/attn_output_ratios/Llama-3.1-8B-Instruct_domain_multi-document_QA_20260530_133738/samples/sample_0000_66f94f9ebb02136c067c4fde/prefill_000_attn_output_hidden_l2_ratio.npz \
#   --hidden_state_pca_npz output_dir/results_longbench/hidden_state_pca/Llama-3.1-8B-Instruct_domain_Code_Repository_Understanding_20260603_154145/samples/sample_0000_66fa208bbb02136c067c5fc1/prefill_000_key_value_state_pca.npz \
#   --group_scheme key_pca_angle \
#   --output hidden_mix_profile/hidden_mix_profile_Llama-3.1-8B-Instruct.json \
#   --key_pca_angle_threshold_mode mean_without_outliers\
#   --key_pca_angle_threshold 10 \
#   --group_threshold 0.85 \
#   --max_group_size 5


# python build_hidden_mix_profile.py \
#   --similarity_npz output_dir/results_longbench/query_window_similarity/mistral-7b_domain_single-document_QA_20260606_073538/samples/sample_0000_66f36490821e116aacb2cc22/prefill_000_hidden_states_query_window_similarity.npz \
#   --attn_output_ratio_npz output_dir/results_longbench/attn_output_ratios/mistral-7b_domain_multi-document_QA_20260606_080345/samples/sample_0000_66f94f9ebb02136c067c4fde/prefill_000_attn_output_hidden_l2_ratio.npz \
#   --hidden_state_pca_npz output_dir/results_longbench/hidden_state_pca/mistral-7b_domain_Code_Repository_Understanding_20260606_080805/samples/sample_0000_66fa208bbb02136c067c5fc1/prefill_000_key_value_state_pca.npz \
#   --group_scheme key_pca_angle \
#   --output hidden_mix_profile/hidden_mix_profile_mistral_7b.json \
#   --key_pca_angle_threshold_mode mean_without_outliers\
#   --key_pca_angle_threshold 10 \
#   --group_threshold 0.9 \
#   --max_group_size 5


# python build_hidden_mix_profile.py \
#   --similarity_npz output_dir/results_longbench/query_window_similarity/Qwen2.5-7B-Instruct_domain_single-document_QA_20260606_093113/samples/sample_0000_66f36490821e116aacb2cc22/prefill_000_hidden_states_query_window_similarity.npz \
#   --attn_output_ratio_npz output_dir/results_longbench/attn_output_ratios/Qwen2.5-7B-Instruct_domain_multi-document_QA_20260606_095412/samples/sample_0000_66f94f9ebb02136c067c4fde/prefill_000_attn_output_hidden_l2_ratio.npz \
#   --hidden_state_pca_npz output_dir/results_longbench/hidden_state_pca/Qwen2.5-7B-Instruct_domain_Code_Repository_Understanding_20260606_095617/samples/sample_0000_66fa208bbb02136c067c5fc1/prefill_000_key_value_state_pca.npz \
#   --group_scheme key_pca_angle \
#   --output hidden_mix_profile/hidden_mix_profile_Qwen2.5-7B-Instruct.json \
#   --key_pca_angle_threshold_mode mean_without_outliers\
#   --key_pca_angle_threshold 10 \
#   --max_group_size 5


##### efficiency benchmark for qwen3.5-9b

# for input_len in 10000 20000 30000 40000 50000 60000; do
#     echo "Running benchmark with input_len=${input_len}"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python run_efficiency.py \
#         --model_path Qwen/Qwen3.5-9B \
#         --batch_size 1 \
#         --input_len "${input_len}" \
#         --output_len 128 \
#         --num_warmups 1 \
#         --num_runs 2
# done

# for input_len in 10000 20000 30000 40000 50000 60000; do
#     echo "Running benchmark with input_len=${input_len}"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python run_efficiency.py \
#         --model_path Qwen/Qwen3.5-9B \
#         --batch_size 1 \
#         --input_len "${input_len}" \
#         --output_len 128 \
#         --compression \
#         --compression_mode "gatekv" \
#         --compression_budget 1024 \
#         --num_warmups 1 \
#         --num_runs 2
# done

# for input_len in 10000 20000 30000 40000 50000 60000; do
#     echo "Running benchmark with input_len=${input_len}"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python run_efficiency.py \
#         --model_path Qwen/Qwen3.5-9B \
#         --batch_size 1 \
#         --input_len "${input_len}" \
#         --output_len 128 \
#         --compression \
#         --compression_mode "snapkv" \
#         --compression_budget 1024 \
#         --num_warmups 1 \
#         --num_runs 2
# done

##### compression benchmark for qwen3.5-9b

# Code Repository Understanding, Long In-context Learning, Long Structured Data Understanding, Long-dialogue History Understanding, Multi-Document QA, Single-Document QA

for domain in \
    "Code Repository Understanding" \
    "Long In-context Learning" \
    "Long Structured Data Understanding" \
    "Long-dialogue History Understanding" \
    "Multi-Document QA" \
    "Single-Document QA"; do
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python run_longbench.py \
    #     --model Llama-3.1-8B-Instruct \
    #     --domain "$domain" \
    #     --model_maxlen 50000

    # CUDA_VISIBLE_DEVICES=0,1,2,3 python run_longbench.py \
    #     --model Llama-3.1-8B-Instruct \
    #     --domain "$domain" \
    #     --compression \
    #     --compression_mode "snapkv" \
    #     --compression_budget 1024 \
    #     --model_maxlen 50000 

    # CUDA_VISIBLE_DEVICES=0,1,2,3 python run_longbench.py \
    #     --model Llama-3.1-8B-Instruct \
    #     --domain "$domain" \
    #     --compression \
    #     --compression_mode "defensivekv" \
    #     --compression_budget 1024 \
    #     --model_maxlen 50000 

    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_longbench.py \
        --model Llama-3.1-8B-Instruct \
        --domain "$domain" \
        --compression \
        --compression_mode "snapkv_hidden_mix_layer" \
        --compression_budget 1024 \
        --model_maxlen 50000 \
        --hidden_mix_profile_path "/home/yangx/new_compression/hidden_mix_profile/hidden_mix_profile_Llama-3.1-8B-Instruct.json"

done
