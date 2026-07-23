import torch
import torch.nn.functional as F

from . import compute_attention_scores


class TridentKVSpatialTemporal:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        record_kept_token_indices=False,
        layer_idx=None,
        model_config=None,
        model_type=None,
        mode=None,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size

        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode

        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []

    def update_kv(self, key_states, query_states, value_states):
        head_dim = query_states.shape[-1]
        bsz, num_key_value_heads, kv_cache_len, _ = key_states.shape
        num_key_value_groups = query_states.shape[1] // num_key_value_heads

        if kv_cache_len < self.budget:
            return key_states, value_states

        attn_weights = compute_attention_scores(query_states, key_states)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(
            attention_mask,
            diagonal=key_states.shape[-2] - self.window_size + 1,
        )
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-self.window_size]

        history_length = kv_cache_len - self.window_size
        scores = attn_weights.view(
            bsz,
            num_key_value_heads,
            num_key_value_groups,
            self.window_size,
            history_length,
        )
        spatial_temporal_scores = scores.permute(0, 4, 1, 2, 3).reshape(
            bsz * history_length,
            1,
            num_key_value_heads * num_key_value_groups,
            self.window_size,
        )
        spatial_temporal_scores = F.max_pool2d(
            spatial_temporal_scores,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        ).reshape(
            bsz,
            history_length,
            num_key_value_heads,
            num_key_value_groups,
            self.window_size,
        )
        attn_weights_sum = spatial_temporal_scores.permute(0, 2, 1, 3, 4).mean(dim=(-1, -2))

        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        indices = attn_cache.topk(self.budget - self.window_size, dim=-1).indices

        if self.record_kept_token_indices:
            indices_cl = indices.clone().squeeze(0).to("cpu")
            attn_weights_sum_analysis = (
                F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                .mean(dim=-2)
                .to(query_states.dtype)
            )
            attn_cache_analysis = F.max_pool1d(
                attn_weights_sum_analysis,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
            recent_window_indices = torch.arange(
                kv_cache_len - self.window_size,
                kv_cache_len,
                device="cpu",
            ).expand(indices_cl.shape[0], -1)
            cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)
            attn_scores = attn_cache_analysis.clone().squeeze(0).to("cpu")
            kept_attn = torch.gather(attn_scores, dim=1, index=cur_indices)

            if self.evicted_token_num > 0:
                prev_indices = self.kept_token_indices[-1]
                mask = cur_indices < self.budget
                for head_idx in range(cur_indices.shape[0]):
                    positions = torch.where(mask[head_idx])[0]
                    for pos in positions:
                        val = cur_indices[head_idx, pos].item()
                        cur_indices[head_idx, pos] = prev_indices[head_idx, val]
                cur_indices[~mask] += self.evicted_token_num

            self.kept_attention_scores.append(kept_attn)
            self.kept_token_indices.append(cur_indices)
            self.evicted_token_num += kv_cache_len - self.budget

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(
            dim=2,
            index=gather_indices,
        )
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(
            dim=2,
            index=gather_indices,
        )
        key_states = torch.cat([k_past_compress, key_states[:, :, -self.window_size :, :]], dim=2)
        value_states = torch.cat([v_past_compress, value_states[:, :, -self.window_size :, :]], dim=2)
        return key_states, value_states
