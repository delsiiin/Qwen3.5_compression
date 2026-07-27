import torch
import torch.nn.functional as F

from . import compute_attention_scores
from .snapkv_ada import SnapKV as SnapKVAda


class TridentKVSpatialTemporal(SnapKVAda):
    """Spatial-temporal scoring with layer-wide adaptive head budgets."""

    def _compute_attn_cache(self, key_states, query_states, valid_mask=None):
        bsz, num_key_value_heads, kv_cache_len, _ = key_states.shape
        hist_len = kv_cache_len - self.window_size
        if hist_len < 1:
            return key_states.new_zeros(bsz, num_key_value_heads, 0)
        if query_states.shape[1] % num_key_value_heads != 0:
            raise ValueError(
                "tridentkv_spatial_temporal requires query heads divisible by "
                "key/value heads."
            )
        if self.kernel_size % 2 == 0:
            raise ValueError("tridentkv_spatial_temporal requires odd kernel_size.")

        num_key_value_groups = query_states.shape[1] // num_key_value_heads
        query_window = min(self.window_size, query_states.shape[-2])
        query_states = query_states[:, :, -query_window:, :]

        attn_weights = compute_attention_scores(query_states, key_states)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(
            attention_mask,
            diagonal=kv_cache_len - query_window + 1,
        )
        attn_weights = attn_weights + attention_mask
        if valid_mask is not None:
            full_valid = valid_mask[:, :, None, :].expand(
                bsz,
                num_key_value_heads,
                num_key_value_groups,
                kv_cache_len,
            )
            full_valid = full_valid.reshape(
                bsz,
                query_states.shape[1],
                kv_cache_len,
            )
            attn_weights = attn_weights.masked_fill(
                ~full_valid[:, :, None, :],
                torch.finfo(attn_weights.dtype).min,
            )
        attn_weights = F.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)
        attn_weights = attn_weights[..., :hist_len]

        scores = attn_weights.view(
            bsz,
            num_key_value_heads,
            num_key_value_groups,
            query_window,
            hist_len,
        )
        spatial_temporal_scores = scores.permute(0, 4, 1, 2, 3).reshape(
            bsz * hist_len,
            1,
            num_key_value_heads * num_key_value_groups,
            query_window,
        )
        spatial_temporal_scores = F.max_pool2d(
            spatial_temporal_scores,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        ).reshape(
            bsz,
            hist_len,
            num_key_value_heads,
            num_key_value_groups,
            query_window,
        )
        attn_weights_sum = spatial_temporal_scores.permute(
            0,
            2,
            1,
            3,
            4,
        ).mean(dim=(-1, -2))
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        if valid_mask is not None:
            hist_valid = valid_mask[:, :, :hist_len].to(
                device=attn_cache.device,
                dtype=torch.bool,
            )
            attn_cache = torch.where(
                hist_valid,
                attn_cache,
                torch.zeros_like(attn_cache),
            )
        return attn_cache
