import torch

from .snapkv_ada import SnapKV as SnapKVAda
from .snapkv_head_cluster import AttentionHeadClusterMixin


class SnapKV(AttentionHeadClusterMixin, SnapKVAda):
    manages_kv_cache = True

    def __init__(self, *args, attn_head_cluster_path="/home/yangx/new_compression/attn_head_clusters_llama_8b.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_head_cluster_path = self._resolve_attn_head_cluster_path(attn_head_cluster_path)
        self.attn_head_cluster_profile = self._load_attn_head_cluster_profile(self.attn_head_cluster_path)

    def _select_layer_head_topk(self, key_states, scores, valid_mask):
        batch_size, num_heads = key_states.shape[:2]
        hist_len = key_states.shape[-2] - self.window_size
        if hist_len < 1:
            return torch.zeros(batch_size, num_heads, 0, dtype=torch.bool, device=key_states.device), 0
        if scores.shape[-1] != hist_len:
            raise ValueError("snapkv_ada_head_cluster attn_cache length must match historical cache length.")

        clusters = self._layer_clusters(num_heads)
        mixed_scores = self._mix_attn_cache_by_cluster(scores, clusters)
        hist_valid = valid_mask[:, :, :hist_len].to(device=scores.device, dtype=torch.bool)
        selected = torch.zeros(batch_size, num_heads, hist_len, dtype=torch.bool, device=scores.device)

        hist_budget_per_head = self.budget - self.window_size
        for cluster in clusters:
            heads = cluster["heads"]
            head_index = torch.tensor(heads, dtype=torch.long, device=scores.device)
            cluster_scores = mixed_scores.index_select(dim=1, index=head_index)
            cluster_valid = hist_valid.index_select(dim=1, index=head_index)
            flat_valid = cluster_valid.reshape(batch_size, -1)
            valid_count = flat_valid.sum(dim=-1)
            total_budget = len(heads) * hist_budget_per_head
            topk = min(int(total_budget), int(valid_count.min().item()))
            if topk <= 0:
                continue

            flat_scores = cluster_scores.reshape(batch_size, -1).masked_fill(
                ~flat_valid,
                torch.finfo(cluster_scores.dtype).min,
            )
            topk_indices = flat_scores.topk(topk, dim=-1).indices
            selected_cluster_flat = torch.zeros_like(flat_valid, dtype=torch.bool)
            selected_cluster_flat.scatter_(dim=-1, index=topk_indices, value=True)
            selected_cluster = selected_cluster_flat.view(batch_size, len(heads), hist_len)
            for offset, head_idx in enumerate(heads):
                selected[:, head_idx, :] = selected_cluster[:, offset, :]

        return selected.to(device=key_states.device), hist_len
