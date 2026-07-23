from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..utils import compute_attention_scores
from .snapkv_ada import SnapKV as SnapKVAda


@dataclass(frozen=True)
class TridentKVHeadClusterResult:
    """Per-compression head-clustering data derived from TridentKV attention."""

    clusters: tuple
    group_similarity: torch.Tensor
    raw_head_weights: torch.Tensor
    raw_head_scores: torch.Tensor
    hist_len: int


def _complete_link_clusters(layer_distance, distance_threshold):
    """Cluster heads with the same deterministic complete-link rule as profiling."""
    head_count = int(layer_distance.shape[0])
    clusters = [[head_idx] for head_idx in range(head_count)]

    while True:
        best_pair = None
        best_distance = None
        for left_idx in range(len(clusters)):
            for right_idx in range(left_idx + 1, len(clusters)):
                complete_distance = float(
                    layer_distance[clusters[left_idx]][:, clusters[right_idx]].max().item()
                )
                if complete_distance > distance_threshold:
                    continue
                if best_distance is None or complete_distance < best_distance:
                    best_distance = complete_distance
                    best_pair = (left_idx, right_idx)

        if best_pair is None:
            break

        left_idx, right_idx = best_pair
        clusters[left_idx] = sorted(clusters[left_idx] + clusters[right_idx])
        del clusters[right_idx]

    return tuple(
        {"cluster_id": cluster_id, "heads": tuple(heads)}
        for cluster_id, heads in enumerate(sorted(clusters, key=lambda cluster: cluster[0]))
    )


class TridentKVHeadClusterer:
    """Build TridentKV GQA clusters from the current query-window attention."""

    def __init__(self, window_size, eps=1e-12):
        self.window_size = int(window_size)
        self.eps = float(eps)

    def _build_raw_head_attention(self, key_states, query_states, valid_mask=None):
        if key_states.ndim != 4 or query_states.ndim != 4:
            raise ValueError("Online attention head clustering requires rank-4 key and query states.")
        if key_states.shape[0] != 1 or query_states.shape[0] != 1:
            raise ValueError("Online attention head clustering only supports batch size 1.")

        _, num_key_value_heads, kv_cache_len, _ = key_states.shape
        hist_len = kv_cache_len - self.window_size
        if hist_len < 1:
            raise ValueError("Online attention head clustering requires at least one historical KV token.")
        if query_states.shape[1] % num_key_value_heads != 0:
            raise ValueError(
                "Online attention head clustering requires query heads to be divisible by key/value heads."
            )

        gqa_group_size = query_states.shape[1] // num_key_value_heads
        query_window = min(self.window_size, query_states.shape[-2])
        query_states = query_states[:, :, -query_window:, :]

        attn_weights = compute_attention_scores(query_states, key_states)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=kv_cache_len - query_window + 1)
        attn_weights = attn_weights + attention_mask
        if valid_mask is not None:
            if valid_mask.shape != (1, num_key_value_heads, kv_cache_len):
                raise ValueError(
                    "Online attention head clustering valid_mask must have shape "
                    "[1, key_value_heads, kv_cache_len]."
                )
            full_valid = valid_mask[:, :, None, :].expand(
                1,
                num_key_value_heads,
                gqa_group_size,
                kv_cache_len,
            ).reshape(1, query_states.shape[1], kv_cache_len)
            attn_weights = attn_weights.masked_fill(
                ~full_valid[:, :, None, :],
                torch.finfo(attn_weights.dtype).min,
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        hist_attention = attn_weights[..., :hist_len]
        raw_head_attention = hist_attention.view(
            1,
            num_key_value_heads,
            gqa_group_size,
            query_window,
            hist_len,
        )[0]
        return raw_head_attention

    def build(self, key_states, query_states, valid_mask=None):
        raw_head_attention = self._build_raw_head_attention(key_states, query_states, valid_mask)
        return self.build_from_raw_head_attention(raw_head_attention)

    def build_from_raw_head_attention(self, raw_head_attention):
        """Build online clusters from precomputed raw GQA attention."""
        if raw_head_attention.ndim != 4:
            raise ValueError(
                "Online attention head clustering raw attention must be rank 4 "
                "[key_value_heads, groups, query_window, history]."
            )
        num_key_value_heads, gqa_group_size, _, hist_len = raw_head_attention.shape
        if hist_len < 1:
            raise ValueError("Online attention head clustering requires at least one historical KV token.")
        raw_head_scores = raw_head_attention.mean(dim=-2).unsqueeze(0)

        group_attention = raw_head_attention.mean(dim=1)
        group_vectors = group_attention.reshape(num_key_value_heads, -1)
        normalized_groups = F.normalize(group_vectors, p=2, dim=-1, eps=self.eps)
        group_similarity = (normalized_groups @ normalized_groups.transpose(0, 1)).clamp(min=-1.0, max=1.0)

        if num_key_value_heads <= 1:
            distance_threshold = 0.0
        else:
            upper = torch.triu_indices(
                num_key_value_heads,
                num_key_value_heads,
                offset=1,
                device=group_similarity.device,
            )
            distance_threshold = float((1.0 - group_similarity[upper[0], upper[1]]).mean().item())
        clusters = _complete_link_clusters(1.0 - group_similarity, distance_threshold)

        raw_vectors = raw_head_attention.reshape(num_key_value_heads, gqa_group_size, -1)
        normalized_raw_heads = F.normalize(raw_vectors, p=2, dim=-1, eps=self.eps)
        normalized_group_means = F.normalize(
            raw_vectors.mean(dim=1, keepdim=True),
            p=2,
            dim=-1,
            eps=self.eps,
        )
        raw_head_similarity = (normalized_raw_heads * normalized_group_means).sum(dim=-1).clamp_min(0.0)
        weight_sums = raw_head_similarity.sum(dim=-1, keepdim=True)
        raw_head_weights = torch.where(
            weight_sums > self.eps,
            raw_head_similarity / weight_sums.clamp_min(self.eps),
            torch.full_like(raw_head_similarity, 1.0 / gqa_group_size),
        )
        return TridentKVHeadClusterResult(
            clusters=clusters,
            group_similarity=group_similarity,
            raw_head_weights=raw_head_weights,
            raw_head_scores=raw_head_scores,
            hist_len=hist_len,
        )

    def build_clusters_from_scores(self, scores, valid_mask=None):
        """Cluster KV heads from final per-head historical token scores."""
        if scores.ndim != 3:
            raise ValueError("Online attention head clustering scores must be rank 3 [batch, heads, history].")
        if scores.shape[0] != 1:
            raise ValueError("Online attention head clustering only supports batch size 1.")
        if scores.shape[-1] < 1:
            raise ValueError("Online attention head clustering requires at least one historical KV token.")

        if valid_mask is not None:
            if valid_mask.shape != scores.shape:
                raise ValueError(
                    "Online attention head clustering score valid_mask must match [1, key_value_heads, history]."
                )
            scores = scores.masked_fill(~valid_mask.to(device=scores.device, dtype=torch.bool), 0)

        vectors = scores[0].to(dtype=torch.float32)
        normalized_vectors = F.normalize(vectors, p=2, dim=-1, eps=self.eps)
        group_similarity = (normalized_vectors @ normalized_vectors.transpose(0, 1)).clamp(
            min=-1.0,
            max=1.0,
        )
        num_key_value_heads = scores.shape[1]
        if num_key_value_heads <= 1:
            distance_threshold = 0.0
        else:
            upper = torch.triu_indices(
                num_key_value_heads,
                num_key_value_heads,
                offset=1,
                device=group_similarity.device,
            )
            distance_threshold = float((1.0 - group_similarity[upper[0], upper[1]]).mean().item())
        return _complete_link_clusters(1.0 - group_similarity, distance_threshold)

    def build_head_cluster_attn_cache(self, key_states, query_states, kernel_size, valid_mask=None):
        result = self.build(key_states, query_states, valid_mask)
        attn_weights_sum = (result.raw_head_scores * result.raw_head_weights[None, :, :, None]).sum(dim=2)
        return result, self._pool_attn_cache(attn_weights_sum, key_states, kernel_size, valid_mask)

    def build_mean_head_cluster_attn_cache(self, key_states, query_states, kernel_size, valid_mask=None):
        """Build a cluster result while mean-pooling raw GQA heads without similarity weights."""
        result = self.build(key_states, query_states, valid_mask)
        attn_weights_sum = result.raw_head_scores.mean(dim=2)
        return result, self._pool_attn_cache(attn_weights_sum, key_states, kernel_size, valid_mask)

    def build_mean_attn_cache_without_head_clustering(
        self,
        key_states,
        query_states,
        kernel_size,
        valid_mask=None,
    ):
        """Mean-pool raw GQA attention without constructing an online head cluster."""
        raw_head_attention = self._build_raw_head_attention(key_states, query_states, valid_mask)
        attn_weights_sum = raw_head_attention.mean(dim=-2).unsqueeze(0).mean(dim=2)
        return self._pool_attn_cache(attn_weights_sum, key_states, kernel_size, valid_mask)

    def _pool_attn_cache(self, attn_weights_sum, key_states, kernel_size, valid_mask):
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            stride=1,
        )
        if valid_mask is not None:
            hist_valid = valid_mask[:, :, : attn_cache.shape[-1]].to(device=attn_cache.device, dtype=torch.bool)
            attn_cache = torch.where(hist_valid, attn_cache, torch.zeros_like(attn_cache))
        return attn_cache.to(dtype=key_states.dtype)


class TridentKVHeadCluster(SnapKVAda):
    manages_kv_cache = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tridentkv_head_clusterer = TridentKVHeadClusterer(self.window_size)
        self._tridentkv_head_cluster_result = None

    def _compute_attn_cache(self, key_states, query_states, valid_mask=None):
        result, attn_cache = self._tridentkv_head_clusterer.build_head_cluster_attn_cache(
            key_states,
            query_states,
            self.kernel_size,
            valid_mask,
        )
        self._tridentkv_head_cluster_result = result
        return attn_cache

    def _select_layer_head_topk(self, key_states, scores, valid_mask):
        batch_size, num_heads = key_states.shape[:2]
        hist_len = key_states.shape[-2] - self.window_size
        if hist_len < 1:
            return torch.zeros(batch_size, num_heads, 0, dtype=torch.bool, device=key_states.device), 0
        if scores.shape[-1] != hist_len:
            raise ValueError("tridentkv_head_cluster attn_cache length must match historical cache length.")

        result = self._tridentkv_head_cluster_result
        if result is None:
            raise RuntimeError("tridentkv_head_cluster requires head clusters from _compute_attn_cache.")
        if result.hist_len != hist_len:
            raise ValueError("tridentkv_head_cluster head-cluster history length does not match current cache.")
        if result.group_similarity.shape[0] != num_heads:
            raise ValueError(
                "tridentkv_head_cluster head count does not match runtime key/value heads."
            )
        clusters = result.clusters
        hist_valid = valid_mask[:, :, :hist_len].to(device=scores.device, dtype=torch.bool)
        selected = torch.zeros(batch_size, num_heads, hist_len, dtype=torch.bool, device=scores.device)

        hist_budget_per_head = self.budget - self.window_size
        for cluster in clusters:
            heads = cluster["heads"]
            head_index = torch.tensor(heads, dtype=torch.long, device=scores.device)
            cluster_scores = scores.index_select(dim=1, index=head_index)
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
