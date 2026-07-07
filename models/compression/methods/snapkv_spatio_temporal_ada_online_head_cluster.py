import torch
import torch.nn.functional as F

from .snapkv_ada_online_head_cluster import SnapKVAdaOnlineHeadCluster


class SnapKVSpatioTemporalAdaOnlineHeadCluster(SnapKVAdaOnlineHeadCluster):
    """SnapKV spatio-temporal scoring with online head-cluster budget sharing."""

    def _compute_attn_cache(self, key_states, query_states, valid_mask=None):
        raw_head_attention = self._online_head_clusterer._build_raw_head_attention(
            key_states,
            query_states,
            valid_mask,
        )
        result = self._online_head_clusterer.build_from_raw_head_attention(raw_head_attention)
        self._online_head_cluster_result = result

        (
            num_key_value_heads,
            num_key_value_groups,
            query_window,
            hist_len,
        ) = raw_head_attention.shape
        if self.kernel_size % 2 == 0:
            raise ValueError("snapkv_spatio_temporal_ada_online_head_cluster requires odd kernel_size.")

        attn_weights_sum = raw_head_attention.new_empty(1, num_key_value_heads, hist_len)
        for cluster in result.clusters:
            heads = cluster["heads"]
            head_index = torch.tensor(heads, dtype=torch.long, device=raw_head_attention.device)
            cluster_head_count = len(heads)
            cluster_attention = raw_head_attention.index_select(dim=0, index=head_index)
            cluster_scores = cluster_attention.permute(3, 1, 0, 2).reshape(
                hist_len,
                1,
                cluster_head_count * num_key_value_groups,
                query_window,
            )
            if cluster_head_count * num_key_value_groups % 2 ==0:
                head_kernel = cluster_head_count * num_key_value_groups - 1
            else:
                head_kernel = cluster_head_count * num_key_value_groups
            cluster_scores = F.max_pool2d(
                cluster_scores,
                kernel_size=(head_kernel, 4),
                stride=1,
                padding=(head_kernel // 2, 2),
            )
            pooled_head_count = cluster_head_count
            pooled_query_window = cluster_scores.shape[-1]
            cluster_scores = cluster_scores.reshape(
                hist_len,
                num_key_value_groups,
                pooled_head_count,
                pooled_query_window,
            )
            cluster_head_scores = cluster_scores.mean(dim=(1, 3)).transpose(0, 1)
            attn_weights_sum[:, head_index, :] = cluster_head_scores.unsqueeze(0)

        return self._online_head_clusterer._pool_attn_cache(
            attn_weights_sum,
            key_states,
            self.kernel_size,
            valid_mask,
        )

