from .snapkv_ada_online_head_cluster import OnlineAttentionHeadCluster
from .snapkv_hidden_mix_no_cos import SnapKVHiddenMix as SnapKVHiddenMixNoCos


class SnapKVHiddenMix(SnapKVHiddenMixNoCos):

    def __init__(self, *args, attn_head_cluster_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Kept for direct-call compatibility. Online clustering does not read profiles.
        self.attn_head_cluster_path = attn_head_cluster_path
        self._online_head_clusterer = OnlineAttentionHeadCluster(self.window_size)

    def _compute_attn_cache(self, key_states, query_states):
        _, attn_cache = self._online_head_clusterer.build_attn_cache(
            key_states,
            query_states,
            self.kernel_size,
        )
        return attn_cache
