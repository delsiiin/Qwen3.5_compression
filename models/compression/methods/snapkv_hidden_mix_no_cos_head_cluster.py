from .snapkv_head_cluster import AttentionHeadClusterMixin
from .snapkv_hidden_mix_no_cos import SnapKVHiddenMix as SnapKVHiddenMixNoCos


class SnapKVHiddenMix(AttentionHeadClusterMixin, SnapKVHiddenMixNoCos):
    attn_head_cluster_method_name = "snapkv_hidden_mix_no_cos_head_cluster"

    def __init__(self, *args, attn_head_cluster_path=None, **kwargs):
        if attn_head_cluster_path is None:
            raise ValueError("snapkv_hidden_mix_no_cos_head_cluster requires attn_head_cluster_path.")
        super().__init__(*args, **kwargs)
        self.attn_head_cluster_path = self._resolve_attn_head_cluster_path(attn_head_cluster_path)
        self.attn_head_cluster_profile = self._load_attn_head_cluster_profile(self.attn_head_cluster_path)

    def _compute_attn_cache(self, key_states, query_states):
        return self._compute_head_cluster_attn_cache(key_states, query_states)
