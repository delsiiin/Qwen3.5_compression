from .snapkv_head_cluster import AttentionHeadClusterMixin
from .snapkv_hidden_mix_no_cos import SnapKVHiddenMix as SnapKVHiddenMixNoCos


class SnapKVHiddenMix(AttentionHeadClusterMixin, SnapKVHiddenMixNoCos):
    attn_head_cluster_method_name = "snapkv_hidden_mix_no_cos_head_cluster"

    def __init__(self, *args, attn_head_cluster_path="/home/yangx/new_compression/attn_head_clusters_llama_8b.json", **kwargs):
        if attn_head_cluster_path is None:
            raise ValueError("snapkv_hidden_mix_no_cos_head_cluster requires attn_head_cluster_path.")
        super().__init__(*args, **kwargs)
        self.attn_head_cluster_path = self._resolve_attn_head_cluster_path(attn_head_cluster_path)
        self.attn_head_cluster_profile = self._load_attn_head_cluster_profile(self.attn_head_cluster_path)

    def _store_group_entry(self, key_states, value_states, attn_cache, layer_cache):
        attn_cache = self._mix_attn_cache_by_cluster(attn_cache)
        return super()._store_group_entry(
            key_states=key_states,
            value_states=value_states,
            attn_cache=attn_cache,
            layer_cache=layer_cache,
        )
