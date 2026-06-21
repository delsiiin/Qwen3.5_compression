import torch

from .snapkv_ada_online_head_cluster import OnlineAttentionHeadCluster
from .snapkv_hidden_mix_no_cos import SnapKVHiddenMix as SnapKVHiddenMixNoCos


class SnapKVHiddenMix(SnapKVHiddenMixNoCos):

    def __init__(self, *args, attn_head_cluster_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Kept for direct-call compatibility. Online clustering does not read profiles.
        self.attn_head_cluster_path = attn_head_cluster_path
        self._online_head_clusterer = OnlineAttentionHeadCluster(self.window_size)

    def update_kv_cache(
        self,
        attention,
        hidden_states,
        position_embeddings,
        key_states,
        query_states,
        value_states,
        past_key_values,
        layer_cache,
    ):
        if self._should_append_to_cache(query_states, layer_cache):
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            self._append_valid_mask(layer_cache, key_states)
        else:
            self._clear_padding_metadata(layer_cache)
            if self.model_config.update_kv is not True or key_states.shape[-2] < self.budget:
                self._set_layer_cache(layer_cache, key_states, value_states)
            if self.model_config.update_kv is not True:
                return key_states, value_states

        if self.model_config.update_kv is not True:
            return key_states, value_states

        self._set_attention_mask_for_current_step(layer_cache, key_states, query_states)
        should_compress = self.model_config.compression is None or query_states.shape[-2] > 1
        if not should_compress or self._valid_token_count(layer_cache, key_states) < self.budget:
            return key_states, value_states

        query_cache = getattr(layer_cache, "query_cache", None)
        if query_cache is None or query_cache.shape[-2] == 0:
            query_cache = query_states[:, :, -self.window_size :, :]
        valid_mask = self._current_valid_mask(layer_cache, key_states)
        attn_cache = self._compute_attn_cache(key_states, query_cache, valid_mask)
        self._store_group_entry(
            key_states=key_states,
            value_states=value_states,
            attn_cache=attn_cache,
            layer_cache=layer_cache,
            valid_mask=valid_mask,
        )
        return key_states, value_states

    def _compute_attn_cache(self, key_states, query_states, valid_mask=None):
        return self._online_head_clusterer.build_mean_attn_cache_without_clustering(
            key_states,
            query_states,
            self.kernel_size,
            valid_mask,
        )

    def _store_group_entry(self, key_states, value_states, attn_cache, layer_cache, valid_mask=None):
        super()._store_group_entry(key_states, value_states, attn_cache, layer_cache)
        if valid_mask is None:
            valid_mask = torch.ones(key_states.shape[:3], dtype=torch.bool, device=key_states.device)
        group_layers = self._group_layers(self.layer_idx)
        entry = self._state()["groups"][tuple(group_layers)][self.layer_idx]
        entry["valid_mask"] = valid_mask

    def _compress_group(self, entries, group_layers):
        entry_by_layer = {int(entry["layer_idx"]): entry for entry in entries}
        for entry in entries:
            target_layer = int(entry["layer_idx"])
            mix = self._mix_for_layer(group_layers, target_layer)
            mixed_cache = None
            for source_layer, weight in zip(mix["sources"], mix["weights"]):
                source_cache = entry_by_layer[int(source_layer)]["attn_cache"]
                source_cache = source_cache.to(device=entry["attn_cache"].device, dtype=entry["attn_cache"].dtype)
                if source_cache.shape != entry["attn_cache"].shape:
                    raise ValueError("snapkv_hidden_mix requires matching attn_cache shapes within a layer group.")
                normalized_cache = self._normalize_attn_cache(source_cache)
                weighted = normalized_cache * float(weight)
                mixed_cache = weighted if mixed_cache is None else mixed_cache + weighted

            valid_mask = entry["valid_mask"].to(device=mixed_cache.device, dtype=torch.bool)
            hist_valid = valid_mask[:, :, : mixed_cache.shape[-1]]
            entry["clusters"] = self._online_head_clusterer.build_clusters_from_scores(
                mixed_cache,
                hist_valid,
            )
            self._pack_layer(entry, mixed_cache)

    def _pack_layer(self, entry, attn_cache):
        key_states = entry["key_states"]
        value_states = entry["value_states"]
        batch_size, num_heads, _, head_dim = key_states.shape
        hist_len = attn_cache.shape[-1]
        valid_mask = entry["valid_mask"].to(device=key_states.device, dtype=torch.bool)
        if valid_mask.shape != key_states.shape[:3]:
            raise ValueError("snapkv_hidden_mix_no_cos_head_cluster valid mask does not match KV cache.")

        selected_hist_mask = torch.zeros(
            batch_size,
            num_heads,
            hist_len,
            dtype=torch.bool,
            device=key_states.device,
        )
        hist_valid = valid_mask[:, :, :hist_len]
        hist_budget_per_head = self.budget - self.window_size
        seen_heads = []
        for cluster in entry["clusters"]:
            heads = cluster["heads"]
            seen_heads.extend(heads)
            head_index = torch.tensor(heads, dtype=torch.long, device=key_states.device)
            cluster_scores = attn_cache.index_select(dim=1, index=head_index)
            cluster_valid = hist_valid.index_select(dim=1, index=head_index)
            flat_valid = cluster_valid.reshape(batch_size, -1)
            topk = min(len(heads) * hist_budget_per_head, int(flat_valid.sum(dim=-1).min().item()))
            if topk <= 0:
                continue
            flat_scores = cluster_scores.reshape(batch_size, -1).masked_fill(
                ~flat_valid,
                torch.finfo(cluster_scores.dtype).min,
            )
            indices = flat_scores.topk(topk, dim=-1).indices
            selected_flat = torch.zeros_like(flat_valid, dtype=torch.bool)
            selected_flat.scatter_(dim=-1, index=indices, value=True)
            selected_hist_mask.index_copy_(
                1,
                head_index,
                selected_flat.view(batch_size, len(heads), hist_len),
            )

        if sorted(seen_heads) != list(range(num_heads)):
            raise ValueError("snapkv_hidden_mix_no_cos_head_cluster clusters must cover every KV head exactly once.")

        keep_indices = []
        lengths = torch.zeros(batch_size, num_heads, dtype=torch.long, device=key_states.device)
        for batch_idx in range(batch_size):
            per_batch = []
            for head_idx in range(num_heads):
                hist_indices = torch.where(selected_hist_mask[batch_idx, head_idx])[0].sort().values
                recent_indices = torch.where(valid_mask[batch_idx, head_idx, hist_len:])[0] + hist_len
                indices = torch.cat([hist_indices, recent_indices], dim=0)
                per_batch.append(indices)
                lengths[batch_idx, head_idx] = indices.numel()
            keep_indices.append(per_batch)

        max_len = max(int(lengths.max().item()), 1)
        packed_keys = key_states.new_zeros(batch_size, num_heads, max_len, head_dim)
        packed_values = value_states.new_zeros(batch_size, num_heads, max_len, head_dim)
        packed_mask = torch.zeros(batch_size, num_heads, max_len, dtype=torch.bool, device=key_states.device)
        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                indices = keep_indices[batch_idx][head_idx]
                if indices.numel() == 0:
                    continue
                packed_keys[batch_idx, head_idx, : indices.numel()] = key_states[batch_idx, head_idx].index_select(0, indices)
                packed_values[batch_idx, head_idx, : indices.numel()] = value_states[batch_idx, head_idx].index_select(0, indices)
                packed_mask[batch_idx, head_idx, : indices.numel()] = True

        self._set_layer_cache(entry["layer_cache"], packed_keys, packed_values)
        entry["layer_cache"].kv_valid_mask = packed_mask
        entry["layer_cache"].kv_lengths = lengths

    def _current_valid_mask(self, layer_cache, key_states):
        valid_mask = getattr(layer_cache, "kv_valid_mask", None)
        if valid_mask is None or valid_mask.shape[-1] != key_states.shape[-2]:
            return torch.ones(key_states.shape[:3], dtype=torch.bool, device=key_states.device)
        return valid_mask.to(device=key_states.device, dtype=torch.bool)

    def _append_valid_mask(self, layer_cache, key_states):
        valid_mask = getattr(layer_cache, "kv_valid_mask", None)
        if valid_mask is None:
            return
        old_len = valid_mask.shape[-1]
        new_len = key_states.shape[-2]
        if new_len <= old_len:
            layer_cache.kv_valid_mask = valid_mask[:, :, :new_len]
        else:
            appended = torch.ones(
                *valid_mask.shape[:2],
                new_len - old_len,
                dtype=torch.bool,
                device=key_states.device,
            )
            layer_cache.kv_valid_mask = torch.cat([valid_mask.to(key_states.device), appended], dim=-1)
        layer_cache.kv_lengths = layer_cache.kv_valid_mask.sum(dim=-1)

    def _valid_token_count(self, layer_cache, key_states):
        return int(self._current_valid_mask(layer_cache, key_states).sum(dim=-1).min().item())

    def _set_attention_mask_for_current_step(self, layer_cache, key_states, query_states):
        if query_states.shape[-2] == 1:
            layer_cache.attention_kv_valid_mask = self._current_valid_mask(layer_cache, key_states)
        elif hasattr(layer_cache, "attention_kv_valid_mask"):
            delattr(layer_cache, "attention_kv_valid_mask")

    def _clear_padding_metadata(self, layer_cache):
        for attr in ("kv_valid_mask", "kv_lengths", "attention_kv_valid_mask"):
            if hasattr(layer_cache, attr):
                delattr(layer_cache, attr)
