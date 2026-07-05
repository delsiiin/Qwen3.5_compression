import torch
import torch.nn.functional as F

from .snapkv_ada_online_head_cluster import OnlineAttentionHeadCluster
from .snapkv_hidden_mix_no_cos import SnapKVHiddenMix as SnapKVHiddenMixNoCos


class SnapKVHiddenMix(SnapKVHiddenMixNoCos):

    def __init__(
        self,
        *args,
        attn_head_cluster_path=None,
        hidden_mix_profile_path=None,
        group_threshold_ema_decay=0.1,
        max_group_size=10,
        mix_temperature=0.1,
        mix_min_weight=0.0,
        **kwargs,
    ):
        if not 0.0 <= float(group_threshold_ema_decay) <= 1.0:
            raise ValueError("group_threshold_ema_decay must be in [0, 1].")
        if int(max_group_size) < 1:
            raise ValueError("max_group_size must be at least 1.")
        if float(mix_temperature) <= 0.0:
            raise ValueError("mix_temperature must be greater than 0.")
        if float(mix_min_weight) < 0.0:
            raise ValueError("mix_min_weight must be non-negative.")

        # Kept for direct-call compatibility. This online variant no longer
        # reads hidden-mix profiles, so avoid passing the path to the parent.
        super().__init__(*args, hidden_mix_profile_path=None, **kwargs)
        self.attn_head_cluster_path = attn_head_cluster_path
        self.hidden_mix_profile_path = hidden_mix_profile_path
        self.hidden_mix_profile = None
        self.group_threshold_ema_decay = float(group_threshold_ema_decay)
        self.max_group_size = int(max_group_size)
        self.mix_temperature = float(mix_temperature)
        self.mix_min_weight = float(mix_min_weight)
        self._online_head_clusterer = OnlineAttentionHeadCluster(self.window_size)
        self._online_window_attention_vector = None

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
            attention_vector=self._online_window_attention_vector,
        )
        return key_states, value_states

    def finalize_after_attention(self, attention, hidden_states, attn_output, layer_cache):
        state = self._state()
        online = state["online"]
        group_key = online["layer_to_group"].get(int(self.layer_idx))
        if group_key is None:
            return
        group = state["groups"].get(group_key)
        if group is None:
            return
        entry = group.get(int(self.layer_idx))
        if entry is None:
            return

        entry["finalized"] = True
        if int(self.layer_idx) == 0:
            if self._close_group(group_key):
                online["last_attention_vector"] = None
                online["last_attn_cache_shape"] = None
            return

        num_layers = getattr(self.model_config, "num_hidden_layers", None)
        if num_layers is not None and int(self.layer_idx) == int(num_layers) - 1:
            self._close_group(group_key)

    def _compute_attn_cache(self, key_states, query_states, valid_mask=None):
        """Build per-KV-head scores with spatio-temporal 2D pooling.

        The 2D grid spans flattened KV/GQA heads and the query window, matching
        ``snapkv_spatio_temporal``.  The resulting scores still feed the
        hidden-mix-specific online clustering and shared-budget selection below.
        """
        raw_head_attention = self._online_head_clusterer._build_raw_head_attention(
            key_states,
            query_states,
            valid_mask,
        )
        (
            num_key_value_heads,
            num_key_value_groups,
            query_window,
            hist_len,
        ) = raw_head_attention.shape
        self._online_window_attention_vector = (
            raw_head_attention.to(dtype=torch.float32)
            .mean(dim=(0, 1))
            .reshape(-1)
            .detach()
        )

        spatiotemporal_scores = raw_head_attention.permute(3, 0, 1, 2).reshape(
            hist_len,
            1,
            num_key_value_heads * num_key_value_groups,
            query_window,
        )
        spatiotemporal_scores = F.max_pool2d(
            spatiotemporal_scores,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        attn_weights_sum = spatiotemporal_scores.reshape(
            hist_len,
            num_key_value_heads,
            num_key_value_groups,
            query_window,
        ).permute(1, 0, 2, 3).mean(dim=(-1, -2)).unsqueeze(0)

        return self._online_head_clusterer._pool_attn_cache(
            attn_weights_sum,
            key_states,
            self.kernel_size,
            valid_mask,
        )

    def _state(self):
        state = super()._state()
        state.setdefault("groups", {})
        state.setdefault(
            "online",
            {
                "active_group_key": None,
                "active_layers": [],
                "next_group_id": 0,
                "last_layer_idx": None,
                "last_attention_vector": None,
                "attention_vector_device": None,
                "last_attn_cache_shape": None,
                "ema": None,
                "ema_initial": None,
                "layer_to_group": {},
            },
        )
        state["online"].setdefault("active_group_key", None)
        state["online"].setdefault("active_layers", [])
        state["online"].setdefault("next_group_id", 0)
        state["online"].setdefault("last_layer_idx", None)
        state["online"].setdefault("last_attention_vector", None)
        state["online"].setdefault("attention_vector_device", None)
        state["online"].setdefault("last_attn_cache_shape", None)
        state["online"].setdefault("ema", None)
        state["online"].setdefault("ema_initial", None)
        state["online"].setdefault("layer_to_group", {})
        return state

    def _reset_online_state(self):
        state = self._state()
        state["groups"] = {}
        state["online"] = {
            "active_group_key": None,
            "active_layers": [],
            "next_group_id": 0,
            "last_layer_idx": None,
            "last_attention_vector": None,
            "attention_vector_device": None,
            "last_attn_cache_shape": None,
            "ema": None,
            "ema_initial": None,
            "layer_to_group": {},
        }
        return state

    def _store_group_entry(
        self,
        key_states,
        value_states,
        attn_cache,
        layer_cache,
        valid_mask=None,
        attention_vector=None,
    ):
        if valid_mask is None:
            valid_mask = torch.ones(key_states.shape[:3], dtype=torch.bool, device=key_states.device)
        raw_attention_vector = attention_vector
        state = self._state()
        online = state["online"]
        layer_idx = int(self.layer_idx)
        if layer_idx == 0 or (
            online["last_layer_idx"] is not None and layer_idx <= int(online["last_layer_idx"])
        ):
            state = self._reset_online_state()
            online = state["online"]

        should_split = False
        if online["active_group_key"] is not None:
            last_shape = online["last_attn_cache_shape"]
            if last_shape is not None and tuple(attn_cache.shape) != tuple(last_shape):
                should_split = True
            if len(online["active_layers"]) >= self.max_group_size:
                should_split = True

            last_vector = online["last_attention_vector"]
            compare_vector = self._prepare_attention_vector(
                raw_attention_vector,
                device=getattr(last_vector, "device", online["attention_vector_device"]),
            )
            if compare_vector is not None and last_vector is not None:
                if tuple(compare_vector.shape) != tuple(last_vector.shape):
                    should_split = True
                else:
                    similarity = self._cosine_similarity(compare_vector, last_vector)
                    if online["ema"] is None:
                        online["ema"] = similarity
                        online["ema_initial"] = similarity
                    else:
                        if similarity < float(online["ema"]):
                            should_split = True
                        online["ema"] = (
                            self.group_threshold_ema_decay * float(online["ema"])
                            + (1.0 - self.group_threshold_ema_decay) * similarity
                        )

        if should_split:
            self._close_group(online["active_group_key"])
            online = state["online"]

        if online["active_group_key"] is None:
            online["active_group_key"] = int(online["next_group_id"])
            online["next_group_id"] = int(online["next_group_id"]) + 1
            online["active_layers"] = []
            online["attention_vector_device"] = self._attention_vector_device(raw_attention_vector)
            state["groups"][online["active_group_key"]] = {}

        group_key = online["active_group_key"]
        group = state["groups"].setdefault(group_key, {})
        attention_vector = self._prepare_attention_vector(
            raw_attention_vector,
            device=online["attention_vector_device"],
        )
        group[layer_idx] = {
            "layer_idx": layer_idx,
            "key_states": key_states,
            "value_states": value_states,
            "attn_cache": attn_cache,
            "attention_vector": attention_vector,
            "layer_cache": layer_cache,
            "valid_mask": valid_mask,
            "finalized": False,
        }
        online["active_layers"].append(layer_idx)
        online["layer_to_group"][layer_idx] = group_key
        online["last_layer_idx"] = layer_idx
        online["last_attention_vector"] = attention_vector
        online["last_attn_cache_shape"] = tuple(attn_cache.shape)

    def _compress_group(self, entries, group_layers):
        entry_by_layer = {int(entry["layer_idx"]): entry for entry in entries}
        online_mix = self._build_online_mix(entries, group_layers)
        for entry in entries:
            target_layer = int(entry["layer_idx"])
            mix = online_mix.get(target_layer, {"sources": (target_layer,), "weights": (1.0,)})
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

    def _close_group(self, group_key):
        if group_key is None:
            return False
        state = self._state()
        group = state["groups"].get(group_key)
        if not group:
            return False
        group_layers = tuple(state["online"]["active_layers"])
        if not group_layers:
            group_layers = tuple(sorted(int(layer_idx) for layer_idx in group))
        if not all(layer_idx in group and group[layer_idx].get("finalized") for layer_idx in group_layers):
            return False

        entries = [group[layer_idx] for layer_idx in group_layers]
        try:
            self._compress_group(entries, group_layers)
        finally:
            for entry in entries:
                state["online"]["layer_to_group"].pop(int(entry["layer_idx"]), None)
                entry.clear()
            state["groups"].pop(group_key, None)
            if state["online"]["active_group_key"] == group_key:
                state["online"]["active_group_key"] = None
                state["online"]["active_layers"] = []
                state["online"]["attention_vector_device"] = None
        return True

    def _build_online_mix(self, entries, group_layers):
        vectors = []
        for entry in entries:
            target_device = vectors[0].device if vectors else None
            vector = self._prepare_attention_vector(entry.get("attention_vector"), device=target_device)
            if vector is None or not torch.is_tensor(vector) or vector.ndim != 1:
                return {
                    int(layer_idx): {"sources": (int(layer_idx),), "weights": (1.0,)}
                    for layer_idx in group_layers
                }
            if vectors and vector.shape != vectors[0].shape:
                return {
                    int(layer_idx): {"sources": (int(layer_idx),), "weights": (1.0,)}
                    for layer_idx in group_layers
                }
            vectors.append(vector)

        if not vectors:
            return {}

        matrix = torch.stack(vectors, dim=0)
        normalized = F.normalize(matrix, p=2, dim=-1, eps=1e-12)
        similarity = (normalized @ normalized.transpose(0, 1)).clamp(min=-1.0, max=1.0)
        group_layers = tuple(int(layer_idx) for layer_idx in group_layers)
        mix_by_layer = {}
        for target_pos, target_layer in enumerate(group_layers):
            logits = similarity[target_pos] / self.mix_temperature
            weights = torch.softmax(logits, dim=0)
            keep = weights >= self.mix_min_weight
            if not bool(keep.any().item()):
                mix_by_layer[target_layer] = {"sources": (target_layer,), "weights": (1.0,)}
                continue
            kept_weights = weights[keep]
            weight_sum = kept_weights.sum()
            if not torch.isfinite(weight_sum) or float(weight_sum.item()) <= 0.0:
                mix_by_layer[target_layer] = {"sources": (target_layer,), "weights": (1.0,)}
                continue
            kept_sources = [
                layer
                for layer, should_keep in zip(group_layers, keep.tolist())
                if bool(should_keep)
            ]
            normalized_weights = (kept_weights / weight_sum).tolist()
            mix_by_layer[target_layer] = {
                "sources": tuple(int(source) for source in kept_sources),
                "weights": tuple(float(weight) for weight in normalized_weights),
            }
        return mix_by_layer

    def _cosine_similarity(self, left, right):
        left = self._prepare_attention_vector(left)
        right = self._prepare_attention_vector(right, device=getattr(left, "device", None))
        if left is None or right is None:
            raise ValueError("snapkv_hidden_mix_no_cos_head_cluster cosine similarity requires tensor vectors.")
        return float(F.cosine_similarity(left, right, dim=0, eps=1e-12).clamp(min=-1.0, max=1.0).item())

    def _attention_vector_device(self, attention_vector):
        if torch.is_tensor(attention_vector):
            return attention_vector.device
        return None

    def _prepare_attention_vector(self, attention_vector, device=None):
        if attention_vector is None:
            return None
        if not torch.is_tensor(attention_vector):
            return attention_vector
        attention_vector = attention_vector.detach()
        if device is None:
            return attention_vector.to(dtype=torch.float32)
        return attention_vector.to(device=device, dtype=torch.float32)

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
