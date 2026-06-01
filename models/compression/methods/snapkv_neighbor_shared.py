import json
import math
import os

import torch
import torch.nn.functional as F

from . import compute_attention_scores
from .snapkv import SnapKV


class SnapKVNeighborShared:
    requires_layer_coordination = True

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
        hidden_mix_profile_path=None,
        hidden_mix_fallback="self",
        flatten_cache="auto",
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        if hidden_mix_fallback != "self":
            raise ValueError("hidden_mix_fallback currently supports only 'self'.")
        if flatten_cache not in ("auto", True, False):
            raise ValueError("flatten_cache must be one of 'auto', True, or False.")
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode
        self.hidden_mix_profile_path = hidden_mix_profile_path
        self.hidden_mix_fallback = hidden_mix_fallback
        self.flatten_cache = flatten_cache
        self.record_kept_token_indices = record_kept_token_indices
        self.hidden_mix_profile = self._load_hidden_mix_profile(hidden_mix_profile_path)
        self.fallback = SnapKV(
            budget=budget,
            window_size=window_size,
            kernel_size=kernel_size,
            record_kept_token_indices=record_kept_token_indices,
            layer_idx=layer_idx,
            model_config=model_config,
            model_type=model_type,
            mode=mode,
            **kwargs,
        )
        if model_config is not None and not hasattr(model_config, "_snapkv_neighbor_shared_state"):
            model_config._snapkv_neighbor_shared_state = {"groups": {}}

    def _load_hidden_mix_profile(self, profile_path):
        if profile_path is None:
            return None
        if self.model_config is None:
            raise ValueError("hidden_mix_profile_path requires model_config.")

        profile_path = os.path.abspath(os.path.expanduser(str(profile_path)))
        cached = getattr(self.model_config, "_snapkv_neighbor_shared_profile", None)
        if cached is not None and cached.get("path") == profile_path:
            return cached["profile"]

        with open(profile_path, "r", encoding="utf-8") as handle:
            raw_profile = json.load(handle)
        profile = self._parse_hidden_mix_profile(raw_profile, profile_path)
        self.model_config._snapkv_neighbor_shared_profile = {
            "path": profile_path,
            "profile": profile,
        }
        return profile

    def _parse_hidden_mix_profile(self, raw_profile, profile_path):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Hidden mix profile must be a JSON object: {profile_path}")
        groups = raw_profile.get("groups")
        if not isinstance(groups, list):
            raise ValueError("Hidden mix profile requires a list field named 'groups'.")

        layer_to_group = {}
        group_profiles = {}
        for group_idx, group_spec in enumerate(groups):
            if not isinstance(group_spec, dict):
                raise ValueError(f"Profile group {group_idx} must be an object.")
            layers = group_spec.get("layers")
            if not isinstance(layers, list) or not layers:
                raise ValueError(f"Profile group {group_idx} requires non-empty 'layers'.")
            layers = tuple(int(layer) for layer in layers)
            if len(set(layers)) != len(layers):
                raise ValueError(f"Profile group {group_idx} contains duplicate layers.")
            for layer in layers:
                if layer in layer_to_group:
                    raise ValueError(f"Layer {layer} appears in multiple hidden mix groups.")
                layer_to_group[layer] = layers

            layer_set = set(layers)
            budget_weight = group_spec.get("budget_weight")
            if budget_weight is not None:
                budget_weight = float(budget_weight)
                if (not math.isfinite(budget_weight)) or budget_weight < 0.0:
                    raise ValueError(f"Profile group {group_idx} budget_weight must be finite and non-negative.")

            mix_spec = group_spec.get("mix", {})
            if mix_spec is None:
                mix_spec = {}
            if not isinstance(mix_spec, dict):
                raise ValueError(f"Profile group {group_idx} field 'mix' must be an object.")
            mix = {}
            for target_key, target_spec in mix_spec.items():
                target_layer = int(target_key)
                if target_layer not in layer_set:
                    raise ValueError(f"Mix target layer {target_layer} is not in group {layers}.")
                if not isinstance(target_spec, dict):
                    raise ValueError(f"Mix target {target_layer} must be an object.")
                sources = target_spec.get("sources")
                weights = target_spec.get("weights")
                if not isinstance(sources, list) or not sources:
                    raise ValueError(f"Mix target {target_layer} requires non-empty sources.")
                if not isinstance(weights, list) or len(weights) != len(sources):
                    raise ValueError(f"Mix target {target_layer} requires one weight per source.")
                sources = tuple(int(source) for source in sources)
                for source in sources:
                    if source not in layer_set:
                        raise ValueError(f"Mix source layer {source} is not in target {target_layer} group {layers}.")
                weights = tuple(float(weight) for weight in weights)
                if any((not math.isfinite(weight)) or weight < 0.0 for weight in weights):
                    raise ValueError(f"Mix target {target_layer} weights must be finite and non-negative.")
                weight_sum = sum(weights)
                if weight_sum <= 0.0:
                    raise ValueError(f"Mix target {target_layer} weights must sum to a positive value.")
                mix[target_layer] = tuple(
                    (source, weight / weight_sum) for source, weight in zip(sources, weights)
                )
            group_profiles[layers] = {"mix": mix, "budget_weight": budget_weight}

        budget_weights = [group["budget_weight"] for group in group_profiles.values()]
        has_budget_weights = all(weight is not None for weight in budget_weights)
        budget_weight_total = float(sum(budget_weights)) if has_budget_weights else 0.0
        use_budget_weights = has_budget_weights and budget_weight_total > 0.0

        return {
            "metric": raw_profile.get("metric"),
            "groups": group_profiles,
            "layer_to_group": layer_to_group,
            "profiled_layer_count": sum(len(layers) for layers in group_profiles),
            "budget_weight_total": budget_weight_total,
            "use_budget_weights": use_budget_weights,
        }

    def update_kv(self, key_states, query_states, value_states):
        return self.fallback.update_kv(key_states, query_states, value_states)

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
        if self._is_flatten_cache(layer_cache):
            if query_states.shape[-2] != 1:
                raise ValueError("snapkv_neighbor_shared flatten cache only supports q_len=1 after prefill.")
            return self._append_flatten_cache(layer_cache, key_states, value_states)

        if self._should_append_to_cache(query_states, layer_cache):
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )
            self._append_valid_mask(layer_cache, key_states)
        else:
            self._clear_padding_metadata(layer_cache)
            if self.model_config.update_kv is not True or key_states.shape[-2] < self.budget:
                self._set_layer_cache(layer_cache, key_states, value_states)
            if self.model_config.update_kv is not True:
                return key_states, value_states

        if self.model_config.update_kv is not True:
            return key_states, value_states

        hidden_window = self._update_hidden_window(layer_cache, hidden_states)
        position_window = self._update_position_window(layer_cache, position_embeddings, hidden_states.shape[-2])
        should_compress = (
            self.model_config.compression is None
            or query_states.shape[-2] > 1
            or self.model_config.compression is True
        )
        self._set_attention_mask_for_current_step(layer_cache, key_states, query_states)
        if not should_compress:
            return key_states, value_states

        if self._is_incomplete_tail_layer():
            compressed_k, compressed_v = self.fallback.update_kv(
                key_states,
                layer_cache.query_cache,
                value_states,
            )
            self._clear_padding_metadata(layer_cache)
            self._set_layer_cache(layer_cache, compressed_k, compressed_v)
            return key_states, value_states

        if self._valid_token_count(layer_cache, key_states) < self.budget:
            return key_states, value_states

        self._set_attention_mask_for_current_step(layer_cache, key_states, query_states)
        self._store_group_entry(
            attention=attention,
            hidden_window=hidden_window,
            position_window=position_window,
            key_states=key_states,
            value_states=value_states,
            layer_cache=layer_cache,
        )
        return key_states, value_states

    def _should_append_to_cache(self, query_states, layer_cache):
        has_existing_cache = (
            getattr(layer_cache, "keys", None) is not None
            and torch.is_tensor(layer_cache.keys)
            and layer_cache.keys.numel() > 0
        )
        if has_existing_cache:
            return True
        return not (self.model_config.compression is None or query_states.shape[-2] > 1)

    def _is_incomplete_tail_layer(self):
        if self.hidden_mix_profile is not None:
            return False
        num_layers = getattr(self.model_config, "num_hidden_layers", None)
        if num_layers is None:
            return False
        return self._group_start(self.layer_idx) + 2 >= int(num_layers)

    def _group_start(self, layer_idx):
        return (int(layer_idx) // 3) * 3

    def _group_layers(self, layer_idx):
        layer_idx = int(layer_idx)
        if self.hidden_mix_profile is not None:
            return self.hidden_mix_profile["layer_to_group"].get(layer_idx, (layer_idx,))
        group_start = self._group_start(layer_idx)
        return (group_start, group_start + 1, group_start + 2)

    def _state(self):
        if not hasattr(self.model_config, "_snapkv_neighbor_shared_state"):
            self.model_config._snapkv_neighbor_shared_state = {"groups": {}}
        return self.model_config._snapkv_neighbor_shared_state

    def _store_group_entry(
        self,
        attention,
        hidden_window,
        position_window,
        key_states,
        value_states,
        layer_cache,
    ):
        group_layers = self._group_layers(self.layer_idx)
        group_key = tuple(group_layers)
        state = self._state()
        if self.layer_idx == group_layers[0]:
            state["groups"][group_key] = {}
        group = state["groups"].setdefault(group_key, {})
        group[self.layer_idx] = {
            "layer_idx": self.layer_idx,
            "attention": attention,
            "hidden_window": hidden_window,
            "position_window": position_window,
            "key_states": key_states,
            "value_states": value_states,
            "valid_mask": self._current_valid_mask(layer_cache, key_states),
            "layer_cache": layer_cache,
        }
        if all(layer_idx in group for layer_idx in group_layers):
            entries = [group[layer_idx] for layer_idx in group_layers]
            try:
                self._compress_group(entries, group_layers)
            finally:
                for entry in entries:
                    entry.clear()
                state["groups"].pop(group_key, None)

    def _compress_group(self, entries, group_layers):
        if any(self._valid_token_count(entry["layer_cache"], entry["key_states"]) < self.budget for entry in entries):
            return

        hidden_len = min(entry["hidden_window"].shape[-2] for entry in entries)
        entry_by_layer = {int(entry["layer_idx"]): entry for entry in entries}
        avg_hidden = None
        if self.hidden_mix_profile is None:
            avg_device = entries[-1]["hidden_window"].device
            avg_dtype = entries[-1]["hidden_window"].dtype
            avg_hidden = sum(
                entry["hidden_window"][:, -hidden_len:, :].to(device=avg_device, dtype=avg_dtype)
                for entry in entries
            ) / len(entries)
        score_tensors = []
        valid_tensors = []
        hist_lengths = []
        for entry in entries:
            key_states = entry["key_states"]
            valid_mask = entry["valid_mask"]
            hist_len = key_states.shape[-2] - self.window_size
            if hist_len < 1:
                return
            mixed_hidden = avg_hidden
            if mixed_hidden is None:
                mixed_hidden = self._mixed_hidden_window(entry, entry_by_layer, group_layers, hidden_len)
            query_states = self._project_query_window(
                entry["attention"],
                mixed_hidden,
                self._slice_position_window(entry["position_window"], hidden_len),
            )
            attn_scores = compute_attention_scores(query_states, key_states)
            hist_valid = valid_mask[:, :, :hist_len]
            query_scores = attn_scores[:, :, -min(self.window_size, attn_scores.shape[-2]) :, :hist_len]
            query_scores = query_scores.masked_fill(~hist_valid[:, :, None, :], torch.finfo(query_scores.dtype).min)
            attn_probs = F.softmax(query_scores, dim=-1, dtype=torch.float32)
            attn_probs = torch.where(hist_valid[:, :, None, :], attn_probs, torch.zeros_like(attn_probs))
            attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            pooled_scores = F.max_pool1d(
                attn_probs.mean(dim=-2).to(query_states.dtype),
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
            pooled_scores = pooled_scores.masked_fill(~hist_valid, torch.finfo(pooled_scores.dtype).min)
            # Put each layer/head distribution on a common scale before the shared top-k.
            pooled_scores = self._normalize_scores_for_global_rank(pooled_scores, hist_valid)
            score_tensors.append(pooled_scores)
            valid_tensors.append(hist_valid)
            hist_lengths.append(hist_len)

        batch_size = entries[0]["key_states"].shape[0]
        num_kv_heads = entries[0]["key_states"].shape[1]
        total_budget = self._group_historical_budget(group_layers, num_kv_heads)
        rank_device = score_tensors[-1].device
        rank_dtype = score_tensors[-1].dtype
        flat_scores = torch.cat(
            [scores.reshape(batch_size, -1).to(device=rank_device, dtype=rank_dtype) for scores in score_tensors],
            dim=-1,
        )
        flat_valid = torch.cat(
            [valid.reshape(batch_size, -1).to(device=rank_device) for valid in valid_tensors],
            dim=-1,
        )
        valid_count = flat_valid.sum(dim=-1)
        topk = min(int(total_budget), int(valid_count.min().item()))
        if topk < 1:
            return

        flat_scores = flat_scores.masked_fill(~flat_valid, torch.finfo(flat_scores.dtype).min)
        selected_flat = torch.zeros_like(flat_valid, dtype=torch.bool)
        topk_indices = flat_scores.topk(topk, dim=-1).indices
        selected_flat.scatter_(dim=-1, index=topk_indices, value=True)

        offset = 0
        for entry, hist_len in zip(entries, hist_lengths):
            width = num_kv_heads * hist_len
            selected = selected_flat[:, offset : offset + width].view(batch_size, num_kv_heads, hist_len)
            selected = selected.to(device=entry["key_states"].device)
            self._pack_layer(entry, selected, hist_len)
            offset += width

    def _normalize_scores_for_global_rank(self, scores, valid_mask):
        rank_scores = scores.to(dtype=torch.float32)
        valid_mask = valid_mask.to(device=rank_scores.device, dtype=torch.bool)
        zeros = torch.zeros_like(rank_scores)
        valid_scores = torch.where(valid_mask, rank_scores, zeros)
        valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = valid_scores.sum(dim=-1, keepdim=True) / valid_count
        centered = torch.where(valid_mask, rank_scores - mean, zeros)
        variance = centered.square().sum(dim=-1, keepdim=True) / valid_count
        std = variance.sqrt().clamp_min(torch.finfo(rank_scores.dtype).eps)
        normalized = centered / std
        return normalized.masked_fill(~valid_mask, torch.finfo(normalized.dtype).min)

    def _group_historical_budget(self, group_layers, num_kv_heads):
        per_layer_history = self.budget - self.window_size
        if self.hidden_mix_profile is None or not self.hidden_mix_profile.get("use_budget_weights"):
            return len(group_layers) * num_kv_heads * per_layer_history

        group = self.hidden_mix_profile["groups"].get(tuple(group_layers))
        budget_weight = None if group is None else group.get("budget_weight")
        total_weight = self.hidden_mix_profile.get("budget_weight_total", 0.0)
        profiled_layer_count = self.hidden_mix_profile.get("profiled_layer_count", len(group_layers))
        if budget_weight is None or total_weight <= 0.0 or profiled_layer_count < 1:
            return len(group_layers) * num_kv_heads * per_layer_history
        total_hist_budget = int(profiled_layer_count) * num_kv_heads * per_layer_history
        return max(1, int(round(total_hist_budget * float(budget_weight) / float(total_weight))))

    def _mixed_hidden_window(self, entry, entry_by_layer, group_layers, hidden_len):
        target_layer = int(entry["layer_idx"])
        mix = self._target_mix(group_layers, target_layer)
        if mix is None:
            return entry["hidden_window"][:, -hidden_len:, :]

        device = entry["hidden_window"].device
        dtype = entry["hidden_window"].dtype
        mixed_hidden = None
        for source_layer, weight in mix:
            source_hidden = entry_by_layer[int(source_layer)]["hidden_window"][:, -hidden_len:, :]
            source_hidden = source_hidden.to(device=device, dtype=dtype)
            weighted = source_hidden * weight
            mixed_hidden = weighted if mixed_hidden is None else mixed_hidden + weighted
        return mixed_hidden

    def _target_mix(self, group_layers, target_layer):
        if self.hidden_mix_profile is None:
            return None
        group = self.hidden_mix_profile["groups"].get(tuple(group_layers))
        if group is None:
            return None
        return group["mix"].get(int(target_layer))

    def _pack_layer(self, entry, selected_hist_mask, hist_len):
        key_states = entry["key_states"]
        value_states = entry["value_states"]
        valid_mask = entry["valid_mask"]
        batch_size, num_heads, _, head_dim = key_states.shape
        keep_indices = []
        lengths = torch.zeros(batch_size, num_heads, dtype=torch.long, device=key_states.device)
        for batch_idx in range(batch_size):
            per_batch = []
            for head_idx in range(num_heads):
                hist_idx = torch.where(selected_hist_mask[batch_idx, head_idx])[0]
                hist_idx = hist_idx.sort().values
                recent_valid = valid_mask[batch_idx, head_idx, hist_len:]
                recent_idx = torch.where(recent_valid)[0] + hist_len
                cur_indices = torch.cat([hist_idx, recent_idx], dim=0)
                per_batch.append(cur_indices)
                lengths[batch_idx, head_idx] = cur_indices.numel()
            keep_indices.append(per_batch)

        layer_cache = entry["layer_cache"]
        if self._should_use_flatten_cache(entry["attention"], key_states, value_states, lengths):
            self._set_flatten_layer_cache(layer_cache, key_states, value_states, keep_indices, lengths)
            return

        max_len = max(int(lengths.max().item()), 1)
        packed_keys = key_states.new_zeros(batch_size, num_heads, max_len, head_dim)
        packed_values = value_states.new_zeros(batch_size, num_heads, max_len, head_dim)
        packed_mask = torch.zeros(batch_size, num_heads, max_len, dtype=torch.bool, device=key_states.device)
        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                cur_indices = keep_indices[batch_idx][head_idx]
                cur_len = int(cur_indices.numel())
                if cur_len == 0:
                    continue
                packed_keys[batch_idx, head_idx, :cur_len] = key_states[batch_idx, head_idx].index_select(0, cur_indices)
                packed_values[batch_idx, head_idx, :cur_len] = value_states[batch_idx, head_idx].index_select(0, cur_indices)
                packed_mask[batch_idx, head_idx, :cur_len] = True

        self._set_layer_cache(layer_cache, packed_keys, packed_values)
        layer_cache.kv_valid_mask = packed_mask
        layer_cache.kv_lengths = lengths

    def _set_layer_cache(self, layer_cache, key_states, value_states):
        self._clear_flatten_metadata(layer_cache)
        layer_cache.keys = key_states
        layer_cache.values = value_states
        layer_cache.dtype = key_states.dtype
        layer_cache.device = key_states.device
        layer_cache.is_initialized = True

    def _should_use_flatten_cache(self, attention, key_states, value_states, lengths):
        if self.flatten_cache is False:
            return False

        reasons = []
        if getattr(self.model_config, "update_kv", None) is not True:
            reasons.append("update_kv must be True")
        if getattr(attention.config, "_attn_implementation", None) != "flash_attention_2":
            reasons.append("attn_implementation must be flash_attention_2")
        if key_states.shape[0] != 1:
            reasons.append("batch_size must be 1")
        if key_states.device.type != "cuda" or value_states.device.type != "cuda":
            reasons.append("key/value states must be CUDA tensors")
        if key_states.dtype not in (torch.float16, torch.bfloat16):
            reasons.append("key/value dtype must be float16 or bfloat16")
        if lengths.shape[0] != 1:
            reasons.append("flatten metadata supports one batch only")

        if reasons:
            if self.flatten_cache is True:
                raise ValueError("flatten_cache=True cannot be enabled: " + "; ".join(reasons) + ".")
            return False
        return True

    def _set_flatten_layer_cache(self, layer_cache, key_states, value_states, keep_indices, lengths):
        batch_size, num_heads, _, head_dim = key_states.shape
        if batch_size != 1:
            raise ValueError("snapkv_neighbor_shared flatten cache only supports batch_size=1.")

        flat_keys = []
        flat_values = []
        for head_idx in range(num_heads):
            cur_indices = keep_indices[0][head_idx]
            if cur_indices.numel() == 0:
                continue
            flat_keys.append(key_states[0, head_idx].index_select(0, cur_indices))
            flat_values.append(value_states[0, head_idx].index_select(0, cur_indices))

        if flat_keys:
            layer_cache.keys = torch.cat(flat_keys, dim=0).contiguous()
            layer_cache.values = torch.cat(flat_values, dim=0).contiguous()
        else:
            layer_cache.keys = key_states.new_zeros(0, head_dim)
            layer_cache.values = value_states.new_zeros(0, head_dim)
        self._set_flatten_metadata(layer_cache, lengths[0])
        layer_cache.dtype = key_states.dtype
        layer_cache.device = key_states.device
        layer_cache.is_initialized = True
        self._clear_padding_metadata(layer_cache)

    def _append_flatten_cache(self, layer_cache, key_states, value_states):
        if key_states.shape[0] != 1 or key_states.shape[-2] != 1:
            raise ValueError("snapkv_neighbor_shared flatten cache decode update requires shape [1, heads, 1, dim].")
        if key_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("snapkv_neighbor_shared flatten cache requires float16 or bfloat16 key/value tensors.")
        if key_states.device.type != "cuda" or value_states.device.type != "cuda":
            raise ValueError("snapkv_neighbor_shared flatten cache requires CUDA key/value tensors.")

        from tiny_api_cuda import update_flatten_view

        head_lens = layer_cache.kv_head_lens.to(device=key_states.device, dtype=torch.int32).contiguous()
        cu_lens = layer_cache.kv_cu_lens.to(device=key_states.device, dtype=torch.int32).contiguous()
        _, num_heads, _, head_dim = key_states.shape
        if head_lens.numel() != num_heads:
            raise ValueError("flatten cache head count does not match the decode key/value tensors.")

        flat_new_keys = key_states.contiguous().view(-1, head_dim)
        flat_new_values = value_states.contiguous().view(-1, head_dim)
        with torch.cuda.device(key_states.device):
            layer_cache.keys = update_flatten_view(
                layer_cache.keys.contiguous().view(-1, head_dim),
                flat_new_keys,
                head_lens,
                cu_lens,
            )
            layer_cache.values = update_flatten_view(
                layer_cache.values.contiguous().view(-1, head_dim),
                flat_new_values,
                head_lens,
                cu_lens,
            )
        self._set_flatten_metadata(layer_cache, head_lens.to(dtype=torch.long) + 1)
        layer_cache.dtype = key_states.dtype
        layer_cache.device = key_states.device
        layer_cache.is_initialized = True
        return layer_cache.keys, layer_cache.values

    def _set_flatten_metadata(self, layer_cache, head_lens):
        from types import MethodType

        head_lens = head_lens.to(device=layer_cache.keys.device, dtype=torch.int32).contiguous()
        cu_lens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=head_lens.device),
                torch.cumsum(head_lens, dim=0, dtype=torch.int32),
            ],
            dim=0,
        )
        layer_cache.kv_head_lens = head_lens
        layer_cache.kv_cu_lens = cu_lens
        layer_cache.kv_max_seqlen = int(head_lens.max().item()) if head_lens.numel() else 0
        layer_cache.kv_flatten_enabled = True
        layer_cache.kv_num_heads = int(head_lens.numel())

        def _flatten_get_seq_length(cache_self):
            return int(getattr(cache_self, "kv_max_seqlen", 0))

        layer_cache.get_seq_length = MethodType(_flatten_get_seq_length, layer_cache)

    def _is_flatten_cache(self, layer_cache):
        return bool(getattr(layer_cache, "kv_flatten_enabled", False))

    def _set_attention_mask_for_current_step(self, layer_cache, key_states, query_states):
        if query_states.shape[-2] == 1:
            layer_cache.attention_kv_valid_mask = self._current_valid_mask(layer_cache, key_states)
        elif hasattr(layer_cache, "attention_kv_valid_mask"):
            delattr(layer_cache, "attention_kv_valid_mask")

    def _valid_token_count(self, layer_cache, key_states):
        valid_mask = getattr(layer_cache, "kv_valid_mask", None)
        if valid_mask is None or valid_mask.shape[-1] != key_states.shape[-2]:
            return key_states.shape[-2]
        return int(valid_mask.sum(dim=-1).min().item())

    def _current_valid_mask(self, layer_cache, key_states):
        valid_mask = getattr(layer_cache, "kv_valid_mask", None)
        if valid_mask is None or valid_mask.shape[-1] != key_states.shape[-2]:
            return torch.ones(
                key_states.shape[:3],
                dtype=torch.bool,
                device=key_states.device,
            )
        return valid_mask

    def _append_valid_mask(self, layer_cache, key_states):
        valid_mask = getattr(layer_cache, "kv_valid_mask", None)
        if valid_mask is None:
            return
        old_len = valid_mask.shape[-1]
        new_len = key_states.shape[-2]
        if new_len <= old_len:
            layer_cache.kv_valid_mask = valid_mask[:, :, :new_len]
            layer_cache.kv_lengths = layer_cache.kv_valid_mask.sum(dim=-1)
            return
        appended = torch.ones(
            *valid_mask.shape[:2],
            new_len - old_len,
            dtype=torch.bool,
            device=key_states.device,
        )
        layer_cache.kv_valid_mask = torch.cat([valid_mask.to(key_states.device), appended], dim=-1)
        layer_cache.kv_lengths = layer_cache.kv_valid_mask.sum(dim=-1)

    def _clear_padding_metadata(self, layer_cache):
        if hasattr(layer_cache, "kv_valid_mask"):
            delattr(layer_cache, "kv_valid_mask")
        if hasattr(layer_cache, "kv_lengths"):
            delattr(layer_cache, "kv_lengths")
        if hasattr(layer_cache, "attention_kv_valid_mask"):
            delattr(layer_cache, "attention_kv_valid_mask")

    def _clear_flatten_metadata(self, layer_cache):
        for attr in (
            "kv_head_lens",
            "kv_cu_lens",
            "kv_max_seqlen",
            "kv_flatten_enabled",
            "kv_num_heads",
        ):
            if hasattr(layer_cache, attr):
                delattr(layer_cache, attr)
        if "get_seq_length" in getattr(layer_cache, "__dict__", {}):
            delattr(layer_cache, "get_seq_length")

    def _update_hidden_window(self, layer_cache, hidden_states):
        if hidden_states.shape[-2] >= self.window_size or not hasattr(layer_cache, "neighbor_hidden_window"):
            hidden_window = hidden_states[:, -self.window_size :, :].detach().clone()
        else:
            hidden_window = torch.cat([layer_cache.neighbor_hidden_window.to(hidden_states.device), hidden_states.detach()], dim=1)
            hidden_window = hidden_window[:, -self.window_size :, :].clone()
        layer_cache.neighbor_hidden_window = hidden_window
        return hidden_window

    def _update_position_window(self, layer_cache, position_embeddings, seq_len):
        if position_embeddings is None:
            return None
        cos, sin = position_embeddings
        cos = self._slice_last_positions(cos.detach(), seq_len).clone()
        sin = self._slice_last_positions(sin.detach(), seq_len).clone()
        if cos.shape[-2] >= self.window_size or not hasattr(layer_cache, "neighbor_position_window"):
            cos_window = self._slice_last_positions(cos, self.window_size).clone()
            sin_window = self._slice_last_positions(sin, self.window_size).clone()
        else:
            prev_cos, prev_sin = layer_cache.neighbor_position_window
            cos_window = torch.cat([prev_cos.to(cos.device), cos], dim=-2)
            sin_window = torch.cat([prev_sin.to(sin.device), sin], dim=-2)
            cos_window = self._slice_last_positions(cos_window, self.window_size).clone()
            sin_window = self._slice_last_positions(sin_window, self.window_size).clone()
        layer_cache.neighbor_position_window = (cos_window, sin_window)
        return layer_cache.neighbor_position_window

    def _project_query_window(self, attention, hidden_window, position_embeddings):
        device = next(attention.parameters()).device
        dtype = next(attention.parameters()).dtype
        hidden_window = hidden_window.to(device=device, dtype=dtype)
        input_shape = hidden_window.shape[:-1]
        head_dim = attention.head_dim
        num_attention_heads = attention.config.num_attention_heads
        q_proj = attention.q_proj(hidden_window)
        if q_proj.shape[-1] == num_attention_heads * head_dim * 2:
            query_states, _ = torch.chunk(
                q_proj.view(*input_shape, num_attention_heads, head_dim * 2),
                2,
                dim=-1,
            )
        else:
            query_states = q_proj.view(*input_shape, num_attention_heads, head_dim)
        q_norm = getattr(attention, "q_norm", None)
        if q_norm is not None:
            query_states = q_norm(query_states)
        query_states = query_states.transpose(1, 2)
        if position_embeddings is None:
            return query_states
        cos, sin = position_embeddings
        cos = self._slice_last_positions(cos.to(device=device, dtype=dtype), query_states.shape[-2])
        sin = self._slice_last_positions(sin.to(device=device, dtype=dtype), query_states.shape[-2])
        return self._apply_rotary(query_states, cos, sin)

    def _slice_position_window(self, position_window, seq_len):
        if position_window is None:
            return None
        cos, sin = position_window
        return self._slice_last_positions(cos, seq_len), self._slice_last_positions(sin, seq_len)

    def _slice_last_positions(self, tensor, seq_len):
        if tensor.ndim == 2:
            return tensor[-seq_len:, :]
        return tensor[..., -seq_len:, :]

    def _apply_rotary(self, states, cos, sin):
        cos = self._unsqueeze_position_embedding(cos, states)
        sin = self._unsqueeze_position_embedding(sin, states)
        return (states * cos) + (self._rotate_half(states) * sin)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _unsqueeze_position_embedding(self, position_embedding, states):
        if position_embedding.ndim == 2:
            position_embedding = position_embedding.unsqueeze(0)
        while position_embedding.ndim < states.ndim:
            position_embedding = position_embedding.unsqueeze(1)
        return position_embedding


def masked_eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    kv_valid_mask,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    scaling = module.head_dim**-0.5 if scaling is None else scaling
    num_key_value_groups = query.shape[1] // key.shape[1]
    if num_key_value_groups != 1:
        key = _repeat_kv(key, num_key_value_groups)
        value = _repeat_kv(value, num_key_value_groups)
        kv_valid_mask = kv_valid_mask[:, :, None, :].expand(
            kv_valid_mask.shape[0],
            kv_valid_mask.shape[1],
            num_key_value_groups,
            kv_valid_mask.shape[-1],
        )
        kv_valid_mask = kv_valid_mask.reshape(kv_valid_mask.shape[0], query.shape[1], kv_valid_mask.shape[-1])

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=attn_weights.device)
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        if causal_mask.shape[-1] < key.shape[-2]:
            causal_mask = F.pad(causal_mask, (0, key.shape[-2] - causal_mask.shape[-1]), value=0.0)
        attn_weights = attn_weights + causal_mask
    attn_weights = attn_weights.masked_fill(~kv_valid_mask[:, :, None, :], torch.finfo(attn_weights.dtype).min)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights if kwargs.get("output_attentions", False) else None


def _repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
