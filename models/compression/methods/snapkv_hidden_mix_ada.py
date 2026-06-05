import json
import math
import os

import torch
import torch.nn.functional as F

from . import compute_attention_scores
from .snapkv import SnapKV


class SnapKVHiddenMix:
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
        importance_epsilon=1e-12,
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
        self.importance_epsilon = float(importance_epsilon)
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []

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
        if model_config is not None and not hasattr(model_config, "_snapkv_hidden_mix_state"):
            model_config._snapkv_hidden_mix_state = {"groups": {}}

    def _load_hidden_mix_profile(self, profile_path):
        if profile_path is None:
            return None
        if self.model_config is None:
            raise ValueError("hidden_mix_profile_path requires model_config.")

        profile_path = os.path.abspath(os.path.expanduser(str(profile_path)))
        cached = getattr(self.model_config, "_snapkv_hidden_mix_profile", None)
        if cached is not None and cached.get("path") == profile_path:
            return cached["profile"]

        with open(profile_path, "r", encoding="utf-8") as handle:
            raw_profile = json.load(handle)
        profile = self._parse_hidden_mix_profile(raw_profile, profile_path)
        self.model_config._snapkv_hidden_mix_profile = {"path": profile_path, "profile": profile}
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

            raw_mix = group_spec.get("mix", {})
            if raw_mix is None:
                raw_mix = {}
            if not isinstance(raw_mix, dict):
                raise ValueError(f"Profile group {group_idx} mix must be an object.")
            mix = {}
            for target_text, mix_spec in raw_mix.items():
                target_layer = int(target_text)
                if target_layer not in layers:
                    raise ValueError(f"Profile group {group_idx} has mix target outside layers: {target_layer}.")
                if not isinstance(mix_spec, dict):
                    raise ValueError(f"Profile group {group_idx} mix for layer {target_layer} must be an object.")
                sources = mix_spec.get("sources")
                weights = mix_spec.get("weights")
                if not isinstance(sources, list) or not isinstance(weights, list) or len(sources) != len(weights):
                    raise ValueError(f"Profile group {group_idx} mix for layer {target_layer} needs equal sources/weights lists.")
                if not sources:
                    raise ValueError(f"Profile group {group_idx} mix for layer {target_layer} must not be empty.")
                parsed_sources = tuple(int(source) for source in sources)
                parsed_weights = tuple(float(weight) for weight in weights)
                if any(source not in layers for source in parsed_sources):
                    raise ValueError(f"Profile group {group_idx} mix source must stay within its group.")
                weight_sum = float(sum(parsed_weights))
                if (not math.isfinite(weight_sum)) or weight_sum <= 0.0:
                    raise ValueError(f"Profile group {group_idx} mix weights must have a positive finite sum.")
                if any((not math.isfinite(weight)) or weight < 0.0 for weight in parsed_weights):
                    raise ValueError(f"Profile group {group_idx} mix weights must be finite and non-negative.")
                mix[target_layer] = {
                    "sources": parsed_sources,
                    "weights": tuple(weight / weight_sum for weight in parsed_weights),
                }
            group_profiles[layers] = {"mix": mix}

        return {
            "metric": raw_profile.get("metric"),
            "groups": group_profiles,
            "layer_to_group": layer_to_group,
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
                raise ValueError("snapkv_hidden_mix flatten cache only supports q_len=1 after prefill.")
            return self._append_flatten_cache(layer_cache, key_states, value_states)

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

        should_compress = (
            self.model_config.compression is None
            or query_states.shape[-2] > 1
        )
        self._set_attention_mask_for_current_step(layer_cache, key_states, query_states)
        if not should_compress or self._valid_token_count(layer_cache, key_states) < self.budget:
            return key_states, value_states

        query_cache = getattr(layer_cache, "query_cache", None)
        if query_cache is None or query_cache.shape[-2] == 0:
            query_cache = query_states[:, :, -self.window_size :, :]
        attn_cache = self._compute_attn_cache(
            key_states,
            query_cache,
            self._current_valid_mask(layer_cache, key_states),
        )
        self._store_group_entry(
            attention=attention,
            key_states=key_states,
            value_states=value_states,
            attn_cache=attn_cache,
            layer_cache=layer_cache,
        )
        return key_states, value_states

    def finalize_after_attention(self, attention, hidden_states, attn_output, layer_cache):
        group_layers = self._group_layers(self.layer_idx)
        group_key = tuple(group_layers)
        group = self._state()["groups"].get(group_key)
        if group is None:
            return
        entry = group.get(self.layer_idx)
        if entry is None:
            return

        importance = self._compute_cosine_importance(
            attention=attention,
            hidden_states=hidden_states,
            attn_output=attn_output,
            num_key_value_heads=entry["attn_cache"].shape[1],
            hist_len=entry["attn_cache"].shape[-1],
        )
        importance = importance.to(device=entry["attn_cache"].device, dtype=entry["attn_cache"].dtype)
        if importance.shape != entry["attn_cache"].shape:
            raise ValueError("snapkv_hidden_mix cosine importance must match attn_cache shape.")
        entry["cosine_importance"] = importance
        entry["finalized"] = True
        if all(layer_idx in group and group[layer_idx].get("finalized") for layer_idx in group_layers):
            entries = [group[layer_idx] for layer_idx in group_layers]
            try:
                self._compress_group(entries, group_layers)
            finally:
                for cur_entry in entries:
                    cur_entry.clear()
                self._state()["groups"].pop(group_key, None)

    def _compute_cosine_importance(
        self,
        attention,
        hidden_states,
        attn_output,
        num_key_value_heads,
        hist_len,
    ):
        if (
            hidden_states is None
            or attn_output is None
            or not torch.is_tensor(hidden_states)
            or not torch.is_tensor(attn_output)
            or hidden_states.ndim != 3
            or attn_output.ndim != 3
            or hist_len < 1
        ):
            if torch.is_tensor(hidden_states):
                device = hidden_states.device
            elif torch.is_tensor(attn_output):
                device = attn_output.device
            else:
                device = torch.device("cpu")
            return torch.ones(1, int(num_key_value_heads), int(hist_len), device=device)

        batch_size = hidden_states.shape[0]
        actual_len = min(int(hidden_states.shape[1]), int(attn_output.shape[1]), int(hist_len))
        device = hidden_states.device
        importance = torch.ones(
            batch_size,
            int(num_key_value_heads),
            int(hist_len),
            dtype=torch.float32,
            device=device,
        )
        if actual_len < 1:
            return importance

        input_vectors = hidden_states[:, :actual_len, :].detach().to(dtype=torch.float32)
        output_vectors = attn_output[:, :actual_len, :].detach().to(device=device, dtype=torch.float32)
        cosine = self._token_cosine(input_vectors, output_vectors)

        num_attention_heads = getattr(getattr(attention, "config", None), "num_attention_heads", None)
        head_dim = getattr(attention, "head_dim", None)
        hidden_size = input_vectors.shape[-1]
        if (
            num_attention_heads is not None
            and head_dim is not None
            and int(num_attention_heads) > 0
            and int(num_key_value_heads) > 0
            and int(num_attention_heads) % int(num_key_value_heads) == 0
            and hidden_size == int(num_attention_heads) * int(head_dim)
            and output_vectors.shape[-1] == hidden_size
        ):
            num_attention_heads = int(num_attention_heads)
            head_dim = int(head_dim)
            num_key_value_groups = num_attention_heads // int(num_key_value_heads)
            head_input = input_vectors.view(batch_size, actual_len, num_attention_heads, head_dim)
            head_output = output_vectors.view(batch_size, actual_len, num_attention_heads, head_dim)
            head_cosine = self._token_cosine(head_input, head_output)
            cosine = head_cosine.transpose(1, 2).reshape(
                batch_size,
                int(num_key_value_heads),
                num_key_value_groups,
                actual_len,
            ).mean(dim=2)
        else:
            cosine = cosine[:, None, :].expand(batch_size, int(num_key_value_heads), actual_len)

        importance[:, :, :actual_len] = torch.exp(cosine.clamp(min=-1.0, max=1.0))
        return importance

    def _token_cosine(self, input_vectors, output_vectors):
        dot = (input_vectors * output_vectors).sum(dim=-1)
        input_norm = input_vectors.square().sum(dim=-1).sqrt()
        output_norm = output_vectors.square().sum(dim=-1).sqrt()
        denom = (input_norm * output_norm).clamp_min(self.importance_epsilon)
        return (dot / denom).clamp(min=-1.0, max=1.0)

    def _should_append_to_cache(self, query_states, layer_cache):
        has_existing_cache = (
            getattr(layer_cache, "keys", None) is not None
            and torch.is_tensor(layer_cache.keys)
            and layer_cache.keys.numel() > 0
        )
        if has_existing_cache:
            return True
        return not (self.model_config.compression is None or query_states.shape[-2] > 1)

    def _compute_attn_cache(self, key_states, query_states, valid_mask=None):
        bsz, num_key_value_heads, kv_cache_len, _ = key_states.shape
        hist_len = kv_cache_len - self.window_size
        if hist_len < 1:
            return key_states.new_zeros(bsz, num_key_value_heads, 0)

        num_key_value_groups = query_states.shape[1] // num_key_value_heads
        query_window = min(self.window_size, query_states.shape[-2])
        query_states = query_states[:, :, -query_window:, :]

        attn_weights = compute_attention_scores(query_states, key_states)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=kv_cache_len - query_window + 1)
        attn_weights = attn_weights + attention_mask
        if valid_mask is not None:
            full_valid = valid_mask[:, :, None, :].expand(
                bsz,
                num_key_value_heads,
                num_key_value_groups,
                kv_cache_len,
            )
            full_valid = full_valid.reshape(bsz, query_states.shape[1], kv_cache_len)
            attn_weights = attn_weights.masked_fill(
                ~full_valid[:, :, None, :],
                torch.finfo(attn_weights.dtype).min,
            )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :hist_len]

        scores = attn_weights.view(
            bsz,
            num_key_value_heads,
            num_key_value_groups,
            query_window,
            hist_len,
        )
        attn_weights_sum = scores.mean(dim=2).mean(dim=-2)
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        if valid_mask is not None:
            hist_valid = valid_mask[:, :, :hist_len].to(device=attn_cache.device, dtype=torch.bool)
            attn_cache = torch.where(hist_valid, attn_cache, torch.zeros_like(attn_cache))
        return attn_cache

    def _group_layers(self, layer_idx):
        layer_idx = int(layer_idx)
        if self.hidden_mix_profile is not None:
            return self.hidden_mix_profile["layer_to_group"].get(layer_idx, (layer_idx,))
        return (layer_idx,)

    def _state(self):
        if not hasattr(self.model_config, "_snapkv_hidden_mix_state"):
            self.model_config._snapkv_hidden_mix_state = {"groups": {}}
        return self.model_config._snapkv_hidden_mix_state

    def _store_group_entry(self, attention, key_states, value_states, attn_cache, layer_cache):
        group_layers = self._group_layers(self.layer_idx)
        group_key = tuple(group_layers)
        state = self._state()
        if self.layer_idx == group_layers[0]:
            state["groups"][group_key] = {}
        group = state["groups"].setdefault(group_key, {})
        group[self.layer_idx] = {
            "layer_idx": self.layer_idx,
            "attention": attention,
            "key_states": key_states,
            "value_states": value_states,
            "attn_cache": attn_cache,
            "valid_mask": self._current_valid_mask(layer_cache, key_states),
            "layer_cache": layer_cache,
            "cosine_importance": None,
            "finalized": False,
        }

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
            importance = entry.get("cosine_importance")
            if importance is not None:
                importance = importance.to(device=mixed_cache.device, dtype=mixed_cache.dtype)
                if importance.shape != mixed_cache.shape:
                    raise ValueError("snapkv_hidden_mix cosine importance must match mixed attn_cache shape.")
                mixed_cache = mixed_cache * importance
            selected_hist_mask, hist_len = self._select_layer_head_topk(entry, mixed_cache)
            self._pack_layer(entry, selected_hist_mask, hist_len, mixed_cache)

    def _select_layer_head_topk(self, entry, scores):
        key_states = entry["key_states"]
        batch_size, num_heads = key_states.shape[:2]
        hist_len = key_states.shape[-2] - self.window_size
        if hist_len < 1:
            return torch.zeros(batch_size, num_heads, 0, dtype=torch.bool, device=key_states.device), 0
        if scores.shape[-1] != hist_len:
            raise ValueError("snapkv_hidden_mix attn_cache length must match historical cache length.")

        valid_mask = entry["valid_mask"]
        hist_valid = valid_mask[:, :, :hist_len].to(device=scores.device, dtype=torch.bool)
        flat_valid = hist_valid.reshape(batch_size, -1)
        valid_count = flat_valid.sum(dim=-1)
        total_budget = num_heads * (self.budget - self.window_size)
        topk = min(int(total_budget), int(valid_count.min().item()))
        selected_flat = torch.zeros_like(flat_valid, dtype=torch.bool)
        if topk > 0:
            flat_scores = scores.reshape(batch_size, -1).masked_fill(
                ~flat_valid,
                torch.finfo(scores.dtype).min,
            )
            topk_indices = flat_scores.topk(topk, dim=-1).indices
            selected_flat.scatter_(dim=-1, index=topk_indices, value=True)
        return selected_flat.view(batch_size, num_heads, hist_len).to(device=key_states.device), hist_len

    def _normalize_attn_cache(self, attn_cache):
        rank_cache = attn_cache.to(dtype=torch.float32)
        mean = rank_cache.mean(dim=-1, keepdim=True)
        centered = rank_cache - mean
        std = centered.square().mean(dim=-1, keepdim=True).sqrt()
        std = std.clamp_min(torch.finfo(rank_cache.dtype).eps)
        return (centered / std).to(dtype=attn_cache.dtype)

    def _mix_for_layer(self, group_layers, target_layer):
        if self.hidden_mix_profile is None:
            return {"sources": (target_layer,), "weights": (1.0,)}
        group = self.hidden_mix_profile["groups"].get(tuple(group_layers), {})
        mix = group.get("mix", {}).get(int(target_layer))
        if mix is None:
            return {"sources": (target_layer,), "weights": (1.0,)}
        return mix

    def _pack_layer(self, entry, selected_hist_mask, hist_len, scores):
        key_states = entry["key_states"]
        value_states = entry["value_states"]
        valid_mask = entry["valid_mask"]
        batch_size, num_heads, _, head_dim = key_states.shape
        keep_indices = []
        lengths = torch.zeros(batch_size, num_heads, dtype=torch.long, device=key_states.device)
        for batch_idx in range(batch_size):
            per_batch = []
            for head_idx in range(num_heads):
                hist_idx = torch.where(selected_hist_mask[batch_idx, head_idx])[0].sort().values
                recent_valid = valid_mask[batch_idx, head_idx, hist_len:].to(device=key_states.device, dtype=torch.bool)
                recent_idx = torch.where(recent_valid)[0] + hist_len
                cur_indices = torch.cat([hist_idx, recent_idx], dim=0)
                per_batch.append(cur_indices)
                lengths[batch_idx, head_idx] = cur_indices.numel()
            keep_indices.append(per_batch)

        if self.record_kept_token_indices:
            self._record_kept_indices(selected_hist_mask, scores, hist_len, key_states.shape[-2], lengths)

        layer_cache = entry["layer_cache"]
        if self._should_use_flatten_cache(entry.get("attention"), key_states, value_states, lengths):
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

    def _record_kept_indices(self, selected_hist_mask, scores, hist_len, kv_cache_len, lengths):
        batch_size, num_heads = selected_hist_mask.shape[:2]
        max_len = max(int(lengths.max().item()), 1)
        cur_indices = torch.full((batch_size, num_heads, max_len), -1, dtype=torch.long, device="cpu")
        kept_attn = torch.zeros(batch_size, num_heads, max(int(selected_hist_mask.sum(dim=-1).max().item()), 1), device="cpu")
        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                hist_idx = torch.where(selected_hist_mask[batch_idx, head_idx])[0].sort().values.to("cpu")
                recent_idx = torch.arange(hist_len, kv_cache_len, device="cpu")
                combined = torch.cat([hist_idx, recent_idx], dim=0)
                cur_indices[batch_idx, head_idx, : combined.numel()] = combined
                if hist_idx.numel() > 0:
                    cur_scores = scores[batch_idx, head_idx].detach().to("cpu")
                    kept_attn[batch_idx, head_idx, : hist_idx.numel()] = cur_scores.index_select(0, hist_idx)

        if self.evicted_token_num > 0:
            valid = cur_indices >= 0
            cur_indices[valid] += self.evicted_token_num
        self.kept_attention_scores.append(kept_attn.squeeze(0))
        self.kept_token_indices.append(cur_indices.squeeze(0))
        self.evicted_token_num += kv_cache_len - int(lengths.min().item())

    def _should_use_flatten_cache(self, attention, key_states, value_states, lengths):
        if self.flatten_cache is False:
            return False

        reasons = []
        attention_config = getattr(attention, "config", None)
        if getattr(self.model_config, "update_kv", None) is not True:
            reasons.append("update_kv must be True")
        if getattr(attention_config, "_attn_implementation", None) != "flash_attention_2":
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
            raise ValueError("snapkv_hidden_mix flatten cache only supports batch_size=1.")

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
            raise ValueError("snapkv_hidden_mix flatten cache decode update requires shape [1, heads, 1, dim].")
        if key_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("snapkv_hidden_mix flatten cache requires float16 or bfloat16 key/value tensors.")
        if key_states.device.type != "cuda" or value_states.device.type != "cuda":
            raise ValueError("snapkv_hidden_mix flatten cache requires CUDA key/value tensors.")

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
        return valid_mask.to(device=key_states.device, dtype=torch.bool)

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

    def _set_layer_cache(self, layer_cache, key_states, value_states):
        self._clear_flatten_metadata(layer_cache)
        self._clear_padding_metadata(layer_cache)
        layer_cache.keys = key_states
        layer_cache.values = value_states
        layer_cache.dtype = key_states.dtype
        layer_cache.device = key_states.device
        layer_cache.is_initialized = True
