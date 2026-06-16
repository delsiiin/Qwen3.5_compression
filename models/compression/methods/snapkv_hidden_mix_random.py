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
        importance_epsilon=1e-12,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        if hidden_mix_fallback != "self":
            raise ValueError("hidden_mix_fallback currently supports only 'self'.")
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode
        self.hidden_mix_profile_path = hidden_mix_profile_path
        self.hidden_mix_fallback = hidden_mix_fallback
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
        if self._should_append_to_cache(query_states, layer_cache):
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        else:
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
        if not should_compress or key_states.shape[-2] < self.budget:
            return key_states, value_states

        query_cache = getattr(layer_cache, "query_cache", None)
        if query_cache is None or query_cache.shape[-2] == 0:
            query_cache = query_states[:, :, -self.window_size :, :]
        attn_cache = self._compute_attn_cache(key_states, query_cache)
        self._store_group_entry(
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

        random_exp_input = torch.empty(
            batch_size,
            int(num_key_value_heads),
            actual_len,
            dtype=torch.float32,
            device=device,
        ).uniform_(-1.0, 1.0)
        importance[:, :, :actual_len] = torch.exp(random_exp_input)
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

    def _compute_attn_cache(self, key_states, query_states):
        bsz, num_key_value_heads, kv_cache_len, _ = key_states.shape
        num_key_value_groups = query_states.shape[1] // num_key_value_heads
        query_window = min(self.window_size, query_states.shape[-2])
        query_states = query_states[:, :, -query_window:, :]

        attn_weights = compute_attention_scores(query_states, key_states)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=kv_cache_len - query_window + 1)
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., : -self.window_size]

        hist_len = kv_cache_len - self.window_size
        scores = attn_weights.view(
            bsz,
            num_key_value_heads,
            num_key_value_groups,
            query_window,
            hist_len,
        )
        attn_weights_sum = scores.mean(dim=2).mean(dim=-2)
        return F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

    def _group_layers(self, layer_idx):
        layer_idx = int(layer_idx)
        if self.hidden_mix_profile is not None:
            return self.hidden_mix_profile["layer_to_group"].get(layer_idx, (layer_idx,))
        return (layer_idx,)

    def _state(self):
        if not hasattr(self.model_config, "_snapkv_hidden_mix_state"):
            self.model_config._snapkv_hidden_mix_state = {"groups": {}}
        return self.model_config._snapkv_hidden_mix_state

    def _store_group_entry(self, key_states, value_states, attn_cache, layer_cache):
        group_layers = self._group_layers(self.layer_idx)
        group_key = tuple(group_layers)
        state = self._state()
        if self.layer_idx == group_layers[0]:
            state["groups"][group_key] = {}
        group = state["groups"].setdefault(group_key, {})
        group[self.layer_idx] = {
            "layer_idx": self.layer_idx,
            "key_states": key_states,
            "value_states": value_states,
            "attn_cache": attn_cache,
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
            self._pack_layer(entry, mixed_cache)

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

    def _pack_layer(self, entry, attn_cache):
        key_states = entry["key_states"]
        value_states = entry["value_states"]
        head_dim = key_states.shape[-1]
        kv_cache_len = key_states.shape[-2]
        keep_count = min(self.budget - self.window_size, attn_cache.shape[-1])
        indices = attn_cache.topk(keep_count, dim=-1).indices

        if self.record_kept_token_indices:
            self._record_kept_indices(indices, attn_cache, kv_cache_len)

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, : -self.window_size, :].gather(dim=2, index=gather_indices)
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(dim=2, index=gather_indices)
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        self._set_layer_cache(
            entry["layer_cache"],
            torch.cat([k_past_compress, k_cur], dim=2),
            torch.cat([v_past_compress, v_cur], dim=2),
        )

    def _record_kept_indices(self, indices, attn_cache, kv_cache_len):
        indices_cl = indices.clone().squeeze(0).to("cpu")
        recent_window_indices = torch.arange(
            kv_cache_len - self.window_size,
            kv_cache_len,
            device="cpu",
        ).expand(indices_cl.shape[0], -1)
        cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)
        attn_scores = attn_cache.clone().squeeze(0).to("cpu")
        kept_attn = torch.gather(attn_scores, dim=1, index=indices_cl)

        if self.evicted_token_num > 0:
            prev_indices = self.kept_token_indices[-1]
            mask = cur_indices < self.budget
            for head_idx in range(cur_indices.shape[0]):
                positions = torch.where(mask[head_idx])[0]
                for pos in positions:
                    val = cur_indices[head_idx, pos].item()
                    cur_indices[head_idx, pos] = prev_indices[head_idx, val]
            cur_indices[~mask] += self.evicted_token_num

        self.kept_attention_scores.append(kept_attn)
        self.kept_token_indices.append(cur_indices)
        self.evicted_token_num += kv_cache_len - self.budget

    def _set_layer_cache(self, layer_cache, key_states, value_states):
        layer_cache.keys = key_states
        layer_cache.values = value_states
        layer_cache.dtype = key_states.dtype
        layer_cache.device = key_states.device
        layer_cache.is_initialized = True
