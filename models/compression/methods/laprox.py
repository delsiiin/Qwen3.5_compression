import torch
import torch.nn.functional as F


class LaProx:
    requires_layer_coordination = True

    def __init__(
        self,
        budget=128,
        window_size=8,
        record_kept_token_indices=False,
        layer_idx=None,
        model_config=None,
        model_type=None,
        mode=None,
        importance_epsilon=1e-12,
        max_projected_elements=16 * 1024 * 1024,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = int(budget)
        self.window_size = int(window_size)
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode
        self.importance_epsilon = float(importance_epsilon)
        self.max_projected_elements = int(max_projected_elements)

        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.kept_token_indices = []

        if model_config is not None and not hasattr(model_config, "_laprox_state"):
            model_config._laprox_state = {"entries": {}}

    def update_kv(self, key_states, query_states, value_states):
        kv_cache_len = key_states.shape[-2]
        if kv_cache_len <= self.budget:
            return key_states, value_states
        return key_states[:, :, -self.budget :, :], value_states[:, :, -self.budget :, :]

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
        if getattr(self.model_config, "update_kv", None) is not True:
            return past_key_values.update(key_states, value_states, self.layer_idx)

        had_existing_cache = self._has_existing_cache(layer_cache)
        if had_existing_cache:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )
            self._append_valid_mask(layer_cache, key_states)
        else:
            self._clear_padding_metadata(layer_cache)

        self._set_attention_mask_for_current_step(layer_cache, key_states, query_states)

        should_compress = (
            self.model_config.compression is None
            or query_states.shape[-2] > 1
            or self.model_config.compression is True
        )
        if not should_compress:
            if not had_existing_cache:
                self._set_layer_cache(layer_cache, key_states, value_states)
            return key_states, value_states

        if not had_existing_cache and self._max_valid_token_count(layer_cache, key_states) <= self.budget:
            self._set_layer_cache(layer_cache, key_states, value_states)
            return key_states, value_states

        cached_queries = getattr(layer_cache, "query_cache", None)
        if cached_queries is None:
            cached_queries = query_states
        self._store_model_entry(
            attention=attention,
            key_states=key_states,
            query_states=cached_queries,
            value_states=value_states,
            layer_cache=layer_cache,
        )
        return key_states, value_states

    def _store_model_entry(self, attention, key_states, query_states, value_states, layer_cache):
        model_layers = self._model_layers()
        state = self._state()
        if int(self.layer_idx) == model_layers[0]:
            state["entries"] = {}

        state["entries"][int(self.layer_idx)] = {
            "layer_idx": int(self.layer_idx),
            "attention": attention,
            "key_states": key_states,
            "query_states": query_states,
            "value_states": value_states,
            "valid_mask": self._current_valid_mask(layer_cache, key_states),
            "layer_cache": layer_cache,
        }

        if all(layer_idx in state["entries"] for layer_idx in model_layers):
            entries = [state["entries"][layer_idx] for layer_idx in model_layers]
            try:
                self._compress_model(entries)
            finally:
                for entry in entries:
                    entry.clear()
                state["entries"] = {}

    def _compress_model(self, entries):
        score_tensors = []
        valid_tensors = []
        hist_lengths = []
        for entry in entries:
            key_states = entry["key_states"]
            value_states = entry["value_states"]
            valid_mask = entry["valid_mask"]
            kv_cache_len = key_states.shape[-2]
            recent_window = min(self.window_size, kv_cache_len)
            hist_len = kv_cache_len - recent_window
            if hist_len < 1:
                return

            scores = self._laprox_scores(
                entry["attention"],
                key_states,
                entry["query_states"],
                value_states,
                valid_mask,
                hist_len,
                recent_window,
            )
            hist_valid = valid_mask[:, :, :hist_len]
            scores = torch.where(hist_valid, scores, torch.zeros_like(scores))
            scores = self._normalize_layer_scores(scores, hist_valid)
            score_tensors.append(scores)
            valid_tensors.append(hist_valid)
            hist_lengths.append(hist_len)

        batch_size = entries[0]["key_states"].shape[0]
        total_budget = self._global_historical_budget(entries)
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
            selected_flat = torch.zeros_like(flat_valid, dtype=torch.bool)
        else:
            flat_scores = flat_scores.masked_fill(~flat_valid, torch.finfo(flat_scores.dtype).min)
            selected_flat = torch.zeros_like(flat_valid, dtype=torch.bool)
            topk_indices = flat_scores.topk(topk, dim=-1).indices
            selected_flat.scatter_(dim=-1, index=topk_indices, value=True)

        offset = 0
        for entry, hist_len in zip(entries, hist_lengths):
            batch_size, num_kv_heads = entry["key_states"].shape[:2]
            width = num_kv_heads * hist_len
            selected = selected_flat[:, offset : offset + width].view(batch_size, num_kv_heads, hist_len)
            selected = selected.to(device=entry["key_states"].device)
            self._pack_layer(entry, selected, hist_len)
            offset += width

    def _laprox_scores(
        self,
        attention,
        key_states,
        query_states,
        value_states,
        valid_mask,
        hist_len,
        recent_window,
    ):
        query_states = query_states[:, :, -recent_window:, :]
        key_states_for_attention = self._repeat_kv(key_states, query_states.shape[1] // key_states.shape[1])
        attn_scores = torch.matmul(query_states, key_states_for_attention.transpose(2, 3))
        attn_scores = attn_scores / (query_states.shape[-1] ** 0.5)

        attention_mask = torch.ones_like(attn_scores) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=key_states.shape[-2] - query_states.shape[-2] + 1)
        attn_scores = attn_scores + attention_mask

        expanded_valid = self._expand_valid_mask(valid_mask, query_states.shape[1])
        attn_scores = attn_scores.masked_fill(
            ~expanded_valid[:, :, None, :],
            torch.finfo(attn_scores.dtype).min,
        )
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        hist_probs = attn_probs[..., :hist_len]

        batch_size, num_kv_heads = key_states.shape[:2]
        num_key_value_groups = query_states.shape[1] // num_kv_heads
        attn_l2 = hist_probs.reshape(
            batch_size,
            num_kv_heads,
            num_key_value_groups,
            query_states.shape[-2],
            hist_len,
        ).float()
        attn_l2 = torch.linalg.vector_norm(attn_l2, ord=2, dim=-2)

        value_output_l2 = self._value_output_l2_norm(
            attention,
            value_states[:, :, :hist_len, :],
            num_key_value_groups,
        ).to(device=attn_l2.device, dtype=attn_l2.dtype)
        return ((attn_l2 + self.importance_epsilon) * value_output_l2).sum(dim=2)

    def _value_output_l2_norm(self, attention, value_states, num_key_value_groups):
        o_proj = getattr(attention, "o_proj", None)
        if o_proj is None or not hasattr(o_proj, "weight"):
            raise ValueError("LaProx requires attention.o_proj.weight to compute projected value scores.")

        batch_size, num_kv_heads, seq_len, head_dim = value_states.shape
        weight = o_proj.weight.transpose(0, 1)
        if weight.shape[0] % head_dim != 0:
            raise ValueError("attention.o_proj input dimension must be divisible by value head_dim.")

        num_attention_heads = weight.shape[0] // head_dim
        if num_attention_heads != num_kv_heads * num_key_value_groups:
            raise ValueError("attention.o_proj head layout does not match key/value head layout.")

        hidden_size = weight.shape[-1]
        weight = weight.to(device=value_states.device)
        weight = weight.view(num_attention_heads, head_dim, hidden_size)
        repeated_values = self._repeat_kv(value_states, num_key_value_groups)

        head_norms = []
        for head_idx in range(repeated_values.size(1)):
            projected = repeated_values[:, head_idx, :, :].matmul(weight[head_idx, :, :].unsqueeze(0))
            head_norms.append(torch.linalg.vector_norm(projected.float(), ord=2, dim=-1))

        value_output_l2 = torch.stack(head_norms, dim=1)
        return value_output_l2.view(batch_size, num_kv_heads, num_key_value_groups, seq_len)

    def _normalize_layer_scores(self, scores, valid_mask):
        rank_scores = scores.to(dtype=torch.float32)
        valid_mask = valid_mask.to(device=rank_scores.device, dtype=torch.bool)
        valid_scores = torch.where(valid_mask, rank_scores, torch.zeros_like(rank_scores))
        denom = valid_scores.sum(dim=(1, 2), keepdim=True).clamp_min(self.importance_epsilon)
        normalized = rank_scores / denom
        return normalized.masked_fill(~valid_mask, torch.finfo(normalized.dtype).min)

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

        layer_cache = entry["layer_cache"]
        self._set_layer_cache(layer_cache, packed_keys, packed_values)
        layer_cache.kv_valid_mask = packed_mask
        layer_cache.kv_lengths = lengths
        if self.record_kept_token_indices:
            self.kept_token_indices.append(keep_indices)

    def _global_historical_budget(self, entries):
        if not entries:
            return 0
        num_kv_heads = entries[0]["key_states"].shape[1]
        return len(entries) * num_kv_heads * (self.budget - self.window_size)

    def _model_layers(self):
        num_layers = getattr(self.model_config, "num_hidden_layers", None)
        if num_layers is None:
            return (int(self.layer_idx),)
        return tuple(range(int(num_layers)))

    def _state(self):
        if not hasattr(self.model_config, "_laprox_state"):
            self.model_config._laprox_state = {"entries": {}}
        return self.model_config._laprox_state

    def _has_existing_cache(self, layer_cache):
        keys = getattr(layer_cache, "keys", None)
        return torch.is_tensor(keys) and keys.numel() > 0

    def _set_layer_cache(self, layer_cache, key_states, value_states):
        self._clear_flatten_metadata(layer_cache)
        layer_cache.keys = key_states
        layer_cache.values = value_states
        layer_cache.dtype = key_states.dtype
        layer_cache.device = key_states.device
        layer_cache.is_initialized = True

    def _max_valid_token_count(self, layer_cache, key_states):
        valid_mask = getattr(layer_cache, "kv_valid_mask", None)
        if valid_mask is None or valid_mask.shape[-1] != key_states.shape[-2]:
            return key_states.shape[-2]
        return int(valid_mask.sum(dim=-1).max().item())

    def _current_valid_mask(self, layer_cache, key_states):
        valid_mask = getattr(layer_cache, "kv_valid_mask", None)
        if valid_mask is None or valid_mask.shape[-1] != key_states.shape[-2]:
            return torch.ones(key_states.shape[:3], dtype=torch.bool, device=key_states.device)
        return valid_mask.to(device=key_states.device)

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

    def _set_attention_mask_for_current_step(self, layer_cache, key_states, query_states):
        if query_states.shape[-2] == 1:
            layer_cache.attention_kv_valid_mask = self._current_valid_mask(layer_cache, key_states)
        elif hasattr(layer_cache, "attention_kv_valid_mask"):
            delattr(layer_cache, "attention_kv_valid_mask")

    def _clear_padding_metadata(self, layer_cache):
        for attr in ("kv_valid_mask", "kv_lengths", "attention_kv_valid_mask"):
            if hasattr(layer_cache, attr):
                delattr(layer_cache, attr)

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

    def _expand_valid_mask(self, valid_mask, num_attention_heads):
        num_key_value_groups = num_attention_heads // valid_mask.shape[1]
        if num_key_value_groups == 1:
            return valid_mask
        valid_mask = valid_mask[:, :, None, :].expand(
            valid_mask.shape[0],
            valid_mask.shape[1],
            num_key_value_groups,
            valid_mask.shape[-1],
        )
        return valid_mask.reshape(valid_mask.shape[0], num_attention_heads, valid_mask.shape[-1])

    @staticmethod
    def _repeat_kv(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch,
            num_key_value_heads,
            n_rep,
            seq_len,
            head_dim,
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)
