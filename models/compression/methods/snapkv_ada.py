import torch
import torch.nn as nn
import torch.nn.functional as F

from . import compute_attention_scores


class SnapKV:
    manages_kv_cache = True

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
        flatten_cache="auto",
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        if flatten_cache not in ("auto", True, False):
            raise ValueError("flatten_cache must be one of 'auto', True, or False.")
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.flatten_cache = flatten_cache

        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode

        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
    ):
        head_dim = query_states.shape[-1]
        bsz, num_key_value_heads, kv_cache_len, _ = key_states.shape
        num_key_value_groups = query_states.shape[1] // num_key_value_heads

        if kv_cache_len < self.budget:
            return key_states, value_states

        attn_weights = compute_attention_scores(query_states, key_states)

        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=key_states.shape[-2] - self.window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-self.window_size]

        scores = attn_weights.view(
            bsz, num_key_value_heads, num_key_value_groups, self.window_size, kv_cache_len - self.window_size
        )
        attn_weights_sum = scores.mean(dim=2).mean(dim=-2)

        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        indices = attn_cache.topk(self.budget - self.window_size, dim=-1).indices

        if self.record_kept_token_indices:
            indices_cl = indices.clone().squeeze(0).to("cpu")

            attn_weights_sum_analysis = (
                nn.functional.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            )

            attn_cache_analysis = F.max_pool1d(
                attn_weights_sum_analysis,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )

            recent_window_indices = torch.arange(
                kv_cache_len - self.window_size, kv_cache_len, device="cpu"
            ).expand(indices_cl.shape[0], -1)
            cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)

            attn_scores = attn_cache_analysis.clone().squeeze(0).to("cpu")
            kept_attn = torch.gather(attn_scores, dim=1, index=cur_indices)

            if self.evicted_token_num > 0:
                prev_indices = self.kept_token_indices[-1]
                mask = cur_indices < self.budget

                for i in range(cur_indices.shape[0]):
                    positions = torch.where(mask[i])[0]
                    for pos in positions:
                        val = cur_indices[i, pos].item()
                        cur_indices[i, pos] = prev_indices[i, val]

                cur_indices[~mask] += self.evicted_token_num

            self.kept_attention_scores.append(kept_attn)
            self.kept_token_indices.append(cur_indices)
            self.evicted_token_num += kv_cache_len - self.budget

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states

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
                raise ValueError("snapkv flatten cache only supports q_len=1 after prefill.")
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
        selected_hist_mask, hist_len = self._select_layer_head_topk(
            key_states,
            attn_cache,
            self._current_valid_mask(layer_cache, key_states),
        )
        self._pack_layer(
            attention=attention,
            key_states=key_states,
            value_states=value_states,
            selected_hist_mask=selected_hist_mask,
            hist_len=hist_len,
            scores=attn_cache,
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

    def _select_layer_head_topk(self, key_states, scores, valid_mask):
        batch_size, num_heads = key_states.shape[:2]
        hist_len = key_states.shape[-2] - self.window_size
        if hist_len < 1:
            return torch.zeros(batch_size, num_heads, 0, dtype=torch.bool, device=key_states.device), 0
        if scores.shape[-1] != hist_len:
            raise ValueError("snapkv attn_cache length must match historical cache length.")

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

    def _pack_layer(
        self,
        attention,
        key_states,
        value_states,
        selected_hist_mask,
        hist_len,
        scores,
        layer_cache,
    ):
        valid_mask = self._current_valid_mask(layer_cache, key_states)
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

        if self._should_use_flatten_cache(attention, key_states, value_states, lengths):
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
        kept_attn = torch.zeros(
            batch_size,
            num_heads,
            max(int(selected_hist_mask.sum(dim=-1).max().item()), 1),
            device="cpu",
        )
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
            raise ValueError("snapkv flatten cache only supports batch_size=1.")

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
            raise ValueError("snapkv flatten cache decode update requires shape [1, heads, 1, dim].")
        if key_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("snapkv flatten cache requires float16 or bfloat16 key/value tensors.")
        if key_states.device.type != "cuda" or value_states.device.type != "cuda":
            raise ValueError("snapkv flatten cache requires CUDA key/value tensors.")

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
