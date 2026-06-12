import torch
import torch.nn.functional as F

from . import compute_attention_scores
from .snapkv import SnapKV


class SnapKVHiddenMix:
    requires_layer_coordination = False
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
        importance_epsilon=1e-12,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode
        self.importance_epsilon = float(importance_epsilon)
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []

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
        self._store_layer_entry(
            key_states=key_states,
            value_states=value_states,
            attn_cache=attn_cache,
            layer_cache=layer_cache,
        )
        return key_states, value_states

    def finalize_after_attention(self, attention, hidden_states, attn_output, layer_cache):
        entry = getattr(layer_cache, "snapkv_hidden_wo_mix_entry", None)
        if entry is None:
            return

        try:
            importance = self._compute_cosine_importance(
                attention=attention,
                hidden_states=hidden_states,
                attn_output=attn_output,
                num_key_value_heads=entry["attn_cache"].shape[1],
                hist_len=entry["attn_cache"].shape[-1],
            )
            scores = entry["attn_cache"]
            importance = importance.to(device=scores.device, dtype=scores.dtype)
            if importance.shape != scores.shape:
                raise ValueError("snapkv_hidden_wo_mix cosine importance must match attn_cache shape.")
            self._pack_layer(entry, scores)
        finally:
            entry.clear()
            layer_cache.snapkv_hidden_wo_mix_entry = None

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

    def _store_layer_entry(self, key_states, value_states, attn_cache, layer_cache):
        layer_cache.snapkv_hidden_wo_mix_entry = {
            "layer_idx": self.layer_idx,
            "key_states": key_states,
            "value_states": value_states,
            "attn_cache": attn_cache,
            "layer_cache": layer_cache,
        }

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
