import torch
import torch.nn.functional as F

from .tridentkv_head_cluster import TridentKVHeadCluster


class TridentKV(TridentKVHeadCluster):
    """TridentKV spatial-temporal scoring with head-cluster budget sharing."""

    requires_layer_coordination = True

    def __init__(
        self,
        *args,
        prefill_layer_budget="dissimilarity",
        prefill_layer_budget_reduction="nearest",
        prefill_layer_budget_layers=None,
        prefill_layer_budget_min_history=1,
        prefill_layer_budget_temperature=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if prefill_layer_budget not in (None, False, "none", "dissimilarity"):
            raise ValueError(
                "prefill_layer_budget must be one of None, False, 'none', or 'dissimilarity'."
            )
        if prefill_layer_budget_reduction not in ("nearest", "farthest", "mean"):
            raise ValueError("prefill_layer_budget_reduction must be 'nearest', 'farthest', or 'mean'.")
        if int(prefill_layer_budget_min_history) < 0:
            raise ValueError("prefill_layer_budget_min_history must be non-negative.")
        if float(prefill_layer_budget_temperature) <= 0.0:
            raise ValueError("prefill_layer_budget_temperature must be positive.")

        self.prefill_layer_budget = prefill_layer_budget
        self.prefill_layer_budget_reduction = prefill_layer_budget_reduction
        self.prefill_layer_budget_layers = (
            None
            if prefill_layer_budget_layers is None
            else tuple(int(layer) for layer in prefill_layer_budget_layers)
        )
        if self.prefill_layer_budget_layers is not None and not self.prefill_layer_budget_layers:
            raise ValueError("prefill_layer_budget_layers must not be empty.")
        self.prefill_layer_budget_min_history = int(prefill_layer_budget_min_history)
        self.prefill_layer_budget_temperature = float(prefill_layer_budget_temperature)
        if model_config := self.model_config:
            if not hasattr(model_config, "_tridentkv_layer_budget_state"):
                model_config._tridentkv_layer_budget_state = {"layers": {}}

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
        if not self._should_coordinate_prefill_layer_budget(query_states):
            return super().update_kv_cache(
                attention,
                hidden_states,
                position_embeddings,
                key_states,
                query_states,
                value_states,
                past_key_values,
                layer_cache,
            )

        if self._is_flatten_cache(layer_cache):
            return super().update_kv_cache(
                attention,
                hidden_states,
                position_embeddings,
                key_states,
                query_states,
                value_states,
                past_key_values,
                layer_cache,
            )

        layer_indices = self._prefill_layer_budget_layers()
        if int(self.layer_idx) not in layer_indices:
            return super().update_kv_cache(
                attention,
                hidden_states,
                position_embeddings,
                key_states,
                query_states,
                value_states,
                past_key_values,
                layer_cache,
            )
        if self.layer_idx == layer_indices[0]:
            self._layer_budget_state()["layers"] = {}

        if self._should_append_to_cache(query_states, layer_cache):
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            self._append_valid_mask(layer_cache, key_states)
        else:
            self._clear_padding_metadata(layer_cache)
            if self.model_config.update_kv is not True:
                self._set_layer_cache(layer_cache, key_states, value_states)
                return key_states, value_states

        should_compress = (
            self.model_config.compression is None
            or query_states.shape[-2] > 1
        )
        self._set_attention_mask_for_current_step(layer_cache, key_states, query_states)
        valid_mask = self._current_valid_mask(layer_cache, key_states)
        if not should_compress or self._valid_token_count(layer_cache, key_states) < self.budget:
            self._set_layer_cache(layer_cache, key_states, value_states)
            self._store_prefill_layer_budget_entry(
                layer_idx=self.layer_idx,
                entry={"layer_idx": self.layer_idx, "skip": True},
            )
            self._try_compress_prefill_layer_budget_group(layer_indices)
            return key_states, value_states

        query_cache = getattr(layer_cache, "query_cache", None)
        if query_cache is None or query_cache.shape[-2] == 0:
            query_cache = query_states[:, :, -self.window_size :, :]
        observation_raw_attention = None
        if self._observes_head_budget_attention():
            observation_raw_attention = self._compute_head_cluster_observation_attention(
                key_states,
                query_cache,
                valid_mask,
            )
        attn_cache = self._compute_attn_cache(key_states, query_cache, valid_mask)
        hist_len = key_states.shape[-2] - self.window_size
        result = self._tridentkv_head_cluster_result
        layer_vector = self._layer_attention_distribution(attn_cache, valid_mask, hist_len)
        entry = {
            "layer_idx": self.layer_idx,
            "skip": False,
            "attention": attention,
            "key_states": key_states,
            "value_states": value_states,
            "scores": attn_cache,
            "valid_mask": valid_mask,
            "hist_len": hist_len,
            "layer_cache": layer_cache,
            "tridentkv_head_cluster_result": result,
            "layer_vector": layer_vector,
        }
        if self._head_cluster_observation_enabled:
            entry["observation_raw_attention"] = observation_raw_attention
        self._store_prefill_layer_budget_entry(
            layer_idx=self.layer_idx,
            entry=entry,
        )
        self._try_compress_prefill_layer_budget_group(layer_indices)
        return key_states, value_states

    def _compute_attn_cache(self, key_states, query_states, valid_mask=None):
        raw_head_attention = self._tridentkv_head_clusterer._build_raw_head_attention(
            key_states,
            query_states,
            valid_mask,
        )
        result = self._tridentkv_head_clusterer.build_from_raw_head_attention(raw_head_attention)
        self._tridentkv_head_cluster_result = result
        if (
            self._head_cluster_observation_enabled
            and self._head_cluster_observation_submode == "head_cluster_pca"
        ):
            self._capture_head_cluster_pca_observation(raw_head_attention, result)

        (
            num_key_value_heads,
            num_key_value_groups,
            query_window,
            hist_len,
        ) = raw_head_attention.shape
        if self.kernel_size % 2 == 0:
            raise ValueError("tridentkv requires odd kernel_size.")

        attn_weights_sum = raw_head_attention.new_empty(1, num_key_value_heads, hist_len)
        for cluster in result.clusters:
            heads = cluster["heads"]
            head_index = torch.tensor(heads, dtype=torch.long, device=raw_head_attention.device)
            cluster_head_count = len(heads)
            cluster_attention = raw_head_attention.index_select(dim=0, index=head_index)
            cluster_scores = cluster_attention.permute(3, 1, 0, 2).reshape(
                hist_len,
                1,
                cluster_head_count * num_key_value_groups,
                query_window,
            )
            if cluster_head_count * num_key_value_groups % 2 ==0:
                head_kernel = cluster_head_count * num_key_value_groups - 1
            else:
                head_kernel = cluster_head_count * num_key_value_groups
            cluster_scores = F.max_pool2d(
                cluster_scores,
                kernel_size=(head_kernel, 4),
                stride=1,
                padding=(head_kernel // 2, 2),
            )
            pooled_head_count = cluster_head_count
            pooled_query_window = cluster_scores.shape[-1]
            cluster_scores = cluster_scores.reshape(
                hist_len,
                num_key_value_groups,
                pooled_head_count,
                pooled_query_window,
            )
            cluster_head_scores = cluster_scores.mean(dim=(1, 3)).transpose(0, 1)
            attn_weights_sum[:, head_index, :] = cluster_head_scores.unsqueeze(0)

        return self._tridentkv_head_clusterer._pool_attn_cache(
            attn_weights_sum,
            key_states,
            self.kernel_size,
            valid_mask,
        )

    def _should_coordinate_prefill_layer_budget(self, query_states):
        if self.prefill_layer_budget in (None, False, "none"):
            return False
        if self.model_config is None or self.model_config.update_kv is not True:
            return False
        if self.layer_idx is None:
            return False
        return query_states.shape[-2] > 1

    def _prefill_layer_budget_layers(self):
        configured_layers = getattr(self.model_config, "prefill_layer_budget_layers", None)
        if self.prefill_layer_budget_layers is not None:
            return self.prefill_layer_budget_layers
        if configured_layers is not None:
            layers = tuple(int(layer) for layer in configured_layers)
            if not layers:
                raise ValueError("prefill_layer_budget_layers must not be empty.")
            return layers
        num_layers = getattr(self.model_config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(self.model_config, "num_layers", None)
        if num_layers is None:
            return (int(self.layer_idx),)
        return tuple(range(int(num_layers)))

    def _layer_budget_state(self):
        if not hasattr(self.model_config, "_tridentkv_layer_budget_state"):
            self.model_config._tridentkv_layer_budget_state = {"layers": {}}
        return self.model_config._tridentkv_layer_budget_state

    def _store_prefill_layer_budget_entry(self, layer_idx, entry):
        self._layer_budget_state()["layers"][int(layer_idx)] = entry

    def _try_compress_prefill_layer_budget_group(self, layer_indices):
        state = self._layer_budget_state()
        layers = state.setdefault("layers", {})
        if not all(int(layer_idx) in layers for layer_idx in layer_indices):
            return

        entries = [layers[int(layer_idx)] for layer_idx in layer_indices]
        try:
            compressible = [entry for entry in entries if not entry.get("skip", False)]
            budgets = self._allocate_layer_historical_budgets(compressible)
            state["last_budgets"] = dict(budgets)
            for entry in compressible:
                hist_budget = budgets.get(int(entry["layer_idx"]), self.budget - self.window_size)
                selected_hist_mask, hist_len = self._select_layer_head_topk_with_budget(
                    entry["key_states"],
                    entry["scores"],
                    entry["valid_mask"],
                    entry["tridentkv_head_cluster_result"],
                    hist_budget,
                )
                self._pack_layer(
                    attention=entry["attention"],
                    key_states=entry["key_states"],
                    value_states=entry["value_states"],
                    selected_hist_mask=selected_hist_mask,
                    hist_len=hist_len,
                    scores=entry["scores"],
                    layer_cache=entry["layer_cache"],
                    observation_raw_attention=entry.get("observation_raw_attention"),
                )
        finally:
            state["layers"] = {}

    def _layer_attention_distribution(self, scores, valid_mask, hist_len):
        hist_scores = scores[..., :hist_len].detach().to(dtype=torch.float32)
        if valid_mask is not None:
            hist_valid = valid_mask[:, :, :hist_len].to(device=hist_scores.device, dtype=torch.bool)
            hist_scores = hist_scores.masked_fill(~hist_valid, 0.0)
        vector = hist_scores.reshape(-1).clamp_min(0.0)
        total = vector.sum()
        if total <= torch.finfo(vector.dtype).eps:
            return torch.full_like(vector, 1.0 / max(vector.numel(), 1))
        return vector / total

    def _allocate_layer_historical_budgets(self, entries):
        if not entries:
            return {}
        if len(entries) == 1:
            return {int(entries[0]["layer_idx"]): self.budget - self.window_size}

        vectors = self._align_layer_vectors([entry["layer_vector"] for entry in entries])
        normalized = F.normalize(vectors, p=2, dim=-1, eps=1e-12)
        similarity = (normalized @ normalized.transpose(0, 1)).clamp(min=-1.0, max=1.0)
        eye = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        off_diag = similarity.masked_fill(eye, 0.0)
        if self.prefill_layer_budget_reduction == "nearest":
            peer_similarity = similarity.masked_fill(eye, -1.0).max(dim=-1).values
            layer_scores = (1.0 - peer_similarity).clamp_min(0.0)
        elif self.prefill_layer_budget_reduction == "farthest":
            peer_similarity = similarity.masked_fill(eye, 1.0).min(dim=-1).values
            layer_scores = (1.0 - peer_similarity).clamp_min(0.0)
        else:
            peer_count = max(similarity.shape[0] - 1, 1)
            mean_similarity = off_diag.sum(dim=-1) / peer_count
            layer_scores = (1.0 - mean_similarity).clamp_min(0.0)

        if self.prefill_layer_budget_temperature != 1.0:
            exponent = 1.0 / self.prefill_layer_budget_temperature
            layer_scores = layer_scores.clamp_min(0.0).pow(exponent)
        return self._integer_layer_budgets(entries, layer_scores)

    def _align_layer_vectors(self, vectors):
        sizes = {int(vector.numel()) for vector in vectors}
        if len(sizes) == 1:
            return torch.stack([vector.to(dtype=torch.float32) for vector in vectors], dim=0)

        min_size = min(sizes)
        return torch.stack(
            [vector[:min_size].to(dtype=torch.float32) for vector in vectors],
            dim=0,
        )

    def _integer_layer_budgets(self, entries, layer_scores):
        layer_count = len(entries)
        per_layer_history = self.budget - self.window_size
        total_budget = layer_count * per_layer_history
        min_history = min(int(self.prefill_layer_budget_min_history), per_layer_history)
        caps = [max(0, int(entry["hist_len"])) for entry in entries]
        budgets = [min(min_history, cap) for cap in caps]
        assigned = sum(budgets)
        remaining = min(total_budget, sum(caps)) - assigned
        if remaining <= 0:
            return {int(entry["layer_idx"]): budget for entry, budget in zip(entries, budgets)}

        weights = layer_scores.detach().to(dtype=torch.float32, device="cpu").clamp_min(0.0)
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)
        raw_extras = remaining * weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        remainders = []
        for offset, raw_extra in enumerate(raw_extras.tolist()):
            capacity = caps[offset] - budgets[offset]
            extra = min(int(raw_extra), max(capacity, 0))
            budgets[offset] += extra
            assigned += extra
            if capacity > extra:
                remainders.append((raw_extra - int(raw_extra), -offset, offset))

        leftover = min(total_budget, sum(caps)) - assigned
        while leftover > 0:
            candidates = [
                (fraction, neg_offset, offset)
                for fraction, neg_offset, offset in remainders
                if budgets[offset] < caps[offset]
            ]
            if not candidates:
                break
            for _fraction, _neg_offset, offset in sorted(candidates, reverse=True)[:leftover]:
                budgets[offset] += 1
                leftover -= 1
                if leftover <= 0:
                    break
        return {int(entry["layer_idx"]): budget for entry, budget in zip(entries, budgets)}

    def _select_layer_head_topk_with_budget(
        self,
        key_states,
        scores,
        valid_mask,
        tridentkv_head_cluster_result,
        hist_budget_per_head,
    ):
        batch_size, num_heads = key_states.shape[:2]
        hist_len = key_states.shape[-2] - self.window_size
        if hist_len < 1:
            return torch.zeros(batch_size, num_heads, 0, dtype=torch.bool, device=key_states.device), 0
        if scores.shape[-1] != hist_len:
            raise ValueError("tridentkv scores length must match history.")
        if tridentkv_head_cluster_result is None:
            raise RuntimeError("tridentkv requires head clusters.")
        if tridentkv_head_cluster_result.hist_len != hist_len:
            raise ValueError("tridentkv head-cluster history length mismatch.")

        hist_valid = valid_mask[:, :, :hist_len].to(device=scores.device, dtype=torch.bool)
        selected = torch.zeros(batch_size, num_heads, hist_len, dtype=torch.bool, device=scores.device)
        hist_budget_per_head = max(0, int(hist_budget_per_head))
        for cluster in tridentkv_head_cluster_result.clusters:
            heads = cluster["heads"]
            head_index = torch.tensor(heads, dtype=torch.long, device=scores.device)
            cluster_scores = scores.index_select(dim=1, index=head_index)
            cluster_valid = hist_valid.index_select(dim=1, index=head_index)
            flat_valid = cluster_valid.reshape(batch_size, -1)
            valid_count = flat_valid.sum(dim=-1)
            total_budget = len(heads) * hist_budget_per_head
            topk = min(int(total_budget), int(valid_count.min().item()))
            if topk <= 0:
                continue

            flat_scores = cluster_scores.reshape(batch_size, -1).masked_fill(
                ~flat_valid,
                torch.finfo(cluster_scores.dtype).min,
            )
            topk_indices = flat_scores.topk(topk, dim=-1).indices
            selected_cluster_flat = torch.zeros_like(flat_valid, dtype=torch.bool)
            selected_cluster_flat.scatter_(dim=-1, index=topk_indices, value=True)
            selected_cluster = selected_cluster_flat.view(batch_size, len(heads), hist_len)
            for offset, head_idx in enumerate(heads):
                selected[:, head_idx, :] = selected_cluster[:, offset, :]

        return selected.to(device=key_states.device), hist_len
