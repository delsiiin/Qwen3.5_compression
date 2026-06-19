import json
import math
import os

import torch
import torch.nn.functional as F

from ..utils import compute_attention_scores


class AttentionHeadClusterMixin:
    attn_head_cluster_cache_attr = "_snapkv_ada_head_cluster_profile"
    attn_head_cluster_method_name = "snapkv_ada_head_cluster"

    def _resolve_attn_head_cluster_path(self, profile_path):
        if profile_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            profile_path = os.path.join(repo_root, "attn_head_clusters.json")
        return os.path.abspath(os.path.expanduser(str(profile_path)))

    def _load_attn_head_cluster_profile(self, profile_path):
        cache_attr = self.attn_head_cluster_cache_attr
        if self.model_config is not None:
            cached = getattr(self.model_config, cache_attr, None)
            if cached is not None and cached.get("path") == profile_path:
                return cached["profile"]

        with open(profile_path, "r", encoding="utf-8") as handle:
            raw_profile = json.load(handle)
        profile = self._parse_attn_head_cluster_profile(raw_profile, profile_path)

        if self.model_config is not None:
            setattr(
                self.model_config,
                cache_attr,
                {
                    "path": profile_path,
                    "profile": profile,
                },
            )
        return profile

    def _parse_attn_head_cluster_profile(self, raw_profile, profile_path):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Attention head cluster profile must be a JSON object: {profile_path}")
        if raw_profile.get("cluster_scope") != "per_layer":
            raise ValueError("Attention head cluster profile requires cluster_scope='per_layer'.")
        if raw_profile.get("unit") != "gqa_group":
            raise ValueError("Attention head cluster profile requires unit='gqa_group'.")
        if raw_profile.get("head_mean_similarity_unit") != "raw_head_to_gqa_group_mean_before_grouping":
            raise ValueError(
                "Attention head cluster profile requires "
                "head_mean_similarity_unit='raw_head_to_gqa_group_mean_before_grouping'."
            )
        if raw_profile.get("head_mean_similarity_metric") != "cosine_similarity":
            raise ValueError("Attention head cluster profile requires head_mean_similarity_metric='cosine_similarity'.")
        if (
            raw_profile.get("head_mean_similarity_reduction")
            != "raw_head_to_mean_gqa_group_attention_distribution_per_layer"
        ):
            raise ValueError(
                "Attention head cluster profile requires "
                "head_mean_similarity_reduction='raw_head_to_mean_gqa_group_attention_distribution_per_layer'."
            )
        gqa_group_count = self._parse_positive_int(raw_profile.get("gqa_group_count"), "gqa_group_count")
        gqa_group_size = self._parse_positive_int(raw_profile.get("gqa_group_size"), "gqa_group_size")
        attention_head_count = raw_profile.get("attention_head_count")
        if attention_head_count is not None and int(attention_head_count) != gqa_group_count * gqa_group_size:
            raise ValueError("Attention head cluster profile attention_head_count must equal gqa_group_count * gqa_group_size.")

        raw_clusters_by_layer = raw_profile.get("clusters_by_layer")
        if not isinstance(raw_clusters_by_layer, dict):
            raw_clusters_by_layer = self._clusters_by_layer_from_layers(raw_profile.get("layers"))
        raw_head_to_cluster = raw_profile.get("head_to_cluster_by_layer", {})
        if raw_head_to_cluster is None:
            raw_head_to_cluster = {}
        if not isinstance(raw_head_to_cluster, dict):
            raise ValueError("Attention head cluster profile head_to_cluster_by_layer must be an object.")

        clusters_by_layer = {}
        for layer_text, raw_clusters in raw_clusters_by_layer.items():
            layer_idx = int(layer_text)
            clusters_by_layer[layer_idx] = self._parse_layer_clusters(
                layer_idx,
                raw_clusters,
                gqa_group_size,
                raw_head_to_cluster.get(str(layer_idx)),
            )

        if not clusters_by_layer:
            raise ValueError("Attention head cluster profile must contain at least one layer.")
        return {
            "gqa_group_count": gqa_group_count,
            "gqa_group_size": gqa_group_size,
            "clusters_by_layer": clusters_by_layer,
        }

    def _parse_positive_int(self, value, field_name):
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"Attention head cluster profile requires integer {field_name}.") from None
        if value < 1:
            raise ValueError(f"Attention head cluster profile {field_name} must be at least 1.")
        return value

    def _clusters_by_layer_from_layers(self, raw_layers):
        if not isinstance(raw_layers, list):
            raise ValueError("Attention head cluster profile requires clusters_by_layer or layers.")
        clusters_by_layer = {}
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, dict) or "layer_idx" not in raw_layer:
                raise ValueError("Each attention head cluster layer entry requires layer_idx.")
            clusters_by_layer[str(int(raw_layer["layer_idx"]))] = raw_layer.get("clusters")
        return clusters_by_layer

    def _parse_layer_clusters(self, layer_idx, raw_clusters, gqa_group_size, raw_head_to_cluster=None):
        if not isinstance(raw_clusters, list) or not raw_clusters:
            raise ValueError(f"Layer {layer_idx} requires a non-empty cluster list.")

        clusters = []
        seen_heads = set()
        for raw_cluster in raw_clusters:
            if not isinstance(raw_cluster, dict):
                raise ValueError(f"Layer {layer_idx} cluster entries must be objects.")
            cluster_id = int(raw_cluster.get("cluster_id", len(clusters)))
            raw_heads = raw_cluster.get("heads")
            if not isinstance(raw_heads, list) or not raw_heads:
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} requires non-empty heads.")
            heads = tuple(int(head) for head in raw_heads)
            if len(set(heads)) != len(heads):
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} contains duplicate heads.")
            duplicate_heads = seen_heads.intersection(heads)
            if duplicate_heads:
                raise ValueError(f"Layer {layer_idx} head appears in multiple clusters: {sorted(duplicate_heads)}.")
            seen_heads.update(heads)

            raw_head_weights = self._parse_head_mean_similarities(
                layer_idx,
                cluster_id,
                heads,
                raw_cluster.get("raw_heads_by_gqa_group"),
                raw_cluster.get("head_mean_similarities"),
                gqa_group_size,
            )
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "heads": heads,
                    "raw_head_weights": raw_head_weights,
                }
            )

        if raw_head_to_cluster is not None:
            if not isinstance(raw_head_to_cluster, list):
                raise ValueError(f"Layer {layer_idx} head_to_cluster must be a list.")
            expected_heads = set(range(len(raw_head_to_cluster)))
            if seen_heads != expected_heads:
                raise ValueError(
                    f"Layer {layer_idx} cluster heads must match head_to_cluster range "
                    f"0..{len(raw_head_to_cluster) - 1}."
                )
            for head_idx, cluster_id in enumerate(raw_head_to_cluster):
                matching = [cluster for cluster in clusters if head_idx in cluster["heads"]]
                if len(matching) != 1 or matching[0]["cluster_id"] != int(cluster_id):
                    raise ValueError(f"Layer {layer_idx} head_to_cluster disagrees at head {head_idx}.")

        return tuple(clusters)

    def _parse_head_mean_similarities(
        self,
        layer_idx,
        cluster_id,
        heads,
        raw_heads_by_gqa_group,
        raw_head_mean_similarities,
        gqa_group_size,
    ):
        if raw_heads_by_gqa_group is None or raw_head_mean_similarities is None:
            raise ValueError(
                f"Layer {layer_idx} cluster {cluster_id} requires head_mean_similarities "
                "and raw_heads_by_gqa_group; pair_similarities-only profiles are unsupported."
            )
        if not isinstance(raw_heads_by_gqa_group, dict):
            raise ValueError(f"Layer {layer_idx} cluster {cluster_id} raw_heads_by_gqa_group must be an object.")
        if not isinstance(raw_head_mean_similarities, list):
            raise ValueError(f"Layer {layer_idx} cluster {cluster_id} head_mean_similarities must be a list.")

        expected_raw_heads = tuple(range(int(gqa_group_size)))
        head_set = set(heads)
        raw_heads_by_group = {}
        for gqa_group_text, raw_heads in raw_heads_by_gqa_group.items():
            gqa_group_id = int(gqa_group_text)
            if gqa_group_id not in head_set:
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} has raw heads for non-cluster GQA group.")
            if not isinstance(raw_heads, list):
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} raw head entries must be lists.")
            raw_head_tuple = tuple(int(head_id) for head_id in raw_heads)
            if raw_head_tuple != expected_raw_heads:
                raise ValueError(
                    f"Layer {layer_idx} cluster {cluster_id} raw heads for GQA group {gqa_group_id} "
                    f"must be 0..{int(gqa_group_size) - 1}."
                )
            raw_heads_by_group[gqa_group_id] = raw_head_tuple
        if set(raw_heads_by_group) != head_set:
            raise ValueError(f"Layer {layer_idx} cluster {cluster_id} raw_heads_by_gqa_group must match heads.")

        similarities = {head: [None] * int(gqa_group_size) for head in heads}
        for raw_entry in raw_head_mean_similarities:
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} head_mean_similarities entries must be objects.")
            gqa_group_id = int(raw_entry["gqa_group_id"])
            raw_head_id = int(raw_entry["head_id"])
            if gqa_group_id not in head_set or raw_head_id < 0 or raw_head_id >= int(gqa_group_size):
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} has invalid head_mean_similarities entry.")
            similarity = float(raw_entry["similarity"])
            if not math.isfinite(similarity) or similarity < 0.0:
                raise ValueError(
                    f"Layer {layer_idx} cluster {cluster_id} head_mean similarity must be finite and non-negative."
                )
            if similarities[gqa_group_id][raw_head_id] is not None:
                raise ValueError(
                    f"Layer {layer_idx} cluster {cluster_id} duplicates GQA group {gqa_group_id} raw head {raw_head_id}."
                )
            similarities[gqa_group_id][raw_head_id] = similarity

        raw_head_weights = {}
        for gqa_group_id, group_similarities in similarities.items():
            if any(similarity is None for similarity in group_similarities):
                raise ValueError(
                    f"Layer {layer_idx} cluster {cluster_id} is missing head_mean_similarities "
                    f"for GQA group {gqa_group_id}."
                )
            weight_sum = float(sum(group_similarities))
            if weight_sum <= 0.0:
                raise ValueError(
                    f"Layer {layer_idx} cluster {cluster_id} head_mean_similarities "
                    f"for GQA group {gqa_group_id} must sum to a positive value."
                )
            raw_head_weights[gqa_group_id] = tuple(float(similarity) / weight_sum for similarity in group_similarities)
        return raw_head_weights

    def _layer_clusters(self, num_heads):
        if self.layer_idx is None:
            raise ValueError(f"{self.attn_head_cluster_method_name} requires layer_idx.")
        layer_idx = int(self.layer_idx)
        clusters = self.attn_head_cluster_profile["clusters_by_layer"].get(layer_idx)
        if clusters is None:
            raise ValueError(f"{self.attn_head_cluster_method_name} has no cluster profile for layer {layer_idx}.")

        seen_heads = []
        for cluster in clusters:
            seen_heads.extend(cluster["heads"])
        expected_heads = list(range(int(num_heads)))
        if sorted(seen_heads) != expected_heads or len(set(seen_heads)) != len(seen_heads):
            raise ValueError(
                f"Layer {layer_idx} cluster heads must match runtime key/value heads 0..{int(num_heads) - 1}."
            )

        profiled_head_count = self.attn_head_cluster_profile.get("gqa_group_count")
        if profiled_head_count is not None and int(profiled_head_count) != int(num_heads):
            raise ValueError(
                f"Layer {layer_idx} profile gqa_group_count={int(profiled_head_count)} "
                f"does not match runtime key/value heads={int(num_heads)}."
            )
        return clusters

    def _raw_head_weight_tensor(self, num_heads, clusters, device, dtype):
        gqa_group_size = int(self.attn_head_cluster_profile["gqa_group_size"])
        weights = torch.zeros((int(num_heads), gqa_group_size), dtype=torch.float32, device=device)
        for cluster in clusters:
            for head_idx, raw_head_weights in cluster["raw_head_weights"].items():
                weights[int(head_idx)] = torch.tensor(raw_head_weights, dtype=torch.float32, device=device)
        return weights.to(dtype=dtype)

    def _compute_head_cluster_attn_cache(self, key_states, query_states, valid_mask=None):
        bsz, num_key_value_heads, kv_cache_len, _ = key_states.shape
        hist_len = kv_cache_len - self.window_size
        if hist_len < 1:
            return key_states.new_zeros(bsz, num_key_value_heads, 0)

        clusters = self._layer_clusters(num_key_value_heads)
        gqa_group_size = int(self.attn_head_cluster_profile["gqa_group_size"])
        expected_query_heads = num_key_value_heads * gqa_group_size
        if query_states.shape[1] != expected_query_heads:
            raise ValueError(
                f"{self.attn_head_cluster_method_name} expected {expected_query_heads} query heads "
                f"from gqa_group_count={num_key_value_heads} and gqa_group_size={gqa_group_size}, "
                f"got {query_states.shape[1]}."
            )

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
                gqa_group_size,
                kv_cache_len,
            )
            full_valid = full_valid.reshape(bsz, query_states.shape[1], kv_cache_len)
            attn_weights = attn_weights.masked_fill(
                ~full_valid[:, :, None, :],
                torch.finfo(attn_weights.dtype).min,
            )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :hist_len]

        raw_head_scores = attn_weights.view(
            bsz,
            num_key_value_heads,
            gqa_group_size,
            query_window,
            hist_len,
        ).mean(dim=-2)
        raw_head_weights = self._raw_head_weight_tensor(
            num_key_value_heads,
            clusters,
            device=attn_weights.device,
            dtype=raw_head_scores.dtype,
        )
        attn_weights_sum = (raw_head_scores * raw_head_weights[None, :, :, None]).sum(dim=2)
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
