import json
import math
import os

import torch

from .snapkv_ada import SnapKV as SnapKVAda


class SnapKV(SnapKVAda):
    manages_kv_cache = True

    def __init__(self, *args, attn_head_cluster_path="/home/yangx/new_compression/attn_head_clusters_qwen2_5_7b.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_head_cluster_path = self._resolve_attn_head_cluster_path(attn_head_cluster_path)
        self.attn_head_cluster_profile = self._load_attn_head_cluster_profile(self.attn_head_cluster_path)

    def _resolve_attn_head_cluster_path(self, profile_path):
        if profile_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            profile_path = os.path.join(repo_root, "attn_head_clusters.json")
        return os.path.abspath(os.path.expanduser(str(profile_path)))

    def _load_attn_head_cluster_profile(self, profile_path):
        if self.model_config is not None:
            cached = getattr(self.model_config, "_snapkv_ada_head_cluster_profile", None)
            if cached is not None and cached.get("path") == profile_path:
                return cached["profile"]

        with open(profile_path, "r", encoding="utf-8") as handle:
            raw_profile = json.load(handle)
        profile = self._parse_attn_head_cluster_profile(raw_profile, profile_path)

        if self.model_config is not None:
            self.model_config._snapkv_ada_head_cluster_profile = {
                "path": profile_path,
                "profile": profile,
            }
        return profile

    def _parse_attn_head_cluster_profile(self, raw_profile, profile_path):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Attention head cluster profile must be a JSON object: {profile_path}")
        if raw_profile.get("cluster_scope") != "per_layer":
            raise ValueError("Attention head cluster profile requires cluster_scope='per_layer'.")
        if raw_profile.get("unit") != "gqa_group":
            raise ValueError("Attention head cluster profile requires unit='gqa_group'.")

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
                raw_head_to_cluster.get(str(layer_idx)),
            )

        if not clusters_by_layer:
            raise ValueError("Attention head cluster profile must contain at least one layer.")
        return {
            "gqa_group_count": raw_profile.get("gqa_group_count"),
            "clusters_by_layer": clusters_by_layer,
        }

    def _clusters_by_layer_from_layers(self, raw_layers):
        if not isinstance(raw_layers, list):
            raise ValueError("Attention head cluster profile requires clusters_by_layer or layers.")
        clusters_by_layer = {}
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, dict) or "layer_idx" not in raw_layer:
                raise ValueError("Each attention head cluster layer entry requires layer_idx.")
            clusters_by_layer[str(int(raw_layer["layer_idx"]))] = raw_layer.get("clusters")
        return clusters_by_layer

    def _parse_layer_clusters(self, layer_idx, raw_clusters, raw_head_to_cluster=None):
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

            similarities = self._parse_pair_similarities(
                layer_idx,
                cluster_id,
                heads,
                raw_cluster.get("pair_similarities", []),
            )
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "heads": heads,
                    "similarities": similarities,
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

    def _parse_pair_similarities(self, layer_idx, cluster_id, heads, raw_pair_similarities):
        if raw_pair_similarities is None:
            raw_pair_similarities = []
        if not isinstance(raw_pair_similarities, list):
            raise ValueError(f"Layer {layer_idx} cluster {cluster_id} pair_similarities must be a list.")

        head_set = set(heads)
        similarities = {}
        for raw_pair in raw_pair_similarities:
            if not isinstance(raw_pair, dict):
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} pair similarity entries must be objects.")
            head_i = int(raw_pair["head_i"])
            head_j = int(raw_pair["head_j"])
            if head_i == head_j or head_i not in head_set or head_j not in head_set:
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} has invalid similarity pair.")
            similarity = float(raw_pair["similarity"])
            if not math.isfinite(similarity) or similarity < 0.0:
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} similarity must be finite and non-negative.")
            key = tuple(sorted((head_i, head_j)))
            if key in similarities:
                raise ValueError(f"Layer {layer_idx} cluster {cluster_id} has duplicate similarity pair {key}.")
            similarities[key] = similarity

        for offset, head_i in enumerate(heads):
            for head_j in heads[offset + 1 :]:
                key = tuple(sorted((head_i, head_j)))
                if key not in similarities:
                    raise ValueError(
                        f"Layer {layer_idx} cluster {cluster_id} is missing similarity pair {key}."
                    )
        return similarities

    def _select_layer_head_topk(self, key_states, scores, valid_mask):
        batch_size, num_heads = key_states.shape[:2]
        hist_len = key_states.shape[-2] - self.window_size
        if hist_len < 1:
            return torch.zeros(batch_size, num_heads, 0, dtype=torch.bool, device=key_states.device), 0
        if scores.shape[-1] != hist_len:
            raise ValueError("snapkv_ada_head_cluster attn_cache length must match historical cache length.")

        clusters = self._layer_clusters(num_heads)
        mixed_scores = self._mix_attn_cache_by_cluster(scores, clusters)
        hist_valid = valid_mask[:, :, :hist_len].to(device=scores.device, dtype=torch.bool)
        selected = torch.zeros(batch_size, num_heads, hist_len, dtype=torch.bool, device=scores.device)

        hist_budget_per_head = self.budget - self.window_size
        for cluster in clusters:
            heads = cluster["heads"]
            head_index = torch.tensor(heads, dtype=torch.long, device=scores.device)
            cluster_scores = mixed_scores.index_select(dim=1, index=head_index)
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

    def _layer_clusters(self, num_heads):
        if self.layer_idx is None:
            raise ValueError("snapkv_ada_head_cluster requires layer_idx.")
        layer_idx = int(self.layer_idx)
        clusters = self.attn_head_cluster_profile["clusters_by_layer"].get(layer_idx)
        if clusters is None:
            raise ValueError(f"snapkv_ada_head_cluster has no cluster profile for layer {layer_idx}.")

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

    def _mix_attn_cache_by_cluster(self, attn_cache, clusters=None):
        if clusters is None:
            clusters = self._layer_clusters(attn_cache.shape[1])

        mixed_cache = attn_cache.clone()
        for cluster in clusters:
            heads = cluster["heads"]
            if len(heads) == 1:
                continue

            weights = attn_cache.new_zeros((len(heads), len(heads)), dtype=torch.float32)
            for target_offset, target_head in enumerate(heads):
                for source_offset, source_head in enumerate(heads):
                    if target_head == source_head:
                        weights[target_offset, source_offset] = 1.0
                    else:
                        key = tuple(sorted((target_head, source_head)))
                        weights[target_offset, source_offset] = float(cluster["similarities"][key])
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            weights = weights.to(device=attn_cache.device, dtype=attn_cache.dtype)

            head_index = torch.tensor(heads, dtype=torch.long, device=attn_cache.device)
            cluster_cache = attn_cache.index_select(dim=1, index=head_index)
            mixed_cluster = torch.einsum("ts,bsl->btl", weights, cluster_cache)
            for offset, head_idx in enumerate(heads):
                mixed_cache[:, head_idx, :] = mixed_cluster[:, offset, :]
        return mixed_cache
