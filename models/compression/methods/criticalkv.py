import torch

from . import compute_attention_scores


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class CriticalKV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        epsilon=1e-4,
        first_stage_ratio=0.5,
        record_kept_token_indices=False,
        layer_idx=None,
        model_config=None,
        model_type=None,
        mode=None,
        **kwargs,
    ):
        assert budget > 0, "budget must be positive"
        if not 0 <= first_stage_ratio <= 1:
            raise ValueError("first_stage_ratio must be in [0, 1].")
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.first_stage_ratio = first_stage_ratio

        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode
        self.attention = None

        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []

    def bind_attention(self, attention):
        self.attention = attention

    @staticmethod
    def vwl1norm(values, module):
        bsz, num_key_value_heads, q_len, _ = values.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        Wo = module.o_proj.weight.transpose(0, 1)
        Wo = Wo.view(module.config.num_attention_heads, module.head_dim, module.config.hidden_size)

        V = repeat_kv(values, num_key_value_groups)

        head_WoV_norm_list = []
        for head in range(V.size(1)):
            head_WoV = V[:, head, :, ...].matmul(Wo[head, ...].unsqueeze(0))
            head_WoV_norm = torch.norm(head_WoV, p=1, dim=-1)
            head_WoV_norm_list.append(head_WoV_norm)

        WoV_norm = torch.stack(head_WoV_norm_list, dim=1)
        WoV_norm = WoV_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, q_len).mean(dim=2)
        return WoV_norm

    def score(self, key_states, query_states):
        attn_weights = compute_attention_scores(query_states, key_states)
        query_window = min(self.window_size, attn_weights.shape[-2])
        scores = torch.softmax(
            attn_weights[:, :, -query_window:, :],
            dim=-1,
            dtype=torch.float32,
        )
        return scores.mean(dim=-2).to(query_states.dtype)

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
    ):
        if self.attention is None:
            raise ValueError("CriticalKV requires bind_attention(attention) before update_kv().")

        head_dim = key_states.shape[-1]
        kv_cache_len = key_states.shape[-2]
        if kv_cache_len <= self.budget:
            return key_states, value_states

        scores = self.score(key_states, query_states)
        selection_budget = int(self.budget * self.first_stage_ratio)
        if selection_budget > 0:
            top_k_index = torch.topk(scores, selection_budget, sorted=True, dim=-1).indices

        projected_norm = self.vwl1norm(value_states, self.attention).to(device=scores.device, dtype=scores.dtype)
        scores = (scores + self.epsilon) * projected_norm

        if selection_budget > 0:
            scores.scatter_(-1, top_k_index, torch.finfo(scores.dtype).max)

        indices = torch.topk(scores, self.budget, sorted=True, dim=-1).indices

        if self.record_kept_token_indices:
            self._record_kept_tokens(indices, scores, kv_cache_len)

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        key_states = key_states.gather(dim=2, index=indices)
        value_states = value_states.gather(dim=2, index=indices)
        return key_states, value_states

    def _record_kept_tokens(self, indices, scores, kv_cache_len):
        indices_cl = indices.clone().squeeze(0).to("cpu")
        score_cl = scores.clone().squeeze(0).to("cpu")
        kept_scores = torch.gather(score_cl, dim=1, index=indices_cl)

        cur_indices = indices_cl
        if self.evicted_token_num > 0:
            prev_indices = self.kept_token_indices[-1]
            mask = cur_indices < self.budget

            for i in range(cur_indices.shape[0]):
                positions = torch.where(mask[i])[0]
                for pos in positions:
                    val = cur_indices[i, pos].item()
                    cur_indices[i, pos] = prev_indices[i, val]

            cur_indices[~mask] += self.evicted_token_num

        self.kept_attention_scores.append(kept_scores)
        self.kept_token_indices.append(cur_indices)
        self.evicted_token_num += kv_cache_len - self.budget
