import torch

from . import compute_attention_scores
import torch.nn.functional as F
import math
from .vw_norm import vw_l1norm

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


class DefensiveKV:
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

    ## Following CriticalKV, we select tokens jointly using value norms and attention scores in a two-stage procedure.
    def vwl1norm(self, values, module, scores, window_bias, ave_attn_weights):
        bsz, num_key_value_heads, q_len, _ = values.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        Wo = module.o_proj.weight.transpose(0, 1)

        # Wo = Wo.view(module.config.num_attention_heads, module.config.head_dim, module.config.hidden_size)
        Wo = Wo.view(module.config.num_attention_heads, module.head_dim, module.config.hidden_size)

        V = repeat_kv(values, num_key_value_groups)

        # We use head-wise computation instead of direct matmul to reduce the memory usage of WoV.
        # Future kernel fusion optimization could eliminate this intermediate variables to enhance performance.
        head_WoV_norm_list = []
        for head in range(V.size(1)):
            head_WoV = V[:, head, :, ...].matmul(Wo[head, ...].unsqueeze(0))
            head_WoV_norm = torch.norm(head_WoV, p=1, dim=-1)
            head_WoV_norm_list.append(head_WoV_norm)

        # b_size, num_heads, q_len , k_len
        WoV_norm = torch.stack(head_WoV_norm_list, dim=1)
        WoV_norm = WoV_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, q_len).mean(dim=2)

        projected_norm_normalization = WoV_norm / WoV_norm.sum(dim=-2, keepdim=True)
        scores = scores * projected_norm_normalization

        ## Specifically, we first select tokens until the cumulative attention mass reaches a 90% threshold (CriticalKV stage 1), and then jointly select additional tokens based on value norms (stage 2).
        threshold = 0.9 - window_bias
        normalized_scores = ave_attn_weights / ave_attn_weights.sum(dim=-1, keepdim=True)
        batch_size, num_heads = normalized_scores.shape[:2]
        score_mask = torch.zeros_like(normalized_scores, dtype=torch.bool)
        # calculate cumsum
        sorted_scores, sorted_indices = torch.sort(normalized_scores, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=-1)
        mask = cumsum >= threshold
        count = torch.argmax(mask.to(torch.int32), dim=-1)
        count.clamp_(max=int(self.budget - self.window_size))
        for b in range(batch_size):
            for h in range(num_heads):
                k = count[b, h].item()
                indices = sorted_indices[b, h, :k]
                score_mask[b, h, indices] = True
        scores = torch.where(score_mask, scores.max().item(), scores)

        return scores

   ## Following CriticalKV, we select tokens jointly using value norms and attention scores in a two-stage procedure. (Triton version)
    def vwl1norm_triton(self, values, module, scores, window_bias, ave_attn_weights):
        bsz, num_key_value_heads, q_len, _ = values.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        Wo = module.o_proj.weight.transpose(0, 1)

        # Wo = Wo.view(module.config.num_attention_heads, module.config.head_dim, module.config.hidden_size)
        Wo = Wo.view(module.config.num_attention_heads, module.head_dim, module.config.hidden_size)

        V = repeat_kv(values, num_key_value_groups)
        # Kernel fusion optimization
        WoV_norm = vw_l1norm(V, Wo)
        WoV_norm = WoV_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, q_len).mean(dim=2)

        projected_norm_normalization = WoV_norm / WoV_norm.sum(dim=-2, keepdim=True)
        scores = scores * projected_norm_normalization

        ## Specifically, we first select tokens until the cumulative attention mass reaches a 90% threshold (CriticalKV stage 1), and then jointly select additional tokens based on value norms (stage 2).
        threshold = 0.9 - window_bias
        normalized_scores = ave_attn_weights / ave_attn_weights.sum(dim=-1, keepdim=True)
        batch_size, num_heads = normalized_scores.shape[:2]
        score_mask = torch.zeros_like(normalized_scores, dtype=torch.bool)
        # calculate cumsum
        sorted_scores, sorted_indices = torch.sort(normalized_scores, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=-1)
        mask = cumsum >= threshold
        count = torch.argmax(mask.to(torch.int32), dim=-1)
        count.clamp_(max=int(self.budget - self.window_size))
        for b in range(batch_size):
            for h in range(num_heads):
                k = count[b, h].item()
                indices = sorted_indices[b, h, :k]
                score_mask[b, h, indices] = True
        scores = torch.where(score_mask, scores.max().item(), scores)

        return scores

    def score(self, key_states, query_states, recent_window):
        batch_size, q_heads, q_len, head_dim = query_states.shape
        kv_heads = key_states.shape[1]
        query_group_size = q_heads // kv_heads

        key_states = repeat_kv(key_states, query_group_size)

        if query_group_size == 1:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
        else:
            # shape: [batch_size, kv_heads, query_group_size, q_len, head_dim]
            # query_states = query_states.view(
            #     batch_size, kv_heads, query_group_size, q_len, head_dim
            # )

            # shape: [batch_size, kv_heads, 1, kv_len, head_dim]
            # key_states = key_states.unsqueeze(2)

            # shape: [batch_size, kv_heads, query_group_size, q_len, kv_len]
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
            
        history_len = attn_weights.shape[-1] - recent_window
        query_window = min(self.window_size, attn_weights.shape[-2])
        attn_weights = torch.softmax(
            attn_weights[..., -query_window:, :history_len],
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        window_bias = attn_weights[..., -recent_window:].sum(dim=-1).mean().item()

        return attn_weights, window_bias

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
    ):
        if self.attention is None:
            raise ValueError("DefensiveKV requires bind_attention(attention) before update_kv().")

        bsz, num_key_value_heads, q_len, head_dim = key_states.shape
        num_key_value_groups = self.attention.config.num_attention_heads // num_key_value_heads

        head_dim = key_states.shape[-1]
        kv_cache_len = key_states.shape[-2]
        if kv_cache_len <= self.budget:
            return key_states, value_states

        recent_window = min(self.window_size, kv_cache_len)
        history_len = kv_cache_len - recent_window
        history_budget = self.budget - recent_window

        if history_budget <= 0:
            key_states = key_states[:, :, history_len:, :]
            value_states = value_states[:, :, history_len:, :]
            if self.record_kept_token_indices:
                empty_scores = key_states.new_empty(key_states.shape[0], key_states.shape[1], 0)
                empty_indices = torch.empty(
                    key_states.shape[0],
                    key_states.shape[1],
                    0,
                    dtype=torch.long,
                    device=key_states.device,
                )
                self._record_kept_tokens(empty_indices, empty_scores, kv_cache_len, recent_window)
            return key_states, value_states

        attn_weights, window_bias = self.score(
            key_states, query_states, self.window_size
        )

        # Average per grioup (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = attn_weights.view(
            bsz, num_key_value_heads, num_key_value_groups, self.window_size, q_len - self.window_size
        )
        ave_attn_weights = scores.mean(dim=2).mean(dim=-2)
        scores_base = scores.sum(dim=-1, keepdim=True)
        scores = F.avg_pool1d(
            scores.view(bsz * num_key_value_heads * num_key_value_groups, self.window_size, q_len - self.window_size),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).view(bsz, num_key_value_heads, num_key_value_groups, self.window_size, q_len - self.window_size)
        scores = (scores / scores.sum(dim=-1, keepdim=True)) * scores_base

        ## Defensive Mechanism
        # the first max is for num_key_value_groups, the second max is for window_size
        max_scores = scores.max(dim=2).values.max(dim=-2).values
        scores = max_scores.clamp(min=max_scores.mean(dim=-1, keepdim=True))
        ## Defensive Mechanism End

        ## Borrowed from CriticalKV
        scores = self.vwl1norm(value_states[..., : -self.window_size, :],self.attention, scores, window_bias, ave_attn_weights)

        indices = torch.topk(scores, history_budget, sorted=True, dim=-1).indices

        if self.record_kept_token_indices:
            self._record_kept_tokens(indices, scores, kv_cache_len, recent_window)

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, :history_len, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :history_len, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, history_len:, :]
        v_cur = value_states[:, :, history_len:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states

    def _record_kept_tokens(self, indices, scores, kv_cache_len, recent_window):
        indices_cl = indices.clone().squeeze(0).to("cpu")
        score_cl = scores.clone().squeeze(0).to("cpu")
        recent_window_indices = torch.arange(
            kv_cache_len - recent_window, kv_cache_len, device="cpu"
        ).expand(indices_cl.shape[0], -1)
        cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)

        recent_scores = score_cl.new_full(
            (score_cl.shape[0], recent_window),
            torch.finfo(score_cl.dtype).max,
        )
        score_cl = torch.cat([score_cl, recent_scores], dim=-1)
        kept_scores = torch.gather(score_cl, dim=1, index=cur_indices)

        if self.evicted_token_num > 0:
            prev_indices = self.kept_token_indices[-1]
            mask = cur_indices < prev_indices.shape[-1]

            for i in range(cur_indices.shape[0]):
                positions = torch.where(mask[i])[0]
                for pos in positions:
                    val = cur_indices[i, pos].item()
                    cur_indices[i, pos] = prev_indices[i, val]

            cur_indices[~mask] += self.evicted_token_num

        self.kept_attention_scores.append(kept_scores)
        self.kept_token_indices.append(cur_indices)
        self.evicted_token_num += kv_cache_len - cur_indices.shape[-1]
