import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import SiGLU


class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, d_ff)
        self.W2 = nn.Linear(hidden_size, d_ff)
        self.W3 = nn.Linear(d_ff, hidden_size)
        self.SiGLU = SiGLU(d_ff, d_ff)

    def forward(self, x) -> torch.Tensor:
        return self.W3(self.SiGLU(self.W1(x) * self.W2(x)))


class MoeLayer(nn.Module):
    def __init__(self, hidden_size, d_ff, num_experts, n_experts_per_token):
        super().__init__()

        if n_experts_per_token < 1:
            raise ValueError("n_experts_per_token must be >= 1")
        if n_experts_per_token > num_experts:
            raise ValueError("n_experts_per_token must be <= num_experts")

        self.num_experts = num_experts
        self.n_experts_per_token = n_experts_per_token

        self.experts = nn.ModuleList(
            [FeedForward(hidden_size, d_ff) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # x: (batch_size, seq_length, hidden_size)
        # (batch_size, seq_length, num_experts)
        gated = self.gate(x)
        topk = torch.topk(gated, self.n_experts_per_token)
        # (batch_size, seq_length, n_experts_per_token)
        weights = F.softmax(topk.values, dim=-1)

        # (batch_size, seq_length, hidden_size)
        out = torch.zeros_like(x, device=x.device)
        for i, expert in enumerate(self.experts):
            # find the indexes of the hidden states that should be routed to the current expert
            mask = topk.indices == i
            if mask.sum() == 0:
                continue
            # Get concrete indices where expert i was selected
            batch_idx, token_idx, expert_idx = torch.where(mask)
            # (num_selected, hidden_size)
            expert_out = expert(x[batch_idx, token_idx])
            # add the expert output to the final output, weighted by the gate values
            out[batch_idx, token_idx] += expert_out * weights[
                batch_idx, token_idx, expert_idx
            ].unsqueeze(-1)

        return out


if __name__ == "__main__":
    bs = 2
    sl = 3
    hs = 4
    d_ff = 5
    num_experts = 4
    n_experts_per_token = 2
    l = MoeLayer(hs, d_ff, num_experts, n_experts_per_token)
    x = torch.rand(bs, sl, hs)
    l(x)
