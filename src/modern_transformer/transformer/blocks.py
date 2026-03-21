import torch.nn as nn

from ..components.norm_layers import RMSNorm
from ..components.attentions import EfficientSlidingWindowMultiheadAttention
from ..components.moe import MoeLayer


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        window_size,
        d_ff,
        num_experts,
        n_experts_per_token,
        rotation_matrix,
    ) -> None:
        super().__init__()

        self.attention = EfficientSlidingWindowMultiheadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            rotation_matrix=rotation_matrix,
        )
        self.norm1 = RMSNorm(hidden_size)
        self.moe = MoeLayer(
            hidden_size=hidden_size,
            d_ff=d_ff,
            num_experts=num_experts,
            n_experts_per_token=n_experts_per_token,
        )
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x
