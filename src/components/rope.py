import torch
import torch.nn as nn
from typing import Tuple


def get_rotation_matrix(dim: int, context_size: int, period: float) -> torch.Tensor:
    # TODO: compute a tensor of frequencies
    freqs = 1 / period ** (torch.arange(0, dim, 2) / dim)
    # TODO: compute a tensor of token indexes
    token_indexes = torch.arange(0, context_size)
    # TODO: compute the matrix thetas
    thetas = torch.outer(token_indexes, freqs)
    # TODO: create the rotation matrix
    rotation_matrix = torch.polar(torch.ones_like(thetas), thetas)
    return rotation_matrix


class RoPE(nn.Module):
    def __init__(self, rotation_matrix):
        super().__init__()
        self.rotation_matrix = rotation_matrix

    def forward(self, queries, keys):
        batch_size, num_heads, seq_length, head_dim = queries.size()

        # TODO: reshape to [batch_size, num_heads, seq_length, head_dim // 2 , 2]
        queries = queries.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)
        keys = keys.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)

        # TODO: transform into a complex tensor
        queries_complex = torch.view_as_complex(queries)
        keys_complex = torch.view_as_complex(keys)

        # TODO: rotate the queries and keys
        queries_rotated = queries_complex * self.rotation_matrix[:seq_length, :]
        keys_rotated = keys_complex * self.rotation_matrix[:seq_length, :]

        # TODO: conver to read and reshape back to [batch_size, num_heads, seq_length, head_dim]
        new_queries = torch.view_as_real(queries_rotated).reshape(
            batch_size, num_heads, seq_length, head_dim
        )
        new_keys = torch.view_as_real(keys_rotated).reshape(
            batch_size, num_heads, seq_length, head_dim
        )

        return new_queries, new_keys


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
