import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPE


class EfficientSlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size, rotation_matrix):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

        # position embedding attribute with RoPE
        self.pos_emb = RoPE(rotation_matrix=rotation_matrix)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # Compute Q, K, V
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(
            0, 2, 1, 3
        )  # Reorder to (batch_size, num_heads, seq_length, 3 * head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)
        print(keys.size())

        # Rotate with RoPE
        queries, keys = self.pos_emb(queries, keys)

        # Pad sequence for windowed attention (batch_size, num_heads, seq_length + 2 * padding, head_dim)
        keys_padded = F.pad(keys, (0, 0, padding, padding), "constant", 0)
        values_padded = F.pad(values, (0, 0, padding, padding), "constant", 0)
        print(keys_padded.size())

        # Windows (batch_size, num_heads, seq_length, head_dim, window_size)
        keys_windows = keys_padded.unfold(-2, self.window_size, 1)
        values_windows = values_padded.unfold(-2, self.window_size, 1)
        print(keys_windows.size())

        # Attention (batch_size, num_heads, seq_length, window_size)
        scores = torch.einsum("bnsh,bnshw->bnsw", queries, keys_windows)
        scores = scores / (self.head_dim**0.5)
        attention = F.softmax(scores, dim=-1)

        # Context (batch_size, seq_length, num_heads, head_dim)
        context = torch.einsum("bnsw,bnshw->bsnh", attention, values_windows)

        # Reshape context to (batch_size, seq_length, num_heads * head_dim)
        context = context.reshape(batch_size, seq_length, self.hidden_size)

        # Final linear layer
        output = self.out(context)
        return output


class SlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # Compute Q, K, V
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(
            0, 2, 1, 3
        )  # Reorder to (batch_size, num_heads, seq_length, 3 * head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Pad sequence for windowed attention
        keys = F.pad(keys, (0, 0, padding, padding), "constant", 0)
        values = F.pad(values, (0, 0, padding, padding), "constant", 0)

        # Initialize context tensors
        context = torch.zeros_like(queries, device=x.device)

        # Compute attention for each sliding window
        for i in range(seq_length):
            # Determine the start and end of the window
            start = i
            end = i + self.window_size

            # Compute scores
            scores = torch.matmul(
                queries[:, :, i : i + 1, :], keys[:, :, start:end, :].transpose(-2, -1)
            )
            scores = scores / (self.head_dim**0.5)
            attention = F.softmax(scores, dim=-1)

            # Apply attention to values and add to context
            context[:, :, i : i + 1, :] += torch.matmul(
                attention, values[:, :, start:end, :]
            )

        # Reshape context to (batch_size, seq_length, num_heads * head_dim)
        context = context.permute(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.hidden_size
        )

        # Final linear layer
        output = self.out(context)
        return output
