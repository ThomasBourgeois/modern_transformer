import torch
import pytest

from modern_transformer.components.attentions import (
    SlidingWindowMultiheadAttention,
    EfficientSlidingWindowMultiheadAttention,
)


def _identity_rotation_matrix(seq_len: int, head_dim: int) -> torch.Tensor:
    return torch.ones((seq_len, head_dim // 2), dtype=torch.complex64)


def test_sliding_attention_shape_and_finite() -> None:
    torch.manual_seed(0)
    batch, seq_len, hidden_size, num_heads, window = 2, 7, 12, 3, 5
    m = SlidingWindowMultiheadAttention(hidden_size, num_heads, window)
    x = torch.randn(batch, seq_len, hidden_size)
    out = m(x)
    assert out.shape == (batch, seq_len, hidden_size)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_efficient_matches_sliding_with_identity_rope() -> None:
    torch.manual_seed(0)
    batch, seq_len, hidden_size, num_heads, window = 2, 7, 12, 3, 5
    head_dim = hidden_size // num_heads
    rot = _identity_rotation_matrix(seq_len, head_dim)

    eff = EfficientSlidingWindowMultiheadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        window_size=window,
        rotation_matrix=rot,
    )
    sl = SlidingWindowMultiheadAttention(
        hidden_size=hidden_size, num_heads=num_heads, window_size=window
    )

    with torch.no_grad():
        eff.qkv_linear.weight.copy_(sl.qkv_linear.weight)
        eff.qkv_linear.bias.copy_(sl.qkv_linear.bias)
        eff.out.weight.copy_(sl.out.weight)
        eff.out.bias.copy_(sl.out.bias)

    x = torch.randn(batch, seq_len, hidden_size)
    out_eff = eff(x)
    out_sl = sl(x)

    assert out_eff.shape == out_sl.shape
    assert torch.allclose(out_eff, out_sl, atol=1e-5)


def test_zeroed_weights_produce_zero_output() -> None:
    torch.manual_seed(0)
    batch, seq_len, hidden_size, num_heads, window = 1, 5, 8, 2, 3
    eff = EfficientSlidingWindowMultiheadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        window_size=window,
        rotation_matrix=_identity_rotation_matrix(seq_len, (hidden_size // num_heads)),
    )
    with torch.no_grad():
        eff.qkv_linear.weight.zero_()
        eff.qkv_linear.bias.zero_()
        eff.out.weight.zero_()
        eff.out.bias.zero_()

    x = torch.randn(batch, seq_len, hidden_size)
    out = eff(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_invalid_hidden_divisible_by_heads_raises() -> None:
    with pytest.raises(AssertionError):
        SlidingWindowMultiheadAttention(hidden_size=10, num_heads=3, window_size=3)
