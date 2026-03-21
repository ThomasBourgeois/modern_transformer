import torch
import pytest

from modern_transformer.components.rope import (
    get_rotation_matrix,
    RoPE,
    apply_rotary_emb,
)


def test_get_rotation_matrix_shape_and_dtype() -> None:
    dim = 8
    context = 16
    period = 10000.0
    rot = get_rotation_matrix(dim, context, period)
    assert rot.shape == (context, dim // 2)
    assert torch.is_complex(rot)


def test_rope_forward_preserves_shape_and_changes_values() -> None:
    torch.manual_seed(0)
    batch, num_heads, seq_length, head_dim = 2, 3, 10, 8
    # head_dim must be even
    assert head_dim % 2 == 0

    rotation = get_rotation_matrix(head_dim, seq_length, 10000.0)
    rope = RoPE(rotation)

    queries = torch.randn(batch, num_heads, seq_length, head_dim)
    keys = torch.randn(batch, num_heads, seq_length, head_dim)

    q_rot, k_rot = rope(queries.clone(), keys.clone())

    assert q_rot.shape == queries.shape
    assert k_rot.shape == keys.shape
    assert q_rot.dtype == queries.dtype
    assert k_rot.dtype == keys.dtype
    assert torch.isfinite(q_rot).all()
    assert torch.isfinite(k_rot).all()
    # rotation should change values for non-zero inputs
    if not torch.allclose(queries, torch.zeros_like(queries)):
        assert not torch.allclose(q_rot, queries)


def test_apply_rotary_emb_matches_manual_complex_mult() -> None:
    torch.manual_seed(0)
    seq_len = 7
    batch = 2
    heads = 1
    head_dim = 6  # even

    xq = torch.randn(batch, heads, seq_len, head_dim)
    xk = torch.randn(batch, heads, seq_len, head_dim)

    freqs = get_rotation_matrix(head_dim, seq_len, 10000.0)
    # apply_rotary_emb expects freqs shaped (seq_len, head_dim//2)

    out_q, out_k = apply_rotary_emb(xq, xk, freqs)

    # Manual computation using complex views
    def to_complex(t: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(t.float().reshape(*t.shape[:-1], -1, 2))

    xq_c = to_complex(xq)
    xk_c = to_complex(xk)

    # apply_rotary_emb internally broadcasts `freqs` to align with queries/keys
    # and then flattens the final real view from dim=2 onward. Reproduce that
    # sequence here to produce an expected tensor with the same layout.
    freqs_c = freqs[:, None, :]

    expected_q = torch.view_as_real(xq_c * freqs_c).flatten(2).type_as(xq)
    expected_k = torch.view_as_real(xk_c * freqs_c).flatten(2).type_as(xk)

    assert out_q.shape == expected_q.shape
    assert out_k.shape == expected_k.shape
    torch.testing.assert_close(out_q, expected_q, atol=1e-6, rtol=0)
    torch.testing.assert_close(out_k, expected_k, atol=1e-6, rtol=0)
