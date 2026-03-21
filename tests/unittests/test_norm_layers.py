import torch
import pytest

from modern_transformer.components.norm_layers import RMSNorm


def test_rmsnorm_shape_dtype_device() -> None:
    torch.manual_seed(0)
    batch, seq, hidden = 2, 4, 6
    m = RMSNorm(hidden)
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    out = m(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_rmsnorm_matches_manual_computation() -> None:
    torch.manual_seed(0)
    batch, seq, hidden = 2, 3, 5
    eps = 1e-6
    m = RMSNorm(hidden, eps=eps)
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)

    out = m(x)

    variance = x.pow(2).mean(-1, keepdim=True)
    expected = (x / torch.sqrt(variance + eps)).type_as(x) * m.weight
    assert torch.allclose(out, expected, atol=1e-6)


def test_backward_produces_gradients() -> None:
    torch.manual_seed(0)
    batch, seq, hidden = 3, 2, 4
    m = RMSNorm(hidden)
    x = torch.randn(batch, seq, hidden, requires_grad=True)

    out = m(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert m.weight.grad is not None
    assert torch.isfinite(m.weight.grad).all()
    assert m.weight.grad.abs().sum().item() > 0.0


def test_preserves_input_dtype() -> None:
    torch.manual_seed(0)
    batch, seq, hidden = 1, 2, 4
    m = RMSNorm(hidden)
    x = torch.randn(batch, seq, hidden, dtype=torch.float64)
    out = m(x)
    assert out.dtype == torch.float64


def test_zero_input_stable_and_not_nan() -> None:
    batch, seq, hidden = 2, 3, 4
    m = RMSNorm(hidden)
    x = torch.zeros(batch, seq, hidden, dtype=torch.float32)
    out = m(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
