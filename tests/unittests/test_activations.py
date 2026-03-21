import torch
import pytest

from modern_transformer.components.activations import SiGLU


def test_siglu_shape_dtype_device() -> None:
    torch.manual_seed(0)
    batch, seq, in_f, out_f = 2, 3, 5, 7
    m = SiGLU(in_f, out_f)
    x = torch.randn(batch, seq, in_f)
    out = m(x)
    assert out.shape == (batch, seq, out_f)
    assert out.dtype == torch.float32
    assert out.device == x.device


def test_siglu_matches_manual_computation() -> None:
    torch.manual_seed(0)
    batch, seq, in_f, out_f = 2, 4, 6, 8
    m = SiGLU(in_f, out_f)
    x = torch.randn(batch, seq, in_f)

    out = m(x)
    expected = m.W(x) * torch.sigmoid(m.Wg(x))
    assert torch.allclose(out, expected, atol=1e-6)


def test_backward_produces_gradients() -> None:
    torch.manual_seed(0)
    batch, seq, in_f, out_f = 3, 2, 4, 4
    m = SiGLU(in_f, out_f)
    x = torch.randn(batch, seq, in_f, requires_grad=True)

    out = m(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert m.W.weight.grad is not None
    assert m.Wg.weight.grad is not None
    assert torch.isfinite(m.W.weight.grad).all()
    assert torch.isfinite(m.Wg.weight.grad).all()
    assert m.W.weight.grad.abs().sum() > 0.0 or m.Wg.weight.grad.abs().sum() > 0.0


def test_zeroed_weights_produce_zero_output() -> None:
    batch, seq, in_f, out_f = 2, 2, 3, 3
    m = SiGLU(in_f, out_f)
    with torch.no_grad():
        m.W.weight.zero_()
        m.Wg.weight.zero_()
    x = torch.randn(batch, seq, in_f)
    out = m(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
