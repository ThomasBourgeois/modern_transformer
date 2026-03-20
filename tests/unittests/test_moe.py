import pytest
import torch
import torch.nn as nn

from src.components.moe import FeedForward, MoeLayer

class ScaleExpert(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _set_known_gate(layer: MoeLayer) -> None:
    # Configure a deterministic gate that yields different top-k routing per token.
    with torch.no_grad():
        layer.gate.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [3.0, 1.0, 0.0],
                ]
            )
        )
        layer.gate.bias.zero_()


def test_feedforward_preserves_last_dimension() -> None:
    layer = FeedForward(hidden_size=8, d_ff=16)
    x = torch.randn(2, 3, 8)

    out = layer(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_moe_preserves_shape_dtype_and_device() -> None:
    layer = MoeLayer(hidden_size=8, d_ff=16, num_experts=4, n_experts_per_token=2)
    x = torch.randn(2, 5, 8, dtype=torch.float32)

    out = layer(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_moe_rejects_invalid_n_experts_per_token() -> None:
    with pytest.raises(ValueError):
        MoeLayer(hidden_size=8, d_ff=16, num_experts=2, n_experts_per_token=0)

    with pytest.raises(ValueError):
        MoeLayer(hidden_size=8, d_ff=16, num_experts=2, n_experts_per_token=3)


def test_weighted_aggregation_matches_reference() -> None:
    x = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    scales = [10.0, 100.0, 1000.0]

    layer = MoeLayer(hidden_size=3, d_ff=6, num_experts=3, n_experts_per_token=2)
    layer.experts = nn.ModuleList([ScaleExpert(s) for s in scales])
    _set_known_gate(layer)

    out = layer(x)

    logits = layer.gate(x)
    topk = torch.topk(logits, 2, dim=-1)
    weights = torch.softmax(topk.values, dim=-1)

    expected = torch.zeros_like(x)
    for token_idx in range(x.shape[1]):
        for choice_idx in range(2):
            expert_id = int(topk.indices[0, token_idx, choice_idx].item())
            w = weights[0, token_idx, choice_idx]
            expected[0, token_idx] += x[0, token_idx] * scales[expert_id] * w

    assert torch.allclose(out, expected, atol=1e-6)


def test_softmax_is_applied_per_token_choice_axis() -> None:
    # If each selected expert returns the same value, output should equal input exactly.
    x = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    layer = MoeLayer(hidden_size=3, d_ff=6, num_experts=3, n_experts_per_token=2)
    layer.experts = nn.ModuleList([ScaleExpert(1.0), ScaleExpert(1.0), ScaleExpert(1.0)])
    _set_known_gate(layer)

    out = layer(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_backward_produces_gradients() -> None:
    torch.manual_seed(42)
    layer = MoeLayer(hidden_size=4, d_ff=8, num_experts=3, n_experts_per_token=2)
    x = torch.randn(2, 3, 4, requires_grad=True)

    out = layer(x)
    loss = out.pow(2).mean()
    loss.backward()

    gate_grad = layer.gate.weight.grad
    assert gate_grad is not None
    assert torch.isfinite(gate_grad).all()
    assert gate_grad.abs().sum().item() > 0.0

    any_expert_grad = False
    for expert in layer.experts:
        for param in expert.parameters():
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                any_expert_grad = True
                break
        if any_expert_grad:
            break

    assert any_expert_grad
