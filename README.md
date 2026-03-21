# modern_transformer 

[![codecov](https://codecov.io/gh/ThomasBourgeois/modern_transformer/branch/main/graph/badge.svg)](https://codecov.io/gh/ThomasBourgeois/modern_transformer)

`modern_transformer` is a PyTorch learning project for building a decoder-style Transformer with modern components:

- Mixture of Experts (MoE) feed-forward layers
- Sliding-window multi-head attention
- Rotary Position Embeddings (RoPE)
- RMSNorm-based block structure

The codebase is organized to keep reusable components in `src/components/` and model assembly in `src/transformer/`.

## Current Status

The following components are implemented and covered by unit tests in `tests/unittests/`:

- **Activation:** `SiGLU` implemented in `src/components/activations.py` and tested.
- **Normalization:** `RMSNorm` implemented in `src/components/norm_layers.py` and tested.
- **Attention:** `SlidingWindowMultiheadAttention` and `EfficientSlidingWindowMultiheadAttention` implemented in `src/components/attentions.py` and tested.
- **Rotary Embeddings:** `RoPE` and `apply_rotary_emb` in `src/components/rope.py` and tested.
- **Mixture-of-Experts:** `FeedForward` and `MoeLayer` in `src/components/moe.py` and comprehensively tested.
- **Transformer assembly:** `TransformerBlock` and `Transformer` implemented in `src/transformer/blocks.py` and `src/transformer/model.py` (basic forward pass implemented; integration tests present).

Note: the project requires Python >= 3.14 for installation and running the test suite (see Installation section).

## Project Structure

```text
modern_transformer/
	LICENSE
	pyproject.toml
	pytest.ini
	README.md
	src/
		modern_transformer/
			__init__.py
			components/
				__init__.py
				activations.py
				attentions.py
				moe.py
				norm_layers.py
				rope.py
				rope.py.bak
			transformer/
				__init__.py
				blocks.py
				model.py
	tests/
		__init__.py
		unittests/
			test_activations.py
			test_attentions.py
			test_moe.py
			test_norm_layers.py
			test_rope.py
		integration_tests/
			test_code.py
```

## Core Modules

- `src/components/moe.py`
	MoE routing layer with top-k expert selection and weighted aggregation.

- `src/components/attentions.py`
	Sliding-window attention implementations, including an efficient variant using tensor windowing.

- `src/components/rope.py`
	RoPE utilities for complex-valued query/key rotation.

- `src/transformer/blocks.py` and `src/transformer/model.py`
	High-level Transformer block/model composition (work in progress).

## Installation

This project uses `pyproject.toml` (`hatchling` backend) and requires Python >= 3.14.

Create and activate a Python 3.14 environment (example using conda), then install the package with development extras:

```bash
conda create -y -n modern_transformer python=3.14
conda activate modern_transformer
pip install --upgrade pip
pip install -e '.[dev]'
```

## Running Tests

Run the full test suite from the project root (in the Python 3.14 env):

```bash
pytest -q
```

Run only unit tests for MoE:

```bash
pytest tests/unittests/test_moe.py -q
```

If you prefer the `conda run` form (example):

```bash
conda run -n modern_transformer pytest -q
```

## Package Metadata

- Name: `modern_transformer`
- Version: `0.0.1`
- Python: `>=3.14`
- License: MIT

## CI / Codecov notes

- The GitHub Actions workflow uses Python 3.14 (see `.github/workflows/ci.yml`).
- If your repository's default branch is protected, Codecov uploads require a Codecov upload token.
	Add the token as a repository secret named `CODECOV_TOKEN` (Settings → Secrets → Actions) or via the `gh` CLI:

```bash
gh secret set CODECOV_TOKEN --body "<your-codecov-token>"
```

You can also disable or remove the Codecov step in the workflow if you do not wish to publish coverage.

