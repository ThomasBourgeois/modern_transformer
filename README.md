# modern_transformer 

`modern_transformer` is a PyTorch learning project for building a decoder-style Transformer with modern components:

- Mixture of Experts (MoE) feed-forward layers
- Sliding-window multi-head attention
- Rotary Position Embeddings (RoPE)
- RMSNorm-based block structure

The codebase is organized to keep reusable components in `src/components/` and model assembly in `src/transformer/`.

## Current Status

This package is under active development.

- Implemented and tested: `MoeLayer` in `src/components/moe.py`
- Implemented: attention and RoPE helpers in `src/components/attentions.py` and `src/components/rope.py`
- In progress (contains TODOs): `TransformerBlock`, `Transformer`, and `RMSNorm._norm`

## Project Structure

```text
modern_transformer/
	src/
		components/
			activations.py
			attentions.py
			moe.py
			norm_layers.py
			rope.py
		transformer/
			blocks.py
			model.py
	tests/
		unittests/
			test_moe.py
		integration_tests/
			test_code.py
	pyproject.toml
	pytest.ini
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

This project uses `pyproject.toml` (`hatchling` backend).

```bash
conda activate <your_env>
pip install -e .
```

## Running Tests

Run tests with `pytest` from the project root.

```bash
conda run -n <your_env> python -m pytest -q
```

Run only unit tests for MoE:

```bash
conda run -n aiedge python -m pytest tests/unittests/test_moe.py -q
```

## Package Metadata

- Name: `modern_transformer`
- Version: `0.0.1`
- Python: `>=3.14`
- License: MIT

