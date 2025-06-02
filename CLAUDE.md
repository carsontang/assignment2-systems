# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Running Tests
```bash
# Run all tests
uv run pytest -v ./tests

# Run a single test file
uv run pytest -v ./tests/test_attention.py

# Run a specific test
uv run pytest -v ./tests/test_attention.py::test_flashattention_pytorch

# Create submission file
./test_and_make_submission.sh
```

### Linting
```bash
# Run ruff linter (configured with line-length=120)
uv run ruff check .
uv run ruff format .
```

### Dependencies
- This project uses `uv` for dependency management
- Dependencies are specified in `pyproject.toml`
- The project depends on `cs336-basics` module from the `./cs336-basics` directory

## Architecture Overview

This is CS336 Assignment 2: Systems, focused on implementing optimized Transformer components and distributed training.

### Key Implementation Requirements

The `cs336_systems` module needs to implement four main components:

1. **Flash Attention** (`test_attention.py`)
   - `get_flashattention_autograd_function_pytorch()`: PyTorch-based implementation
   - `get_flashattention_autograd_function_triton()`: Triton kernel-based implementation
   - Must handle causal masking and save log-sum-exp values for backward pass

2. **DDP with Bucketing** (`test_ddp.py`)
   - `get_ddp_bucketed()`: Returns DDP wrapper that groups gradients into buckets
   - `ddp_bucketed_on_train_batch_start()`: Pre-training batch hook
   - `ddp_bucketed_on_after_backward()`: Post-backward hook
   - Must handle parameter broadcasting and gradient synchronization

3. **DDP with Individual Parameters** (`test_ddp_individual_parameters.py`)
   - `get_ddp_individual_parameters()`: Returns DDP wrapper for individual parameter sync
   - `ddp_individual_parameters_on_after_backward()`: Post-backward hook

4. **Sharded Optimizer** (`test_sharded_optimizer.py`)
   - `get_sharded_optimizer()`: Returns optimizer that shards states across ranks
   - Must maintain correctness with AdamW optimization

### Important Implementation Notes

- All distributed implementations must handle:
  - Models with tied weights (shared parameters)
  - Parameters that don't require gradients
  - Proper synchronization across distributed ranks
- Test fixtures are provided in `tests/fixtures/` with pre-generated data
- The `cs336-basics` module contains the base language model implementation from Assignment 1
- You should copy needed code from `cs336-basics` as a starting point for your implementations in `cs336_systems`

### Project Structure

```
assignment2-systems/
├── cs336-basics/          # Base LM implementation from Assignment 1
├── cs336_systems/         # Your implementation directory (currently empty)
├── tests/                 # Test files that define the required API
└── pyproject.toml        # Project configuration and dependencies
```