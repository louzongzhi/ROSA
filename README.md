# ROSA: Rapid Online Suffix Automaton

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

ROSA (Rapid Online Suffix Automaton) is a high-performance, parameter-free attention alternative that implements a "neurosymbolic infinite-range lossless information propagator" as proposed by Peng Bo (BlinkDL) in the RWKV-8 architecture.

![RWKV-8-ROSA Architecture](RWKV-8-ROSA.png)

## Background: What is ROSA?

ROSA is a groundbreaking mechanism that replaces the standard attention mechanism with what is described as a "neurosymbolic infinite-range lossless information propagator."

The core idea is to predict the next token in a sequence based on the longest exact match found in the history. For a given sequence `x`, the output `y_i` is determined by finding the longest suffix of `x` ending at `i-1` that matches a previous substring. If such a match is found at index `j`, the output is the token that followed that match, `x_{j+1}`.

This mechanism has several powerful properties:
- **Parameter-Free**: The core logic has no trainable weights.
- **No Dot Product / Softmax**: It operates on discrete tokens, eliminating the quadratic complexity of standard attention.
- **No Float KV Cache**: It only needs to store the history of discrete tokens.
- **Efficient Inference**: The underlying Suffix Automaton can be processed very quickly on a CPU, in parallel with GPU-based layers.

## Installation & Usage

### Installing ROSA

```bash
pip install --no-build-isolation .
```

### Importing the Operator

```python
from rosa import rosa_bits_ops
```

## Project Structure

- **`rosa/`**: Python package with PyTorch interface and differentiable extensions
- **`test/`**: Unit tests for the ROSA implementation
- **`setup.py`**: Python package installation script
- **`pyproject.toml`**: Python project configuration
- **`RWKV-8-ROSA.png`**: ROSA architecture diagram

## Acknowledgments

This work is heavily inspired by the original research and innovations of Peng Bo (BlinkDL) in the RWKV project. The ROSA concept was first proposed in the [RWKV](https://github.com/BlinkDL/RWKV-LM). This implementation is based on the `rosa_cpp` component from [rosa_soft](https://github.com/wjie98/rosa_soft).

## License

[Apache License](LICENSE)
