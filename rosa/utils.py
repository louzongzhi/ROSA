"""
Utility functions for quantization and dequantization.
"""

from typing import Union

import torch
from torch import Tensor


def quantize(x: Tensor) -> Tensor:
    """
    Quantize a floating point tensor to an integer representation.

    Args:
        x: Input tensor with shape (..., num_bits).

    Returns:
        Quantized tensor with shape (...).
    """
    assert x.is_floating_point()
    num_bits = x.size(-1)
    if num_bits <= 8:
        dtype = torch.uint8
    elif num_bits <= 16:
        dtype = torch.int16
    elif num_bits <= 32:
        dtype = torch.int32
    else:
        dtype = torch.int64

    r = torch.arange(num_bits, device=x.device)
    x = ((x > 0).to(dtype) << r).sum(dim=-1)
    return x


def dequantize(x: Tensor, v: Union[Tensor, int]) -> Tensor:
    """
    Dequantize an integer tensor back to binary bits.

    Args:
        x: Input tensor.
        v: Reference tensor or integer specifying the number of bits.

    Returns:
        Dequantized binary tensor.
    """
    assert not x.is_floating_point()

    if isinstance(v, Tensor):
        num_bits = v.size(-1)
    else:
        num_bits = int(v)

    r = torch.arange(num_bits, device=x.device)
    x = (x.unsqueeze(-1) >> r) & 1

    if isinstance(v, Tensor):
        return x.type_as(v)
    else:
        return x
