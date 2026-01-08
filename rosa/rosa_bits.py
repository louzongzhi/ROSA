"""
ROSA Bits operations.
"""

import math
from typing import Dict, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from .rosa_sam import RosaContext, RosaWork

__all__ = [
    "rosa_bits_ops",
]


def rosa_bits_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    suffix_window: int = 8,
    suffix_factor: Optional[float] = None,
    attention_mask: Optional[Tensor] = None,
    attention_tau: float = 0.1,
    async_op: bool = False,
) -> Union[Tensor, "RosaBitsWork"]:
    work = RosaBitsWork()
    work._future = RosaContext().update(
        query, key, value, 0, inspect=True, async_op=True
    )
    work._params = RosaBitsParams(
        suffix_window=suffix_window,
        suffix_factor=suffix_factor,
        attention_mask=attention_mask,
        attention_tau=attention_tau,
    )
    work._query_key_value = (query, key, value)

    if async_op:
        return work
    return work.wait()


class RosaBitsWork:
    def __init__(self):
        self._future: Optional[RosaWork] = None
        self._params: Optional["RosaBitsParams"] = None
        self._query_key_value: Optional[Tuple[Tensor, Tensor, Tensor]] = None

    def wait(self) -> Tensor:
        if self._future is None:
            raise RuntimeError("wait() called twice")

        work = self._future
        params = self._params
        query, key, value = self._query_key_value

        x_hard, info = work.wait()
        params.info["x_hard"] = x_hard
        params.info.update(info)

        self._future = None
        self._params = None
        self._query_key_value = None

        return RosaBitsFunction.apply(query, key, value, params)


class RosaBitsParams:
    def __init__(
        self,
        suffix_window: int,
        suffix_factor: Optional[float],
        attention_mask: Optional[Tensor],
        attention_tau: float,
    ):
        self.suffix_window = suffix_window
        self.suffix_factor = suffix_factor

        if isinstance(attention_mask, Tensor):
            self.attention_mask = attention_mask.detach()
        else:
            self.attention_mask = None
        self.attention_tau = attention_tau

        self.info: Dict[str, Tensor] = {}
        self._ctx: Optional[RosaContext] = None


class RosaBitsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query: Tensor, key: Tensor, value: Tensor, params: RosaBitsParams
    ) -> Tensor:
        if "x_hard" in params.info:
            x_hard = params.info.pop("x_hard")
        else:
            x_hard, info = RosaContext().update(query, key, value, 0, inspect=True)
            params.info.update(info)

        ctx.save_for_backward(
            query.detach(),
            key.detach(),
            value.detach(),
        )
        ctx.saved_params = params

        return x_hard

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, value = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        params: RosaBitsParams = ctx.saved_params

        length = params.info.pop("length")
        endpos = params.info.pop("endpos")

        with torch.enable_grad():
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

            x_soft = suffix_attention_proxy(
                query,
                key,
                value,
                length=length,
                endpos=endpos,
                suffix_window=params.suffix_window,
                suffix_factor=params.suffix_factor,
                attention_mask=params.attention_mask,
                attention_tau=params.attention_tau,
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                outputs=x_soft,
                inputs=(query, key, value),
                grad_outputs=grad_output,
                retain_graph=False,
                only_inputs=True,
            )
        return grad_query, grad_key, grad_value, None


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    bsz, num_heads, seq_len, head_dim = hidden_states.size()
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, -1, seq_len, head_dim)


def unfold_qk(hidden_states: Tensor, win_size: int, offset: int = 0) -> Tensor:
    hidden_states = F.pad(hidden_states, (0, 0, win_size - 1 + offset, -offset))
    hidden_states = hidden_states.unfold(-2, win_size, 1).transpose(-2, -1)
    return hidden_states


def decay_qk(
    xq: Tensor, xk: Tensor, decay_factor: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    bsz, num_heads, seq_len, win_size, num_bits = xq.size()

    if win_size > 1:
        inds = torch.arange(win_size, device=xq.device)
        inds = win_size - 1 - inds

        if decay_factor is None:
            wd = max(0.1, 1.0 / win_size) ** (1.0 / win_size)
        else:
            wd = max(0.1, min(decay_factor, 0.99))

        qk_w = torch.pow(wd, inds)
        qk_w = qk_w / qk_w.sum()
        qk_w_sqrt = torch.sqrt(qk_w)

        xq = xq * qk_w_sqrt.view(-1, 1).type_as(xq)
        xk = xk * qk_w_sqrt.view(-1, 1).type_as(xk)
    return xq, xk


def gather_x(x: Tensor, endpos: Tensor, offset: int = 1) -> Tensor:
    bsz, num_heads, seq_len, num_bits = x.size()
    with torch.no_grad():
        epos = endpos.view(bsz, num_heads, seq_len)

    mask = (epos >= 0).view(bsz, num_heads, seq_len, 1).type_as(x)
    inds = (epos + offset).clamp(0, seq_len - 1)
    inds = inds.view(bsz, num_heads, seq_len, 1).expand_as(x)

    return x.gather(-2, inds) * mask


def suffix_attention_proxy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    length: Tensor,
    endpos: Tensor,
    suffix_window: int,
    suffix_factor: Optional[float],
    attention_mask: Optional[Tensor],
    attention_tau: float,
) -> Tensor:
    bsz, num_heads, seq_len, num_q_bits = query.size()
    bsz, num_kv_heads, seq_len, num_k_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    num_qk_bits = num_q_bits
    assert num_q_bits == num_k_bits, "query and key must have the same number of bits"

    MAX_ATTENTION_HEAD_DIM = 256
    assert 0 < num_qk_bits * suffix_window <= MAX_ATTENTION_HEAD_DIM

    xq = F.softsign(query)
    xk = F.softsign(key)
    xv = F.softsign(value)

    attention_scale = 1.0 / num_qk_bits / attention_tau

    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    xq = unfold_qk(xq, win_size=suffix_window, offset=0)
    xk = unfold_qk(xk, win_size=suffix_window, offset=1)

    xq, xk = decay_qk(xq, xk, decay_factor=suffix_factor)

    xq = xq.reshape(bsz, num_heads, seq_len, num_q_bits * suffix_window)
    xk = xk.reshape(bsz, num_heads, seq_len, num_k_bits * suffix_window)

    xo = F.scaled_dot_product_attention(
        xq,
        xk,
        xv,
        scale=attention_scale,
        is_causal=True,
        attn_mask=attention_mask,
    )

    pk = gather_x(xk, endpos)
    pv = gather_x(xv, endpos)

    gg = torch.sum(xq * pk, dim=-1, keepdim=True)
    gg = torch.sigmoid(gg * attention_scale)

    xo = xo * (1 - gg) + pv * gg
    return xo
