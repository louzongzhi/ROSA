"""
Test script for Torch ROSA Extension.

This script verifies the correctness of the ROSA SAM and Bits operations
by comparing the C++/CUDA implementation against a slow reference Python
implementation. It also checks the gradients of the backward pass.
"""

import torch

from rosa import RosaContext, rosa_bits_ops


def samx_qkv_slow(qqq, kkk, vvv):
    """
    Slow reference implementation of the SAM QKV operation.
    Source: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py

    Args:
        qqq: Query sequence (list of integers).
        kkk: Key sequence (list of integers).
        vvv: Value sequence (list of integers).

    Returns:
        list: The output sequence.
    """
    n = len(qqq)
    y = [-1] * n
    s = 2 * n + 1
    t = [None] * s
    f = [-1] * s
    m = [0] * s
    r = [-1] * s
    t[0] = {}
    g = 0
    u = 1
    w = h = 0
    assert n == len(kkk) == len(vvv)
    for i, (q, k) in enumerate(zip(qqq, kkk)):
        p, x = w, h
        while p != -1 and q not in t[p]:
            x = m[p] if x > m[p] else x
            p = f[p]
        p, x = (t[p][q], x + 1) if p != -1 else (0, 0)
        v = p
        while f[v] != -1 and m[f[v]] >= x:
            v = f[v]
        while v != -1 and (m[v] <= 0 or r[v] < 0):
            v = f[v]
        y[i] = vvv[r[v] + 1] if v != -1 else -1
        w, h = p, x
        j = u
        u += 1
        t[j] = {}
        m[j] = m[g] + 1
        p = g
        while p != -1 and k not in t[p]:
            t[p][k] = j
            p = f[p]
        if p == -1:
            f[j] = 0
        else:
            d = t[p][k]
            if m[p] + 1 == m[d]:
                f[j] = d
            else:
                b = u
                u += 1
                t[b] = t[d].copy()
                m[b] = m[p] + 1
                f[b] = f[d]
                r[b] = r[d]
                f[d] = f[j] = b
                while p != -1 and t[p][k] == d:
                    t[p][k] = b
                    p = f[p]
        v = g = j
        while v != -1 and r[v] < i:
            r[v] = i
            v = f[v]
    return [max(0, y) for y in y]  # use "0" for both "no-match" and matched "0"


if __name__ == "__main__":
    print("=== Testing Forward Pass ===")
    try:
        for _ in range(10):
            # Generate random sequences of 0s and 1s (packed integers)
            q_list = torch.randint(0, 2, size=(8,)).tolist()
            k_list = torch.randint(0, 2, size=(8,)).tolist()
            v_list = torch.randint(0, 2, size=(8,)).tolist()

            # Reference result
            o1 = torch.tensor(samx_qkv_slow(q_list, k_list, v_list))

            # Prepare tensor inputs for ROSA
            # Shape transformation: (T) -> (1, 1, T, 1)
            # Bit unpacking: shift and mask to create (1, 1, T, 4) channels
            query = (torch.tensor([q_list]).view(1, 1, -1, 1) >> torch.arange(4)) & 1
            key = (torch.tensor([k_list]).view(1, 1, -1, 1) >> torch.arange(4)) & 1
            value = (torch.tensor([v_list]).view(1, 1, -1, 1) >> torch.arange(4)) & 1

            query = query.float()
            key = key.float()
            value = value.float()

            # Result from RosaContext
            o2 = RosaContext().update(query, key, value, mismatch=0)
            o2 = ((o2 > 0) << torch.arange(4)).sum(dim=-1).squeeze()

            # Result from rosa_bits_ops
            o3 = rosa_bits_ops(query, key, value)
            o3 = ((o3 > 0) << torch.arange(4)).sum(dim=-1).squeeze()

            # Debug output (optional)
            # print("Ref:", o1)
            # print("Ctx:", o2)
            # print("Ops:", o3)
            # print()

            assert (o1 == o2).all(), f"Mismatch in RosaContext: {o1} vs {o2}"
            assert (o1 == o3).all(), f"Mismatch in rosa_bits_ops: {o1} vs {o3}"

        print("✅ Forward Pass Passed!")
    except AssertionError as e:
        print("❌ Forward Pass Failed!")
        print(e)
    print()

    print("=== Testing Backward Pass ===")
    try:
        for _ in range(10):
            # Generate inputs with gradients enabled
            # Shape: (T, 2) -> (1, 1, T, 2)
            q = torch.randint(0, 2, size=(8, 2)).float().view(1, 1, -1, 2).requires_grad_(True)
            k = torch.randint(0, 2, size=(8, 2)).float().view(1, 1, -1, 2).requires_grad_(True)
            v = torch.randint(0, 2, size=(8, 2)).float().view(1, 1, -1, 2).requires_grad_(True)

            # Forward pass
            o = rosa_bits_ops(q, k, v)
            # Backward pass
            o.sum().backward()

            # Check for NaNs or Infs in gradients
            assert not q.grad.isnan().any(), "NaN detected in Query gradient"
            assert not k.grad.isnan().any(), "NaN detected in Key gradient"
            assert not v.grad.isnan().any(), "NaN detected in Value gradient"

            assert not q.grad.isinf().any(), "Inf detected in Query gradient"
            assert not k.grad.isinf().any(), "Inf detected in Key gradient"
            assert not v.grad.isinf().any(), "Inf detected in Value gradient"

        print("✅ Backward Pass Passed!")
    except AssertionError as e:
        print("❌ Backward Pass Failed!")
        print(e)
    print()
