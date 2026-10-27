"""Shared helpers for numerical-equivalence tests.

The contract mirrors the reference projects: a batched or optimized path must
produce the same tensors as the straightforward per-sample path, within a
declared tolerance. fp32 CPU is the strict case; bf16 tolerances are recorded
for completeness even though the fake models run in fp32.
"""

from __future__ import annotations

from typing import Sequence

import torch

TOL_CPU_FP32 = (1e-5, 1e-5)
TOL_BF16 = (1e-2, 1e-2)


def assert_tensor_close(actual, expected, atol=TOL_CPU_FP32[0], rtol=TOL_CPU_FP32[1]):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_batch_matches_singles(
    batch: torch.Tensor,
    singles: Sequence[torch.Tensor],
    atol=TOL_CPU_FP32[0],
    rtol=TOL_CPU_FP32[1],
):
    assert batch.shape[0] == len(singles), (
        f"batch rows ({batch.shape[0]}) != references ({len(singles)})"
    )
    for i, single in enumerate(singles):
        assert_tensor_close(batch[i], single, atol=atol, rtol=rtol)


def assert_generation_equal(first, second, atol=TOL_CPU_FP32[0], rtol=TOL_CPU_FP32[1]):
    """Compare the ``(ids, cetts, texts, confs, ents)`` tuples from generation."""
    assert first[0] == second[0]
    assert first[2] == second[2]
    assert len(first[1]) == len(second[1])
    for a, b in zip(first[1], second[1]):
        assert_tensor_close(a, b, atol=atol, rtol=rtol)
    for a, b in zip(first[3], second[3]):
        assert abs(a - b) <= atol
    for a, b in zip(first[4], second[4]):
        assert abs(a - b) <= atol
