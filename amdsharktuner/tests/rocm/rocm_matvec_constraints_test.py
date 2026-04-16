# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import z3  # type: ignore

from iree.compiler.dialects import iree_gpu  # type: ignore

from amdsharktuner import common
from amdsharktuner.rocm import rocm_dispatch_constraints

from amdsharktuner.test_utils import tuner_ctx  # noqa: F401


@pytest.fixture
def gpu_target_info(tuner_ctx: common.TunerContext) -> iree_gpu.TargetInfo:
    context = tuner_ctx.mlir_ctx
    return iree_gpu.TargetInfo(
        context=context,
        arch="gfx942",
        subgroup_size_choices=[64],
        max_workgroup_sizes=[1024, 1024, 1024],
        max_thread_count_per_workgroup=1024,
        max_workgroup_memory_bytes=65536,
        workgroup_count=304,
        simds_per_workgroup=4,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        ],
    )


def _solve(
    gpu_target_info: iree_gpu.TargetInfo,
    parallel_bounds: list[int],
    reduction_bound: int,
    bitwidth: int,
) -> z3.ModelRef | None:
    subgroup_size = z3.Int("subgroup_size")
    thread_loads = z3.Int("thread_loads")
    workgroup_size = z3.Int("workgroup_size")
    num_parallel_reductions = z3.Int("num_parallel_reductions")

    constraints = rocm_dispatch_constraints.generate_matvec_vector_distribute_constraints(
        parallel_bounds=parallel_bounds,
        reduction_bound=reduction_bound,
        largest_operand_bitwidth=bitwidth,
        subgroup_size=subgroup_size,
        thread_loads=thread_loads,
        workgroup_size=workgroup_size,
        num_parallel_reductions=num_parallel_reductions,
        gpu_target_info=gpu_target_info,
    )

    solver = z3.Solver()
    solver.add(constraints)
    if solver.check() != z3.sat:
        return None
    return solver.model()


def test_matvec_constraints_basic(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    model = _solve(gpu_target_info, parallel_bounds=[4096], reduction_bound=4096, bitwidth=16)
    assert model is not None
    subgroup = model[z3.Int("subgroup_size")].as_long()
    tl = model[z3.Int("thread_loads")].as_long()
    wg = model[z3.Int("workgroup_size")].as_long()
    npr = model[z3.Int("num_parallel_reductions")].as_long()

    assert subgroup == 64
    assert tl in (1, 2, 4, 8)
    assert wg % subgroup == 0
    assert wg <= 1024
    assert (4096 % (wg * tl)) == 0
    assert npr & (npr - 1) == 0  # power of two
    assert npr * 4 <= wg
    assert (4096 % npr) == 0


def test_matvec_constraints_gemv_shape(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    model = _solve(gpu_target_info, parallel_bounds=[1], reduction_bound=8192, bitwidth=16)
    assert model is not None
    npr = model[z3.Int("num_parallel_reductions")].as_long()
    assert npr == 1


def test_matvec_constraints_batched(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    model = _solve(gpu_target_info, parallel_bounds=[8, 1], reduction_bound=4096, bitwidth=16)
    assert model is not None
    npr = model[z3.Int("num_parallel_reductions")].as_long()
    assert 8 % npr == 0


def test_matvec_constraints_f32(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    model = _solve(gpu_target_info, parallel_bounds=[4096], reduction_bound=4096, bitwidth=32)
    assert model is not None
    tl = model[z3.Int("thread_loads")].as_long()
    assert tl in (1, 2, 4)


def test_matvec_constraints_unsatisfiable_small_reduction(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    model = _solve(gpu_target_info, parallel_bounds=[4096], reduction_bound=7, bitwidth=16)
    assert model is None
