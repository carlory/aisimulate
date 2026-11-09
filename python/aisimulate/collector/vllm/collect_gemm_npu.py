# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM GEMM collector for Huawei Ascend NPU devices (vllm-ascend).

The NPU counterpart to the CUDA vLLM GEMM collector: it builds
RowParallelLinear layers, prepares supported FP8 (W8A8FP8) paths, expands
YAML-backed matrix shapes and logs perf rows through the device-agnostic
helpers. Eager only, no power sampling.

The pinned version is the vllm-ascend release, not base vLLM: vllm-ascend
ships the Ascend kernels, so it is the kernel-source identity of these rows
and the version axis of the parquet directory (manifest key ``vllm_npu``).
"""

__compat__ = "vllm_ascend==0.30.0rc2"

import importlib.metadata as im
import os

import torch
from collector.case_generator import get_gemm_case_specs, get_gemm_type_specs
from collector.helper import benchmark_with_power, get_device_module, log_perf

# The stock vLLM harness utilities reused here are device-neutral:
# create_vllm_config builds a plain VllmConfig from a local fake HF model and
# setup_distributed does a single-rank FileStore init whose process-group
# backend comes from vLLM's own platform plugin (HCCL under vllm-ascend).
# The XPU fork (utils_xpu) stays off-limits to non-XPU collectors
# (tests/unit/collector/test_vllm_xpu_utils.py), and its env://+port rendezvous
# is prone to port conflicts in multi-worker runs; the stock FileStore
# rendezvous is the robust default here.
# TODO: on-device campaign — confirm that the single-rank init selects the HCCL PG.
from collector.vllm.utils import create_vllm_config, setup_distributed, with_exit_stack
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm_ascend.quantization.configs import AscendModelSlimConfig

vllm_ascend_version = im.version("vllm-ascend")

# Max device-memory fraction for the packed op copies.
_LOOP_MEM_BUDGET_FRACTION = 0.45

# Upper bound on GEMM ops packed into one measurement loop. Small GEMMs are
# launch-overhead dominated, so packing more per measurement amortizes it and
# stabilizes the per-op latency (total/count). The memory budget drops this
# lower for large shapes.
_MAX_OPS_PER_LOOP = 64


# Per-worker cache: VllmConfig is identical for every gemm case, but building
# it re-scans package entry_points (slow). Build once, reuse.
_VLLM_CONFIG_CACHE = {}


def _get_cached_vllm_config(dtype):
    key = str(dtype)
    if key not in _VLLM_CONFIG_CACHE:
        model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
        _VLLM_CONFIG_CACHE[key] = create_vllm_config(model_name=model, dtype=dtype)
    return _VLLM_CONFIG_CACHE[key]


def _gemm_peak_footprint_bytes(gemm_type: str, m: int, n: int, k: int, copies: int = 1) -> int:
    """Peak GEMM case footprint (bytes): input + copies*(weight+output) bf16,
    plus fp8 int8 staging (n*k).

    ``copies=0`` returns just the copy-independent portion (input + staging).
    Over-reserves for fp8 paths (weight/output sized as bf16) on purpose.
    """
    input_bytes = m * k * 2
    per_copy = (n * k + m * n) * 2
    footprint = input_bytes + per_copy * max(copies, 0)
    if gemm_type == "fp8":
        footprint += n * k
    return footprint


def get_gemm_test_cases():
    gemm_list = get_gemm_type_specs("vllm_npu")
    if not gemm_list:
        raise RuntimeError("collector/cases/base_ops/gemm.yaml must define vllm_npu gemm_types")

    test_cases = []

    for gemm_common_testcase in get_gemm_case_specs("vllm_npu"):
        x = gemm_common_testcase.x
        n = gemm_common_testcase.n
        k = gemm_common_testcase.k
        for gemm_type in gemm_list:
            test_cases.append([gemm_type, x, n, k])

    return test_cases


@with_exit_stack
def run_gemm(exit_stack, gemm_type, m, n, k, *, perf_filename, device="npu:0"):
    if gemm_type not in ("bfloat16", "fp8"):
        # The YAML partition declares bfloat16 and fp8 (W8A8FP8); anything else
        # would be benchmarked as unquantized bf16 under the wrong dtype label.
        raise RuntimeError(f"vllm_npu gemm collects bfloat16 and fp8 only, got {gemm_type!r}")

    setup_distributed(device)

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    get_device_module(device).set_device(device)
    # TODO: on-device campaign — NPU visibility is governed by
    # ASCEND_RT_VISIBLE_DEVICES (CUDA_VISIBLE_DEVICES has no effect);
    # multi-card parallel collection must pin its workers with the former.

    x = torch.randn((m, k), dtype=dtype, device=torch.device(device))

    if gemm_type == "fp8":
        # Base vLLM's Fp8Config dispatches to choose_scaled_mm_linear_kernel
        # (no ScaledMM backend on NPU -> ValueError), and AscendFp8Config rejects
        # weight_block_size=None while re-encoding block-wise fp8 to W8A8_MXFP8
        # on Ascend 950 (fp8_block.py:AscendFp8BlockLinearMethod). Route
        # per-channel dynamic W8A8FP8 through AscendModelSlimConfig ->
        # AscendLinearMethod(AscendW8A8FP8DynamicLinearMethod) (prefix="" maps
        # to key ".weight" in get_quant_type_for_layer). Note: as with fp8_block
        # auto-requantizing to W8A8_MXFP8 on 950, future vllm-ascend releases
        # may similarly remap W8A8*_DYNAMIC paths on A5 hardware.
        qc = AscendModelSlimConfig({".weight": "W8A8FP8_DYNAMIC"})
    else:
        qc = None

    def create_gemm():
        gemm = RowParallelLinear(
            input_size=k,
            output_size=n,
            bias=False,
            skip_bias_add=True,
            params_dtype=dtype,
            quant_config=qc,
            prefix="",
            return_bias=True,
            disable_tp=True,
        )
        # vLLM >=0.16 may create layers on the meta device; use to_empty()
        # then fill with random data.
        try:
            gemm.to(torch.device(device))
        except NotImplementedError:
            gemm = gemm.to_empty(device=torch.device(device))
            with torch.no_grad():
                for param in gemm.parameters():
                    if param.dtype.is_floating_point:
                        param.normal_()
                    else:
                        param.zero_()

        if gemm_type == "fp8" and hasattr(gemm, "weight"):
            with torch.no_grad():
                # CANN aclnnInplaceFillScalar rejects DT_FLOAT8_E4M3FN, so fill
                # a BF16 tensor first and copy the converted FP8 values in.
                gemm.weight.copy_(
                    torch.full(
                        gemm.weight.shape,
                        0.01,
                        dtype=dtype,
                        device=gemm.weight.device,
                    ).to(gemm.weight.dtype)
                )
                if hasattr(gemm, "weight_scale"):
                    gemm.weight_scale.fill_(1.0)
                if hasattr(gemm, "weight_offset"):
                    gemm.weight_offset.zero_()
            # Use process_weights_after_loading() to transpose weights and flatten scales
            quant_method = getattr(gemm, "quant_method", None)
            if quant_method is not None and hasattr(quant_method, "process_weights_after_loading"):
                quant_method.process_weights_after_loading(gemm)

        gemm.forward(x)  # noqa: F821  # dry run to init

        return gemm

    # VllmConfig is identical across gemm cases; build once per worker and
    # reuse (construction re-scans package entry_points, which is slow).
    vllm_config = _get_cached_vllm_config(dtype)
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    # Ops per measurement: pack as many as the memory budget allows, capped so
    # tiny shapes don't create an absurd number. Large shapes -> few
    # (memory-bound); tiny shapes -> the cap, diluting fixed launch overhead.
    total_mem = get_device_module(device).get_device_properties(device).total_memory
    per_copy = (n * k + m * n) * 2
    fixed_bytes = _gemm_peak_footprint_bytes(gemm_type, m, n, k, copies=0)
    budget = int(total_mem * _LOOP_MEM_BUDGET_FRACTION)
    mem_cap = (budget - fixed_bytes) // max(per_copy, 1)
    outside_loop_count = max(1, min(_MAX_OPS_PER_LOOP, mem_cap))

    op_list = []
    try:
        for i in range(outside_loop_count):
            op_list.append(create_gemm())

        def kernel_func():
            for op in op_list:  # noqa: F821
                op.forward(x)  # noqa: F821

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=3,
            num_runs=6,
            repeat_n=1,
            # MVP is eager: ACL graph capture is a follow-up, and
            # benchmark_with_power has no NPU graph arm (helper.py use_graph is
            # CUDA/XPU only), so `used_cuda_graph` is recorded False below.
            use_cuda_graph=False,
            allow_graph_fail=False,
        ) as results:
            pass
    finally:
        # Free device weights after each case; a packed op_list left cached
        # OOMs later shapes.
        del op_list
        del x
        import gc

        gc.collect()
        get_device_module(device).empty_cache()

    # torch_npu 2.10.0.post7 API notes (read from the pinned image's own
    # wheel): get_device_name validates the ordinal and returns the SoC model
    # name (no per-card distinction), which matches what this informational
    # column records.
    # get_device_capability is never called: it returns None unless
    # TORCH_NPU_DEVICE_CAPABILITY is set, and then a configured fake value.
    log_perf(
        item_list=[
            {
                "gemm_dtype": gemm_type,
                "m": m,
                "n": n,
                "k": k,
                "latency": results["latency_ms"] / outside_loop_count,
                # Shared vLLM gemm_perf schema column; always False while the
                # lane is eager-only.
                "used_cuda_graph": results["used_cuda_graph"],
            }
        ],
        framework="VLLM",
        version=vllm_ascend_version,
        device_name=get_device_module(device).get_device_name(device),
        op_name="gemm",
        # The vllm-ascend dispatch for an unquantized linear is not traced yet;
        # op_backend_facts.yaml records the slice as backend "unverified".
        kernel_source="vllm_default",
        perf_filename=perf_filename,
        # No NVML-equivalent on CANN: power is never sampled on this lane.
        power_stats=None,
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_gemm_test_cases()
    for test_case in test_cases[:10]:
        run_gemm(*test_case, perf_filename=PerfFile.GEMM)
