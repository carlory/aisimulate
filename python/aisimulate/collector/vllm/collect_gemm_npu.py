# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM GEMM collector for Huawei Ascend NPU devices (vllm-ascend).

The NPU counterpart to the CUDA vLLM GEMM collector: it builds
RowParallelLinear layers, prepares supported MXFP4 (`mxfp4` ->
`W4A4_MXFP4`) paths, expands YAML-backed matrix shapes and logs perf
rows through the device-agnostic helpers. Eager only, no power sampling.

The pinned version is the vllm-ascend release, not base vLLM: vllm-ascend
ships the Ascend kernels, so it is the kernel-source identity of these rows
and the version axis of the parquet directory (manifest key ``vllm_npu``).
"""

__compat__ = "vllm_ascend==0.30.0rc2"

import importlib.metadata as im
import os
import sys

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
from vllm_ascend.ascend_config import init_ascend_config
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
    plus mxfp4 (W4A4_MXFP4) bf16-to-mxfp4 quantization staging (n*k*3).

    ``copies=0`` returns just the copy-independent portion (input + staging).
    Over-reserves for quantized paths (weight/output sized as bf16) on purpose.
    """
    input_bytes = m * k * 2
    per_copy = (n * k + m * n) * 2
    footprint = input_bytes + per_copy * max(copies, 0)
    if gemm_type == "mxfp4":
        footprint += n * k * 3
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
    if gemm_type not in ("bfloat16", "mxfp4"):
        # The YAML partition declares bfloat16 and mxfp4 (W4A4_MXFP4);
        # anything else would be benchmarked as unquantized bf16 under the
        # wrong dtype label.
        raise RuntimeError(f"vllm_npu gemm collects bfloat16 and mxfp4 only, got {gemm_type!r}")

    setup_distributed(device)

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    get_device_module(device).set_device(device)
    # TODO: on-device campaign — NPU visibility is governed by
    # ASCEND_RT_VISIBLE_DEVICES (CUDA_VISIBLE_DEVICES has no effect);
    # multi-card parallel collection must pin its workers with the former.

    x = torch.randn((m, k), dtype=dtype, device=torch.device(device))

    if gemm_type == "mxfp4":
        # On Ascend 950DT, vllm-ascend registers AscendW4A4MXFP4DynamicLinearMethod
        # under ("W4A4_MXFP4", "linear") (vllm_ascend/quantization/methods/w4a4/
        # w4a4_mxfp4.py:67-197): 4-bit FP4 E2M1 weights packed 2 per uint8 byte
        # (n, k // 2) + 8-bit E8M0 scale (n, ceil(k / 32)) with group_size=32,
        # dynamic activation quantization via npu_dynamic_mx_quant(dst_type=
        # float4_e2m1fn_x2, round_mode="round") and npu_quant_matmul with
        # x1_dtype=x2_dtype=float4_e2m1fn_x2, scale_dtype=float8_e8m0fnu,
        # group_sizes=[1, 1, 32]. Route mxfp4 through AscendModelSlimConfig
        # ({".weight": "W4A4_MXFP4", "group_size": 32}) -> AscendLinearMethod
        # (prefix="" maps to key ".weight" in get_quant_type_for_layer).
        qc = AscendModelSlimConfig({".weight": "W4A4_MXFP4", "group_size": 32})
    else:
        qc = None

    # VllmConfig is identical across gemm cases; build once per worker and
    # reuse (construction re-scans package entry_points, which is slow).
    # Enter set_current_vllm_config and call init_ascend_config BEFORE
    # RowParallelLinear construction: AscendW4A4MXFP4DynamicLinearMethod.__init__
    # (w4a4_mxfp4.py:90) directly reads
    # get_current_vllm_config().quant_config.quant_description.
    vllm_config = _get_cached_vllm_config(dtype)
    vllm_config.quant_config = qc
    if gemm_type == "mxfp4":
        init_ascend_config(vllm_config)
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

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

        if gemm_type == "mxfp4" and hasattr(gemm, "weight"):
            with torch.no_grad():
                # Synthesize a full (n, k) BF16 weight tensor and dynamically
                # MX-quantize it to packed FP4 (float4_e2m1fn_x2, 2 values per
                # uint8 byte -> shape (n, k // 2)) + E8M0 scale (uint8,
                # group_size=32 -> shape (n, ceil(k / 32))) via
                # torch_npu.npu_dynamic_mx_quant.
                raw_weight = torch.full(
                    (n, k),
                    0.01,
                    dtype=dtype,
                    device=gemm.weight.device,
                )
                npu_mod = get_device_module(device)
                torch_npu_mod = sys.modules.get("torch_npu")
                mx_quant = (
                    getattr(torch_npu_mod, "npu_dynamic_mx_quant", None)
                    or getattr(npu_mod, "npu_dynamic_mx_quant", None)
                    or getattr(torch.ops.npu, "npu_dynamic_mx_quant", None)
                )
                fp4_dtype = (
                    getattr(torch_npu_mod, "float4_e2m1fn_x2", None)
                    or getattr(npu_mod, "float4_e2m1fn_x2", None)
                    or getattr(torch, "float4_e2m1fn_x2", gemm.weight.dtype)
                )
                if callable(mx_quant):
                    q_weight, mx_scale = mx_quant(
                        raw_weight,
                        dst_type=fp4_dtype,
                        round_mode="round",
                    )
                    gemm.weight.copy_(q_weight.view(torch.uint8).reshape(gemm.weight.shape))
                    if hasattr(gemm, "weight_scale"):
                        # CANN npu_dynamic_mx_quant emits scales paired as
                        # (n, ceil(k / 64), 2), whereas process_weights_after_loading
                        # expects the loader layout (n, ceil(k / 32)) and pairs
                        # adjacent scales itself; flatten(-2) and slice to the
                        # unpadded group count before reshaping.
                        gemm.weight_scale.copy_(
                            mx_scale.view(torch.uint8)
                            .flatten(-2)[:, : gemm.weight_scale.shape[1]]
                            .reshape(gemm.weight_scale.shape)
                        )
                else:
                    # 0x22 packs two +1.0 FP4 E2M1 nibbles (0x2 = +1.0);
                    # E8M0 exponent 127 encodes 2^(127-127) = 1.0.
                    gemm.weight.fill_(0x22)
                    if hasattr(gemm, "weight_scale"):
                        gemm.weight_scale.fill_(127)
            # Transform weight to (k // 2, n) ND transpose view (must stay ND,
            # not contiguous or FRACTAL_NZ) and weight_scale to (ceil(k/64), n, 2).
            quant_method = getattr(gemm, "quant_method", None)
            if quant_method is not None and hasattr(quant_method, "process_weights_after_loading"):
                quant_method.process_weights_after_loading(gemm)

        gemm.forward(x)  # noqa: F821  # dry run to init

        return gemm

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
