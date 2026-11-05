# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM Fused-MoE collector for Huawei Ascend NPU devices (vllm-ascend).

The NPU counterpart to the CUDA vLLM MoE collector: it runs vllm-ascend's
out-of-tree ``FusedMoEFactory`` -> ``AscendMoERunner`` / ``AscendRoutedExperts``
path (`vllm_ascend/patch/platform/patch_fused_moe.py` and
`vllm_ascend/ops/fused_moe/fused_moe.py` @ v0.30.0rc2), expands YAML-backed
`moe_vllm_npu` cases, and logs perf rows through the device-agnostic helpers.

Stage-2 MVP scope and execution boundary:
- Precision: ``bfloat16`` (`AscendUnquantizedFusedMoEMethod`, `QuantType.NONE`)
  only; eager execution (`use_cuda_graph=False`), no power sampling.
- Single-rank local expert compute: `moe_perf.parquet` prices per-rank local
  expert compute (routing top-k + `npu_moe_init_routing_v2` permute +
  `npu_grouped_matmul` GMM1 + SwiGLU/activation + `npu_grouped_matmul` GMM2 +
  `npu_moe_token_unpermute`), while cross-rank TP/EP collective communication is
  modeled separately by the SDK communication / MoE-dispatch tables.
- Multi-rank HCCL `MC2` / `AlltoAll` / `FusedMC2` paths are NOT exercised or
  claimed as verified by this single-rank collector. In `vllm-ascend`,
  `AscendMoERunner.__init__` unconditionally calls `get_mc2_group()` and
  `setup_moe_comm_method(self.moe_config)` (which instantiates `MC2CommImpl`
  and `AlltoAllCommImpl` when `ep_size > 1`), and `AscendRoutedExperts.forward_impl`
  reads `_EXTRA_CTX.moe_comm_type`. On a single physical rank
  (`setup_distributed(device)` with `world_size=1`), `_single_rank_allgather_comm_only`
  binds `AllGatherCommImpl` (`MoECommType.ALLGATHER`, whose `prepare`/`finalize`
  are local no-ops when `dp_size=1, pcp_size=1, is_sequence_parallel=False` and
  whose `TokenDispatcherWithAllGather` filters tokens to the local expert shard
  `[0, num_experts // moe_ep_size)` via `expert_map`).

Deviation note (serving parity): vllm-ascend defaults to `FULL_AND_PIECEWISE`
ACLGraph capture in serving (`vllm_ascend/platform.py`), whereas this MVP
collector runs in eager mode (`use_cuda_graph=False`) until `benchmark_with_power`
adds an `NPUGraph` capture arm.

The pinned version is the vllm-ascend release, not base vLLM: vllm-ascend
ships the Ascend MoE runner and kernels, so it is the kernel-source identity
of these rows and the version axis of the parquet directory (manifest key
`vllm_npu`).
"""

__compat__ = "vllm_ascend==0.30.0rc2"

import contextlib
import gc
import importlib.metadata as im
from types import SimpleNamespace

import torch
from collector.case_generator import (
    get_moe_backend_test_cases,
    get_moe_quantization_modes,
    get_moe_quantization_module_config,
    moe_model_allows_quantization,
)
from collector.helper import (
    balanced_logits,
    benchmark_with_power,
    get_device_module,
    log_perf,
    power_law_logits_v3,
)
from collector.vllm.collect_moe import _resolve_moe_runtime_config
from collector.vllm.utils import setup_distributed
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.worker.workspace import init_workspace_manager

vllm_ascend_version = im.version("vllm-ascend")

# Max device-memory fraction for a single MoE benchmark case (weights +
# transient post-load transpose buffer + forward routing/activation buffers).
_MOE_MEM_BUDGET_FRACTION = 0.80


def _moe_case_does_not_fit(
    moe_type: str,
    max_num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    total_memory: int,
) -> str | None:
    """Return a classified capacity refusal reason if the case exceeds device memory, else None.

    Follows `.claude/rules/collector/layer_permissions.md` ("Capacity: refuse at
    run time, before allocation"): plans do not depend on the device, and the
    worker refuses a shape that exceeds the live device memory budget with the
    `FIXME(kernel-limit): does not fit` prefix before allocating weights.
    """
    local_experts = max(num_experts // max(moe_ep_size, 1), 1)
    local_inter = max(inter_size // max(moe_tp_size, 1), 1)
    n_elem = 3 * local_experts * hidden_size * local_inter

    # BF16 steady-state weights (w13: 2*I*H, w2: H*I, 2 bytes/elem) and
    # process_weights_after_loading transpose copy coexistence (~3.4 bytes/elem).
    if moe_type == "bfloat16":
        steady_weight_bytes = 2 * n_elem
        init_peak_bytes = int(3.4 * n_elem)
    else:
        steady_weight_bytes = 2 * n_elem
        init_peak_bytes = 4 * n_elem

    t = max(int(max_num_tokens), 1)
    tk = t * topk
    router_bytes = 5 * t * num_experts * 4
    io_bytes = 4 * t * hidden_size
    # Permuted input + GMM1 gate_up (bf16) + SwiGLU out + GMM2 out.
    act_bytes = tk * (4 * hidden_size + 6 * local_inter + 8)
    peak_bytes = max(init_peak_bytes, steady_weight_bytes + router_bytes + io_bytes + act_bytes)

    allowed_bytes = int(total_memory * _MOE_MEM_BUDGET_FRACTION)
    if peak_bytes <= allowed_bytes:
        return None
    return (
        f"FIXME(kernel-limit): does not fit: MoE case needs {peak_bytes / 2**30:.1f} GiB, "
        f"the device allows {allowed_bytes / 2**30:.1f} GiB ({_MOE_MEM_BUDGET_FRACTION:g} of its memory)"
    )


def _ensure_ascend_moe_registered(vllm_config: VllmConfig) -> None:
    """Run vllm-ascend worker patches and OOT CustomOp registrations before importing FusedMoEFactory.

    Serving population site: `vllm_ascend/worker/worker_v1.py` (`NPUWorker.__init__`)
    calls `adapt_patch(is_global_patch=True)`, `adapt_patch(is_global_patch=False)`
    (which runs `vllm_ascend/patch/platform/patch_fused_moe.py` to replace
    `FusedMoEFactory` with `_ascend_FusedMoE`), `register_dummy_fusion_op()`,
    `init_ascend_config(vllm_config)` (required by `AscendRoutedExperts.__init__`),
    and `register_ascend_customop(vllm_config)` (binds `MoERunner` ->
    `AscendMoERunner` and `RoutedExperts` -> `AscendRoutedExperts` via
    `CustomOp.register_oot`) @ v0.30.0rc2.
    """
    from vllm_ascend import ops as ascend_ops
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.utils import adapt_patch, enable_custom_op, register_ascend_customop

    adapt_patch(is_global_patch=True)
    adapt_patch(is_global_patch=False)
    if hasattr(ascend_ops, "register_dummy_fusion_op"):
        ascend_ops.register_dummy_fusion_op()
    init_ascend_config(vllm_config)
    register_ascend_customop(vllm_config)
    enable_custom_op()


@contextlib.contextmanager
def _single_rank_allgather_comm_only():
    """Bind single-rank MoE execution to AllGatherCommImpl without multi-rank HCCL communicators.

    In `vllm_ascend/ops/fused_moe/fused_moe.py:98,114` and
    `vllm_ascend/ops/fused_moe/moe_comm_method.py:61-66` @ v0.30.0rc2,
    `AscendMoERunner.__init__` calls `get_mc2_group()` and `setup_moe_comm_method`,
    which unconditionally instantiates `MC2CommImpl` and `AlltoAllCommImpl` when
    `moe_config.ep_size > 1`. On a single-rank collector process (`world_size=1`),
    this context manager stubs `get_mc2_group` to `None` and registers only
    `AllGatherCommImpl(moe_config)` under `MoECommType.ALLGATHER` so local expert
    sharding (`num_local_experts = num_experts // moe_ep_size` via `expert_map`
    and `TokenDispatcherWithAllGather`) executes on-device without multi-rank HCCL.
    """
    import vllm_ascend.ops.fused_moe.fused_moe as ascend_fused_moe_mod
    import vllm_ascend.ops.fused_moe.moe_comm_method as comm_mod
    from vllm_ascend.ascend_forward_context import MoECommType

    orig_setup_comm = comm_mod.setup_moe_comm_method
    orig_fused_setup_comm = getattr(ascend_fused_moe_mod, "setup_moe_comm_method", None)
    orig_get_mc2_group = getattr(ascend_fused_moe_mod, "get_mc2_group", None)

    def _setup_allgather_only(moe_config):
        comm_impl = comm_mod.AllGatherCommImpl(moe_config)
        if hasattr(comm_mod, "_MoECommMethods") and hasattr(comm_mod, "_moe_config_key"):
            comm_mod._MoECommMethods[comm_mod._moe_config_key(MoECommType.ALLGATHER, moe_config)] = comm_impl
        return comm_impl

    comm_mod.setup_moe_comm_method = _setup_allgather_only
    if orig_fused_setup_comm is not None:
        ascend_fused_moe_mod.setup_moe_comm_method = _setup_allgather_only
    if orig_get_mc2_group is not None:
        ascend_fused_moe_mod.get_mc2_group = lambda: None
    try:
        yield
    finally:
        comm_mod.setup_moe_comm_method = orig_setup_comm
        if orig_fused_setup_comm is not None:
            ascend_fused_moe_mod.setup_moe_comm_method = orig_fused_setup_comm
        if orig_get_mc2_group is not None:
            ascend_fused_moe_mod.get_mc2_group = orig_get_mc2_group


def _initialize_routed_expert_parameters(routed_experts, device) -> None:
    """Populate synthetic expert parameters safely on Ascend NPU.

    Lesson from Ascend 950DT GEMM validation (`110511d46`): CANN
    `aclnnInplaceFillScalar` rejects low-precision dtypes such as
    `DT_FLOAT8_E4M3FN` when `parameter.fill_(...)` is called directly. Always
    initialize floating-point parameters in `torch.bfloat16` on device and
    copy/cast into `parameter` (`torch.full(..., dtype=torch.bfloat16).to(parameter.dtype)`).
    """
    with torch.no_grad():
        for name, parameter in routed_experts.named_parameters(recurse=False):
            e8m0_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
            if parameter.dtype in (torch.uint8, e8m0_dtype):
                if "scale" in name:
                    parameter.view(torch.uint8).fill_(127)
                else:
                    parameter.view(torch.uint8).random_(0, 256)
            elif parameter.dtype == torch.int32:
                parameter.random_(0, 16)
            elif parameter.dtype == torch.int8:
                parameter.random_(-8, 8)
            elif parameter.is_floating_point():
                fill_val = 1.0 if "scale" in name else 0.01
                if parameter.dtype == torch.bfloat16:
                    parameter.fill_(fill_val)
                else:
                    bf16_buf = torch.full(
                        parameter.shape,
                        fill_val,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    parameter.copy_(bf16_buf.to(parameter.dtype))
            else:
                parameter.zero_()


def get_moe_npu_test_cases():
    return get_moe_backend_test_cases("vllm_npu")


def get_moe_test_cases():
    """Generate MoE test cases for the vllm_npu lane from cases/base_ops/moe.yaml."""
    enabled_moe_types = get_moe_quantization_modes(
        "vllm_npu",
        sm_version=0,
        runtime_version=vllm_ascend_version,
        runtime_features={"torch_fp8_e4m3fn": hasattr(torch, "float8_e4m3fn")},
    )
    if not enabled_moe_types:
        raise RuntimeError("collector/cases/base_ops/moe.yaml must define moe_vllm_npu quantization_modes")

    test_cases = []
    seen_consumer_keys: dict[tuple, str] = {}

    for common_moe_testcase in get_moe_npu_test_cases():
        if common_moe_testcase.token_expert_distribution != "power_law":
            continue

        # vLLM flattens TP x DP into EP when expert parallelism is enabled, so
        # simultaneous moe_tp_size > 1 and moe_ep_size > 1 is not representable.
        if common_moe_testcase.tp > 1 and common_moe_testcase.ep > 1:
            continue

        model_name = common_moe_testcase.model_name
        for moe_type in enabled_moe_types:
            if not moe_model_allows_quantization("vllm_npu", model_name, moe_type):
                continue

            consumer_key = (
                moe_type,
                tuple(common_moe_testcase.num_tokens_list),
                common_moe_testcase.hidden_size,
                common_moe_testcase.inter_size,
                common_moe_testcase.topk,
                common_moe_testcase.num_experts,
                common_moe_testcase.tp,
                common_moe_testcase.ep,
                common_moe_testcase.token_expert_distribution,
                common_moe_testcase.power_law_alpha,
            )
            owner = seen_consumer_keys.get(consumer_key)
            if owner is not None:
                raise ValueError(
                    f"vllm_npu MoE rows {owner!r} and {model_name!r} collide on persisted consumer key {consumer_key}"
                )
            seen_consumer_keys[consumer_key] = model_name

            test_cases.append(
                [
                    moe_type,
                    common_moe_testcase.num_tokens_list,
                    common_moe_testcase.hidden_size,
                    common_moe_testcase.inter_size,
                    common_moe_testcase.topk,
                    common_moe_testcase.num_experts,
                    common_moe_testcase.tp,
                    common_moe_testcase.ep,
                    common_moe_testcase.model_name,
                    common_moe_testcase.token_expert_distribution,
                    common_moe_testcase.power_law_alpha,
                ]
            )

    return test_cases


def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0.0,
    *,
    perf_filename,
    device="npu:0",
):
    """Run vLLM-Ascend FusedMoE performance benchmarking on a single NPU rank."""
    if moe_type != "bfloat16":
        # Stage-2 MVP collects bfloat16 only; fail closed so no other dtype is
        # benchmarked as unquantized bf16 under a mislabeled moe_dtype.
        raise RuntimeError(f"vllm_npu moe collects bfloat16 only, got {moe_type!r}")

    if moe_tp_size > 1 and moe_ep_size > 1:
        raise ValueError("vLLM MoE collector does not combine logical TP and EP")

    # Lesson from Ascend 950DT GEMM/Attention validation (`17ca2a908`):
    # vLLM / vllm-ascend layers and workspace manager query distributed groups
    # (`get_tp_group()`, `get_ep_group()`, `get_dp_group()`), so single-rank
    # FileStore distributed init must run before any layer/workspace setup.
    setup_distributed(device)

    dev_mod = get_device_module(device)
    dev_mod.set_device(device)
    init_workspace_manager(torch.device(device))
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)

    total_memory = dev_mod.get_device_properties(device).total_memory
    reason = _moe_case_does_not_fit(
        moe_type=moe_type,
        max_num_tokens=max(num_tokens_lists),
        hidden_size=hidden_size,
        inter_size=inter_size,
        topk=topk,
        num_experts=num_experts,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        total_memory=total_memory,
    )
    if reason is not None:
        raise RuntimeError(reason)

    gc.collect()
    dev_mod.empty_cache()

    module_config = get_moe_quantization_module_config("vllm_npu", moe_type, model_name=model_name)
    runtime_config = _resolve_moe_runtime_config(model_name, module_config)
    if runtime_config["use_grouped_topk"]:
        num_expert_group = runtime_config["num_expert_group"]
        topk_group = runtime_config["topk_group"]
        if num_experts <= num_expert_group or num_experts % num_expert_group != 0:
            raise ValueError(
                f"vLLM grouped-topk model {model_name!r} cannot divide {num_experts} experts "
                f"into {num_expert_group} groups"
            )
        experts_per_group = num_experts // num_expert_group
        if topk > topk_group * experts_per_group:
            raise ValueError(
                f"vLLM grouped-topk model {model_name!r} requests topk={topk}, but "
                f"topk_group={topk_group} exposes only {topk_group * experts_per_group} experts"
            )

    quant_config = None
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(model_type=""),
        hf_config=SimpleNamespace(model_type=""),
        model="collector_dummy",
        quantization_config=None,
        runner_type=None,
    )
    # AscendMoEScheme subclasses read get_current_vllm_config().quant_config;
    # keep vllm_config.quant_config synchronized with the FusedMoE argument.
    vllm_config.quant_config = quant_config
    vllm_config.scheduler_config.max_num_batched_tokens = max(num_tokens_lists)
    vllm_config.parallel_config.enable_expert_parallel = moe_ep_size > 1
    logical_parallel_size = moe_ep_size if moe_ep_size > 1 else moe_tp_size

    _ensure_ascend_moe_registered(vllm_config)

    # Import FusedMoEFactory AFTER _ensure_ascend_moe_registered() has applied
    # adapt_patch(), which replaces FusedMoEFactory with _ascend_FusedMoE.
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory as FusedMoE
    except ImportError:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

    e_score_correction_bias = (
        torch.zeros(num_experts, dtype=torch.float32, device=device) if runtime_config["use_routing_bias"] else None
    )
    router_logits_dtype = torch.float32 if runtime_config["router_logits_float32"] else None

    situ_kwargs = {}
    if runtime_config["activation"] == "situ":
        from inspect import signature as _sig

        if "activation_situ_beta" in _sig(FusedMoE).parameters:
            situ_kwargs = {
                "activation_situ_beta": runtime_config["activation_situ_beta"],
                "activation_situ_linear_beta": runtime_config["activation_situ_linear_beta"],
            }

    moe_module = None
    try:
        with set_current_vllm_config(vllm_config), _single_rank_allgather_comm_only():
            moe_module = FusedMoE(
                num_experts=num_experts,
                top_k=topk,
                hidden_size=hidden_size,
                intermediate_size=inter_size,
                params_dtype=torch.bfloat16,
                renormalize=runtime_config["renormalize"],
                use_grouped_topk=runtime_config["use_grouped_topk"],
                num_expert_group=runtime_config["num_expert_group"],
                topk_group=runtime_config["topk_group"],
                quant_config=quant_config,
                tp_size=logical_parallel_size,
                dp_size=1,
                pcp_size=1,
                prefix="collector_moe",
                scoring_func=runtime_config["scoring_func"],
                routed_scaling_factor=runtime_config["routed_scaling_factor"],
                swiglu_limit=runtime_config["swiglu_limit"],
                e_score_correction_bias=e_score_correction_bias,
                apply_router_weight_on_input=runtime_config["apply_router_weight_on_input"],
                has_bias=runtime_config["has_bias"],
                activation=runtime_config["activation"],
                router_logits_dtype=router_logits_dtype,
                apply_routed_scale_to_output=runtime_config["apply_routed_scale_to_output"],
                **situ_kwargs,
            )

            try:
                moe_module.to(torch.device(device))
            except NotImplementedError:
                moe_module = moe_module.to_empty(device=torch.device(device))

            moe_module.eval()
            moe_module.requires_grad_(False)

            routed_experts = moe_module.routed_experts
            _initialize_routed_expert_parameters(routed_experts, device)

            quant_method = routed_experts.quant_method
            if quant_method is not None and hasattr(quant_method, "process_weights_after_loading"):
                quant_method.process_weights_after_loading(routed_experts)

            comm_impl = get_moe_comm_method(MoECommType.ALLGATHER, moe_module.moe_config)
            device_name = dev_mod.get_device_name(device)
            # Reuses the registered "vllm_default" label (mapped to backend: unverified
            # in collector/kernel_source_backends.yaml) until on-device Ascend 950DT
            # campaigns trace and register the exact Ascend MoE kernel labels.
            source = "vllm_default"

            for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
                hidden_states = torch.randn([num_tokens, hidden_size], dtype=torch.bfloat16, device=device)
                num_iter = 5 if distributed == "power_law" else 1
                logits_dtype = router_logits_dtype or torch.bfloat16

                if distributed == "power_law":
                    previous_dtype = torch.get_default_dtype()
                    torch.set_default_dtype(torch.float32)
                    try:
                        router_logits_list = [
                            power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha)
                            .to(logits_dtype)
                            .to(device)
                            for _ in range(num_iter)
                        ]
                    finally:
                        torch.set_default_dtype(previous_dtype)
                elif distributed == "balanced":
                    router_logits_list = [balanced_logits(num_tokens, num_experts, topk).to(logits_dtype).to(device)]
                else:
                    raise ValueError(f"Unsupported distributed mode: {distributed}")

                num_warmups = 1 if distributed == "power_law" else 3
                num_runs = 1 if distributed == "power_law" else 6

                def run_single_iteration(
                    _moe_module=moe_module,
                    _hidden_states=hidden_states,
                    _router_logits_list=router_logits_list,
                    _comm_impl=comm_impl,
                ):
                    for router_logits in _router_logits_list:
                        forward_context = get_forward_context()
                        if forward_context is not None:
                            forward_context.moe_layer_index = 0
                        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
                        _EXTRA_CTX.moe_comm_method = _comm_impl
                        _EXTRA_CTX.in_profile_run = False
                        _EXTRA_CTX.use_mega_moe = False
                        _moe_module(_hidden_states, router_logits)

                try:
                    with (
                        set_forward_context({}, vllm_config),
                        benchmark_with_power(
                            device=device,
                            kernel_func=run_single_iteration,
                            num_warmups=num_warmups,
                            num_runs=num_runs,
                            repeat_n=1,
                            # MVP is eager: ACLGraph capture is a follow-up, and
                            # benchmark_with_power has no NPUGraph arm yet.
                            use_cuda_graph=False,
                            allow_graph_fail=False,
                        ) as results,
                    ):
                        pass
                except torch.OutOfMemoryError:
                    if num_tokens_idx > 0:
                        break
                    raise
                finally:
                    del hidden_states, router_logits_list

                latency = results["latency_ms"] / num_iter

                log_perf(
                    item_list=[
                        {
                            "moe_dtype": moe_type,
                            "num_tokens": num_tokens,
                            "hidden_size": hidden_size,
                            "inter_size": inter_size,
                            "topk": topk,
                            "num_experts": num_experts,
                            "moe_tp_size": moe_tp_size,
                            "moe_ep_size": moe_ep_size,
                            "distribution": (
                                "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed
                            ),
                            "latency": latency,
                        }
                    ],
                    framework="VLLM",
                    version=vllm_ascend_version,
                    device_name=device_name,
                    op_name="moe",
                    kernel_source=source,
                    perf_filename=perf_filename,
                    # No NVML-equivalent on CANN: power is never sampled on this lane.
                    power_stats=None,
                )
    finally:
        del moe_module
        gc.collect()
        dev_mod.empty_cache()


if __name__ == "__main__":
    import traceback

    from collector.registry_types import PerfFile

    test_cases = get_moe_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    for test_case in test_cases[:4]:
        print(f"Running test case: {test_case}")
        try:
            run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
        except Exception as error:
            traceback.print_exc()
            print(f"Test case failed ({type(error).__name__}): {test_case}")
