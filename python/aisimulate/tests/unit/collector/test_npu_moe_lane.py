# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU Fused-MoE lane — manifest pin, registry entry, cases, and collector module.

Verifies the Stage-2 single-rank BF16 MoE wiring for the ``vllm_npu`` lane
without requiring an Ascend device, ``torch_npu``, or ``vllm_ascend`` on CI/dev
machines.
"""

import ast
import importlib
import importlib.metadata
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from collector.case_generator import (
    get_moe_backend_test_cases,
    get_moe_quantization_modes,
    moe_model_allows_quantization,
)
from collector.framework_manifest import (
    get_collector_runtime,
    require_collector_runtime,
    resolve_op_runtime,
    validate_resolution,
)
from collector.model_cases import build_collection_case_plan
from collector.registry_types import PerfFile
from collector.vllm.registry import REGISTRY, REGISTRY_NPU, REGISTRY_XPU

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE = "collector.vllm.collect_moe_npu"


# ---------------------------------------------------------------------------
# Manifest & Registry
# ---------------------------------------------------------------------------


def test_every_registry_lane_including_npu_moe_resolves():
    assert validate_resolution() == []


def test_moe_resolves_through_the_npu_lane_and_enforces_the_pin():
    assert resolve_op_runtime("vllm_npu", "moe").version == "0.30.0rc2"

    runtime = require_collector_runtime("vllm_npu", "0.30.0rc2", requested_ops={"moe"}, wideep_ops=set())
    assert runtime.framework == "vllm_npu"

    with pytest.raises(RuntimeError, match=r"vllm_npu stock collector requires exactly 0\.30\.0rc2"):
        require_collector_runtime("vllm_npu", "0.30.0", requested_ops={"moe"}, wideep_ops=set())


def test_registry_npu_carries_moe_entry_with_standard_entrypoints():
    entries = {entry.op: entry for entry in REGISTRY_NPU}
    assert {"gemm", "moe"} <= set(entries)

    cuda_moe = next(entry for entry in REGISTRY if entry.op == "moe")
    xpu_moe = next(entry for entry in REGISTRY_XPU if entry.op == "moe")
    npu_moe = entries["moe"]

    assert npu_moe.module == MODULE
    assert (npu_moe.get_func, npu_moe.run_func, npu_moe.perf_filename) == (
        cuda_moe.get_func,
        cuda_moe.run_func,
        PerfFile.MOE,
    )
    assert npu_moe.module not in {cuda_moe.module, xpu_moe.module}
    assert npu_moe.unverified is True
    assert npu_moe.unverified_sms == ()


def test_npu_moe_collector_is_exactly_pinned_to_the_manifest_version():
    expected = f'__compat__ = "vllm_ascend=={get_collector_runtime("vllm_npu").version}"'
    source = (REPO_ROOT / f"{MODULE.replace('.', '/')}.py").read_text(encoding="utf-8")
    declarations = [line.strip() for line in source.splitlines() if line.startswith("__compat__")]
    assert declarations == [expected]


# ---------------------------------------------------------------------------
# Case partition & Consumer-key deduplication
# ---------------------------------------------------------------------------


def test_moe_vllm_npu_partition_is_defined_in_base_op_yaml(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    assert get_moe_quantization_modes("vllm_npu", sm_version=0, runtime_features={}) == ["bfloat16"]

    cases = get_moe_backend_test_cases("vllm_npu")
    assert cases
    assert {case.model_name for case in cases} == {
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B",
        "MiniMaxAI/MiniMax-M3",
        "zai-org/GLM-5",
        "deepseek-ai/DeepSeek-V3",
    }
    assert not any(case.model_name == "Qwen/Qwen3-30B-A3B" and case.tp >= 8 for case in cases)
    assert all(moe_model_allows_quantization("vllm_npu", case.model_name, "bfloat16") for case in cases)

    # Targeted model alias resolution maps onto the canonical shape owner.
    for alias, owner in (
        ("Qwen/Qwen3-30B-A3B-FP8", "Qwen/Qwen3-30B-A3B"),
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen/Qwen3-30B-A3B"),
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B"),
        ("zai-org/GLM-5.2", "zai-org/GLM-5"),
        ("deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3"),
    ):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", alias)
        targeted = get_moe_backend_test_cases("vllm_npu")
        assert targeted
        assert {case.model_name for case in targeted} == {owner}


def test_model_case_plan_for_vllm_npu_selects_moe_for_moe_models():
    plan = build_collection_case_plan(backend="vllm_npu", model_path="Qwen/Qwen3-235B-A22B")
    assert "moe" in plan.ops
    assert "gemm" in plan.ops


# ---------------------------------------------------------------------------
# Collector module
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, shape, dtype="bfloat16", device="npu:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.filled_value = None
        self.copied_from = None
        self.random_range = None
        self.zeroed = False

    def to(self, arg):
        if isinstance(arg, str) and not arg.startswith("npu"):
            return _FakeTensor(self.shape, dtype=arg, device=self.device)
        return _FakeTensor(self.shape, dtype=self.dtype, device=str(arg))

    def view(self, dtype):
        return _FakeTensor(self.shape, dtype=dtype, device=self.device)

    def fill_(self, val):
        self.filled_value = val
        return self

    def copy_(self, other):
        self.copied_from = other
        return self

    def random_(self, low, high):
        self.random_range = (low, high)
        return self

    def zero_(self):
        self.zeroed = True
        return self

    def is_floating_point(self):
        return self.dtype in ("bfloat16", "float32", "float8_e4m3fn")


@pytest.fixture
def collect_moe_npu(monkeypatch):
    """Import the collector against stubbed torch / vLLM / vllm-ascend."""
    mock_torch = MagicMock(name="torch")
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.float32 = "float32"
    mock_torch.float8_e4m3fn = "float8_e4m3fn"
    mock_torch.uint8 = "uint8"
    mock_torch.int32 = "int32"
    mock_torch.int8 = "int8"
    mock_torch.OutOfMemoryError = RuntimeError
    mock_torch.device.side_effect = lambda d: str(d)
    mock_torch.get_default_dtype.return_value = "bfloat16"
    mock_torch.randn.side_effect = lambda shape, **kw: _FakeTensor(
        shape,
        dtype=kw.get("dtype", "bfloat16"),
        device=kw.get("device", "npu:0"),
    )
    mock_torch.zeros.side_effect = lambda shape, **kw: _FakeTensor(
        (shape,) if isinstance(shape, int) else shape,
        dtype=kw.get("dtype", "float32"),
        device=kw.get("device", "npu:0"),
    )
    mock_torch.full.side_effect = lambda shape, val, **kw: _FakeTensor(
        shape,
        dtype=kw.get("dtype", "bfloat16"),
        device=kw.get("device", "npu:0"),
    )

    @contextmanager
    def _no_grad():
        yield

    mock_torch.no_grad = _no_grad

    for name in (
        "torch",
        "vllm",
        "vllm.config",
        "vllm.forward_context",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.layer",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.workspace",
        "vllm.version",
    ):
        monkeypatch.setitem(sys.modules, name, mock_torch if name == "torch" else MagicMock(name=name))

    sys.modules["vllm.version"].__version__ = "0.30.0"

    @contextmanager
    def _dummy_cm(*_a, **_kw):
        yield

    sys.modules["vllm.config"].set_current_vllm_config = _dummy_cm
    sys.modules["vllm.forward_context"].set_forward_context = _dummy_cm

    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda pkg: "0.30.0rc2" if pkg == "vllm-ascend" else real_version(pkg),
    )

    utils = MagicMock(name="utils")
    monkeypatch.setitem(sys.modules, "collector.vllm.utils", utils)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    module = importlib.import_module(MODULE)
    try:
        yield module
    finally:
        sys.modules.pop(MODULE, None)


def test_get_moe_test_cases_produces_unique_bf16_consumer_keys(collect_moe_npu, monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    cases = collect_moe_npu.get_moe_test_cases()
    assert cases
    assert {case[0] for case in cases} == {"bfloat16"}
    assert all(not (case[6] > 1 and case[7] > 1) for case in cases)
    assert all(case[9] == "power_law" for case in cases)

    # Every token-expanded persisted consumer key is unique across the full NPU MoE plan.
    consumer_keys = [
        (case[0], f"{case[9]}_{case[10]}", case[4], case[5], case[2], case[3], case[6], case[7], num_tokens)
        for case in cases
        for num_tokens in case[1]
    ]
    assert len(consumer_keys) == len(set(consumer_keys))


def test_moe_capacity_guard_refuses_oversized_footprint():
    source = (REPO_ROOT / f"{MODULE.replace('.', '/')}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_moe_case_does_not_fit")
        or (isinstance(node, ast.Assign) and node.targets[0].id.startswith("_MOE_"))
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), MODULE, "exec"), namespace)
    does_not_fit = namespace["_moe_case_does_not_fit"]

    # Qwen3-30B-A3B (E=128, H=2048, I=768, K=8) easily fits on a 96 GiB 950DT card.
    assert does_not_fit("bfloat16", 16384, 2048, 768, 8, 128, 1, 1, 96 * 2**30) is None
    # DeepSeek-V4-Pro (E=384, H=7168, I=3072, K=6) at TP1/EP1 in BF16 (~86 GiB init peak)
    # exceeds the 80% budget of a 96 GiB card (76.8 GiB).
    reason = does_not_fit("bfloat16", 16384, 7168, 3072, 6, 384, 1, 1, 96 * 2**30)
    assert reason is not None
    assert reason.startswith("FIXME(kernel-limit): does not fit:")


def test_initialize_routed_expert_parameters_uses_bf16_buffer_before_cast_for_non_bf16(collect_moe_npu):
    bf16_w = _FakeTensor((4, 16, 8), dtype="bfloat16")
    fp8_w = _FakeTensor((4, 16, 8), dtype="float8_e4m3fn")
    uint8_scale = _FakeTensor((4, 16, 1), dtype="uint8")

    routed_experts = SimpleNamespace(
        named_parameters=lambda recurse=False: [
            ("w13_weight", bf16_w),
            ("w2_weight", fp8_w),
            ("w13_weight_scale", uint8_scale),
        ]
    )

    collect_moe_npu._initialize_routed_expert_parameters(routed_experts, "npu:0")

    # BF16 weight fills directly; FP8 weight initializes a BF16 tensor then casts/copies
    # (never calling fill_ on DT_FLOAT8_E4M3FN).
    assert bf16_w.filled_value == 0.01
    assert fp8_w.filled_value is None
    assert fp8_w.copied_from is not None
    assert fp8_w.copied_from.dtype == "float8_e4m3fn"


def test_run_moe_torch_defaults_to_npu_0_and_refuses_non_bf16_and_tp_ep(collect_moe_npu):
    signature = inspect.signature(collect_moe_npu.run_moe_torch)
    assert signature.parameters["device"].default == "npu:0"

    with pytest.raises(RuntimeError, match="vllm_npu moe collects bfloat16 only"):
        collect_moe_npu.run_moe_torch(
            "fp8_block",
            [1, 8],
            2048,
            768,
            8,
            128,
            1,
            1,
            "Qwen/Qwen3-30B-A3B",
            perf_filename=PerfFile.MOE,
        )

    with pytest.raises(ValueError, match="does not combine logical TP and EP"):
        collect_moe_npu.run_moe_torch(
            "bfloat16",
            [1, 8],
            2048,
            768,
            8,
            128,
            2,
            2,
            "Qwen/Qwen3-30B-A3B",
            perf_filename=PerfFile.MOE,
        )


def test_run_moe_torch_executes_single_rank_allgather_path_and_logs_standard_moe_schema(
    collect_moe_npu,
    monkeypatch,
):
    logged_calls = []
    benchmark_kwargs = {}
    observed = {}

    device_module = MagicMock()
    device_module.get_device_properties.return_value.total_memory = 96 * 2**30
    device_module.get_device_name.return_value = "Ascend950DT"

    fake_extra_ctx = SimpleNamespace(
        moe_comm_type=None,
        moe_comm_method=None,
        in_profile_run=True,
        use_mega_moe=True,
    )
    fake_comm_type = SimpleNamespace(ALLGATHER="ALLGATHER")
    fake_comm_impl = object()

    ascend_fwd_mod = MagicMock(_EXTRA_CTX=fake_extra_ctx, MoECommType=fake_comm_type)
    comm_mod = MagicMock()
    comm_mod.AllGatherCommImpl.return_value = fake_comm_impl
    comm_mod.get_moe_comm_method.return_value = fake_comm_impl
    comm_mod._MoECommMethods = {}
    comm_mod._moe_config_key.side_effect = lambda comm_type, cfg: (comm_type, id(cfg))

    ascend_fused_moe_mod = MagicMock()
    ascend_fused_moe_mod.get_mc2_group = lambda: (_ for _ in ()).throw(
        AssertionError("get_mc2_group must be stubbed on single-rank EP>1")
    )

    vllm_ascend_pkg = MagicMock()
    vllm_ascend_ops_pkg = MagicMock()
    vllm_ascend_fused_moe_pkg = MagicMock()
    vllm_ascend_pkg.ops = vllm_ascend_ops_pkg
    vllm_ascend_ops_pkg.fused_moe = vllm_ascend_fused_moe_pkg
    vllm_ascend_fused_moe_pkg.fused_moe = ascend_fused_moe_mod
    vllm_ascend_fused_moe_pkg.moe_comm_method = comm_mod

    for mod_name, mod_obj in (
        ("vllm_ascend", vllm_ascend_pkg),
        ("vllm_ascend.ops", vllm_ascend_ops_pkg),
        ("vllm_ascend.ops.fused_moe", vllm_ascend_fused_moe_pkg),
        ("vllm_ascend.ops.fused_moe.fused_moe", ascend_fused_moe_mod),
        ("vllm_ascend.ops.fused_moe.moe_comm_method", comm_mod),
        ("vllm_ascend.ascend_config", MagicMock()),
        ("vllm_ascend.ascend_forward_context", ascend_fwd_mod),
        ("vllm_ascend.utils", MagicMock()),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    w13 = _FakeTensor((64, 1536, 2048), dtype="bfloat16")
    w2 = _FakeTensor((64, 2048, 768), dtype="bfloat16")

    class FakeQuantMethod:
        def process_weights_after_loading(self, layer):
            observed["processed_layer"] = layer

    routed_experts = SimpleNamespace(
        named_parameters=lambda recurse=False: [("w13_weight", w13), ("w2_weight", w2)],
        quant_method=FakeQuantMethod(),
    )

    class FakeMoERunner:
        def __init__(self, **kwargs):
            # Verify get_mc2_group is stubbed to None inside _single_rank_allgather_comm_only.
            assert ascend_fused_moe_mod.get_mc2_group() is None
            ascend_fused_moe_mod.setup_moe_comm_method("moe_cfg")
            observed["fused_moe_kwargs"] = kwargs
            self.routed_experts = routed_experts
            self.moe_config = "moe_cfg"

        def to(self, dev):
            observed["module_device"] = dev
            return self

        def eval(self):
            observed["eval_called"] = True

        def requires_grad_(self, flag):
            observed["requires_grad"] = flag

        def __call__(self, hidden_states, router_logits):
            observed.setdefault("forward_calls", []).append(
                (
                    hidden_states.shape,
                    router_logits.shape,
                    fake_extra_ctx.moe_comm_type,
                    fake_extra_ctx.moe_comm_method,
                    fake_extra_ctx.in_profile_run,
                    fake_extra_ctx.use_mega_moe,
                )
            )

    sys.modules["vllm.model_executor.layers.fused_moe.layer"].FusedMoEFactory = FakeMoERunner
    sys.modules["vllm.forward_context"].get_forward_context.return_value = SimpleNamespace(moe_layer_index=-1)

    @contextmanager
    def fake_benchmark(**kwargs):
        benchmark_kwargs.update(kwargs)
        kwargs["kernel_func"]()
        yield {"latency_ms": 10.0, "used_cuda_graph": False}

    monkeypatch.setattr(
        collect_moe_npu,
        "setup_distributed",
        lambda device: observed.setdefault("setup_distributed_device", device),
    )
    monkeypatch.setattr(collect_moe_npu, "get_device_module", lambda device=None: device_module)
    monkeypatch.setattr(
        collect_moe_npu,
        "power_law_logits_v3",
        lambda num_tokens, num_experts, topk, ep, alpha: _FakeTensor((num_tokens, num_experts), dtype="float32"),
    )
    monkeypatch.setattr(collect_moe_npu, "benchmark_with_power", fake_benchmark)
    monkeypatch.setattr(collect_moe_npu, "log_perf", lambda **kwargs: logged_calls.append(kwargs))

    collect_moe_npu.run_moe_torch(
        "bfloat16",
        [4, 16],
        2048,
        768,
        8,
        128,
        1,
        2,
        "Qwen/Qwen3-30B-A3B",
        distributed="power_law",
        power_law_alpha=1.01,
        perf_filename=PerfFile.MOE,
    )

    assert observed["setup_distributed_device"] == "npu:0"
    assert observed["fused_moe_kwargs"]["num_experts"] == 128
    assert observed["fused_moe_kwargs"]["top_k"] == 8
    assert observed["fused_moe_kwargs"]["tp_size"] == 2  # logical_parallel_size = moe_ep_size
    assert observed["fused_moe_kwargs"]["quant_config"] is None
    assert observed["processed_layer"] is routed_experts
    assert w13.filled_value == 0.01
    assert w2.filled_value == 0.01

    # 2 token counts x 5 power_law iterations = 10 forward calls, all on ALLGATHER.
    assert len(observed["forward_calls"]) == 10
    assert all(call[2] == "ALLGATHER" and call[3] is fake_comm_impl for call in observed["forward_calls"])
    assert all(call[4] is False and call[5] is False for call in observed["forward_calls"])

    assert benchmark_kwargs["use_cuda_graph"] is False
    assert benchmark_kwargs["allow_graph_fail"] is False
    assert len(logged_calls) == 2
    for idx, num_tokens in enumerate((4, 16)):
        logged = logged_calls[idx]
        assert logged["framework"] == "VLLM"
        assert logged["version"] == "0.30.0rc2"
        assert logged["device_name"] == "Ascend950DT"
        assert logged["op_name"] == "moe"
        assert logged["kernel_source"] == "vllm_default"
        assert logged["perf_filename"] is PerfFile.MOE
        assert logged["power_stats"] is None
        assert logged["item_list"] == [
            {
                "moe_dtype": "bfloat16",
                "num_tokens": num_tokens,
                "hidden_size": 2048,
                "inter_size": 768,
                "topk": 8,
                "num_experts": 128,
                "moe_tp_size": 1,
                "moe_ep_size": 2,
                "distribution": "power_law_1.01",
                "latency": pytest.approx(2.0),  # 10.0 ms / 5 power_law iterations
            }
        ]
    device_module.get_device_capability.assert_not_called()
