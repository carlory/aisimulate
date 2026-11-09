# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU dense attention lane — manifest resolution, registry entries, cases, module.

Verifies the context/generation attention wiring for the ``vllm_npu`` lane
without requiring an Ascend device, ``torch_npu``, or ``vllm_ascend`` on CI/dev
machines.
"""

import ast
import functools
import importlib
import importlib.metadata
import inspect
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
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
MODULE = "collector.vllm.collect_attn_npu"
ATTENTION_OPS = ("attention_context", "attention_generation")


# ---------------------------------------------------------------------------
# Manifest & Registry
# ---------------------------------------------------------------------------


def test_every_registry_lane_including_npu_attention_resolves():
    assert validate_resolution() == []


@pytest.mark.parametrize("op", ATTENTION_OPS)
def test_attention_ops_resolve_through_the_npu_lane_and_enforce_the_pin(op):
    assert resolve_op_runtime("vllm_npu", op).version == "0.30.0rc2"

    runtime = require_collector_runtime("vllm_npu", "0.30.0rc2", requested_ops={op}, wideep_ops=set())
    assert runtime.framework == "vllm_npu"

    with pytest.raises(RuntimeError, match=r"vllm_npu stock collector requires exactly 0\.30\.0rc2"):
        require_collector_runtime("vllm_npu", "0.30.0", requested_ops={op}, wideep_ops=set())


def test_registry_npu_carries_both_attention_entries_with_standard_entrypoints():
    entries = {entry.op: entry for entry in REGISTRY_NPU}
    assert {"gemm", "attention_context", "attention_generation"} <= set(entries)

    for op, perf_file in (
        ("attention_context", PerfFile.CONTEXT_ATTENTION),
        ("attention_generation", PerfFile.GENERATION_ATTENTION),
    ):
        cuda_entry = next(entry for entry in REGISTRY if entry.op == op)
        xpu_entry = next(entry for entry in REGISTRY_XPU if entry.op == op)
        npu_entry = entries[op]

        assert npu_entry.module == MODULE
        assert (npu_entry.get_func, npu_entry.run_func, npu_entry.perf_filename) == (
            cuda_entry.get_func,
            cuda_entry.run_func,
            perf_file,
        )
        assert npu_entry.module not in {cuda_entry.module, xpu_entry.module}
        assert npu_entry.unverified is True
        assert npu_entry.unverified_sms == ()


def test_npu_attention_collector_is_exactly_pinned_to_the_manifest_version():
    expected = f'__compat__ = "vllm_ascend=={get_collector_runtime("vllm_npu").version}"'
    source = (REPO_ROOT / f"{MODULE.replace('.', '/')}.py").read_text(encoding="utf-8")
    declarations = [line.strip() for line in source.splitlines() if line.startswith("__compat__")]
    assert declarations == [expected]


# ---------------------------------------------------------------------------
# Case partition
# ---------------------------------------------------------------------------


def test_npu_attention_shape_sweeps_are_defined_in_base_op_yaml():
    (npu_ctx,) = get_attention_context_shape_sweeps("vllm_npu")
    (npu_gen,) = get_attention_generation_shape_sweeps("vllm_npu")

    assert npu_ctx["batch_sizes"] == [1, 2, 4, 8, 16, 32]
    assert npu_ctx["kv_head_options"] == [1, 2, 4, 8]
    assert npu_ctx["head_dims"] == [128, 64]
    assert npu_ctx["window_sizes"] == [0, 128]
    assert [p["id"] for p in npu_ctx["head_profiles"]] == ["default_head64", "default_head128"]

    assert npu_gen["head_dims"] == [128, 64]
    assert npu_gen["window_sizes"] == [0, 128]
    assert [p["id"] for p in npu_gen["head_profiles"]] == ["default_head64", "default_head128"]


def test_model_case_plan_for_vllm_npu_selects_attention_ops_for_dense_models():
    plan = build_collection_case_plan(backend="vllm_npu", model_path="Qwen/Qwen3-32B")
    assert plan.ops == ["attention_context", "attention_generation", "gemm"]


# ---------------------------------------------------------------------------
# Collector module
# ---------------------------------------------------------------------------


def _with_exit_stack(func):
    """Stand-in for collector.vllm.utils.with_exit_stack (injects an ExitStack)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with ExitStack() as stack:
            return func(stack, *args, **kwargs)

    return wrapper


class _FakeTensor:
    def __init__(self, shape, dtype="bfloat16"):
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, idx):
        return _FakeTensor((1, *self.shape[1:]), dtype=self.dtype)

    def cpu(self):
        return self

    def tolist(self):
        return [self.shape[0]]

    def numel(self):
        return self.shape[0] if self.shape else 0


@pytest.fixture
def collect_attn_npu(monkeypatch):
    """Import the collector against stubbed torch / vLLM / vllm-ascend."""
    mock_torch = MagicMock(name="torch")
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.int32 = "int32"
    mock_torch.int64 = "int64"
    mock_torch.cat.side_effect = lambda tensors, dim=0: _FakeTensor(
        (sum(getattr(t, "shape", (1,))[0] for t in tensors), 4, 128),
        dtype="bfloat16",
    )
    mock_torch.randn.side_effect = lambda *shape, **kwargs: _FakeTensor(
        shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape,
        dtype=kwargs.get("dtype", "bfloat16"),
    )
    mock_torch.empty_like.side_effect = lambda t: _FakeTensor(getattr(t, "shape", (1, 4, 128)))

    for name in (
        "torch",
        "vllm",
        "vllm.config",
        "vllm.platforms",
        "vllm.utils",
        "vllm.utils.import_utils",
        "vllm.utils.torch_utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backend",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.registry",
        "vllm.v1.attention.selector",
    ):
        monkeypatch.setitem(sys.modules, name, mock_torch if name == "torch" else MagicMock(name=name))

    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda pkg: "0.30.0rc2" if pkg == "vllm-ascend" else real_version(pkg),
    )

    utils = MagicMock(name="utils")
    utils.with_exit_stack = _with_exit_stack

    class _BatchSpec:
        def __init__(self, seq_lens, query_lens):
            self.seq_lens = seq_lens
            self.query_lens = query_lens

    utils.BatchSpec = _BatchSpec
    monkeypatch.setitem(sys.modules, "collector.vllm.utils", utils)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    module = importlib.import_module(MODULE)
    try:
        yield module
    finally:
        sys.modules.pop(MODULE, None)


def test_context_and_generation_cases_are_bf16_kv_only(collect_attn_npu, monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    unit_cases = collect_attn_npu.get_context_attention_test_cases(if_unit_test=True)
    assert unit_cases
    assert all(case[5] is False and case[6] is True for case in unit_cases)

    ctx_cases = collect_attn_npu.get_context_attention_test_cases()
    gen_cases = collect_attn_npu.get_generation_attention_test_cases()

    assert ctx_cases
    assert gen_cases
    # Every case has use_fp8_kv_cache == False and the matching phase flag.
    assert {case[5] for case in ctx_cases} == {False}
    assert {case[6] for case in ctx_cases} == {True}
    assert {case[5] for case in gen_cases} == {False}
    assert {case[6] for case in gen_cases} == {False}
    # GQA ratio cap (<= 16) and (window > 0, head_dim == 128) exclusion hold.
    assert all(case[2] // case[3] <= 16 for case in ctx_cases + gen_cases)
    assert not any(case[7] > 0 and case[4] == 128 for case in ctx_cases + gen_cases)


def test_kv_pool_capacity_guard_refuses_oversized_pool():
    source = (REPO_ROOT / f"{MODULE.replace('.', '/')}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_kv_pool_does_not_fit")
        or (isinstance(node, ast.Assign) and node.targets[0].id.startswith("_KV_"))
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), MODULE, "exec"), namespace)
    does_not_fit = namespace["_kv_pool_does_not_fit"]

    # At block_size=128, 8 KV heads x hd128 needs 8192 * 128 * 8 * 128 * 4 = 4 GiB.
    assert does_not_fit(8, 128, 96 * 2**30, block_size=128) is None
    reason = does_not_fit(64, 128, 24 * 2**30, block_size=128)
    assert reason is not None
    assert reason.startswith("FIXME(kernel-limit): does not fit:")


def test_run_attention_torch_defaults_to_npu_0_and_refuses_fp8_kv(collect_attn_npu):
    signature = inspect.signature(collect_attn_npu.run_attention_torch.__wrapped__)
    assert signature.parameters["device"].default == "npu:0"

    with pytest.raises(RuntimeError, match="collects bfloat16 kv_cache_dtype only"):
        collect_attn_npu.run_attention_torch(
            1,
            64,
            4,
            4,
            128,
            True,
            True,
            0,
            False,
            perf_filename=PerfFile.CONTEXT_ATTENTION,
        )


@pytest.mark.parametrize(
    ("is_context_phase", "perf_file", "expected_op", "expected_isl", "expected_step", "expected_seq_len"),
    [
        (True, PerfFile.CONTEXT_ATTENTION, "context_attention", 64, 0, 64),
        (False, PerfFile.GENERATION_ATTENTION, "generation_attention", 1, 64, 65),
    ],
)
def test_run_attention_torch_logs_eager_row_and_passes_ascend_impl_contract(
    collect_attn_npu,
    monkeypatch,
    is_context_phase,
    perf_file,
    expected_op,
    expected_isl,
    expected_step,
    expected_seq_len,
):
    logged = {}
    benchmark_kwargs = {}
    observed = {}

    device_module = MagicMock()
    device_module.get_device_properties.return_value.total_memory = 96 * 2**30
    device_module.get_device_name.return_value = "Ascend950DT"

    class FakeImpl:
        def __init__(
            self,
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks=None,
        ):
            observed["impl_kwargs"] = {
                "num_heads": num_heads,
                "head_size": head_size,
                "num_kv_heads": num_kv_heads,
                "sliding_window": sliding_window,
                "kv_cache_dtype": kv_cache_dtype,
                "logits_soft_cap": logits_soft_cap,
                "attn_type": attn_type,
                "kv_sharing_target_layer_name": kv_sharing_target_layer_name,
                "sinks": sinks,
            }

        def do_kv_cache_update(self, layer, k, v, kv_cache, slot_mapping):
            observed["history_updated"] = True

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
            assert layer.layer_name == "placeholder"
            assert layer._k_scale_float == 1.0
            assert layer._v_scale_float == 1.0
            observed["forward_called"] = True
            return output

    class FakeBuilder:
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            observed["builder_layer_names"] = layer_names

        def build(self, common_prefix_len, common_attn_metadata):
            observed["built_metadata"] = common_attn_metadata
            return SimpleNamespace(slot_mapping=common_attn_metadata.slot_mapping)

    class FakeBackend:
        forward_includes_kv_cache_update = True

        @staticmethod
        def get_name():
            return "CUSTOM"

        @staticmethod
        def get_supported_kernel_block_sizes():
            return [128]

        @staticmethod
        def get_builder_cls():
            return FakeBuilder

        @staticmethod
        def get_impl_cls():
            return FakeImpl

    def fake_create_vllm_config(**kwargs):
        observed["vllm_config_kwargs"] = kwargs
        return SimpleNamespace(
            cache_config=SimpleNamespace(block_size=kwargs["block_size"], num_gpu_blocks=kwargs["num_gpu_blocks"]),
            model_config=SimpleNamespace(get_sliding_window=lambda: kwargs.get("sliding_window")),
        )

    def fake_create_common_metadata(batch_spec, block_size, device):
        observed["batch_spec"] = batch_spec
        observed["block_size"] = block_size
        return SimpleNamespace(
            query_start_loc=_FakeTensor((len(batch_spec.seq_lens) + 1,)),
            query_start_loc_cpu=_FakeTensor((len(batch_spec.seq_lens) + 1,)),
            seq_lens=_FakeTensor((len(batch_spec.seq_lens),)),
            seq_lens_cpu_upper_bound=_FakeTensor((len(batch_spec.seq_lens),)),
            num_reqs=len(batch_spec.seq_lens),
            num_actual_tokens=sum(batch_spec.query_lens),
            max_query_len=max(batch_spec.query_lens),
            max_seq_len=max(batch_spec.seq_lens),
            block_table_tensor=_FakeTensor((len(batch_spec.seq_lens), 1)),
            slot_mapping=_FakeTensor((sum(batch_spec.query_lens),)),
        )

    def fake_create_kv_cache(**kwargs):
        observed["kv_cache_kwargs"] = kwargs
        history_slots = SimpleNamespace(numel=lambda: 0 if is_context_phase else 64)
        return ("k_cache", "v_cache"), history_slots

    @contextmanager
    def fake_benchmark(**kwargs):
        benchmark_kwargs.update(kwargs)
        kwargs["kernel_func"]()
        yield {"latency_ms": 2.5, "used_cuda_graph": False}

    monkeypatch.setattr(
        collect_attn_npu,
        "setup_distributed",
        lambda device: observed.setdefault("setup_distributed_device", device),
    )
    monkeypatch.setattr(collect_attn_npu, "get_device_module", lambda device=None: device_module)
    monkeypatch.setattr(collect_attn_npu, "_select_attn_backend_cls", lambda **_kw: FakeBackend)
    monkeypatch.setattr(collect_attn_npu, "framework_kv_block_size", lambda _cls: 128)
    monkeypatch.setattr(collect_attn_npu, "create_vllm_config", fake_create_vllm_config)
    monkeypatch.setattr(collect_attn_npu, "create_standard_kv_cache_spec", lambda *_a, **_kw: SimpleNamespace())
    monkeypatch.setattr(collect_attn_npu, "create_common_attn_metadata", fake_create_common_metadata)
    monkeypatch.setattr(collect_attn_npu, "create_kv_cache_and_block_mappings", fake_create_kv_cache)
    monkeypatch.setattr(collect_attn_npu, "benchmark_with_power", fake_benchmark)
    monkeypatch.setattr(collect_attn_npu, "log_perf", lambda **kwargs: logged.update(kwargs))

    collect_attn_npu.run_attention_torch(
        2,
        64,
        8,
        2,
        128,
        False,
        is_context_phase,
        128,
        True,
        perf_filename=perf_file,
    )

    assert observed["setup_distributed_device"] == "npu:0"
    assert observed["batch_spec"].seq_lens == [expected_seq_len, expected_seq_len]
    assert observed["block_size"] == 128
    assert observed["impl_kwargs"]["num_heads"] == 8
    assert observed["impl_kwargs"]["num_kv_heads"] == 2
    assert observed["impl_kwargs"]["sliding_window"] == 128
    assert observed["impl_kwargs"]["kv_cache_dtype"] == "auto"
    assert observed["impl_kwargs"]["logits_soft_cap"] is None
    assert observed["impl_kwargs"]["kv_sharing_target_layer_name"] is None
    assert observed["impl_kwargs"]["sinks"] is not None
    assert observed["forward_called"] is True
    assert observed.get("history_updated", False) is (not is_context_phase)

    assert benchmark_kwargs["use_cuda_graph"] is False
    assert benchmark_kwargs["allow_graph_fail"] is False
    assert logged["framework"] == "VLLM"
    assert logged["version"] == "0.30.0rc2"
    assert logged["device_name"] == "Ascend950DT"
    assert logged["op_name"] == expected_op
    assert logged["kernel_source"] == "vllm_custom"
    assert logged["perf_filename"] is perf_file
    assert logged["power_stats"] is None
    assert logged["item_list"] == [
        {
            "batch_size": 2,
            "isl": expected_isl,
            "num_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "beam_width": 1,
            "attn_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "window_size": 128,
            "step": expected_step,
            "latency": pytest.approx(2.5),
            "used_cuda_graph": False,
        }
    ]
    device_module.get_device_capability.assert_not_called()
