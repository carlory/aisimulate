# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU gemm lane — manifest pin, registry entry, cases, module.

The three wiring surfaces only work together (an op entry without its manifest
pin is unresolvable, a manifest entry without ops fails ``validate_resolution``),
so they are asserted together here. Neither an Ascend device, ``torch_npu`` nor
``vllm_ascend`` exists on CI or dev machines, so the collector module itself is
imported against stubbed ``torch``/``vllm``/``vllm_ascend`` modules; the point is
its declared shape (bfloat16 and fp8_block, eager, npu:0, no power), not its timings.
"""

import functools
import importlib
import importlib.metadata
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from collector.case_generator import get_gemm_case_specs, get_gemm_type_specs
from collector.framework_manifest import (
    get_collector_runtime,
    require_collector_runtime,
    resolve_op_runtime,
    validate_resolution,
)
from collector.registry_types import PerfFile
from collector.vllm.registry import REGISTRY, REGISTRY_NPU, REGISTRY_XPU

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE = "collector.vllm.collect_gemm_npu"

IMAGE = (
    "quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler"
    "@sha256:2fb630483fee3ee74c882ad490e2f2586e683a507b8365306f8d25e4df3e33ea"
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_pins_the_vllm_ascend_runtime_identity():
    runtime = get_collector_runtime("vllm_npu")

    assert runtime.framework == "vllm_npu"
    # The version axis is the vllm-ascend release, not base vLLM.
    assert runtime.version == "0.30.0rc2"
    assert runtime.images == {"default": IMAGE}
    assert runtime.image() == IMAGE
    # Device-specific vLLM lanes still publish under the shared "vllm" data_backend directory.
    assert runtime.data_backend == "vllm"
    assert runtime.workload == "default"


def test_manifest_records_the_image_abi_read_from_the_image_layers():
    """The A5 image's daily-deps step overrides requirements.txt and the
    published versioning-policy matrix; abi: must describe what is
    installed."""
    abi = get_collector_runtime("vllm_npu").abi

    assert abi["cann"] == "9.2.0_20260930220649"
    assert abi["torch"] == "2.10.0"
    assert abi["torch_npu"] == "2.10.0.post7"  # not the matrix's 2.10.0.post4
    assert abi["triton_ascend"] == "3.6.0+dev20260923230212"  # not 3.2.2
    assert abi["triton"] == "3.6.0"  # uninstalled by the Dockerfile, pulled back by triton-ascend
    assert abi["vllm"] == "0.30.0"
    assert abi["soc_version"] == "ascend950dt_9582"


def test_every_registry_lane_including_npu_resolves():
    assert validate_resolution() == []


def test_gemm_resolves_through_the_npu_lane_and_enforces_the_pin():
    assert resolve_op_runtime("vllm_npu", "gemm").version == "0.30.0rc2"

    runtime = require_collector_runtime("vllm_npu", "0.30.0rc2", requested_ops={"gemm"}, wideep_ops=set())
    assert runtime.framework == "vllm_npu"

    with pytest.raises(RuntimeError, match=r"vllm_npu stock collector requires exactly 0\.30\.0rc2"):
        require_collector_runtime("vllm_npu", "0.30.0", requested_ops={"gemm"}, wideep_ops=set())


def test_npu_collectors_are_exactly_pinned_to_the_manifest_version():
    """REGISTRY_NPU uses unrouted OpEntry items, so __compat__ must match the manifest pin."""
    expected = f'__compat__ = "vllm_ascend=={get_collector_runtime("vllm_npu").version}"'
    assert all(not entry.versions for entry in REGISTRY_NPU)

    for module in sorted({entry.module for entry in REGISTRY_NPU}):
        source = (REPO_ROOT / f"{module.replace('.', '/')}.py").read_text(encoding="utf-8")
        declarations = [line.strip() for line in source.splitlines() if line.startswith("__compat__")]
        assert declarations == [expected], module


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_npu_carries_only_the_gemm_op():
    assert [entry.op for entry in REGISTRY_NPU] == ["gemm"]

    gemm = REGISTRY_NPU[0]
    assert gemm.module == MODULE
    assert (gemm.get_func, gemm.run_func) == ("get_gemm_test_cases", "run_gemm")
    # Reuses the shared table name: the NPU lane adds no new perf file.
    assert gemm.perf_filename is PerfFile.GEMM
    assert gemm.unverified is True
    assert gemm.unverified_sms == ()


def test_npu_gemm_entry_uses_the_standard_vllm_gemm_entrypoints():
    """NPU gemm shares the standard vLLM gemm entrypoints and table name, with its own module."""
    cuda_gemm = next(entry for entry in REGISTRY if entry.op == "gemm")
    xpu_gemm = next(entry for entry in REGISTRY_XPU if entry.op == "gemm")
    npu_gemm = REGISTRY_NPU[0]

    assert (npu_gemm.get_func, npu_gemm.run_func, npu_gemm.perf_filename) == (
        cuda_gemm.get_func,
        cuda_gemm.run_func,
        cuda_gemm.perf_filename,
    )
    assert npu_gemm.module not in {cuda_gemm.module, xpu_gemm.module}


# ---------------------------------------------------------------------------
# Case partition
# ---------------------------------------------------------------------------


def test_npu_gemm_cases_declare_bfloat16_and_fp8_block():
    # The vllm_npu YAML partition declares bfloat16 and fp8_block (W8A8_MXFP8).
    assert get_gemm_type_specs("vllm_npu") == ["bfloat16", "fp8_block"]


def test_npu_gemm_shape_sweep_expands():
    specs = get_gemm_case_specs("vllm_npu")

    assert specs, "cases/base_ops/gemm.yaml must define a vllm_npu shape sweep"
    assert all(spec.x > 0 and spec.n > 0 and spec.k > 0 for spec in specs)
    # Framework-specific transformer GEMM sweep (21 token counts x 21 k x 23 n = 10,143 base
    # grid shapes plus model-specific shapes = 10,857), narrower than the all-frameworks base
    # sweep (which adds +1 alignment-probe token counts).
    assert 21 * 21 * 23 <= len(specs) < len(get_gemm_case_specs())


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


@pytest.fixture
def collect_gemm_npu(monkeypatch):
    """Import the collector against stubbed torch / vLLM / vllm-ascend."""
    for name in (
        "torch",
        "vllm",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm_ascend",
        "vllm_ascend.ascend_config",
        "vllm_ascend.quantization",
    ):
        monkeypatch.setitem(sys.modules, name, MagicMock(name=name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", MagicMock(name="linear"))
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.configs", MagicMock(name="configs"))
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda pkg: "0.30.0rc2" if pkg == "vllm-ascend" else real_version(pkg),
    )
    # utils pulls in the full vLLM attention stack; stub it, but keep a real
    # with_exit_stack so run_gemm keeps its decorated signature.
    utils = MagicMock(name="utils")
    utils.with_exit_stack = _with_exit_stack
    monkeypatch.setitem(sys.modules, "collector.vllm.utils", utils)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    module = importlib.import_module(MODULE)
    try:
        yield module
    finally:
        sys.modules.pop(MODULE, None)


def test_test_cases_expand_bfloat16_and_fp8_block(collect_gemm_npu):
    cases = collect_gemm_npu.get_gemm_test_cases()
    specs = get_gemm_case_specs("vllm_npu")

    assert cases
    assert {gemm_type for gemm_type, *_ in cases} == {"bfloat16", "fp8_block"}
    assert len(cases) == 2 * len(specs)
    assert [(m, n, k) for gemm_type, m, n, k in cases if gemm_type == "bfloat16"] == [
        (spec.x, spec.n, spec.k) for spec in specs
    ]
    assert [(m, n, k) for gemm_type, m, n, k in cases if gemm_type == "fp8_block"] == [
        (spec.x, spec.n, spec.k) for spec in specs
    ]


def test_gemm_peak_footprint_bytes_adds_fp8_block_staging(collect_gemm_npu):
    bf16_fixed = collect_gemm_npu._gemm_peak_footprint_bytes("bfloat16", 16, 32, 64, copies=0)
    mxfp8_fixed = collect_gemm_npu._gemm_peak_footprint_bytes("fp8_block", 16, 32, 64, copies=0)

    assert bf16_fixed == 16 * 64 * 2
    assert mxfp8_fixed == bf16_fixed + 32 * 64 * 3


def test_version_logged_is_the_vllm_ascend_version(collect_gemm_npu):
    """Rows and the parquet <version> directory share the manifest's axis."""
    assert collect_gemm_npu.vllm_ascend_version == "0.30.0rc2"


def test_run_gemm_defaults_to_npu_0_and_refuses_other_dtypes(collect_gemm_npu):
    import inspect

    signature = inspect.signature(collect_gemm_npu.run_gemm.__wrapped__)
    assert signature.parameters["device"].default == "npu:0"

    with pytest.raises(RuntimeError, match="collects bfloat16 and fp8_block only"):
        collect_gemm_npu.run_gemm("fp8", 16, 16, 16, perf_filename=PerfFile.GEMM)


def test_run_gemm_logs_eager_row_without_power(collect_gemm_npu, monkeypatch):
    """Eager, no power sampling, untraced kernel label, standard gemm_perf columns."""
    from contextlib import contextmanager

    logged = {}
    benchmark_kwargs = {}
    linear_calls = []
    device_module = MagicMock()
    device_module.get_device_properties.return_value.total_memory = 64 * 2**30
    device_module.get_device_name.return_value = "Ascend950DT"

    def fake_linear(**kwargs):
        linear_calls.append(kwargs)
        return MagicMock(name="gemm_layer")

    @contextmanager
    def fake_benchmark(**kwargs):
        benchmark_kwargs.update(kwargs)
        kwargs["kernel_func"]()
        yield {"latency_ms": 64.0, "used_cuda_graph": False}

    monkeypatch.setattr(collect_gemm_npu, "setup_distributed", lambda device: None)
    monkeypatch.setattr(collect_gemm_npu, "get_device_module", lambda device=None: device_module)
    monkeypatch.setattr(collect_gemm_npu, "RowParallelLinear", fake_linear)
    monkeypatch.setattr(collect_gemm_npu, "benchmark_with_power", fake_benchmark)
    monkeypatch.setattr(collect_gemm_npu, "log_perf", lambda **kwargs: logged.update(kwargs))

    collect_gemm_npu.run_gemm("bfloat16", 16, 32, 64, perf_filename=PerfFile.GEMM)

    assert linear_calls and all(call["quant_config"] is None for call in linear_calls)
    assert benchmark_kwargs["use_cuda_graph"] is False
    assert benchmark_kwargs["allow_graph_fail"] is False
    assert logged["framework"] == "VLLM"
    assert logged["version"] == "0.30.0rc2"
    assert logged["device_name"] == "Ascend950DT"
    assert logged["op_name"] == "gemm"
    assert logged["kernel_source"] == "vllm_default"
    assert logged["perf_filename"] is PerfFile.GEMM
    assert logged["power_stats"] is None
    assert logged["item_list"] == [
        {
            "gemm_dtype": "bfloat16",
            "m": 16,
            "n": 32,
            "k": 64,
            "latency": pytest.approx(1.0),
            "used_cuda_graph": False,
        }
    ]
    assert not hasattr(collect_gemm_npu, "xpu_graph_measure_enabled")
    device_module.get_device_capability.assert_not_called()


def test_run_gemm_fp8_block_builds_mxfp8_config_quantizes_weights_and_logs_same_schema(
    collect_gemm_npu,
    monkeypatch,
):
    """fp8_block (W8A8_MXFP8) builds AscendModelSlimConfig, quantizes BF16 weights via
    npu_dynamic_mx_quant(group=32), and logs identical schema.
    """
    from contextlib import contextmanager

    logged = {}
    modelslim_config_calls = []
    init_ascend_config_calls = []
    linear_calls = []
    processed_layers = []
    mx_quant_calls = []
    device_module = MagicMock()
    device_module.get_device_properties.return_value.total_memory = 64 * 2**30
    device_module.get_device_name.return_value = "Ascend950DT"

    q_weight = MagicMock(name="q_weight")
    mx_scale = MagicMock(name="mx_scale")

    def fake_mx_quant(tensor, dst_type=None):
        mx_quant_calls.append((tensor, dst_type))
        return q_weight, mx_scale

    device_module.npu_dynamic_mx_quant = fake_mx_quant
    collect_gemm_npu.torch.ops.npu.npu_dynamic_mx_quant = None

    def fake_modelslim_config(quant_config=None):
        modelslim_config_calls.append(quant_config)
        return ("ascend_modelslim_config", quant_config)

    def fake_linear(**kwargs):
        # Verify vllm_config.quant_config and init_ascend_config were populated
        # before RowParallelLinear construction.
        assert init_ascend_config_calls, "init_ascend_config must run before RowParallelLinear"
        linear_calls.append(kwargs)
        layer = MagicMock(name="mxfp8_gemm_layer")
        layer.weight = MagicMock(name="weight")
        layer.weight.shape = (32, 64)
        layer.weight.dtype = collect_gemm_npu.torch.float8_e4m3fn
        layer.weight.device = "npu:0"
        layer.weight_scale = MagicMock(name="weight_scale")
        layer.weight_scale.shape = (32, 2)
        layer.quant_method.process_weights_after_loading.side_effect = processed_layers.append
        return layer

    @contextmanager
    def fake_benchmark(**kwargs):
        kwargs["kernel_func"]()
        yield {"latency_ms": 128.0, "used_cuda_graph": False}

    monkeypatch.setattr(collect_gemm_npu, "setup_distributed", lambda device: None)
    monkeypatch.setattr(collect_gemm_npu, "get_device_module", lambda device=None: device_module)
    monkeypatch.setattr(collect_gemm_npu, "AscendModelSlimConfig", fake_modelslim_config)
    monkeypatch.setattr(collect_gemm_npu, "init_ascend_config", init_ascend_config_calls.append)
    monkeypatch.setattr(collect_gemm_npu, "RowParallelLinear", fake_linear)
    monkeypatch.setattr(collect_gemm_npu, "benchmark_with_power", fake_benchmark)
    monkeypatch.setattr(collect_gemm_npu, "log_perf", lambda **kwargs: logged.update(kwargs))

    collect_gemm_npu.run_gemm("fp8_block", 16, 32, 64, perf_filename=PerfFile.GEMM)

    assert modelslim_config_calls == [{".weight": "W8A8_MXFP8", "group_size": 32}]
    assert len(init_ascend_config_calls) == 1
    assert init_ascend_config_calls[0].quant_config == ("ascend_modelslim_config", modelslim_config_calls[0])
    assert len(linear_calls) == 64
    assert all(
        call["input_size"] == 64
        and call["output_size"] == 32
        and call["quant_config"] == ("ascend_modelslim_config", modelslim_config_calls[0])
        and call["prefix"] == ""
        and call["disable_tp"] is True
        for call in linear_calls
    )
    assert len(processed_layers) == len(linear_calls)
    assert len(mx_quant_calls) == len(linear_calls)
    assert all(
        tensor is collect_gemm_npu.torch.full.return_value and dst_type is collect_gemm_npu.torch.float8_e4m3fn
        for tensor, dst_type in mx_quant_calls
    )
    for layer in processed_layers:
        layer.weight.fill_.assert_not_called()
        layer.weight.copy_.assert_called_once_with(q_weight)
        layer.weight_scale.copy_.assert_called_once_with(mx_scale.reshape.return_value.view.return_value)
    mx_scale.reshape.assert_called_with((32, 2))
    mx_scale.reshape.return_value.view.assert_called_with(collect_gemm_npu.torch.uint8)
    collect_gemm_npu.torch.full.assert_called_with(
        (32, 64),
        0.01,
        dtype=collect_gemm_npu.torch.bfloat16,
        device="npu:0",
    )
    assert logged["framework"] == "VLLM"
    assert logged["version"] == "0.30.0rc2"
    assert logged["device_name"] == "Ascend950DT"
    assert logged["op_name"] == "gemm"
    assert logged["kernel_source"] == "vllm_default"
    assert logged["perf_filename"] is PerfFile.GEMM
    assert logged["power_stats"] is None
    assert logged["item_list"] == [
        {
            "gemm_dtype": "fp8_block",
            "m": 16,
            "n": 32,
            "k": 64,
            "latency": pytest.approx(2.0),
            "used_cuda_graph": False,
        }
    ]
