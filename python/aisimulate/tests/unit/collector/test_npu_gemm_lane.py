# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU gemm lane — manifest pin, registry entry, cases, module.

The three wiring surfaces only work together (an op entry without its manifest
pin is unresolvable, a manifest entry without ops fails ``validate_resolution``),
so they are asserted together here. Neither an Ascend device, ``torch_npu`` nor
``vllm_ascend`` exists on CI or dev machines, so the collector module itself is
imported against stubbed ``torch``/``vllm``/``vllm_ascend`` modules; the point is
its declared shape (bfloat16, fp8, fp8_block and mxfp4, eager/ACL graph, npu:0, no power), not its timings.
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


def test_registry_npu_carries_the_gemm_op():
    gemm = next(entry for entry in REGISTRY_NPU if entry.op == "gemm")
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
    npu_gemm = next(entry for entry in REGISTRY_NPU if entry.op == "gemm")

    assert (npu_gemm.get_func, npu_gemm.run_func, npu_gemm.perf_filename) == (
        cuda_gemm.get_func,
        cuda_gemm.run_func,
        cuda_gemm.perf_filename,
    )
    assert npu_gemm.module not in {cuda_gemm.module, xpu_gemm.module}


# ---------------------------------------------------------------------------
# Case partition
# ---------------------------------------------------------------------------


def test_npu_gemm_cases_declare_bfloat16_fp8_fp8_block_and_mxfp4():
    # The vllm_npu YAML partition declares bfloat16, fp8 (W8A8FP8), fp8_block (W8A8_MXFP8), and mxfp4 (W4A4_MXFP4).
    assert get_gemm_type_specs("vllm_npu") == ["bfloat16", "fp8", "fp8_block", "mxfp4"]


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


def test_test_cases_expand_bfloat16_fp8_fp8_block_and_mxfp4(collect_gemm_npu):
    cases = collect_gemm_npu.get_gemm_test_cases()
    specs = get_gemm_case_specs("vllm_npu")

    assert cases
    assert {gemm_type for gemm_type, *_ in cases} == {"bfloat16", "fp8", "fp8_block", "mxfp4"}
    assert len(cases) == 4 * len(specs)
    assert [(m, n, k) for gemm_type, m, n, k in cases if gemm_type == "bfloat16"] == [
        (spec.x, spec.n, spec.k) for spec in specs
    ]
    assert [(m, n, k) for gemm_type, m, n, k in cases if gemm_type == "fp8"] == [
        (spec.x, spec.n, spec.k) for spec in specs
    ]
    assert [(m, n, k) for gemm_type, m, n, k in cases if gemm_type == "fp8_block"] == [
        (spec.x, spec.n, spec.k) for spec in specs
    ]
    assert [(m, n, k) for gemm_type, m, n, k in cases if gemm_type == "mxfp4"] == [
        (spec.x, spec.n, spec.k) for spec in specs
    ]


class _ByteArrayAdapter:
    """Minimal byte-array tensor adapter for layout tests (no torch dependency)."""

    def __init__(self, data, shape):
        self._data = list(data)
        self._shape = tuple(shape)
        size = 1
        for dim in self._shape:
            size *= dim
        if size != len(self._data):
            raise ValueError(f"size mismatch: shape={self._shape} requires {size}, got {len(self._data)}")

    @property
    def shape(self):
        return self._shape

    def view(self, _dtype):
        return self

    def reshape(self, *shape):
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        target = list(target)
        unknown = [i for i, d in enumerate(target) if d == -1]
        if len(unknown) > 1:
            raise ValueError(f"invalid reshape target={target}")
        size = len(self._data)
        known = 1
        for d in target:
            if d != -1:
                known *= d
        if unknown:
            if known == 0 or size % known != 0:
                raise ValueError(f"cannot infer reshape {target} for size {size}")
            target[unknown[0]] = size // known
        out_shape = tuple(int(d) for d in target)
        return _ByteArrayAdapter(self._data, out_shape)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) != len(self._shape):
            raise TypeError(f"index rank mismatch: idx={idx} shape={self._shape}")

        selectors = []
        out_shape = []
        for dim, sel in zip(self._shape, idx, strict=True):
            if isinstance(sel, slice):
                start, stop, step = sel.indices(dim)
                if step != 1:
                    raise ValueError("step slicing is unsupported")
                vals = list(range(start, stop, step))
                selectors.append(vals)
                out_shape.append(len(vals))
            elif isinstance(sel, int):
                if sel < 0:
                    sel += dim
                if sel < 0 or sel >= dim:
                    raise IndexError(sel)
                selectors.append([sel])
            else:
                raise TypeError(type(sel))

        strides = []
        stride = 1
        for dim in reversed(self._shape):
            strides.insert(0, stride)
            stride *= dim

        out = []

        def walk(axis, offset):
            if axis == len(selectors):
                out.append(self._data[offset])
                return
            for v in selectors[axis]:
                walk(axis + 1, offset + v * strides[axis])

        walk(0, 0)
        return _ByteArrayAdapter(out, tuple(out_shape))

    def rows(self):
        if len(self._shape) != 2:
            raise ValueError(f"rows() requires rank-2, got {self._shape}")
        n, w = self._shape
        return [self._data[i * w : (i + 1) * w] for i in range(n)]


@pytest.mark.parametrize("loader_width", [1, 2, 3, 4])
def test_mx_scale_to_loader_layout_preserves_expected_bytes_for_k32_k64_k96_k128(collect_gemm_npu, loader_width):
    n = 2
    expected_flat_width = 2 * ((loader_width + 1) // 2)
    paired_groups = expected_flat_width // 2
    # Deterministic bytes per row: row0 0.., row1 100..
    data = []
    for row in range(n):
        base = row * 100
        data.extend(range(base, base + expected_flat_width))
    mx_scale = _ByteArrayAdapter(data, (n, paired_groups, 2))

    collect_gemm_npu.torch.uint8 = object()
    out = collect_gemm_npu._mx_scale_to_loader_layout(mx_scale, (n, loader_width))

    assert out.shape == (n, loader_width)
    assert out.rows() == [
        list(range(0, loader_width)),
        list(range(100, 100 + loader_width)),
    ]


def test_mx_scale_to_loader_layout_rejects_rank_mismatch_or_nonpaired_tail(collect_gemm_npu):
    collect_gemm_npu.torch.uint8 = object()

    with pytest.raises(RuntimeError, match=r"loader_shape=\(2, 3, 1\).+rank-2"):
        collect_gemm_npu._mx_scale_to_loader_layout(_ByteArrayAdapter(range(8), (2, 2, 2)), (2, 3, 1))

    with pytest.raises(RuntimeError, match=r"shape=\(2, 1, 3\).+loader_shape=\(2, 2\).+axis == 2"):
        collect_gemm_npu._mx_scale_to_loader_layout(_ByteArrayAdapter(range(6), (2, 1, 3)), (2, 2))


def test_mx_scale_to_loader_layout_rejects_leading_dim_mismatch(collect_gemm_npu):
    collect_gemm_npu.torch.uint8 = object()
    with pytest.raises(RuntimeError, match=r"shape=\(3, 2, 2\).+loader_shape=\(2, 4\).+expected n == 2"):
        collect_gemm_npu._mx_scale_to_loader_layout(_ByteArrayAdapter(range(12), (3, 2, 2)), (2, 4))


@pytest.mark.parametrize(
    ("shape", "loader_shape"),
    [
        ((2, 1, 2), (2, 3)),  # narrow: flat width 2 < expected 4
        ((2, 3, 2), (2, 3)),  # over-wide: flat width 6 > expected 4
    ],
)
def test_mx_scale_to_loader_layout_rejects_noncontract_flat_width(collect_gemm_npu, shape, loader_shape):
    collect_gemm_npu.torch.uint8 = object()
    size = 1
    for dim in shape:
        size *= dim
    with pytest.raises(RuntimeError, match=r"invalid mx_scale flat width .+ shape=.+ loader_shape=.+ expected"):
        collect_gemm_npu._mx_scale_to_loader_layout(_ByteArrayAdapter(range(size), shape), loader_shape)


def test_gemm_peak_footprint_bytes_adds_quantized_staging(collect_gemm_npu):
    bf16_fixed = collect_gemm_npu._gemm_peak_footprint_bytes("bfloat16", 16, 32, 64, copies=0)
    fp8_fixed = collect_gemm_npu._gemm_peak_footprint_bytes("fp8", 16, 32, 64, copies=0)
    mxfp8_fixed = collect_gemm_npu._gemm_peak_footprint_bytes("fp8_block", 16, 32, 64, copies=0)
    mxfp4_fixed = collect_gemm_npu._gemm_peak_footprint_bytes("mxfp4", 16, 32, 64, copies=0)

    assert bf16_fixed == 16 * 64 * 2
    assert fp8_fixed == bf16_fixed + 32 * 64
    assert mxfp8_fixed == bf16_fixed + 32 * 64 * 3
    assert mxfp4_fixed == bf16_fixed + 32 * 64 * 3


def test_version_logged_is_the_vllm_ascend_version(collect_gemm_npu):
    """Rows and the parquet <version> directory share the manifest's axis."""
    assert collect_gemm_npu.vllm_ascend_version == "0.30.0rc2"


def test_run_gemm_defaults_to_npu_0_and_refuses_other_dtypes(collect_gemm_npu):
    import inspect

    signature = inspect.signature(collect_gemm_npu.run_gemm.__wrapped__)
    assert signature.parameters["device"].default == "npu:0"

    with pytest.raises(RuntimeError, match="collects bfloat16, fp8, fp8_block, and mxfp4 only"):
        collect_gemm_npu.run_gemm("nvfp4", 16, 16, 16, perf_filename=PerfFile.GEMM)


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
    assert hasattr(collect_gemm_npu, "npu_graph_measure_enabled")
    device_module.get_device_capability.assert_not_called()


def test_run_gemm_fp8_builds_dynamic_fp8_config_quantizes_weights_and_logs_same_schema(collect_gemm_npu, monkeypatch):
    """FP8 (W8A8FP8) builds AscendModelSlimConfig, prepares weights/scales, and logs identical schema."""
    from contextlib import contextmanager

    logged = {}
    modelslim_config_calls = []
    linear_calls = []
    processed_layers = []
    device_module = MagicMock()
    device_module.get_device_properties.return_value.total_memory = 64 * 2**30
    device_module.get_device_name.return_value = "Ascend950DT"

    def fake_modelslim_config(quant_config=None):
        modelslim_config_calls.append(quant_config)
        return ("ascend_modelslim_config", quant_config)

    def fake_linear(**kwargs):
        linear_calls.append(kwargs)
        layer = MagicMock(name="fp8_gemm_layer")
        layer.weight = MagicMock(name="weight")
        layer.weight.shape = (32, 64)
        layer.weight.dtype = collect_gemm_npu.torch.float8_e4m3fn
        layer.weight.device = "npu:0"
        layer.weight_scale = MagicMock(name="weight_scale")
        layer.weight_offset = MagicMock(name="weight_offset")
        layer.quant_method.process_weights_after_loading.side_effect = processed_layers.append
        return layer

    @contextmanager
    def fake_benchmark(**kwargs):
        kwargs["kernel_func"]()
        yield {"latency_ms": 128.0, "used_cuda_graph": False}

    monkeypatch.setattr(collect_gemm_npu, "setup_distributed", lambda device: None)
    monkeypatch.setattr(collect_gemm_npu, "get_device_module", lambda device=None: device_module)
    monkeypatch.setattr(collect_gemm_npu, "AscendModelSlimConfig", fake_modelslim_config)
    monkeypatch.setattr(collect_gemm_npu, "RowParallelLinear", fake_linear)
    monkeypatch.setattr(collect_gemm_npu, "benchmark_with_power", fake_benchmark)
    monkeypatch.setattr(collect_gemm_npu, "log_perf", lambda **kwargs: logged.update(kwargs))

    collect_gemm_npu.run_gemm("fp8", 16, 32, 64, perf_filename=PerfFile.GEMM)

    assert modelslim_config_calls == [{".weight": "W8A8FP8_DYNAMIC"}]
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
    for layer in processed_layers:
        layer.weight.fill_.assert_not_called()
        layer.weight.copy_.assert_called_once_with(collect_gemm_npu.torch.full.return_value.to.return_value)
        layer.weight_scale.fill_.assert_called_once_with(1.0)
        layer.weight_offset.zero_.assert_called_once_with()
    collect_gemm_npu.torch.full.assert_called_with(
        (32, 64),
        0.01,
        dtype=collect_gemm_npu.torch.bfloat16,
        device="npu:0",
    )
    collect_gemm_npu.torch.full.return_value.to.assert_called_with(collect_gemm_npu.torch.float8_e4m3fn)
    assert logged["framework"] == "VLLM"
    assert logged["version"] == "0.30.0rc2"
    assert logged["device_name"] == "Ascend950DT"
    assert logged["op_name"] == "gemm"
    assert logged["kernel_source"] == "vllm_default"
    assert logged["perf_filename"] is PerfFile.GEMM
    assert logged["power_stats"] is None
    assert logged["item_list"] == [
        {
            "gemm_dtype": "fp8",
            "m": 16,
            "n": 32,
            "k": 64,
            "latency": pytest.approx(2.0),
            "used_cuda_graph": False,
        }
    ]


@pytest.mark.parametrize(
    ("k", "loader_scale_shape"),
    [
        (64, (32, 2)),
        (32, (32, 1)),
    ],
)
def test_run_gemm_fp8_block_builds_mxfp8_config_quantizes_weights_and_trims_scale_to_loader_layout(
    collect_gemm_npu,
    monkeypatch,
    k,
    loader_scale_shape,
):
    """fp8_block (W8A8_MXFP8) maps paired MX scales from npu_dynamic_mx_quant
    into the vLLM loader layout (n, ceil(k/32)); this trims k=32's extra paired
    value and preserves k>=64 width.
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
    mx_scale.shape = (32, 1, 2)
    mx_scale.view.return_value.shape = (32, 1, 2)
    mx_scale.view.return_value.reshape.return_value.shape = (32, 2)

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
        layer.weight.shape = (32, k)
        layer.weight.dtype = collect_gemm_npu.torch.float8_e4m3fn
        layer.weight.device = "npu:0"
        layer.weight_scale = MagicMock(name="weight_scale")
        layer.weight_scale.shape = loader_scale_shape
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

    collect_gemm_npu.run_gemm("fp8_block", 16, 32, k, perf_filename=PerfFile.GEMM)

    assert modelslim_config_calls == [{".weight": "W8A8_MXFP8", "group_size": 32}]
    assert len(init_ascend_config_calls) == 1
    assert init_ascend_config_calls[0].quant_config == ("ascend_modelslim_config", modelslim_config_calls[0])
    assert len(linear_calls) == 64
    assert all(
        call["input_size"] == k
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
        layer.weight_scale.copy_.assert_called_once_with(
            mx_scale.view.return_value.reshape.return_value.__getitem__.return_value.reshape.return_value
        )
    mx_scale.view.assert_called_with(collect_gemm_npu.torch.uint8)
    mx_scale.view.return_value.reshape.assert_called_with(32, -1)
    mx_scale.view.return_value.reshape.return_value.__getitem__.return_value.reshape.assert_called_with(
        loader_scale_shape
    )
    collect_gemm_npu.torch.full.assert_called_with(
        (32, k),
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
            "k": k,
            "latency": pytest.approx(2.0),
            "used_cuda_graph": False,
        }
    ]


def test_run_gemm_mxfp4_builds_w4a4_config_quantizes_weights_and_logs_same_schema(
    collect_gemm_npu,
    monkeypatch,
):
    """mxfp4 (W4A4_MXFP4) builds AscendModelSlimConfig, quantizes BF16 weights via
    npu_dynamic_mx_quant(dst_type=float4_e2m1fn_x2, round_mode='round') into packed
    (n, k // 2) uint8 weight + (n, k // 32) uint8 E8M0 scale, and logs identical schema.
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
    fp4_sentinel = MagicMock(name="float4_e2m1fn_x2")
    device_module.float4_e2m1fn_x2 = fp4_sentinel

    q_weight = MagicMock(name="q_weight")
    mx_scale = MagicMock(name="mx_scale")

    def fake_mx_quant(tensor, dst_type=None, round_mode=None):
        mx_quant_calls.append((tensor, dst_type, round_mode))
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
        layer = MagicMock(name="mxfp4_gemm_layer")
        layer.weight = MagicMock(name="weight")
        # Packed FP4 E2M1: 2 values per uint8 byte -> (n, k // 2) = (32, 32)
        layer.weight.shape = (32, 32)
        layer.weight.dtype = collect_gemm_npu.torch.uint8
        layer.weight.device = "npu:0"
        layer.weight_scale = MagicMock(name="weight_scale")
        # E8M0 scale: 1 byte per 32 K elements -> (n, k // 32) = (32, 2)
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

    collect_gemm_npu.run_gemm("mxfp4", 16, 32, 64, perf_filename=PerfFile.GEMM)

    assert modelslim_config_calls == [{".weight": "W4A4_MXFP4", "group_size": 32}]
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
        tensor is collect_gemm_npu.torch.full.return_value and dst_type is fp4_sentinel and round_mode == "round"
        for tensor, dst_type, round_mode in mx_quant_calls
    )
    for layer in processed_layers:
        layer.weight.fill_.assert_not_called()
        layer.weight.copy_.assert_called_once_with(q_weight.view.return_value.reshape.return_value)
        layer.weight_scale.copy_.assert_called_once_with(
            mx_scale.view.return_value.flatten.return_value.__getitem__.return_value.reshape.return_value
        )
    q_weight.view.assert_called_with(collect_gemm_npu.torch.uint8)
    q_weight.view.return_value.reshape.assert_called_with((32, 32))
    mx_scale.view.assert_called_with(collect_gemm_npu.torch.uint8)
    mx_scale.view.return_value.flatten.assert_called_with(-2)
    mx_scale.view.return_value.flatten.return_value.__getitem__.return_value.reshape.assert_called_with((32, 2))
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
            "gemm_dtype": "mxfp4",
            "m": 16,
            "n": 32,
            "k": 64,
            "latency": pytest.approx(2.0),
            "used_cuda_graph": False,
        }
    ]


def test_run_gemm_enables_acl_graph_when_env_flag_is_set(collect_gemm_npu, monkeypatch):
    """When VLLM_NPU_ENABLE_ACL_GRAPH=1, run_gemm passes use_cuda_graph=True and logs used_cuda_graph=True."""
    from contextlib import contextmanager

    logged = {}
    benchmark_kwargs = {}
    device_module = MagicMock()
    device_module.is_available.return_value = True
    device_module.get_device_properties.return_value.total_memory = 64 * 2**30
    device_module.get_device_name.return_value = "Ascend950DT"

    @contextmanager
    def fake_benchmark(**kwargs):
        benchmark_kwargs.update(kwargs)
        kwargs["kernel_func"]()
        yield {"latency_ms": 32.0, "used_cuda_graph": kwargs["use_cuda_graph"]}

    monkeypatch.setenv("VLLM_NPU_ENABLE_ACL_GRAPH", "1")
    monkeypatch.setattr("collector.helper.torch_npu_module", lambda: device_module)
    monkeypatch.setattr(collect_gemm_npu, "setup_distributed", lambda device: None)
    monkeypatch.setattr(collect_gemm_npu, "get_device_module", lambda device=None: device_module)
    monkeypatch.setattr(collect_gemm_npu, "benchmark_with_power", fake_benchmark)
    monkeypatch.setattr(collect_gemm_npu, "log_perf", lambda **kwargs: logged.update(kwargs))

    collect_gemm_npu.run_gemm("bfloat16", 16, 32, 64, perf_filename=PerfFile.GEMM)

    assert benchmark_kwargs["use_cuda_graph"] is True
    assert benchmark_kwargs["allow_graph_fail"] is False
    assert logged["item_list"] == [
        {
            "gemm_dtype": "bfloat16",
            "m": 16,
            "n": 32,
            "k": 64,
            "latency": pytest.approx(0.5),
            "used_cuda_graph": True,
        }
    ]


@pytest.mark.parametrize("has_primary_exc", [True, False])
def test_run_gemm_empty_cache_cleanup_preserves_primary_and_propagates_without_primary(
    collect_gemm_npu,
    monkeypatch,
    caplog,
    has_primary_exc,
):
    """In finally: if a primary exception is active, empty_cache() failure is logged as warning
    and the primary exception is preserved; if no primary exception is active, empty_cache()
    failure must propagate and log_perf must not be called in either case.
    """
    from contextlib import contextmanager

    logged_calls = []
    device_module = MagicMock()
    device_module.get_device_properties.return_value.total_memory = 64 * 2**30
    device_module.get_device_name.return_value = "Ascend950DT"
    device_module.empty_cache.side_effect = RuntimeError("cleanup empty_cache failure")

    @contextmanager
    def fake_benchmark(**kwargs):
        if has_primary_exc:
            raise RuntimeError("primary gemm kernel failure")
        kwargs["kernel_func"]()
        yield {"latency_ms": 32.0, "used_cuda_graph": False}

    monkeypatch.setattr(collect_gemm_npu, "setup_distributed", lambda device: None)
    monkeypatch.setattr(collect_gemm_npu, "get_device_module", lambda device=None: device_module)
    monkeypatch.setattr(collect_gemm_npu, "benchmark_with_power", fake_benchmark)
    monkeypatch.setattr(collect_gemm_npu, "log_perf", lambda **kwargs: logged_calls.append(kwargs))

    expected_match = "primary gemm kernel failure" if has_primary_exc else "cleanup empty_cache failure"
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match=expected_match):
        collect_gemm_npu.run_gemm("bfloat16", 16, 32, 64, perf_filename=PerfFile.GEMM)

    assert logged_calls == []
    if has_primary_exc:
        assert "Suppressing secondary empty_cache() failure after primary RuntimeError" in caplog.text
