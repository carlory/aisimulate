# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the HCCL Ascend NPU communication collector and comm fallback wiring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_ROOT = REPO_ROOT / "collector"
NETWORK_ROOT = COLLECTOR_ROOT / "network"
COLLECT_COMM_SH = NETWORK_ROOT / "collect_comm.sh"

for _p in (str(COLLECTOR_ROOT), str(NETWORK_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_hccl

# ---------------------------------------------------------------------------
# Vocabulary synchronization across the 5 backend-dir / catalog sites
# ---------------------------------------------------------------------------


def test_hccl_is_registered_across_all_backend_dir_vocabularies():
    from aisimulate_core.sdk.common import PerfDataFilename
    from aisimulate_core.sdk.engine_table_view import VIEW_KEY_LAYERS
    from aisimulate_core.sdk.operations.base import _KNOWN_BACKEND_DIRS

    assert PerfDataFilename.hccl.value == "hccl_perf.parquet"
    assert "_hccl_data" in VIEW_KEY_LAYERS
    assert "hccl" in _KNOWN_BACKEND_DIRS

    # Standalone tools that mirror _KNOWN_BACKEND_DIRS / SKIP_BACKEND_DIRS
    create_charts_src = (REPO_ROOT / "tools" / "sanity_check" / "create_charts.py").read_text(encoding="utf-8")
    assert '"hccl"' in create_charts_src

    perf_layout_src = (REPO_ROOT / "tools" / "perf_database" / "perf_data_layout.py").read_text(encoding="utf-8")
    assert '"hccl"' in perf_layout_src

    migrate_src = (REPO_ROOT / "tools" / "perf_database" / "migrate_family_layout.py").read_text(encoding="utf-8")
    assert '"hccl"' in migrate_src

    catalog = yaml.safe_load((COLLECTOR_ROOT / "op_backend_catalog.yaml").read_text(encoding="utf-8"))
    comm_family = next(f for f in catalog["families"] if f["family"] == "comm")
    assert "hccl_perf" in comm_family["op_files"]

    rust_mod_src = (
        REPO_ROOT.parents[1] / "crates" / "core" / "src" / "perfmodel" / "perf_database" / "mod.rs"
    ).read_text(encoding="utf-8")
    assert 'KNOWN_BACKEND_DIRS: [&str; 6] = ["trtllm", "sglang", "vllm", "nccl", "oneccl", "hccl"]' in rust_mod_src


# ---------------------------------------------------------------------------
# Shape sweep & CLI parsing
# ---------------------------------------------------------------------------


def test_get_sweep_byte_sizes_aligns_to_world_and_dtype():
    sizes = collect_hccl.get_sweep_byte_sizes("512,4097,2", num_gpus=2, bytes_per_elem=2)
    assert sizes == [512, 1024, 2048, 4096]
    assert all(s % (2 * 2) == 0 for s in sizes)


def test_parse_range_rejects_malformed_spec():
    with pytest.raises(ValueError, match="--range"):
        collect_hccl._parse_range("512,256,2")
    with pytest.raises(ValueError, match="--range"):
        collect_hccl._parse_range("not_ints")


def test_cli_parser_supports_all_three_ops_and_defaults():
    parser = collect_hccl.build_parser()
    args = parser.parse_args([])
    assert args.hccl_op == "all_gather"
    assert args.dtype == "half"
    assert args.num_gpus == 2
    assert args.perf_filename == "hccl_perf.txt"

    for op in ("all_reduce", "all_gather", "reduce_scatter"):
        parsed = parser.parse_args(["-O", op, "-n", "4", "-t", "int8"])
        assert parsed.hccl_op == op
        assert parsed.num_gpus == 4
        assert parsed.dtype == "int8"


# ---------------------------------------------------------------------------
# Anti-forgery rank validation & output parsing
# ---------------------------------------------------------------------------


def test_parse_worker_output_rejects_rank_mismatch():
    bad_payload = {
        "procs": 1,
        "ranks": [0],
        "records": [{"size_bytes": 512, "message_size": 256, "latency_ms": 0.05}],
    }
    stdout = f"{collect_hccl._RESULT_PREFIX}{json.dumps(bad_payload)}\n"
    with pytest.raises(RuntimeError, match="ran with 1 ranks, expected 2"):
        collect_hccl._parse_worker_output(stdout, "all_reduce", num_gpus=2, bytes_per_elem=2, dtype="half")


def test_parse_worker_output_rejects_forged_rank_list():
    bad_payload = {
        "procs": 2,
        "ranks": [0, 0],
        "records": [{"size_bytes": 512, "message_size": 256, "latency_ms": 0.05}],
    }
    stdout = f"{collect_hccl._RESULT_PREFIX}{json.dumps(bad_payload)}\n"
    with pytest.raises(RuntimeError, match="rank attestation mismatch"):
        collect_hccl._parse_worker_output(stdout, "all_reduce", num_gpus=2, bytes_per_elem=2, dtype="half")


def test_parse_worker_output_supports_tabular_fallback_and_enforces_procs():
    tabular = "# procs: 2\n512 256 bf16 sum 0 25.0 10.0\n1024 512 bf16 sum 0 50.0 10.0\n"
    items = collect_hccl._parse_worker_output(tabular, "all_reduce", num_gpus=2, bytes_per_elem=2, dtype="half")
    assert items == [
        {"nccl_dtype": "half", "num_gpus": 2, "message_size": 256, "latency": pytest.approx(0.025)},
        {"nccl_dtype": "half", "num_gpus": 2, "message_size": 512, "latency": pytest.approx(0.050)},
    ]

    bad_tabular = "# procs: 1\n512 256 bf16 sum 0 25.0 10.0\n"
    with pytest.raises(RuntimeError, match="ran with 1 ranks, expected 2"):
        collect_hccl._parse_worker_output(bad_tabular, "all_reduce", num_gpus=2, bytes_per_elem=2, dtype="half")


# ---------------------------------------------------------------------------
# End-to-end hccl_benchmark driver & worker execution with mocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ["all_reduce", "all_gather", "reduce_scatter"])
def test_hccl_benchmark_logs_perf_rows(monkeypatch, op_name):
    logged = []

    def fake_run(cmd, capture_output, text, env, timeout):
        assert "--worker" in cmd
        assert op_name in cmd
        payload = {
            "procs": 2,
            "ranks": [0, 1],
            "records": [
                {"size_bytes": 512, "message_size": 256, "latency_ms": 0.012},
                {"size_bytes": 1024, "message_size": 512, "latency_ms": 0.024},
            ],
        }
        return SimpleNamespace(returncode=0, stdout=f"{collect_hccl._RESULT_PREFIX}{json.dumps(payload)}\n", stderr="")

    monkeypatch.setattr(collect_hccl.subprocess, "run", fake_run)
    monkeypatch.setattr(collect_hccl, "get_hccl_version", lambda: "9.2.0")
    monkeypatch.setattr(collect_hccl, "get_device_name", lambda: "Ascend950DT")
    monkeypatch.setattr(collect_hccl, "log_perf", lambda **kwargs: logged.append(kwargs))

    collect_hccl.hccl_benchmark(
        dtype="half",
        hccl_op=op_name,
        test_range="512,2048,2",
        num_gpus=2,
        iters=10,
        warmup_iters=2,
    )

    assert len(logged) == 1
    call = logged[0]
    assert call["framework"] == "VLLM"
    assert call["version"] == "9.2.0"
    assert call["device_name"] == "Ascend950DT"
    assert call["op_name"] == op_name
    assert call["kernel_source"] == "HCCL"
    assert call["perf_filename"] == "hccl_perf.txt"
    assert call["item_list"] == [
        {"nccl_dtype": "half", "num_gpus": 2, "message_size": 256, "latency": pytest.approx(0.012)},
        {"nccl_dtype": "half", "num_gpus": 2, "message_size": 512, "latency": pytest.approx(0.024)},
    ]


@pytest.mark.parametrize("op_name", ["all_reduce", "all_gather", "reduce_scatter"])
def test_worker_main_runs_torch_distributed_hccl_and_emits_records(monkeypatch, capsys, op_name):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    collective_calls = []

    class FakeEvent:
        def __init__(self, enable_timing=False):
            self.enable_timing = enable_timing

        def record(self):
            pass

        def elapsed_time(self, other):
            return 0.5  # ms total across iters

    class FakeTensor:
        def __init__(self, shape, data=None):
            self.shape = shape
            self._data = data if data is not None else [0] * (shape[0] if isinstance(shape, tuple) else 1)

        def cpu(self):
            return self

        def tolist(self):
            return list(self._data)

        def item(self):
            return float(self._data[0])

    fake_npu = SimpleNamespace(
        set_device=lambda dev: None,
        synchronize=lambda dev=None: None,
        Event=FakeEvent,
    )
    monkeypatch.setattr(collect_hccl, "torch_npu_module", lambda: fake_npu)

    init_calls = []

    def fake_all_gather_into_tensor(out_t, in_t):
        if out_t.shape == (2,) and in_t._data == [1]:
            out_t._data = [1, 2]
        else:
            collective_calls.append(("all_gather", out_t.shape, in_t.shape))

    def fake_all_reduce(t, op=None):
        if t.shape == (1,):
            return
        collective_calls.append(("all_reduce", t.shape))

    def fake_reduce_scatter_tensor(out_t, in_t, op=None):
        collective_calls.append(("reduce_scatter", out_t.shape, in_t.shape))

    fake_dist = SimpleNamespace(
        ReduceOp=SimpleNamespace(SUM="sum", MAX="max"),
        init_process_group=lambda **kw: init_calls.append(kw),
        get_world_size=lambda: 2,
        all_gather_into_tensor=fake_all_gather_into_tensor,
        all_reduce=fake_all_reduce,
        reduce_scatter_tensor=fake_reduce_scatter_tensor,
        barrier=lambda: None,
        is_initialized=lambda: True,
        destroy_process_group=lambda: None,
    )
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        int8="int8",
        int32="int32",
        float32="float32",
        device=lambda d: d,
        tensor=lambda vals, dtype=None, device=None: FakeTensor((len(vals),), list(vals)),
        zeros=lambda shape, dtype=None, device=None: FakeTensor(shape),
        ones=lambda shape, dtype=None, device=None: FakeTensor(shape),
        empty=lambda shape, dtype=None, device=None: FakeTensor(shape),
        distributed=fake_dist,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)

    collect_hccl._worker_main(
        dtype="half",
        hccl_op=op_name,
        test_range="512,2048,2",
        num_gpus=2,
        iters=10,
        warmup_iters=2,
    )

    assert init_calls == [{"backend": "hccl", "init_method": "env://", "world_size": 2, "rank": 0}]
    assert collective_calls
    out = capsys.readouterr().out
    items = collect_hccl._parse_worker_output(out, op_name, num_gpus=2, bytes_per_elem=2, dtype="half")
    assert len(items) == 2
    assert items[0]["message_size"] == 256
    assert items[0]["latency"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# collect_comm.sh --device npu syntax and branch check
# ---------------------------------------------------------------------------


def test_collect_comm_sh_supports_npu_device():
    res = subprocess.run(["bash", "-n", str(COLLECT_COMM_SH)], capture_output=True, text=True)
    assert res.returncode == 0, f"bash -n collect_comm.sh failed: {res.stderr}"

    content = COLLECT_COMM_SH.read_text(encoding="utf-8")
    assert '"$device" != "npu"' in content
    assert "collect_hccl.py" in content
    assert "npu-smi info -l" in content


# ---------------------------------------------------------------------------
# SDK communication.py HCCL fallback loading
# ---------------------------------------------------------------------------


def test_communication_nccl_loads_hccl_fallback_when_hccl_version_declared(tmp_path, monkeypatch):
    from aisimulate.sdk.operations import communication as comm_mod
    from aisimulate.sdk.operations.communication import NCCL
    from aisimulate_core.sdk import engine_table_view

    fetched = []

    def fake_fetch_table_view(db, attr):
        fetched.append(attr)
        if attr == "_hccl_data":
            return {"half": {"all_reduce": {2: {256: {"latency": 0.05}}}}}
        return None

    monkeypatch.setattr(engine_table_view, "fetch_table_view", fake_fetch_table_view)

    database = SimpleNamespace(
        systems_root=str(tmp_path),
        system="ascend_950dt",
        backend="vllm",
        version="0.30.0rc2",
        enable_shared_layer=True,
        system_spec={"data_dir": "data/ascend_950dt", "misc": {"hccl_version": "9.2.0"}},
    )

    NCCL.clear_cache()
    try:
        comm_mod.NCCL.load_data(database)
        assert database._nccl_data.loaded is False
        assert database._oneccl_data is None
        assert database._hccl_data is not None
        assert database._hccl_data.loaded is True
        assert database._hccl_data.filepath.endswith("hccl_perf.parquet")
        assert "_hccl_data" in fetched
    finally:
        NCCL.clear_cache()


def test_engine_table_view_and_query_nccl_fallback_to_hccl_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate.sdk import common
    from aisimulate.sdk.operations.communication import NCCL
    from aisimulate.sdk.perf_database import PerfDatabase

    systems_root = tmp_path / "systems"
    systems_root.mkdir()
    ascend_yaml = REPO_ROOT / "src" / "aisimulate_core" / "systems" / "ascend_950dt.yaml"
    (systems_root / "ascend_950dt.yaml").write_bytes(ascend_yaml.read_bytes())

    hccl_dir = systems_root / "data" / "ascend_950dt" / "comm" / "hccl" / "9.2.0"
    hccl_dir.mkdir(parents=True)
    (systems_root / "data" / "ascend_950dt" / "gemm" / "vllm" / "0.30.0rc2").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "framework": ["VLLM", "VLLM"],
                "version": ["9.2.0", "9.2.0"],
                "device": ["Ascend950DT", "Ascend950DT"],
                "op_name": ["all_reduce", "all_reduce"],
                "kernel_source": ["HCCL", "HCCL"],
                "nccl_dtype": ["half", "half"],
                "num_gpus": [2, 2],
                "message_size": [256, 512],
                "latency": [0.02, 0.04],
            }
        ),
        hccl_dir / "hccl_perf.parquet",
    )

    NCCL.clear_cache()
    try:
        db = PerfDatabase(
            system="ascend_950dt",
            backend="vllm",
            version="0.30.0rc2",
            systems_root=str(systems_root),
            database_mode="SILICON",
            strict_provenance=False,
        )
        NCCL.load_data(db)
        assert db._hccl_data is not None and db._hccl_data.loaded is True
        assert db._hccl_data[common.CommQuantMode.half]["all_reduce"][2][256]["latency"] == pytest.approx(0.02)
        assert "half" in db.supported_quant_mode["nccl"]

        op = NCCL("ar", 1.0, "all_reduce", 128, 2, common.CommQuantMode.half)
        res = op._engine_query(db, x=2)
        assert float(res) == pytest.approx(0.02)
    finally:
        NCCL.clear_cache()
