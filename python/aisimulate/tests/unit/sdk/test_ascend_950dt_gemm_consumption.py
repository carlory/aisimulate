# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M4 consumption verification for the ascend_950dt gemm lane.

The data behind these assertions is the synthetic staging table described in
``src/aisimulate_core/systems/data/ascend_950dt/gemm/vllm/README.md`` (every
latency is exactly 1.0 ms). What is verified here is the *consumer chain*, not
any performance number: the loader discovers the version directory, the engine
table view returns the collected rows, the backend factory resolves the generic
vLLM backend (never the Intel-fitted XPU subclass), and the support-matrix
machinery enumerates the new (system, backend, version) cell.

M5 replaces the rows with a device collection; these assertions must keep
passing against the real table, except the 1.0 ms latency marker, which exists
only to prove the synthetic rows are the ones being read.
"""

from __future__ import annotations

import pytest
from tools.support_matrix.build_fpe_support_matrix import SYSTEM_ORDER
from tools.support_matrix.support_matrix import SupportMatrix

from aisimulate_core.sdk import common
from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.backends.vllm_backend import VLLMBackend
from aisimulate_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
from aisimulate_core.sdk.operations.gemm import GEMM
from aisimulate_core.sdk.perf_database import (
    get_database,
    get_supported_databases,
    get_version_slots,
)

pytestmark = pytest.mark.unit

SYSTEM = "ascend_950dt"
BACKEND = "vllm"
VERSION = "0.30.0rc2"
# The staged shapes (collector/cases/base_ops/gemm.yaml vllm_npu partition).
SYNTHETIC_SHAPES = ((1, 4096, 4096), (128, 4096, 4096), (2048, 4096, 4096))
SYNTHETIC_LATENCY_MS = 1.0


@pytest.fixture(scope="module")
def database():
    return get_database(SYSTEM, BACKEND, VERSION)


class TestPerfDatabaseConsumption:
    def test_version_directory_is_discovered(self) -> None:
        assert VERSION in get_supported_databases()[SYSTEM][BACKEND]

    def test_database_identity_is_the_ascend_slice(self, database) -> None:
        assert (database.system, database.backend, database.version) == (SYSTEM, BACKEND, VERSION)
        # The spec travels with the database: vendor discrimination downstream
        # reads it rather than inferring from the absent sm_version.
        assert database.system_spec["gpu"]["vendor"] == "ascend"

    def test_gemm_table_loads_as_bfloat16_only(self, database) -> None:
        """bf16 is the whole MVP lane; a stray fp8 key would mean the loader
        picked up rows this directory does not hold."""
        assert GEMM.supported_quant_modes(database) == {common.GEMMQuantMode.bfloat16}

    def test_collected_rows_are_returned_to_the_engine_table_view(self, database) -> None:
        GEMM.load_data(database)
        table = database._gemm_data[common.GEMMQuantMode.bfloat16]
        for m, n, k in SYNTHETIC_SHAPES:
            assert table[m][n][k]["latency"] == SYNTHETIC_LATENCY_MS


class TestMixedQuantMxfp4ConsumerContract:
    """Verify mixed BF16/FP8/MXFP8(fp8_block)/MXFP4(mxfp4) GEMM table consumption,
    weight byte sizing, and query behavior without aliasing mxfp4 to nvfp4."""

    def test_mxfp4_enum_identity_and_weights_distinct_from_nvfp4(self) -> None:
        assert common.GEMMQuantMode.mxfp4.value == common.QuantMapping(0.5, 4, "mxfp4", "fp4")
        assert common.GEMMQuantMode.nvfp4.value == common.QuantMapping(9 / 16, 4, "nvfp4", "fp4")
        assert common.GEMMQuantMode.mxfp4 is not common.GEMMQuantMode.nvfp4

        mxfp4_op = GEMM("mxfp4_gemm", 1.0, 4096, 4096, common.GEMMQuantMode.mxfp4)
        nvfp4_op = GEMM("nvfp4_gemm", 1.0, 4096, 4096, common.GEMMQuantMode.nvfp4)
        assert mxfp4_op._quant_mode is common.GEMMQuantMode.mxfp4
        assert mxfp4_op.get_weights() == 4096 * 4096 * 0.5
        assert nvfp4_op.get_weights() == 4096 * 4096 * (9 / 16)
        assert mxfp4_op.get_weights() != nvfp4_op.get_weights()

    def test_mixed_gemm_table_loads_all_four_quants_and_queries_supported_lanes(
        self,
        tmp_path,
    ) -> None:
        import shutil

        import pyarrow as pa
        import pyarrow.parquet as pq

        from aisimulate_core.sdk.errors import MissingSystemFlopsError, PerfDataNotAvailableError
        from aisimulate_core.sdk.perf_database import PerfDatabase

        systems_root = tmp_path / "systems"
        systems_root.mkdir()
        repo_systems = (
            shutil.copy(
                get_database(SYSTEM, BACKEND, VERSION).systems_root + f"/{SYSTEM}.yaml",
                systems_root / f"{SYSTEM}.yaml",
            ),
        )
        assert repo_systems
        gemm_dir = systems_root / "data" / SYSTEM / "gemm" / BACKEND / VERSION
        gemm_dir.mkdir(parents=True)

        rows = [
            ("bfloat16", 16, 4096, 4096, 0.020),
            ("bfloat16", 128, 4096, 4096, 0.120),
            ("fp8", 16, 4096, 4096, 0.015),
            ("fp8", 128, 4096, 4096, 0.080),
            ("fp8_block", 16, 4096, 4096, 0.018),
            ("fp8_block", 128, 4096, 4096, 0.085),
            ("mxfp4", 16, 4096, 4096, 0.010),
            ("mxfp4", 128, 4096, 4096, 0.050),
        ]
        pq.write_table(
            pa.table(
                {
                    "framework": ["VLLM"] * len(rows),
                    "version": [VERSION] * len(rows),
                    "device": ["Ascend950DT"] * len(rows),
                    "op_name": ["gemm"] * len(rows),
                    "kernel_source": ["vllm_default"] * len(rows),
                    "gemm_dtype": [r[0] for r in rows],
                    "m": pa.array([r[1] for r in rows], type=pa.int64()),
                    "n": pa.array([r[2] for r in rows], type=pa.int64()),
                    "k": pa.array([r[3] for r in rows], type=pa.int64()),
                    "latency": pa.array([r[4] for r in rows], type=pa.float64()),
                }
            ),
            gemm_dir / "gemm_perf.parquet",
        )

        GEMM.clear_cache()
        db = PerfDatabase(SYSTEM, BACKEND, VERSION, systems_root=str(systems_root))
        expected_modes = {
            common.GEMMQuantMode.bfloat16,
            common.GEMMQuantMode.fp8,
            common.GEMMQuantMode.fp8_block,
            common.GEMMQuantMode.mxfp4,
        }
        assert GEMM.supported_quant_modes(db) == expected_modes
        assert set(db.supported_quant_mode["gemm"]) == {"bfloat16", "fp8", "fp8_block", "mxfp4"}
        assert common.GEMMQuantMode.nvfp4 not in db._gemm_data
        assert db._gemm_data[common.GEMMQuantMode.mxfp4][16][4096][4096]["latency"] == pytest.approx(0.010)

        # Exact queries for BF16, FP8, and MXFP8 (fp8_block) succeed on ascend_950dt.yaml.
        for mode, expected_latency in [
            (common.GEMMQuantMode.bfloat16, 0.020),
            (common.GEMMQuantMode.fp8, 0.015),
            (common.GEMMQuantMode.fp8_block, 0.018),
        ]:
            op = GEMM(f"gemm_{mode.name}", 1.0, 4096, 4096, mode)
            assert float(op._engine_query(db, x=16)) == pytest.approx(expected_latency)

        # On current ascend_950dt.yaml (fp4_tc_flops configured), mxfp4 exact query
        # succeeds on its own table slice.
        mxfp4_op = GEMM("gemm_mxfp4", 1.0, 4096, 4096, common.GEMMQuantMode.mxfp4)
        assert float(mxfp4_op._engine_query(db, x=16)) == pytest.approx(0.010)

        # nvfp4 remains a distinct quant mode and still misses (never aliases mxfp4).
        nvfp4_op = GEMM("gemm_nvfp4", 1.0, 4096, 4096, common.GEMMQuantMode.nvfp4)
        with pytest.raises(PerfDataNotAvailableError):
            nvfp4_op._engine_query(db, x=16)

        # Removing fp4_tc_flops from an otherwise identical system spec must restore
        # the fail-closed guard at query entry.
        systems_root_no_fp4 = tmp_path / "systems_no_fp4"
        shutil.copytree(systems_root, systems_root_no_fp4)
        spec_path = systems_root_no_fp4 / f"{SYSTEM}.yaml"
        import yaml

        spec = yaml.safe_load(spec_path.read_text())
        spec["gpu"].pop("fp4_tc_flops", None)
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
        GEMM.clear_cache()
        db_no_fp4 = PerfDatabase(SYSTEM, BACKEND, VERSION, systems_root=str(systems_root_no_fp4))
        with pytest.raises(MissingSystemFlopsError, match="fp4_tc_flops"):
            mxfp4_op._engine_query(db_no_fp4, x=16)


class TestBackendResolution:
    def test_ascend_resolves_to_the_generic_vllm_backend(self, database) -> None:
        """No sm_version must not mean "Intel B60 fitting"."""
        backend = get_backend(BACKEND, SYSTEM, system_spec=database.system_spec)
        assert type(backend) is VLLMBackend
        assert not isinstance(backend, VLLMXPUBackend)

    def test_backend_resolution_by_system_name_agrees(self) -> None:
        assert type(get_backend(BACKEND, SYSTEM)) is VLLMBackend


class TestVersionSlotContainment:
    """The `ascend_950dt` override in `query_versions.yaml` isolates the
    vllm-ascend version axis in both directions. Override systems expose no
    `next` — blocking the fleet-derived `next` (`0.31.0`) and the fleet
    base-vLLM defaults (`0.30.0` / `0.24.0`) from entering `ascend_950dt`'s
    slots — and are excluded from `_derive_fleet_next`, keeping vllm-ascend
    release strings out of the NVIDIA fleet's slots."""

    def test_ascend_is_pinned_to_the_vllm_ascend_release(self) -> None:
        slots = get_version_slots(SYSTEM, BACKEND)
        assert slots == {"current": VERSION}
        assert "0.31.0" not in slots.values()

    @pytest.mark.parametrize(
        "nvidia_system",
        ["b200_sxm", "b300_sxm", "gb200", "gb300", "h100_sxm", "h200_sxm", "l40s", "rtx_pro_6000_server"],
    )
    def test_the_vllm_ascend_version_never_leaks_into_an_nvidia_slot(self, nvidia_system: str) -> None:
        slots = get_version_slots(nvidia_system, BACKEND)
        assert slots is not None
        assert VERSION not in slots.values()


class TestSupportMatrixCoverage:
    def test_the_generator_enumerates_an_ascend_gemm_cell(self) -> None:
        """The support-matrix generators iterate discovered databases, so the
        staged table is what makes ascend_950dt rows appear at the next
        regeneration (the committed CSVs are a CI-produced fleet artifact)."""
        combinations = SupportMatrix().generate_combinations()
        cells = {(system, backend, version) for _model, system, backend, version in combinations}
        assert (SYSTEM, BACKEND, VERSION) in cells

    def test_fpe_matrix_orders_ascend_after_the_nvidia_and_intel_fleet(self) -> None:
        """Without this the Ascend rows land in the unordered tail of the FPE
        web matrix."""
        assert SYSTEM in SYSTEM_ORDER
        assert SYSTEM_ORDER[-2:] == ("b60", SYSTEM)
