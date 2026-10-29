# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendor discrimination for backend selection (`gpu.vendor`).

`VLLMXPUBackend` carries coefficients fitted on Intel B60 hardware. They used
to be selected by "the spec has no sm_version", which is true of *every*
non-NVIDIA device — so any new non-CUDA system (Ascend NPU) would silently
inherit Intel-fitted TTFT/TPOT laws. Selection is now a positive match on the
spec's declared vendor, with the old inference kept only for specs written
before the key existed.
"""

import pytest

from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.backends.vllm_backend import VLLMBackend
from aisimulate_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
from aisimulate_core.sdk.perf_database import (
    VENDOR_ASCEND,
    VENDOR_INTEL,
    VENDOR_NVIDIA,
    get_gpu_vendor,
    get_gpu_vendor_from_spec,
    is_xpu_system,
    is_xpu_system_spec,
)

pytestmark = pytest.mark.unit


def test_b60_declares_intel_and_stays_xpu() -> None:
    assert get_gpu_vendor("b60") == VENDOR_INTEL
    assert is_xpu_system("b60") is True
    assert isinstance(get_backend("vllm", "b60"), VLLMXPUBackend)


def test_nvidia_systems_resolve_from_the_declared_or_legacy_vendor() -> None:
    assert get_gpu_vendor("h200_sxm") == VENDOR_NVIDIA
    assert is_xpu_system("h200_sxm") is False


def test_legacy_spec_without_vendor_keeps_its_old_classification() -> None:
    # Pre-`vendor` specs: sm_version ⇒ NVIDIA, otherwise the only non-NVIDIA
    # system that existed then (Intel XPU).
    assert get_gpu_vendor_from_spec({"gpu": {"sm_version": 90}}) == VENDOR_NVIDIA
    assert get_gpu_vendor_from_spec({"gpu": {"mem_capacity": 1}}) == VENDOR_INTEL
    assert is_xpu_system_spec({"gpu": {"mem_capacity": 1}}) is True


def test_declared_vendor_wins_over_the_absent_sm_version() -> None:
    """The must-fix: a non-NVIDIA, non-Intel device must not be routed to the
    Intel-calibrated backend just because it carries no sm_version."""
    ascend_spec = {"gpu": {"vendor": VENDOR_ASCEND, "mem_capacity": 1}}
    assert get_gpu_vendor_from_spec(ascend_spec) == VENDOR_ASCEND
    assert is_xpu_system_spec(ascend_spec) is False

    backend = get_backend("vllm", "ignored", system_spec=ascend_spec)
    assert type(backend) is VLLMBackend
    assert not isinstance(backend, VLLMXPUBackend)


def test_vendor_is_case_and_whitespace_tolerant() -> None:
    assert get_gpu_vendor_from_spec({"gpu": {"vendor": " Ascend "}}) == VENDOR_ASCEND


def test_unknown_vendor_fails_loud() -> None:
    # An unrecognized vendor must not fall back to a calibration by accident.
    with pytest.raises(ValueError, match="gpu.vendor"):
        get_gpu_vendor_from_spec({"gpu": {"vendor": "acme"}})
    with pytest.raises(ValueError, match="gpu.vendor"):
        get_gpu_vendor_from_spec({"gpu": {"vendor": 7}})


def test_missing_or_unresolvable_specs_behave_as_before() -> None:
    assert get_gpu_vendor(None) is None
    assert get_gpu_vendor_from_spec(None) is None
    assert is_xpu_system(None) is False
    assert is_xpu_system_spec(None) is False
    with pytest.raises(ValueError):
        get_gpu_vendor("nonexistent_typo_system")
    with pytest.raises(ValueError, match="Cannot resolve GPU spec"):
        get_gpu_vendor_from_spec({"gpu": "not-a-mapping"})
