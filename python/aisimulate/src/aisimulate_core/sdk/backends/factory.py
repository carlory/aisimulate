# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aisimulate_core.sdk import common
from aisimulate_core.sdk.backends.base_backend import BaseBackend
from aisimulate_core.sdk.backends.sglang_backend import SGLANGBackend
from aisimulate_core.sdk.backends.trtllm_backend import TRTLLMBackend
from aisimulate_core.sdk.backends.vllm_backend import VLLMBackend
from aisimulate_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
from aisimulate_core.sdk.perf_database import VENDOR_INTEL, get_gpu_vendor, get_gpu_vendor_from_spec


def get_backend(backend_name: str, system_name: str | None = None, system_spec=None) -> BaseBackend:
    """
    Get the backend class by the backend name.

    On an Intel XPU ``system_name``, vLLM is served by the XPU-calibrated
    subclass; otherwise the default backends are returned. The selection is a
    positive vendor match (``gpu.vendor``): other non-NVIDIA devices must not
    inherit the Intel-fitted coefficients just because they carry no
    ``sm_version``.

    Raises:
        ValueError: If the backend name is not found.
    """
    name = common.BackendName[backend_name]
    vendor = get_gpu_vendor_from_spec(system_spec) if system_spec is not None else get_gpu_vendor(system_name)
    if name == common.BackendName.vllm and vendor == VENDOR_INTEL:
        return VLLMXPUBackend()

    backend_map = {
        common.BackendName.trtllm: TRTLLMBackend,
        common.BackendName.sglang: SGLANGBackend,
        common.BackendName.vllm: VLLMBackend,
    }

    backend_class = backend_map.get(name)
    if backend_class is None:
        raise ValueError(f"Unknown backend: {backend_name}")

    return backend_class()
