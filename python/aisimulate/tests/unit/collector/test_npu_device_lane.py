# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU device abstraction: helper branches, probe, and registry dispatch.

No Ascend hardware (and no ``torch_npu`` wheel) exists on CI or dev machines,
so every test here stubs ``torch``/``torch_npu`` in ``sys.modules``. The point
is the branch structure: which device module a request resolves to, in which
order the no-argument fallback probes, and which vLLM registry a host selects.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# collect.py is a top-level script (`from helper import ...`); put collector/ on
# sys.path the same way the rest of the collect.py test suite does.
_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

import helper as helper_mod  # noqa: E402

pytestmark = pytest.mark.unit


def _device_module(name, available=True):
    return SimpleNamespace(__name__=name, is_available=lambda: available)


def _install_torch(monkeypatch, *, cuda=False, xpu=False, npu=None, npu_plugin=False):
    """Install a stub ``torch`` (and optionally the ``torch_npu`` plugin).

    ``npu`` is the module ``torch_npu`` attaches as ``torch.npu``; when
    ``npu_plugin`` is False importing ``torch_npu`` fails, exactly like a host
    without the Ascend wheel.
    """
    torch = SimpleNamespace(cuda=_device_module("cuda", cuda), xpu=_device_module("xpu", xpu))
    monkeypatch.setitem(sys.modules, "torch", torch)

    if npu_plugin:
        # torch_npu attaches torch.npu as an import side effect; with the module
        # already in sys.modules the import is a no-op, so attach it here.
        monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__name__="torch_npu"))
        torch.npu = npu
    else:
        monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
        monkeypatch.setattr(sys, "meta_path", [_BlockTorchNpu(), *sys.meta_path])
    helper_mod.get_device_str.cache_clear()
    return torch


class _BlockTorchNpu:
    """Meta-path finder that makes ``import torch_npu`` raise ImportError."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch_npu":
            raise ImportError("No module named 'torch_npu'")
        return None


def test_torch_npu_module_is_none_without_the_plugin(monkeypatch):
    _install_torch(monkeypatch)
    assert helper_mod.torch_npu_module() is None
    assert helper_mod.npu_available() is False


def test_torch_npu_module_returns_torch_npu_after_plugin_import(monkeypatch):
    npu = _device_module("npu", available=True)
    _install_torch(monkeypatch, npu=npu, npu_plugin=True)
    assert helper_mod.torch_npu_module() is npu
    assert helper_mod.npu_available() is True


def test_npu_available_is_false_when_plugin_sees_no_device(monkeypatch):
    npu = _device_module("npu", available=False)
    _install_torch(monkeypatch, npu=npu, npu_plugin=True)
    assert helper_mod.npu_available() is False


def test_get_device_module_resolves_requested_npu_device(monkeypatch):
    npu = _device_module("npu", available=True)
    # CUDA is "available" too: an explicit device request must win over the
    # availability fallback, exactly like the existing xpu branch.
    torch = _install_torch(monkeypatch, cuda=True, npu=npu, npu_plugin=True)
    assert helper_mod.get_device_module("npu:0") is npu
    assert helper_mod.get_device_module("cuda:0") is torch.cuda


def test_get_device_module_rejects_npu_request_without_plugin(monkeypatch):
    _install_torch(monkeypatch, cuda=True)
    with pytest.raises(RuntimeError, match="torch_npu"):
        helper_mod.get_device_module("npu:0")


@pytest.mark.parametrize(
    ("cuda", "xpu", "expected"),
    [(True, False, "cuda"), (False, True, "xpu"), (False, False, "npu")],
)
def test_no_argument_fallback_probes_npu_last(monkeypatch, cuda, xpu, expected):
    npu = _device_module("npu", available=True)
    torch = _install_torch(monkeypatch, cuda=cuda, xpu=xpu, npu=npu, npu_plugin=True)
    expected_module = {"cuda": torch.cuda, "xpu": torch.xpu, "npu": npu}[expected]
    assert helper_mod.get_device_module() is expected_module
    assert helper_mod.get_device_str() == expected


def test_no_device_raises_naming_all_three_lanes(monkeypatch):
    _install_torch(monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA, XPU or NPU"):
        helper_mod.get_device_module()
    with pytest.raises(RuntimeError, match="CUDA, XPU or NPU"):
        helper_mod.get_device_str()
