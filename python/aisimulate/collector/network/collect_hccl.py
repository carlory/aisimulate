# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collect HCCL communication performance data for Huawei Ascend NPU.

This script uses ``torch.distributed`` with the ``"hccl"`` process-group backend
(spawned across NPU ranks via ``torchrun``) to measure collective communication
latencies on Ascend NPU devices. It produces ``hccl_perf.txt`` compatible with
``nccl_perf.txt`` / ``oneccl_perf.txt`` for use in projection models.

Prerequisites:
  - PyTorch + torch_npu + CANN (HCCL) installed
  - Multiple Ascend NPU devices available

Usage:
  python collector/network/collect_hccl.py --hccl_op all_reduce --dtype half --num_gpus 2
  python collector/network/collect_hccl.py --hccl_op all_gather --dtype half --num_gpus 4
  python collector/network/collect_hccl.py --hccl_op reduce_scatter --dtype half --num_gpus 8
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import log_perf, torch_npu_module

SUPPORTED_OPS = ("all_gather", "reduce_scatter", "all_reduce")

DTYPE_TO_TORCH = {
    "half": "bfloat16",
    "int8": "int8",
}

BYTES_PER_ELEMENT = {
    "half": 2,
    "int8": 1,
}

_RESULT_PREFIX = "HCCL_BENCH_RESULT_JSON:"


def _parse_range(test_range: str) -> tuple[int, int, int]:
    try:
        min_bytes, max_bytes, ratio = (int(part) for part in test_range.split(","))
    except ValueError as exc:
        raise ValueError("--range must be 'min_bytes,max_bytes,multiplicative_ratio' integers") from exc
    if min_bytes <= 0 or max_bytes <= min_bytes or ratio <= 1:
        raise ValueError(f"--range requires min_bytes > 0, max_bytes > min_bytes, and ratio > 1, got {test_range!r}")
    return min_bytes, max_bytes, ratio


def get_sweep_byte_sizes(test_range: str, num_gpus: int, bytes_per_elem: int) -> list[int]:
    """Generate byte sizes in the geometric sweep, aligned to ``num_gpus * bytes_per_elem``."""
    min_bytes, max_bytes, ratio = _parse_range(test_range)
    unit = max(num_gpus * bytes_per_elem, 1)
    sizes: list[int] = []
    size = min_bytes
    while size < max_bytes:
        aligned = max(unit, (size // unit) * unit)
        if not sizes or sizes[-1] != aligned:
            sizes.append(aligned)
        size *= ratio
    return sizes


def get_hccl_version() -> str:
    """Get installed CANN / HCCL version string (env, version.cfg, else torch_npu wheel)."""
    cann_ver = os.environ.get("CANN_VERSION", "").strip()
    if cann_ver:
        return cann_ver.split("_")[0]

    search_roots = []
    for env_key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        val = os.environ.get(env_key, "").strip()
        if val:
            search_roots.append(Path(val))
    search_roots.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/cann"),
        ]
    )
    for root in search_roots:
        for rel in ("version.cfg", "opp/version.info", "hccl/version.info"):
            cfg = root / rel
            try:
                if cfg.is_file():
                    text = cfg.read_text(encoding="utf-8", errors="ignore")
                    match = re.search(r"Version=([0-9A-Za-z._-]+)", text) or re.search(
                        r"\[([0-9]+\.[0-9]+[^\]]*)\]", text
                    )
                    if match:
                        return match.group(1).strip().split("_")[0]
            except OSError:
                continue

    try:
        import importlib.metadata as im

        return im.version("torch-npu")
    except Exception:
        pass
    return "unknown_version"


def get_device_name() -> str:
    """Ascend NPU device name via torch_npu, else npu-smi."""
    try:
        npu = torch_npu_module()
        if npu is not None and hasattr(npu, "get_device_name"):
            name = npu.get_device_name(0)
            if name:
                return str(name).strip()
    except Exception:
        pass
    try:
        result = subprocess.run(["npu-smi", "info", "-t", "board", "-i", "0"], capture_output=True, text=True)
        if result.returncode == 0:
            match = re.search(r"Chip Name\s*:\s*(\S+)", result.stdout)
            if match:
                return match.group(1).strip()
    except Exception:
        pass
    raise RuntimeError("Could not identify the Ascend NPU device name (torch_npu and npu-smi both failed)")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_collective_op(dist, hccl_op: str, tensor, output_tensor, input_tensor) -> None:
    if hccl_op == "all_reduce":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif hccl_op == "all_gather":
        dist.all_gather_into_tensor(output_tensor, input_tensor)
    elif hccl_op == "reduce_scatter":
        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    else:
        raise ValueError(f"Unsupported HCCL operation: {hccl_op!r}")


def _worker_main(
    dtype: str,
    hccl_op: str,
    test_range: str,
    num_gpus: int,
    iters: int,
    warmup_iters: int,
) -> None:
    """Execute the collective sweep inside a torchrun worker process."""
    import torch
    import torch.distributed as dist

    npu = torch_npu_module()
    if npu is None:
        raise RuntimeError("torch_npu is required to run HCCL collective benchmarks on Ascend NPU")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device = torch.device(f"npu:{local_rank}")
    npu.set_device(device)

    dist.init_process_group(backend="hccl", init_method="env://", world_size=env_world_size, rank=rank)
    try:
        world_size = int(dist.get_world_size())
        # Anti-forgery check: verify the live HCCL process group world_size and
        # an all-gather of rank IDs match num_gpus before timing any collective.
        if world_size != num_gpus:
            raise RuntimeError(f"HCCL {hccl_op} ran with {world_size} ranks, expected {num_gpus}")

        rank_probe = torch.tensor([rank + 1], dtype=torch.int32, device=device)
        gathered_ranks = torch.zeros((world_size,), dtype=torch.int32, device=device)
        dist.all_gather_into_tensor(gathered_ranks, rank_probe)
        npu.synchronize(device)
        observed_ranks = [int(v) - 1 for v in gathered_ranks.cpu().tolist()]
        if observed_ranks != list(range(num_gpus)):
            expected_ranks = list(range(num_gpus))
            raise RuntimeError(
                f"HCCL {hccl_op} rank attestation failed: observed ranks {observed_ranks}, expected {expected_ranks}"
            )

        torch_dtype = getattr(torch, DTYPE_TO_TORCH[dtype])
        bytes_per_elem = BYTES_PER_ELEMENT[dtype]
        byte_sizes = get_sweep_byte_sizes(test_range, num_gpus, bytes_per_elem)

        records: list[dict[str, float | int]] = []
        for size_bytes in byte_sizes:
            # message_size is total buffer elements across the collective,
            # uniform across all_reduce / all_gather / reduce_scatter, matching
            # collect_nccl.py and collect_oneccl_xpu.py.
            msg_elements = size_bytes // bytes_per_elem
            chunk_elements = msg_elements // num_gpus

            if hccl_op == "all_reduce":
                tensor = torch.ones((msg_elements,), dtype=torch_dtype, device=device)
                output_tensor = None
                input_tensor = None
            elif hccl_op == "all_gather":
                tensor = None
                input_tensor = torch.ones((chunk_elements,), dtype=torch_dtype, device=device)
                output_tensor = torch.empty((msg_elements,), dtype=torch_dtype, device=device)
            elif hccl_op == "reduce_scatter":
                tensor = None
                input_tensor = torch.ones((msg_elements,), dtype=torch_dtype, device=device)
                output_tensor = torch.empty((chunk_elements,), dtype=torch_dtype, device=device)
            else:
                raise ValueError(f"Unsupported HCCL operation: {hccl_op!r}")

            for _ in range(warmup_iters):
                _run_collective_op(dist, hccl_op, tensor, output_tensor, input_tensor)
            npu.synchronize(device)
            dist.barrier()

            start_event = npu.Event(enable_timing=True)
            end_event = npu.Event(enable_timing=True)
            start_event.record()
            for _ in range(iters):
                _run_collective_op(dist, hccl_op, tensor, output_tensor, input_tensor)
            end_event.record()
            npu.synchronize(device)

            elapsed_ms = float(start_event.elapsed_time(end_event)) / max(iters, 1)
            lat_tensor = torch.tensor([elapsed_ms], dtype=torch.float32, device=device)
            dist.all_reduce(lat_tensor, op=dist.ReduceOp.MAX)
            npu.synchronize(device)
            latency_ms = float(lat_tensor.item())

            if rank == 0:
                records.append(
                    {
                        "size_bytes": size_bytes,
                        "message_size": msg_elements,
                        "latency_ms": latency_ms,
                    }
                )

            del tensor, input_tensor, output_tensor

        if rank == 0:
            payload = {
                "procs": world_size,
                "ranks": observed_ranks,
                "records": records,
            }
            print(f"{_RESULT_PREFIX}{json.dumps(payload)}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_worker_output(stdout: str, hccl_op: str, num_gpus: int, bytes_per_elem: int, dtype: str) -> list[dict]:
    """Parse and validate worker benchmark output, enforcing the rank anti-forgery check."""
    payload = None
    for line in stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            payload = json.loads(line[len(_RESULT_PREFIX) :].strip())

    # Also support "# procs: N" header + tabular lines for compatibility with
    # external benchmark runners.
    if payload is None:
        procs_match = re.search(r"^#\s*procs:\s*(\d+)", stdout, re.MULTILINE)
        procs = int(procs_match.group(1)) if procs_match else None
        if procs != num_gpus:
            raise RuntimeError(f"HCCL {hccl_op} ran with {procs} ranks, expected {num_gpus}")
        items = []
        for line in stdout.splitlines():
            toks = line.split()
            if len(toks) < 6 or not toks[0].isdigit():
                continue
            try:
                size_bytes = int(toks[0])
                latency_ms = float(toks[5]) * 1e-3
            except (ValueError, IndexError):
                continue
            if not math.isfinite(latency_ms) or latency_ms <= 0:
                raise RuntimeError(f"Invalid HCCL latency: {toks[5]}")
            msg_elements = size_bytes // bytes_per_elem
            print(f"    {hccl_op}: {msg_elements} elems, latency={latency_ms:.6f} ms")
            items.append(
                {
                    "nccl_dtype": dtype,
                    "num_gpus": num_gpus,
                    "message_size": msg_elements,
                    "latency": latency_ms,
                }
            )
        if not items:
            raise RuntimeError(f"HCCL {hccl_op} produced no parseable records")
        return items

    procs = payload.get("procs")
    if procs != num_gpus:
        raise RuntimeError(f"HCCL {hccl_op} ran with {procs} ranks, expected {num_gpus}")
    ranks = payload.get("ranks")
    if ranks != list(range(num_gpus)):
        raise RuntimeError(f"HCCL {hccl_op} rank attestation mismatch: got {ranks}, expected {list(range(num_gpus))}")

    items = []
    for rec in payload.get("records", []):
        msg_elements = int(rec["message_size"])
        latency_ms = float(rec["latency_ms"])
        if not math.isfinite(latency_ms) or latency_ms <= 0 or msg_elements <= 0:
            raise RuntimeError(f"Invalid HCCL record: {rec!r}")
        print(f"    {hccl_op}: {msg_elements} elems, latency={latency_ms:.6f} ms")
        items.append(
            {
                "nccl_dtype": dtype,
                "num_gpus": num_gpus,
                "message_size": msg_elements,
                "latency": latency_ms,
            }
        )

    if not items:
        raise RuntimeError(f"HCCL {hccl_op} produced no parseable records")
    return items


def hccl_benchmark(
    dtype: str = "half",
    hccl_op: str = "all_gather",
    test_range: str = "512,536870913,2",
    num_gpus: int = 2,
    iters: int = 100,
    warmup_iters: int = 20,
    perf_filename: str = "hccl_perf.txt",
) -> None:
    """Spawn NPU workers via torchrun, run the HCCL collective sweep, and log results."""
    if hccl_op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported hccl_op {hccl_op!r}; expected one of {SUPPORTED_OPS}")
    if dtype not in DTYPE_TO_TORCH:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {tuple(DTYPE_TO_TORCH)}")
    if num_gpus < 2:
        raise ValueError(f"--num_gpus must be >= 2 for collective benchmarks, got {num_gpus}")

    min_bytes, max_bytes, ratio = _parse_range(test_range)
    ccl_dtype = DTYPE_TO_TORCH[dtype]
    bytes_per_elem = BYTES_PER_ELEMENT[dtype]
    version = get_hccl_version()
    device_name = get_device_name()

    torchrun_bin = shutil.which("torchrun")
    master_port = str(_find_free_port())
    script_path = str(Path(__file__).resolve())
    worker_args = [
        script_path,
        "--worker",
        "--hccl_op",
        hccl_op,
        "--dtype",
        dtype,
        "--range",
        test_range,
        "--num_gpus",
        str(num_gpus),
        "--iters",
        str(iters),
        "--warmup_iters",
        str(warmup_iters),
    ]
    if torchrun_bin:
        cmd = [
            torchrun_bin,
            f"--nproc_per_node={num_gpus}",
            f"--master_port={master_port}",
            *worker_args,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={master_port}",
            *worker_args,
        ]

    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", master_port)

    print(
        f"Running HCCL {hccl_op}: dtype={dtype}({ccl_dtype}), num_gpus={num_gpus}, {min_bytes}..{max_bytes}B x{ratio}"
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"HCCL {hccl_op} did not complete within 600s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"HCCL {hccl_op} failed (exit {result.returncode}): {result.stderr[:500]}")

    items = _parse_worker_output(result.stdout, hccl_op, num_gpus, bytes_per_elem, dtype)

    log_perf(
        item_list=items,
        framework="VLLM",
        version=version,
        device_name=device_name,
        op_name=hccl_op,
        kernel_source="HCCL",
        perf_filename=perf_filename,
    )
    print(f"Done. Results appended to {perf_filename}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Collect HCCL communication performance data for Ascend NPU")
    parser.add_argument(
        "--hccl_op",
        "-O",
        default="all_gather",
        choices=list(SUPPORTED_OPS),
        help="HCCL collective operation to benchmark",
    )
    parser.add_argument(
        "--dtype",
        "-t",
        default="half",
        choices=["half", "int8"],
        help="Data type for the collective operation (half -> bf16, int8 -> int8)",
    )
    parser.add_argument(
        "--range",
        "-r",
        default="512,536870913,2",  # 512B to 512MB, multiply by 2
        help="min_bytes,max_bytes,multiplicative_ratio",
    )
    parser.add_argument("--num_gpus", "-n", default=2, type=int, help="Number of NPUs (ranks)")
    parser.add_argument("--iters", "-i", default=100, type=int, help="Benchmark iterations per size")
    parser.add_argument("--warmup_iters", "-w", default=20, type=int, help="Warmup iterations per size")
    parser.add_argument(
        "--perf-filename",
        default="hccl_perf.txt",
        help="Output performance filename (default: hccl_perf.txt)",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal flag for torchrun worker processes",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.worker:
        _worker_main(
            dtype=args.dtype,
            hccl_op=args.hccl_op,
            test_range=args.range,
            num_gpus=args.num_gpus,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
        )
    else:
        hccl_benchmark(
            dtype=args.dtype,
            hccl_op=args.hccl_op,
            test_range=args.range,
            num_gpus=args.num_gpus,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
            perf_filename=args.perf_filename,
        )
