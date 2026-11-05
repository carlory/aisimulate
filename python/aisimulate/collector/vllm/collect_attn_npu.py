# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM dense attention collector for Huawei Ascend NPU devices (vllm-ascend).

The NPU counterpart to the CUDA vLLM dense attention collector: it resolves the
platform attention backend (`AscendAttentionBackend`, registered under
`AttentionBackendEnum.CUSTOM`), builds `AscendCommonAttentionMetadata` and a
paged KV cache (`[2, num_blocks, block_size, num_kv_heads, head_dim]`), and
logs context/generation attention rows through the device-agnostic helpers.
MVP scope: bf16 compute and bf16 KV cache only, eager execution, no power
sampling.

Deviation note (serving parity): vllm-ascend defaults to `FULL_AND_PIECEWISE`
ACLGraph capture in serving (`vllm_ascend/platform.py`), whereas this MVP
collector runs in eager mode (`use_cuda_graph=False`, recorded as
`used_cuda_graph=False` on every row) until `benchmark_with_power` adds an
`NPUGraph` capture arm.

The pinned version is the vllm-ascend release, not base vLLM: vllm-ascend
ships the Ascend attention backend and kernels, so it is the kernel-source
identity of these rows and the version axis of the parquet directory
(manifest key `vllm_npu`).
"""

__compat__ = "vllm_ascend==0.30.0rc2"

import importlib.metadata as im
import inspect
import os

import torch
import vllm
from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)
from collector.helper import benchmark_with_power, get_device_module, log_perf
from collector.vllm.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_kv_cache_and_block_mappings,
    create_standard_kv_cache_spec,
    create_vllm_config,
    framework_kv_block_size,
    get_attention_backend,
    with_exit_stack,
)
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform

try:
    from vllm.attention.backends.registry import AttentionBackendEnum
except ImportError:
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        AttentionBackendEnum = None  # type: ignore

try:
    from vllm.v1.attention.backend import AttentionType
except ImportError:
    AttentionType = None  # type: ignore

try:
    from vllm.utils import resolve_obj_by_qualname
except ImportError:
    from vllm.utils.import_utils import resolve_obj_by_qualname  # type: ignore

vllm_ascend_version = im.version("vllm-ascend")


class MockAttentionLayer:
    """Mock attention layer satisfying AscendAttentionBackendImpl.forward."""

    def __init__(self, device: torch.device, layer_name: str = "placeholder"):
        # AscendAttentionBackendImpl.forward reads layer.layer_name and asserts
        # layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        # (vllm_ascend/attention/attention_v1.py:704,709 @ v0.30.0rc2).
        self.layer_name = layer_name
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


# The KV pool run_attention_torch allocates: num_gpu_blocks * block_size *
# num_kv_heads * head_dim * bytes(2) * 2(K+V). Refuse a pool that exceeds the
# device memory budget before allocating it.
_KV_POOL_BLOCKS = 8192
_KV_DEFAULT_BLOCK_SIZE = 128
_KV_POOL_MEM_FRACTION = 0.45


def _kv_pool_does_not_fit(
    num_kv_heads: int,
    head_dim: int,
    total_memory: int,
    block_size: int = _KV_DEFAULT_BLOCK_SIZE,
) -> str | None:
    """Why the case's KV pool cannot fit the device, or None when it can."""
    pool = _KV_POOL_BLOCKS * block_size * num_kv_heads * head_dim * 2 * 2
    allowed = int(total_memory * _KV_POOL_MEM_FRACTION)
    if pool <= allowed:
        return None
    return (
        f"FIXME(kernel-limit): does not fit: the KV pool needs {pool / 2**30:.1f} GiB, "
        f"the device allows {allowed / 2**30:.1f} GiB ({_KV_POOL_MEM_FRACTION:g} of its memory)"
    )


def _select_attn_backend_cls(
    head_dim: int,
    dtype: torch.dtype,
    block_size: int | None,
    has_sink: bool,
    num_heads: int,
):
    """Select the platform attention backend class through vLLM's selector."""
    try:
        from vllm.v1.attention.selector import AttentionSelectorConfig

        attn_selector_config = AttentionSelectorConfig(
            head_size=head_dim,
            dtype=dtype,
            kv_cache_dtype="auto",
            block_size=block_size,
            use_mla=False,
            has_sink=has_sink,
            use_sparse=False,
            use_mm_prefix=False,
        )
        try:
            backend = current_platform.get_attn_backend_cls(
                None,
                attn_selector_config,
                num_heads=num_heads,
            )
        except TypeError:
            backend = current_platform.get_attn_backend_cls(None, attn_selector_config)
    except Exception:
        backend = current_platform.get_attn_backend_cls(
            None,
            head_dim,
            dtype,
            kv_cache_dtype=None,
            block_size=block_size or _KV_DEFAULT_BLOCK_SIZE,
            use_mla=False,
            has_sink=has_sink,
            use_sparse=False,
        )
    return resolve_obj_by_qualname(backend)


def _resolve_kv_block_size(backend_cls) -> int:
    """Return the KV block size the backend runs with (128 on AscendAttentionBackend)."""
    try:
        return int(framework_kv_block_size(backend_cls))
    except Exception:
        get_sizes = getattr(backend_cls, "get_supported_kernel_block_sizes", None)
        if callable(get_sizes):
            sizes = get_sizes()
            if sizes:
                return int(sizes[0])
        return _KV_DEFAULT_BLOCK_SIZE


def _build_ascend_common_attn_metadata(
    common_attn_metadata,
    is_context_phase: bool,
):
    """Upgrade CommonAttentionMetadata to AscendCommonAttentionMetadata when available.

    Serving population site: vllm_ascend/worker/model_runner_v1.py:1141-1168
    (_build_attn_state: PrefillNoCache when num_computed_tokens == 0,
    DecodeOnly when num_scheduled_tokens == 1) and :1660-1705
    (AscendCommonAttentionMetadata with _seq_lens_cpu / seq_lens_cpu /
    actual_seq_lengths_q / attn_state) @ v0.30.0rc2.
    """
    seq_lens_cpu = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
    actual_seq_lengths_q = common_attn_metadata.query_start_loc_cpu[1:].tolist()

    try:
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

        attn_state = AscendAttentionState.PrefillNoCache if is_context_phase else AscendAttentionState.DecodeOnly
        return AscendCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=True,
            actual_seq_lengths_q=actual_seq_lengths_q,
            attn_state=attn_state,
            num_input_tokens=common_attn_metadata.num_actual_tokens,
        )
    except ImportError:
        if not hasattr(common_attn_metadata, "_seq_lens_cpu") or common_attn_metadata._seq_lens_cpu is None:
            common_attn_metadata._seq_lens_cpu = seq_lens_cpu
        if not hasattr(common_attn_metadata, "seq_lens_cpu") or common_attn_metadata.seq_lens_cpu is None:
            common_attn_metadata.seq_lens_cpu = seq_lens_cpu
        return common_attn_metadata


def _enter_forward_context(exit_stack, attn_metadata, vllm_config, num_tokens: int, has_sinks: bool) -> None:
    """Enter vllm-ascend's forward context so _EXTRA_CTX.capturing resolves in eager mode."""
    try:
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        exit_stack.enter_context(
            set_ascend_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=vllm_config,
                num_tokens=num_tokens,
                num_actual_tokens=num_tokens,
                has_sinks=has_sinks,
            )
        )
        return
    except ImportError:
        pass

    try:
        from vllm.forward_context import set_forward_context

        exit_stack.enter_context(
            set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=vllm_config,
                num_tokens=num_tokens,
            )
        )
    except ImportError:
        pass


def _instantiate_impl(
    impl_cls,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
    num_kv_heads: int,
    sliding_window: int | None,
    sinks: torch.Tensor | None,
):
    """Instantiate the attention backend impl, supplying AscendAttentionBackendImpl's required args.

    AscendAttentionBackendImpl.__init__ (vllm_ascend/attention/attention_v1.py:306-320 @ v0.30.0rc2)
    declares (num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
    kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name, sinks=None)
    without defaults on logits_soft_cap / attn_type / kv_sharing_target_layer_name.
    """
    decoder_type = getattr(AttentionType, "DECODER", "decoder")
    kwargs = {
        "num_heads": num_heads,
        "head_size": head_dim,
        "scale": scale,
        "num_kv_heads": num_kv_heads,
        "alibi_slopes": None,
        "sliding_window": sliding_window,
        "kv_cache_dtype": "auto",
        "logits_soft_cap": None,
        "attn_type": decoder_type,
        "kv_sharing_target_layer_name": None,
        "sinks": sinks,
    }
    try:
        params = inspect.signature(impl_cls.__init__).parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not accepts_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (ValueError, TypeError):
        pass
    return impl_cls(**kwargs)


def _forward_with_optional_kv_cache_update(
    backend_cls, impl, layer, query, key, value, kv_cache, attn_metadata, output
):
    if not getattr(backend_cls, "forward_includes_kv_cache_update", True):
        impl.do_kv_cache_update(layer, key, value, kv_cache, attn_metadata.slot_mapping)
    return impl.forward(layer, query, key, value, kv_cache, attn_metadata, output=output)


@with_exit_stack
def run_attention_torch(
    exit_stack,
    batch_size,
    input_len,
    num_heads,
    num_kv_heads,
    head_dim,
    use_fp8_kv_cache,
    is_context_phase,
    window_size=0,
    has_sink=False,
    *,
    perf_filename,
    device="npu:0",
):
    if use_fp8_kv_cache:
        # MVP collects bfloat16 KV cache only: AscendAttentionBackendImpl raises
        # ValueError on quantized KV cache without static C8 quantized weights
        # (vllm_ascend/attention/attention_v1.py:345-351 @ v0.30.0rc2).
        raise RuntimeError("vllm_npu attention collects bfloat16 kv_cache_dtype only, got fp8")

    dev_mod = get_device_module(device)
    dev_mod.set_device(device)

    dtype = torch.bfloat16
    model = os.path.join(os.path.dirname(__file__), "fake_hf_model")

    if is_context_phase:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[input_len] * batch_size,
        )
    else:
        batch_spec = BatchSpec(
            # vLLM seq_lens includes the current query token; input_len is the
            # persisted pre-query history (the raw `step` column).
            seq_lens=[input_len + 1] * batch_size,
            query_lens=[1] * batch_size,
        )

    try:
        vllm.utils.torch_utils.set_random_seed(42)
    except AttributeError:
        current_platform.seed_everything(42)

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=None,
        num_gpu_blocks=_KV_POOL_BLOCKS,
        max_num_seqs=batch_size,
        use_fp8_kv_cache=False,
        sliding_window=window_size if window_size > 0 else None,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    backend_cls = _select_attn_backend_cls(
        head_dim=head_dim,
        dtype=dtype,
        block_size=None,
        has_sink=has_sink,
        num_heads=num_heads,
    )
    backend_name_str = backend_cls.get_name()
    members = getattr(AttentionBackendEnum, "__members__", None) if AttentionBackendEnum is not None else None
    if isinstance(members, dict) and backend_name_str in members:
        backend_name = AttentionBackendEnum[backend_name_str]
    else:
        backend_name = backend_name_str

    block_size = _resolve_kv_block_size(backend_cls)
    vllm_config.cache_config.block_size = block_size

    # Refuse a KV pool that exceeds the device memory budget before allocating it.
    reason = _kv_pool_does_not_fit(
        num_kv_heads,
        head_dim,
        dev_mod.get_device_properties(device).total_memory,
        block_size=block_size,
    )
    if reason is not None:
        raise RuntimeError(reason)

    import gc

    gc.collect()
    dev_mod.empty_cache()

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, False)

    required_blocks = 1 + sum((s_len + block_size - 1) // block_size for s_len in batch_spec.seq_lens)
    num_blocks = max(vllm_config.cache_config.num_gpu_blocks or required_blocks, required_blocks)

    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q = torch.randn(q_len, num_heads, head_dim, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)

    base_common_metadata = create_common_attn_metadata(batch_spec, block_size, device)

    # Allocate paged KV cache in AscendAttentionBackend's native layout
    # [2, num_blocks, block_size, num_kv_heads, head_dim]
    # (vllm_ascend/attention/attention_v1.py:95-104 @ v0.30.0rc2).
    kv_cache, history_slot_mapping = create_kv_cache_and_block_mappings(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=dtype,
        device=device,
        num_blocks=num_blocks,
        common_attn_metadata=base_common_metadata,
        randomize_blocks=True,
    )

    common_attn_metadata = _build_ascend_common_attn_metadata(base_common_metadata, is_context_phase)

    if hasattr(backend_cls, "get_builder_cls") and hasattr(backend_cls, "get_impl_cls"):
        builder_cls, impl_cls = backend_cls.get_builder_cls(), backend_cls.get_impl_cls()
    else:
        builder_cls, impl_cls = get_attention_backend(backend_name)

    layer_names = ["placeholder"]
    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    sliding_window = vllm_config.model_config.get_sliding_window()
    scale = 1.0 / (head_dim**0.5)
    # Learned per-head sink bias (e.g. GPT-OSS): non-None sinks routes
    # AscendAttentionBackendImpl to torch_npu.npu_fused_infer_attention_score_v2
    # with learnable_sink=self.sinks (vllm_ascend/attention/attention_v1.py:488-513).
    sinks = torch.randn(num_heads, dtype=dtype, device=device) if has_sink else None
    impl = _instantiate_impl(
        impl_cls,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        num_kv_heads=num_kv_heads,
        sliding_window=sliding_window,
        sinks=sinks,
    )

    mock_layer = MockAttentionLayer(device, layer_name=layer_names[0])
    output = torch.empty_like(query_vllm)

    if history_slot_mapping.numel() > 0:
        impl.do_kv_cache_update(
            mock_layer,
            torch.cat(k_contexts),
            torch.cat(v_contexts),
            kv_cache,
            history_slot_mapping,
        )

    _enter_forward_context(
        exit_stack,
        attn_metadata=attn_metadata,
        vllm_config=vllm_config,
        num_tokens=int(query_vllm.shape[0]),
        has_sinks=has_sink,
    )

    test_ite = int(os.getenv("AIC_ATTN_NUM_RUNS", "6"))
    warm_up = int(os.getenv("AIC_ATTN_WARMUP", "3"))

    def run():
        _forward_with_optional_kv_cache_update(
            backend_cls,
            impl,
            mock_layer,
            query_vllm,  # noqa: F821
            key_vllm,  # noqa: F821
            value_vllm,  # noqa: F821
            kv_cache,  # noqa: F821
            attn_metadata,
            output,  # noqa: F821
        )

    try:
        with benchmark_with_power(
            device=device,
            kernel_func=run,
            num_warmups=warm_up,
            num_runs=test_ite,
            repeat_n=1,
            # MVP is eager: ACLGraph capture is a follow-up, and
            # benchmark_with_power has no NPUGraph arm yet, so used_cuda_graph
            # is recorded False below.
            use_cuda_graph=False,
            allow_graph_fail=False,
        ) as results:
            pass
    finally:
        del query_vllm, key_vllm, value_vllm, kv_cache, output
        del all_q_vllm, all_k_vllm, all_v_vllm, k_contexts, v_contexts
        gc.collect()
        dev_mod.empty_cache()

    latency = results["latency_ms"]

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"

    kernel_source = f"vllm_{backend_name_str}".lower()

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "beam_width": 1,
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "bfloat16",
                "window_size": window_size,
                "step": step,
                "latency": latency,
                "used_cuda_graph": results["used_cuda_graph"],
            }
        ],
        framework="VLLM",
        version=vllm_ascend_version,
        device_name=dev_mod.get_device_name(device),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=None,
    )


def get_context_attention_test_cases(if_unit_test=False):
    test_cases = []

    if if_unit_test:
        shape_sweeps = [
            {
                "batch_sizes": [1],
                "sequence_lengths": [64],
                "query_head_counts": [4],
                "kv_head_options": [0],
                "head_dims": [128, 64],
                "window_sizes": [0, 128],
                "max_tokens_self_attention": 65536,
                "max_tokens_grouped_query_attention": 131072,
                "max_kv_elements": 2147483647,
            }
        ]
    else:
        shape_sweeps = get_attention_context_shape_sweeps("vllm_npu")

    # MVP collects bfloat16 KV cache only.
    kv_cache_dtype_list = [False]
    max_gqa_ratio = 16

    for shape_sweep in shape_sweeps:
        batch_sizes = [int(value) for value in shape_sweep["batch_sizes"]]
        sequence_lengths = [int(value) for value in shape_sweep["sequence_lengths"]]
        supported_head_dims = {int(value) for value in shape_sweep["head_dims"]}
        supported_window_sizes = {int(value) for value in shape_sweep["window_sizes"]}
        max_tokens_self_attention = int(shape_sweep["max_tokens_self_attention"])
        max_tokens_grouped_query_attention = int(shape_sweep["max_tokens_grouped_query_attention"])
        max_kv_elements = int(shape_sweep["max_kv_elements"])

        for head_config in get_attention_head_configs(shape_sweep, phase="context"):
            n = head_config.num_heads
            num_kv_heads = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size
            if head_dim not in supported_head_dims or window_size not in supported_window_sizes:
                continue
            if n // num_kv_heads > max_gqa_ratio:
                continue
            if window_size > 0 and head_dim == 128:
                continue
            for s in sorted(sequence_lengths, reverse=True):
                for b in sorted(batch_sizes, reverse=True):
                    if num_kv_heads == n:
                        if b * s > max_tokens_self_attention or b > 128:
                            continue
                    elif b * s > max_tokens_grouped_query_attention:
                        continue
                    if b * s * num_kv_heads * head_dim * 2 >= max_kv_elements:
                        continue
                    for is_fp8_kv_cache in kv_cache_dtype_list:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                num_kv_heads,
                                head_dim,
                                is_fp8_kv_cache,
                                True,
                                window_size,
                                head_config.has_attention_sink,
                            ]
                        )

    return test_cases


def _generation_target_sequence_lengths(batch_sizes, sequence_lengths, num_heads, max_tokens, shape_sweep):
    b_s_dict = {}
    s_b_dict = {}
    for s in sequence_lengths:
        max_b = max_tokens // s // num_heads
        for b in batch_sizes:
            if b > max_b:
                break
            if s not in s_b_dict:
                s_b_dict[s] = {b}
            else:
                s_b_dict[s].add(b)
    for s, b_set in s_b_dict.items():
        if len(b_set) < int(shape_sweep["min_batch_options_per_sequence"]):
            continue
        for b in b_set:
            if b not in b_s_dict:
                b_s_dict[b] = {s - 1}
            b_s_dict[b].add(s - 1)
    return b_s_dict


def get_generation_attention_test_cases():
    test_cases = []

    # MVP collects bfloat16 KV cache only.
    kv_cache_dtype_list = [False]
    max_gqa_ratio = 16

    for shape_sweep in get_attention_generation_shape_sweeps("vllm_npu"):
        batch_sizes = [int(value) for value in shape_sweep["batch_sizes"]]
        sequence_lengths = [int(value) for value in shape_sweep["sequence_lengths"]]
        supported_head_dims = {int(value) for value in shape_sweep["head_dims"]}
        supported_window_sizes = {int(value) for value in shape_sweep["window_sizes"]}
        min_drop_batch = int(shape_sweep["drop_largest_sequence_for_batch_at_least"])

        for head_config in get_attention_head_configs(shape_sweep, phase="generation"):
            n = head_config.num_heads
            n_kv = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size
            if head_dim not in supported_head_dims or window_size not in supported_window_sizes:
                continue
            if n // n_kv > max_gqa_ratio:
                continue
            if window_size > 0 and head_dim == 128:
                continue
            max_tokens_key = "max_mha_tokens_per_step" if n == n_kv else "max_xqa_tokens_per_step"
            b_s_dict = _generation_target_sequence_lengths(
                batch_sizes,
                sequence_lengths,
                n,
                int(shape_sweep[max_tokens_key]),
                shape_sweep,
            )
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= min_drop_batch:
                    target_s_list = target_s_list[:-1]
                for s in target_s_list:
                    for is_fp8_kv_cache in kv_cache_dtype_list:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                n_kv,
                                head_dim,
                                is_fp8_kv_cache,
                                False,
                                window_size,
                                head_config.has_attention_sink,
                            ]
                        )
    return test_cases


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_context_attention_test_cases()
    for test_case in test_cases[:10]:
        run_attention_torch(*test_case, perf_filename=PerfFile.CONTEXT_ATTENTION)

    test_cases = get_generation_attention_test_cases()
    for test_case in test_cases[:10]:
        run_attention_torch(*test_case, perf_filename=PerfFile.GENERATION_ATTENTION)
