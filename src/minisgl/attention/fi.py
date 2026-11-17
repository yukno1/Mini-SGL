"""基于 FlashInfer 的注意力后端实现。

使用 FlashInfer 的 paged attention wrapper 执行 prefill 与 decode
两阶段的注意力计算，并支持 CUDA Graph 加速 decode：

- prefill / 常规前向：:class:`BatchPrefillWithPagedKVCacheWrapper`；
- 常规 decode：:class:`BatchDecodeWithPagedKVCacheWrapper`；
- CUDA Graph decode：按批次大小为每个 bs 创建
  :class:`CUDAGraphBatchDecodeWithPagedKVCacheWrapper`。

KV cache 以 page_size=1（每 token 一"页"）的方式管理，因此页表
可退化为逐 token 的物理槽位索引（ragged indices）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch

from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )

    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    """返回不小于 n 的最小 2 的幂（用于缓存尺寸的整齐扩容）。"""
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    """FlashInfer 的 CUDA Graph 捕获数据。

    在基类预分配缓冲区的基础上，提供两个语义别名属性，
    将通用缓冲区复用为 FlashInfer 需要的参数形态：

    - ``one_tensor``: 复用 ``seq_lens`` 作为 ``last_page_len`` 缓冲区
      （page_size=1 时每页即 1 个 token，last_page_len 恒为 1）；
    - ``indices``: 复用 ``page_table``（展平成一维）作为
      ``indices_buffer``（ragged 的物理槽位索引）。
    """

    @property
    def one_tensor(self) -> torch.Tensor:
        """全 1 缓冲区，用作 last_page_len（page_size=1 时恒为 1）。"""
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        """展平的页表，用作 FlashInfer 的 ragged indices 缓冲区。"""
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    """FlashInfer 前向所需的注意力元数据。

    FlashInfer 的调用流程分两步：``plan``（根据序列布局做调度规划，
    需要 CPU 上的输入）与 ``run``（实际执行注意力计算，使用 GPU 上的
    缓存指针）。因此元数据中同时保存 CPU / GPU 两份相关张量。

    :param cu_seqlens_q_cpu: query 侧累积序列长度前缀和（CPU，pinned）。
    :param cu_seqlens_k_cpu: key 侧累积序列长度前缀和（CPU，pinned）。
    :param cu_seqlens_q_gpu: query 前缀和的 GPU 副本，用于
        :meth:`get_last_indices` 等需要 GPU 端索引的场景。
    :param indices: 所有请求的物理槽位索引拼接而成的 ragged 索引
        （GPU，一维）。page_size=1 时即逐 token 的 KV 物理位置。
    :param last_page_len_cpu: 每个请求最后一页的有效 token 数
        （CPU）；page_size=1 时恒为 1。
    :param num_qo_heads / num_kv_heads: TP 切分后本进程负责的
        Q 头数与 KV 头数。
    :param head_dim: 每个注意力头的维度。
    :param page_size: KV cache 页大小，当前仅支持 1。
    :param pos_encoding_mode: RoPE 等位置编码由模型层自行完成，
        这里固定为 "NONE"。
    :param seq_lens_cpu: 每个请求的完整 KV 序列长度（CPU，pinned）。
    :param dtype: KV cache 的数据类型。
    :param wrapper: 本次前向使用的 FlashInfer wrapper
        （prefill 或 decode）。
    :param initialized: 该元数据是否已完成 ``plan``。
    """
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        """校验页大小限制及各张量所在设备是否符合约定。"""
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回每个请求最后一个 token 在展平 q 序列中的索引。

        ``cu_seqlens_q_gpu`` 的第 i+1 项是第 i 个请求 q 段的结束位置
        （不含），减 1 即为该请求最后一个 token 的展平索引。

        :param bs: 批次大小。
        :return: 形状 ``(bs,)`` 的 GPU 张量。
        """
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    """FlashInfer 注意力后端。

    生命周期：
    1. 构造时创建 prefill / decode wrapper 及共享 workspace；
    2. 每次前向前由调度器调用 :meth:`prepare_metadata` 构造元数据；
    3. :meth:`forward` 中先 ``plan``（首次使用该元数据时），
       再写入新 KV 并执行注意力计算；
    4. decode 阶段可选用 CUDA Graph：:meth:`init_capture_graph`
       预分配捕获缓冲区，:meth:`prepare_for_capture` /
       :meth:`prepare_for_replay` 完成录制与回放前的准备。

    :param config: 模型配置（头数、head_dim 等）。
    """

    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        # 从全局上下文获取 KV cache 池，并取其设备作为工作设备
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        # FlashInfer 需要的浮点 workspace（128 MiB），prefill/decode 共享
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        # prefill（含 extend）用的 wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        # 常规 decode 用的 wrapper（是否启用 tensor core 见 use_tensor_cores）
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        # 让 prefill 与 decode wrapper 共用同一块整型 workspace，
        # 省去单独分配的显存开销（借用了 FlashInfer 的私有属性）
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        # 张量并行：将 Q 头与 KV 头均分到各 TP rank
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(
            self.config.num_kv_heads, tp_size, allow_replicate=True
        )

        # pinned 内存的全 1 向量缓存，用于构造 last_page_len（恒为 1）
        self.cached_ones_cpu: torch.Tensor = torch.tensor(
            [], dtype=torch.int32, pin_memory=True
        )
        # for cuda graph
        # CUDA Graph 相关状态：待捕获的批次大小列表、最大批次、
        # 每个 bs 对应的图回放 wrapper，以及捕获期占位数据
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        # 用于跟踪异步 plan 的事件：plan 内部会做异步 H2D 拷贝，
        # 记录事件以便在下次 plan 修改 staging buffer 前同步等待
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        """对元数据执行一次 FlashInfer ``plan``（幂等）。

        ``plan`` 根据序列布局选择 kernel 并填充内部调度表，同一份
        元数据只需执行一次（由 ``initialized`` 标记去重）。

        FlashInfer 的 plan 复用一块 pinned host staging buffer 并发起
        异步 H2D 拷贝，因此这里先同步上一次 plan 记录的事件，避免
        下一次 plan 改写 host 缓冲区时上一次拷贝尚未完成。
        """
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            # decode 路径的 plan：每请求 1 个新 q token
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            # prefill / extend 路径的 plan：q 段长度由 cu_seqlens_q 描述
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        # 记录本次 plan 的异步拷贝事件，供下一次 plan 前同步
        self.last_event.record()

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        """返回长度为 bs 的 pinned 全 1 int32 向量（带缓存与 2 的幂扩容）。

        page_size=1 时 last_page_len 恒为 1，因此直接复用缓存的全 1
        向量切片，避免每次前向重新分配 pinned 内存。

        :param bs: 所需长度（批次数）。
        :return: 长度为 bs 的全 1 向量。
        """
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        # 按 2 的幂扩容，减少反复 realloc 的次数
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        """执行一层注意力前向：plan -> 写 KV -> FlashInfer run。

        :param q: 本层查询张量。
        :param k: 本层新产生的 key 张量。
        :param v: 本层新产生的 value 张量。
        :param layer_id: 层编号，用于定位 KV cache 中该层的存储区。
        :param batch: 当前批次。
        :return: 注意力输出张量。
        """

        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            # 将 (num_pages, page_size, head, dim) 的 cache 视图压成
            # (num_tokens, 1, head, dim)：page_size=1 时页与 token 一一对应
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        # 首次使用该元数据时执行 plan（幂等）
        self._initialize_metadata_once(metadata)
        # 把本层新算出的 k/v 写入全局 KV cache 的对应物理槽位
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        # 取出该层的 k/v cache 并压平成 FlashInfer 需要的形状
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        """根据批次内容构造 :class:`FIMetadata` 并写入 ``batch.attn_metadata``。

        处理三种 q 布局：
        - decode（每请求 extend_len == 1）：q 前缀和为 0,1,2,...；
        - 无缓存命中的 prefill：q 段长度等于 k 段长度，两者前缀和相同；
        - 部分缓存命中的 extend prefill：q 段长度为 extend_len。

        :param batch: 待处理的批次（就地写入 attn_metadata）。
        """
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        # 每请求：本次待计算的 q 长度、完整 KV 长度、已缓存长度
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        # pinned CPU int32 张量的统一构造参数（pinned 加速异步 H2D）
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        # 各请求的完整 KV 序列长度
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        # k 侧累积前缀和：[0, l1, l1+l2, ...]
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            # decode：每请求恰 1 个 q token，前缀和即 0,1,2,...
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # 无前缀缓存命中：q 段与 k 段完全相同
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            # 部分命中：q 段为各请求新计算的 extend 部分
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(
                dim=0
            )

        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            # GPU 副本：异步拷贝，供 get_last_indices 使用
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            # 将各请求页表行（前 device_len 个 token 的物理槽位）拼接
            # 成一维 ragged indices
            indices=torch.cat(
                [page_table[req.table_idx, : req.device_len] for req in reqs]
            ),
            # page_size=1 时 last_page_len 恒为 1
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            # RoPE 等位置编码在模型层完成，FlashInfer 不再处理
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            # decode 用 decode wrapper，其余用 prefill wrapper
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获所需的固定缓冲区。

        按最大批次大小与最大序列长度预分配捕获数据（占位元数据），
        并将页表展平成一维以充当 ragged indices 缓冲区。

        :param max_seq_len: 捕获图支持的最大序列长度。
        :param bs_list: 将要捕获的各批次大小。
        """
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        # 将 (max_bs, max_seq_len) 页表展平为一维，作为 ragged indices
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        """是否让 FlashInfer decode 走 tensor core 路径。

        优先使用环境变量覆盖；否则按 GQA 比例启发式决定：
        当 Q 头数 / KV 头数 >= 4 时，batch decode kernel 即使在
        小批次下也值得使用 tensor core 实现。
        """
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(
                f"Overriding FlashInfer tensor core usage to {overriden_value}"
            )
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        """为某个批次大小 bs 创建 CUDA Graph decode wrapper 并完成录制前准备。

        wrapper 的 indptr / indices / last_page_len 均直接绑定到
        捕获缓冲区（:class:`FICaptureData`）的切片上——回放时只需
        将真实数据写入这些固定张量，无需重新 plan。

        :param batch: 用于捕获的占位批次。
        """
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        # 创建绑定固定缓冲区的图 wrapper（缓冲区在回放时被就地更新）
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        # 与常规 wrapper 保持一致：使用 fa2 后端并共享 int workspace
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        # 用占位批次构造元数据并替换 wrapper，随后完成一次 plan
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        """CUDA Graph 回放前的准备：把元数据的 wrapper 切换为该 bs
        对应的图 wrapper 并重新 plan。

        plan 会基于捕获缓冲区的当前内容（已被真实数据覆盖）重建
        调度信息，随后即可回放录制的图。

        :param batch: 本次要回放的真实批次。
        """
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
