"""基于 FlashInfer TRT-LLM kernels 的注意力后端实现。

调用 FlashInfer 暴露的 TensorRT-LLM 风格 batch attention kernels：

- prefill（含 extend）：:func:`trtllm_batch_context_with_kv_cache`；
- decode：:func:`trtllm_batch_decode_with_kv_cache`。

与 :class:`~minisgl.attention.fi.FlashInferBackend`（wrapper + plan/run
两阶段）不同，TRT-LLM kernels 是无状态的单次调用，无需 plan；
同时原生支持 page_size > 1 的分页 KV cache（页表按页索引），
CUDA Graph 捕获只需维护固定缓冲区，无需为每个 bs 创建 wrapper。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class TRTLLMCaptureData(BaseCaptureData):
    """TRT-LLM 后端的 CUDA Graph 捕获数据。

    直接复用基类的预分配缓冲区（seq_lens / cu_seqlens / page_table），
    捕获时绑定为元数据张量，回放时就地写入真实数据。

    注意：页表列数按"页数"分配（``max_seq_len // page_size``），
    因为 TRT-LLM kernel 的 block table 以页为单位索引。
    """

    pass


@dataclass
class TRTLLMMetadata(BaseAttnMetadata):
    """TRT-LLM kernel 所需的注意力元数据。

    所有张量均位于 GPU 上（kernel 直接消费设备端数据，无需 plan）。

    :param cu_seqlens_k: key 侧累积序列长度前缀和，形状 ``(bs+1,)``。
    :param cu_seqlens_q: query 侧累积序列长度前缀和，形状 ``(bs+1,)``。
    :param cache_seqlens: 每个请求的完整 KV 序列长度，形状 ``(bs,)``。
    :param max_seqlen_k: 批次内最大 KV 长度（kernel 调度用）。
    :param max_seqlen_q: 批次内最大 q 长度（kernel 调度用）。
    :param page_table: block table，形状 ``(bs, num_pages)``，
        值为每个逻辑页对应的物理页号（注意是页号而非 token 槽位）。
    """

    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回每个请求最后一个 token 在展平 q 序列中的索引。

        ``cu_seqlens_q`` 第 i+1 项为第 i 个请求 q 段的结束位置（不含），
        减 1 即为该请求最后一个 token 的展平索引。

        :param bs: 批次大小。
        :return: 形状 ``(bs,)`` 的 GPU 张量。
        """
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TensorRTLLMBackend(BaseAttnBackend):
    """TRT-LLM 风格的注意力后端。

    相比 FlashInferBackend 的特点：
    - 无状态 kernel 调用（无 plan 阶段）；
    - 支持任意 page_size（页表按页索引，token 槽位索引需换算成页号）；
    - CUDA Graph 回放通过就地更新固定缓冲区实现，
      无需为每个批次大小创建独立的 wrapper。

    :param config: 模型配置（head_dim 等）。
    """

    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        # KV cache 页大小；TRT-LLM kernel 的 block table 以页为单位
        self.page_size = ctx.page_size
        # CUDA Graph 捕获状态
        self.capture: TRTLLMCaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        # softmax 缩放因子：1 / sqrt(head_dim)
        self.scale = config.head_dim**-0.5
        # TRT-LLM kernel 需要的工作区缓冲区（128 MiB）
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        """执行一层注意力前向：先写入新 KV，再按阶段调用对应 kernel。

        :param q: 本层查询张量。
        :param k: 本层新产生的 key 张量。
        :param v: 本层新产生的 value 张量。
        :param layer_id: 层编号，用于定位 KV cache 中该层的存储区。
        :param batch: 当前批次。
        :return: 注意力输出张量。
        """
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        metadata = batch.attn_metadata
        assert isinstance(metadata, TRTLLMMetadata)
        # 把本层新算出的 k/v 写入全局 KV cache 的对应物理槽位
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))

        if batch.is_prefill:
            # prefill / extend：context kernel，支持变长 q（causal）
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_q_len=metadata.max_seqlen_q,
                max_kv_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                cum_seq_lens_q=metadata.cu_seqlens_q,
                cum_seq_lens_kv=metadata.cu_seqlens_k,
                kv_layout="NHD",
                batch_size=batch.size,
                out_dtype=q.dtype,
            )
        else:
            # decode：每请求仅 1 个新 q token
            return trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_seq_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                kv_layout="NHD",
                out_dtype=q.dtype,
            )

    def prepare_metadata(self, batch: Batch) -> None:
        """根据批次内容构造 :class:`TRTLLMMetadata` 并写入 ``batch.attn_metadata``。

        与 FlashInfer 版的主要差异：所有张量直接放在 GPU 上；
        页表需从"逐 token 物理槽位"（全局页表，page_size=1 视角）
        换算成"逐页物理页号"（按 page_size 抽样并整除）。

        :param batch: 待处理的批次（就地写入 attn_metadata）。
        """
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        # 每请求：本次待计算的 q 长度、完整 KV 长度、已缓存长度
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        # pinned CPU int32 张量的统一构造参数（先建在 CPU，再异步上 GPU）
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        # 各请求的完整 KV 序列长度（GPU）
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        # k 侧累积前缀和：[0, l1, l1+l2, ...]（GPU）
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            # decode：每请求恰 1 个 q token，前缀和即 0,1,2,...（纯 GPU 构造）
            cu_seqlens_q = torch.arange(
                0, padded_size + 1, device=device, dtype=torch.int32
            )
        elif all(i == 0 for i in cached_lens):  # prefill with no cache hit
            # 无前缀缓存命中：q 段与 k 段完全相同，直接复用
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            # 部分命中：q 段为各请求新计算的 extend 部分
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            # 全局页表是 page_size=1 的逐 token 槽位索引；TRT-LLM 的
            # block table 以页为单位，因此每 page_size 个 token 抽 1 个
            # 代表槽位，得到每请求的页级页表
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            # 槽位索引 -> 物理页号：整除页大小
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = TRTLLMMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获所需的固定缓冲区。

        页表列数按"页数"预分配（``max_seq_len // page_size``），
        与 TRT-LLM block table 的页级索引方式一致。

        :param max_seq_len: 捕获图支持的最大序列长度（token 数）。
        :param bs_list: 将要捕获的各批次大小。
        """
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = TRTLLMCaptureData.create(
            max_bs, max_seq_len // self.page_size, self.kvcache.device
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """CUDA Graph 捕获前的准备：让元数据直接引用捕获缓冲区的切片。

        与 FlashInfer 版不同，这里无需创建 wrapper 或 plan——
        捕获录制的是 kernel 本身，回放时只需更新缓冲区数据。

        :param batch: 用于捕获的占位批次。
        """
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        # 元数据张量直接绑定捕获缓冲区切片（回放时被就地覆盖）
        metadata = TRTLLMMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        """CUDA Graph 回放前的准备：将真实元数据拷入捕获缓冲区。

        只需更新 k 前缀和、序列长度与页表三项；decode 时 q 前缀和
        恒为 ``[0, 1, ..., bs]``，无需更新。

        :param batch: 本次要回放的真实批次。
        """
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TRTLLMMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
