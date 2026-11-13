"""注意力后端的辅助数据结构。

提供 CUDA Graph 捕获所需的占位元数据容器 :class:`BaseCaptureData`：
捕获图时使用按最大批量/最大序列长度预分配的固定张量，回放时
只需将真实数据就地写入这些张量（不改变形状与地址），即可复用
已录制的计算图。
"""

from dataclasses import dataclass

import torch


@dataclass
class BaseCaptureData:
    """CUDA Graph 捕获用的固定形状元数据缓冲区（decode 阶段）。

    所有张量均预分配为最大尺寸，捕获与回放共用同一块内存，
    避免回放时重新分配导致图失效。

    :param seq_lens: 每个请求的当前序列长度，形状 ``(max_bs,)``，int32。
    :param positions: 每个请求当前 token 的绝对位置，形状 ``(max_bs,)``，int32。
    :param cu_seqlens_k: key 侧的累积序列长度前缀和（变长批处理布局），
        形状 ``(max_bs + 1,)``。decode 时每个请求通常只新增 1 个 token，
        故初始化为 ``arange(0, max_bs + 1)``。
    :param cu_seqlens_q: query 侧的累积序列长度前缀和，形状与含义同上；
        decode 阶段每请求 1 个新 token，因此与 ``cu_seqlens_k`` 相同。
    :param page_table: 每个请求的逻辑页号 -> 物理页号映射表，
        形状 ``(max_bs, max_seq_len)``，int32（此处按 page_size=1 视角
        逐 token 索引）。
    """

    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        """按最大尺寸预分配各元数据张量，构造捕获缓冲区。

        初始值仅作为捕获时的占位数据（每请求序列长度 1、位置 0），
        回放前会被真实数据覆盖。

        :param max_bs: 支持的最大批次大小。
        :param max_seq_len: 支持的最大序列长度。
        :param device: 张量所在设备（应为 GPU）。
        :param kwargs: 子类的额外字段，透传给 ``cls`` 的构造函数。
        :return: 填好占位数据的 :class:`BaseCaptureData`（或子类）实例。
        """
        return cls(
            # 占位：每个请求序列长度初始为 1（decode 单 token）
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            # 占位：位置初始为 0
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            # decode 时每请求 1 个 kv token，前缀和即 0,1,2,...
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            # decode 时每请求 1 个 q token，与 cu_seqlens_k 相同
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            # 页表占位为全 0，回放时写入真实映射
            page_table=torch.zeros(
                (max_bs, max_seq_len), dtype=torch.int32, device=device
            ),
            **kwargs,
        )
