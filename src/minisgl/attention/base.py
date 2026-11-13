"""注意力后端的抽象基类与混合后端。

定义注意力模块与具体实现（如 FlashAttention、FlashInfer 等）之间的接口：

- :class:`BaseAttnMetadata`: 注意力元数据的抽象基类，由各后端
  根据 :class:`~minisgl.core.Batch` 自行构造；
- :class:`BaseAttnBackend`: 注意力后端的抽象接口，涵盖普通前向
  以及 CUDA Graph 捕获/回放所需的生命周期方法；
- :class:`HybridBackend`: 组合后端——prefill 与 decode 各自使用
  不同的底层实现（例如 prefill 用 FlashAttention、decode 用
  FlashInfer 的 paged attention），按批次阶段自动分发。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch

    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """注意力元数据抽象基类。

    各具体后端会继承本类，存放执行注意力计算所需的派生信息，
    例如：每个请求的序列长度、页表视图、qo/kv 间接索引等。
    元数据由后端的 :meth:`BaseAttnBackend.prepare_metadata` 根据
    批次内容构造，存入 ``Batch.attn_metadata``。

    :func:`get_last_indices` 主要服务于 CUDA Graph 的 padded 批次：
    在补齐后的批次中，真实请求的最后一列/最后一项在展平索引中的
    位置需要由此方法计算。
    """

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回批次中每个请求"最后一个 token"在展平 q 索引中的位置。

        :param bs: 批次大小（batch size）。
        :return: 形状为 ``(bs,)`` 的张量，第 i 项为第 i 个请求
            最后一个 token 的展平索引。
        """
        ...


class BaseAttnBackend(ABC):
    """注意力后端抽象接口。

    实现类负责：根据批次构造元数据、执行注意力前向，以及
    支持 CUDA Graph 的捕获与回放（decode 阶段消除 CPU 启动开销）。
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        """执行一层注意力计算。

        :param q: 查询张量，形状 ``(num_tokens, num_heads, head_dim)``。
        :param k: 键张量，形状 ``(num_kv_tokens, num_kv_heads, head_dim)``。
        :param v: 值张量，形状同 k。
        :param layer_id: 当前模型层编号，用于在 KV cache 池中
            定位该层的 k/v 存储区。
        :param batch: 当前批次（含 ``attn_metadata``）。
        :return: 注意力输出，形状 ``(num_tokens, num_heads, head_dim)``。
        """
        ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """根据批次内容构造注意力元数据，写入 ``batch.attn_metadata``。

        在每次前向前由调度器调用；CUDA Graph 回放路径中由
        :meth:`prepare_for_replay` 更新既有元数据。

        :param batch: 待处理的批次，会被就地修改。
        """
        ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获所需的后端状态。

        :param max_seq_len: 捕获图时支持的最大序列长度。
        :param bs_list: 要捕获的各批次大小（decode 批次的候选尺寸）。
        """
        ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """进入 CUDA Graph 捕获前的准备：用 dummy（占位）数据
        构造元数据，使后续前向可在图捕获模式下录制。

        :param batch: 用于捕获的批次（内容为占位数据）。
        """
        ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """CUDA Graph 回放前的准备：将真实批次的数据更新到
        捕获时使用的固定缓冲区中，随后即可回放已录制的图。

        :param batch: 本次要回放的真实批次。
        """
        ...


class HybridBackend(BaseAttnBackend):
    """混合注意力后端：按批次阶段分发到不同的底层实现。

    典型用法是 prefill 阶段用一种后端（如 FlashAttention 的
    变长批处理）而 decode 阶段用另一种（如 FlashInfer 的
    paged attention + CUDA Graph），两者组合取长补短。

    :param prefill_backend: prefill 阶段使用的后端。
    :param decode_backend: decode 阶段使用的后端。
    """

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        """按批次阶段选择对应后端执行注意力前向。"""
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        """按批次阶段选择对应后端构造元数据。"""
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """CUDA Graph 仅用于 decode 阶段，因此只初始化 decode 后端。"""
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """CUDA Graph 捕获只在 decode 后端上进行。"""
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        """CUDA Graph 回放只在 decode 后端上进行。"""
        self.decode_backend.prepare_for_replay(batch)
