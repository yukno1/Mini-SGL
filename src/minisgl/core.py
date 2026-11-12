"""推理引擎的核心数据结构。

本模块定义了整个系统中共用的基础类型：

- :class:`SamplingParams`: 采样参数（温度、top-k、top-p 等）；
- :class:`Req`: 单个推理请求及其长度状态机；
- :class:`Batch`: 一次模型前向计算所使用的请求批次；
- :class:`Context`: 引擎运行期的全局上下文（页表、注意力后端、
  KV cache 池、当前批次等）。

此外还提供全局上下文 :func:`set_global_ctx` / :func:`get_global_ctx`
的存取接口，方便模型层等深层代码在不逐层传参的情况下访问运行期资源。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

# 仅用于类型检查的导入，避免运行时的循环导入
if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    """采样参数，控制每个请求的生成行为。

    :param temperature: 采样温度。<= 0 表示贪心解码（总是选概率最大的 token）；
        数值越大分布越平坦、结果越随机。
    :param top_k: 只从概率最高的 k 个 token 中采样；-1 表示不启用。
    :param top_p: 核采样（nucleus sampling）阈值，只保留累计概率
        不超过 top_p 的最小 token 集合；1.0 表示不启用。
    :param ignore_eos: 是否忽略 EOS（结束符），为 True 时强制生成到
        ``max_tokens`` 才停止。
    :param max_tokens: 单次请求最多生成的 token 数。
    """

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        """是否等价于贪心解码。

        当温度为 0（或 top_k == 1）且未启用 top_p 时，采样退化为
        取概率最大的 token，属于确定性输出。
        """
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """单个推理请求。

    长度状态说明（一次请求的生命周期内依次增长）::

        [0, cached_len)       已写入 KV cache、无需重复计算的部分
        [cached_len, device_len)  本次（或待）前向计算需要处理的 token
        [device_len, max_device_len)  尚未生成的输出 token 预算

    ``eq=False`` 使实例比较退化为默认的对象身份比较（哈希基于 id），
    从而可以安全地放入 set / 作为 dict 的键。

    :param input_ids: 请求的完整 token 序列（prompt + 已生成的输出），
        位于 CPU 上。
    :param table_idx: 该请求在全局页表 ``Context.page_table`` 中的行索引。
    :param cached_len: 已经算过并存入 KV cache 的 token 数（前缀缓存命中长度）。
    :param output_len: 期望生成的最大输出 token 数。
    :param uid: 请求的唯一标识。
    :param sampling_params: 该请求的采样参数。
    :param cache_handle: KV cache 的句柄，用于在请求结束/被中止时
        释放其占用的显存页。
    """

    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        """初始化后补算派生字段并做一致性校验。"""
        assert self.input_ids.is_cpu
        # 当前实际的 token 总数（prompt 长度 + 已生成数量）
        self.device_len = len(self.input_ids)
        # 该请求最终可能达到的最大长度（受输出预算限制）
        self.max_device_len = len(self.input_ids) + self.output_len
        # 前缀长度必须小于当前长度，当前长度不得超过上限
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """距离长度上限还剩多少 token 可以生成。"""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """本次前向需要新计算的 token 数（未命中 KV cache 的部分）。

        prefill 阶段通常等于 prompt 长度；decode 阶段通常为 1。
        """
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """完成一个 token 的前向计算：推进长度游标。

        当前已计算的部分（含刚算完的这个 token）全部变为"已缓存"，
        同时 ``device_len`` 加 1，表示序列中新增了一个待计算的输出 token。
        """
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """在 CPU 侧把新采样的 token 追加到输入序列末尾。

        :param next_token: 新生成的 token，形状为 (1,) 的 CPU 张量。
        """
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        """是否还能继续解码（仍有剩余输出预算）。"""
        return self.remain_len > 0

    def __repr__(self) -> str:
        """精简的调试字符串，只展示与调度最相关的长度信息。"""
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """一次模型前向计算所处理的请求集合。

    :param reqs: 参与本次前向的请求列表。
    :param phase: 阶段类型，``"prefill"``（预填充，一次处理多个 token）
        或 ``"decode"``（逐 token 解码）。
    """

    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    # 以下字段由调度器在构造批次时填充（init=False，不参与构造参数）
    input_ids: torch.Tensor = field(init=False)  # 展平后的所有 token id（1D）
    positions: torch.Tensor = field(init=False)  # 每个 token 对应的绝对位置索引
    out_loc: torch.Tensor = field(init=False)  # 采样结果写回的槽位映射
    padded_reqs: List[Req] = field(init=False)  # CUDA Graph 场景下补齐后的请求列表
    # this field should be set by attention backend
    # 由注意力后端根据批次信息构造的元数据（页表、seq lens 等）
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        """是否为 prefill 阶段。"""
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        """是否为 decode 阶段。"""
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """批次中真实请求的数量。"""
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """补齐后（CUDA Graph）的请求数量。"""
        return len(self.padded_reqs)


@dataclass
class Context:
    """引擎运行期全局上下文，承载各模块共享的资源与状态。

    通过 :func:`get_global_ctx` 在模型层深处获取，避免层层传参。

    :param page_size: KV cache 页大小（每页容纳的 token 数）。
    """

    page_size: int
    # NOTE: this table always treat page_size = 1
    # 全局页表：行 = 请求的 table_idx，列 = 逻辑页号，
    # 值为物理页号。注意该页表始终按 page_size = 1 来索引。
    page_table: torch.Tensor = field(init=False)
    # 注意力后端，负责构造 attn_metadata 并执行注意力计算
    attn_backend: BaseAttnBackend = field(init=False)
    # MoE（混合专家）后端
    moe_backend: BaseMoeBackend = field(init=False)
    # KV cache 显存池，负责页的分配与回收
    kv_cache: BaseKVCachePool = field(init=False)
    # 当前正在前向计算的批次（由 forward_batch 上下文管理器维护）
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        """获取当前正在前向计算的批次。

        :raises AssertionError: 当前不处于前向计算过程中。
        """
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """上下文管理器：在 ``with`` 块内将 ``batch`` 设为当前批次。

        这是模型层（如 Attention、MoE）访问批次信息的唯一入口，
        块结束后自动清空，保证状态不泄漏到下一次前向。

        用法示例::

            with ctx.forward_batch(batch):
                logits = model(batch.input_ids, batch.positions)

        :param batch: 本次前向要使用的批次。
        :raises AssertionError: 不支持嵌套调用（避免状态被覆盖）。
        """
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            # 无论是否异常都清空，避免后续代码读到过期批次
            self._batch = None


# 进程内唯一的全局上下文，由引擎启动时通过 set_global_ctx 注入
_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """设置进程全局上下文（只能在引擎初始化时调用一次）。

    :param ctx: 要设置的上下文对象。
    :raises AssertionError: 重复设置时抛出，防止多个引擎实例互相覆盖。
    """
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """获取进程全局上下文。

    :raises AssertionError: 尚未调用 :func:`set_global_ctx` 时抛出。
    """
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
