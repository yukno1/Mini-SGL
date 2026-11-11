"""后端消息定义模块。

定义调度器（scheduler）与模型执行后端（backend）之间通过
消息队列/网络传输的消息类型。所有消息均继承自 :class:`BaseBackendMsg`，
并借助 :mod:`minisgl.message.utils` 中的序列化工具在 dict 与
具体消息类型之间相互转换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    """所有后端消息的基类，提供统一的序列化/反序列化接口。

    - ``encoder``: 将自身序列化为可传输的 dict；
    - ``decoder``: 静态方法，将 dict 反序列化回具体的消息子类。
      反序列化时会以本模块 ``globals()`` 作为类型查找表，
      因此每个消息子类必须定义在本模块中才能被正确还原。
    """

    def encoder(self) -> Dict:
        """将当前消息对象序列化为 dict（包含类型信息，便于还原）。"""
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        """将序列化后的 dict 还原为对应的具体消息对象。

        :param json: 由 :meth:`encoder` 生成的 dict。
        :return: 还原后的消息实例（实际类型为某个子类）。
        """
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """批量消息：把多条后端消息打包成一条，减少通信次数、提高吞吐。

    :param data: 打包在一起的子消息列表。
    """

    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    """退出消息：通知后端进程结束运行、优雅退出。"""

    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """用户请求消息：携带一次推理请求的全部输入信息。

    :param uid: 请求的唯一标识，用于后续跟踪/中止该请求。
    :param input_ids: 经过分词后的输入 token id 序列，
        为 CPU 上一维 int32 张量。
    :param sampling_params: 采样参数（温度、top-p 等），
        控制生成行为。
    """

    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    """中止消息：请求后端终止指定 uid 对应的推理请求。

    :param uid: 要中止的请求唯一标识。
    """

    uid: int
