"""前端消息定义模块。

定义模型执行后端/调度器回发给前端（frontend，即面向客户端的一侧）
的消息类型，主要用于流式返回生成结果。所有消息均继承自
:class:`BaseFrontendMsg`，并借助 :mod:`minisgl.message.utils` 中的
序列化工具在 dict 与具体消息类型之间相互转换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """所有前端消息的基类，提供统一的序列化/反序列化接口。

    与 :class:`minisgl.message.backend.BaseBackendMsg` 的区别在于
    ``encoder`` 是静态方法、以消息实例为参数传入。

    - ``encoder``: 将消息对象序列化为可传输的 dict；
    - ``decoder``: 将 dict 反序列化回具体的消息子类。反序列化时
      以本模块 ``globals()`` 作为类型查找表，因此每个消息子类
      必须定义在本模块中才能被正确还原。
    """

    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        """将消息对象序列化为 dict（包含类型信息，便于还原）。"""
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        """将序列化后的 dict 还原为对应的具体消息对象。

        :param json: 由 :meth:`encoder` 生成的 dict。
        :return: 还原后的消息实例（实际类型为某个子类）。
        """
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """批量消息：把多条前端消息打包成一条，减少通信次数、提高吞吐。

    :param data: 打包在一起的子消息列表。
    """

    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    """用户回复消息：向前端流式返回某个请求的增量生成内容。

    :param uid: 对应请求的唯一标识，前端据此区分不同请求的输出。
    :param incremental_output: 本次新增生成的文本片段（增量而非
        全量，前端自行拼接）。
    :param finished: 该请求是否已生成完毕；为 True 时表示这是
        该请求的最后一条回复。
    """

    uid: int
    incremental_output: str
    finished: bool
