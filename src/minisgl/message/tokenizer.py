"""Tokenizer 进程消息定义模块。

定义前端与 tokenizer 进程之间传输的消息类型：

- :class:`TokenizeMsg`: 前端 -> tokenizer，请求将文本编码为 token id；
- :class:`DetokenizeMsg`: 调度器 -> tokenizer，请求将单个 token id
  解码回文本（流式输出场景，逐 token 增量解码）；
- :class:`AbortMsg`: 前端 -> tokenizer，请求中止某个请求。

所有消息均继承自 :class:`BaseTokenizerMsg`，并借助
:mod:`minisgl.message.utils` 中的序列化工具在 dict 与具体消息
类型之间相互转换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    """所有 tokenizer 消息的基类，提供统一的序列化/反序列化接口。

    - ``encoder``: 静态方法，将消息对象序列化为可传输的 dict；
    - ``decoder``: 静态方法，将 dict 反序列化回具体的消息子类。
      反序列化时以本模块 ``globals()`` 作为类型查找表，因此每个
      消息子类必须定义在本模块中才能被正确还原。
    """

    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        """将消息对象序列化为 dict（包含类型信息，便于还原）。"""
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        """将序列化后的 dict 还原为对应的具体消息对象。

        :param json: 由 :meth:`encoder` 生成的 dict。
        :return: 还原后的消息实例（实际类型为某个子类）。
        """
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    """批量消息：把多条 tokenizer 消息打包成一条，减少通信次数、提高吞吐。

    :param data: 打包在一起的子消息列表。
    """

    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """解码请求消息：请求 tokenizer 将新生成的一个 token id 解码为文本。

    逐 token 增量解码是流式输出的关键：调度器每生成一个 token
    就发送一条此类消息，tokenizer 解码出增量文本后回传给前端。

    :param uid: 对应请求的唯一标识。
    :param next_token: 新生成的单个 token id。
    :param finished: 该请求是否已生成完毕（最后一条解码请求）。
    """

    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """编码请求消息：请求 tokenizer 将用户输入编码为 token id 序列。

    :param uid: 请求的唯一标识，用于后续跟踪/中止该请求。
    :param text: 用户输入。既可以是纯文本字符串，也可以是
        OpenAI 风格的对话消息列表（``[{"role": ..., "content": ...}]``），
        由 tokenizer 侧负责套用对话模板。
    :param sampling_params: 采样参数（温度、top-p 等），随请求
        一路传递给调度器/后端。
    """

    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    """中止消息：请求终止指定 uid 对应的推理请求。

    :param uid: 要中止的请求唯一标识。
    """

    uid: int
