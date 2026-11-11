"""消息序列化/反序列化工具模块。

提供一套基于 dict 的通用序列化机制，用于在进程间（如调度器与
模型后端之间）传输自定义对象。核心思路：

- 序列化时在 dict 中写入 ``__type__`` 字段记录对象类型名，
  并递归处理嵌套的 dict / list / tuple 以及 torch.Tensor；
- 反序列化时通过调用方提供的"类型名 -> 类"映射表（``cls_map``，
  通常传入目标模块的 ``globals()``）查回对应类并重建实例。

张量的序列化方式为：转为 numpy 后取原始字节（``tobytes``），
反序列化时再从字节缓冲区还原，避免 JSON 之类逐元素编码的开销。
"""

from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    """递归地将任意值转换为可传输的基础类型。

    处理规则：
    - dict: 逐键值递归序列化；
    - list / tuple: 逐元素递归序列化，并保持原容器类型；
    - 基础类型（int/float/str/None/bool/bytes）: 原样返回；
    - 其他自定义对象: 交给 :func:`serialize_type` 继续展开。

    :param value: 待序列化的任意值。
    :return: 序列化后的结果（dict、容器或基础类型）。
    """
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    """将对象序列化为携带类型信息的 dict。

    特殊处理 torch.Tensor：
    - 仅支持一维张量；
    - 张量本体转为 numpy 原始字节缓冲区，并附带 dtype 字符串，
      便于反序列化时精确还原。

    普通对象：dict 中记录 ``__type__``（类名），
    并逐个序列化 ``self.__dict__`` 中的成员变量。

    :param self: 待序列化的对象（张量或任意带 ``__dict__`` 的对象）。
    :return: 包含 ``__type__`` 键的 dict。
    """
    # find all member variables
    serialized = {}

    # 张量走专用路径：只支持 1D，转 numpy 字节缓冲 + dtype 信息
    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.numpy().tobytes()
        serialized["dtype"] = str(self.dtype)
        return serialized

    # 普通类型：记录类名，递归序列化所有实例属性
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    """递归地将序列化数据还原为原始对象（:func:`_serialize_any` 的逆过程）。

    处理规则：
    - dict 且含 ``__type__`` 键: 是某个自定义对象，交给
      :func:`deserialize_type` 还原；
    - 普通 dict: 逐键值递归反序列化；
    - list / tuple: 逐元素递归反序列化，保持原容器类型；
    - 基础类型: 原样返回；
    - 其他类型: 不支持，抛出 ValueError。

    :param cls_map: 类型名 -> 类对象的映射表，用于还原自定义对象。
    :param data: 待还原的数据。
    :return: 还原后的对象。
    """
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    """将 :func:`serialize_type` 生成的 dict 还原为对象实例。

    张量路径：根据记录的 dtype 字符串（如 ``"torch.int32"``）映射到
    对应的 numpy dtype，从字节缓冲区重建 numpy 数组，再转为
    torch 张量（``.copy()`` 保证张量持有自己的内存，不与缓冲区共享）。

    对象路径：从 ``cls_map`` 中按类名查回类，剔除 ``__type__`` 键后，
    将其余字段递归反序列化并作为关键字参数构造实例。

    :param cls_map: 类型名 -> 类对象的映射表（如模块的 ``globals()``）。
    :param data: 序列化后的 dict。
    :return: 还原出的 torch.Tensor 或类实例。
    """
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        # "torch.int32" -> "int32" -> numpy 对应 dtype
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        # 从字节缓冲区还原 numpy 数组
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        # copy() 复制一份数据，避免 torch 张量引用只读缓冲区
        return torch.from_numpy(np_tensor.copy())

    # 按类名查回类，重建对象
    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
