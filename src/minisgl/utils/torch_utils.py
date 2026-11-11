"""PyTorch 相关的工具函数。

提供两个实用工具：
- :func:`torch_dtype`: 临时切换 PyTorch 默认数据类型的上下文管理器；
- :func:`nvtx_annotate`: 为方法添加 NVIDIA Nsight Systems (NVTX) 标注的装饰器，
  便于在性能分析工具（如 nsys）中定位各层的耗时。
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

# 仅在类型检查时导入 torch，避免运行时不必要的导入开销
if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    """上下文管理器：在 ``with`` 块内临时修改 PyTorch 的默认数据类型。

    常用于模型构建阶段，例如以 ``torch.bfloat16`` 为默认类型来
    创建参数，避免先建 fp32 权重再转换的开销。

    用法示例::

        with torch_dtype(torch.bfloat16):
            model = build_model()  # 此时创建的张量默认为 bfloat16

    :param dtype: 要临时使用的默认数据类型。
    """
    # 延迟导入：函数真正被调用时才加载 torch
    import torch  # real import when used

    # 记录当前默认类型，退出时恢复
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        # 确保无论是否发生异常都恢复原来的默认类型
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    """装饰器工厂：为方法添加 NVTX range，用于 GPU 性能分析。

    被装饰的方法在执行期间会向 NVIDIA Nsight Systems 发出一个
    命名的标注区间，在分析器的时间线上可以直观看到每次调用的耗时。

    :param name: 标注名称，若提供了 ``layer_id_field``，则作为
        格式化模板使用，例如 ``"forward layer {}"``。
    :param layer_id_field: 可选，宿主对象上存放层编号的属性名。
        若该属性存在，会用它替换 ``name`` 中的 ``{}`` 占位符，
        使每个层的标注名称互不相同（如 "forward layer 3"）。
    :return: 装饰器函数。
    """
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            # 若指定了层编号字段且对象上存在该属性，则格式化出带层号的名称
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            # 在 NVTX 区间内执行原方法，记录其执行耗时
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
