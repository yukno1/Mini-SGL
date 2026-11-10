from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    """环境变量描述符的基类。

    所有具体的环境变量包装类（如 EnvVar）都必须继承本类，
    并实现 `_init` 方法。这样 `EnvClassSingleton` 在初始化时
    可以通过 `isinstance` 检查来识别哪些类属性是环境变量，
    并统一调用 `_init` 完成从 os.environ 读取值的工作。
    """

    def _init(self, name: str) -> None:
        """根据完整的环境变量名 `name` 初始化自身的值。

        子类需要重写该方法，通常的实现逻辑是：
        尝试从 os.environ 中读取 `name` 对应的值，
        解析后赋给 `self.value`；读取失败或解析异常时保留默认值。
        """
        raise NotImplementedError


# 泛型类型变量 T，表示环境变量解析后的 Python 值类型（int / float / bool / str 等）
T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    """单个环境变量的包装器。

    将一个环境变量封装为对象：
    - `value`：环境变量解析后的当前值（初始为默认值）；
    - `fn`：把字符串环境变量值解析为目标类型 T 的函数，
      例如 int、float，或自定义的布尔/内存大小解析函数。

    典型用法（在 EnvClassSingleton 中）：
        SHELL_MAX_TOKENS = EnvInt(2048)
    表示环境变量 MINISGL_SHELL_MAX_TOKENS，默认值为 2048。
    """

    def __init__(self, default_value: T, fn: Callable[[str], T]):
        # 当前生效的值，初始为调用方给定的默认值；
        # 只有当环境中存在对应变量且解析成功时才会被覆盖
        self.value = default_value
        # 字符串 -> 目标类型 T 的解析函数
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        """从 os.environ 读取名为 `name` 的环境变量并解析。

        行为说明：
        - 若环境变量不存在，则保持默认值不变；
        - 若存在但解析失败（例如 EnvInt 遇到非法数字字符串），
          捕获异常并静默回退到默认值，保证程序不会因
          用户配置错误而崩溃。
        """
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        """使包装对象可以直接用于条件判断，
        等价于判断其内部值 `self.value` 的真值。"""
        return self.value

    def __str__(self):
        """打印时直接输出内部值的字符串形式，方便调试与日志。"""
        return str(self.value)


def _TO_BOOL(x):
    """把字符串解析为布尔值。

    只有 "1"、"true"、"yes"（大小写不敏感）被解析为 True，
    其余任意值都被解析为 False。
    """
    return x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    """把人类可读的内存大小字符串解析为字节数（int）。

    支持的格式示例：
    - "1024"       -> 1024 字节（纯数字，视为字节）
    - "512K" / "512KB" -> 512 * 1024 字节
    - "256M" / "256MB" -> 256 * 1024**2 字节
    - "1G" / "1GB"     -> 1024**3 字节

    解析细节：
    - 统一转为大写并去除首尾空白，保证大小写和空格容错；
    - 若最后一个字符不是字母，按纯数字（字节）处理；
    - 若以 "B" 结尾则去掉 "B"，只保留单位字母 K/M/G；
    - 数值部分用 float 解析，支持 "1.5G" 这类小数形式。
    """
    mem = mem.strip().upper()
    # 末尾不是字母 -> 视为纯数字，直接按字节解析
    if not mem[-1].isalpha():
        return int(mem)
    # 去掉可选的 "B" 后缀，例如 "512KB" -> "512K"
    if mem.endswith("B"):
        mem = mem[:-1]
    # 单位 -> 字节的换算表（二进制单位：K=1024, M=1024^2, G=1024^3）
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


# 本项目环境变量的统一前缀，最终变量名形如 MINISGL_<属性名>，
# 例如类属性 SHELL_MAX_TOKENS 对应环境变量 MINISGL_SHELL_MAX_TOKENS
MINISGL_ENV_PREFIX = "MINISGL_"

# 常用环境变量类型的便捷构造器（基于 partial 预先绑定了解析函数）：
# - EnvInt   : 解析为 int，例如 "16" -> 16
EnvInt = partial(EnvVar[int], fn=int)
# - EnvFloat : 解析为 float，例如 "0.6" -> 0.6
EnvFloat = partial(EnvVar[float], fn=float)
# - EnvBool  : 解析为 bool，仅 "1"/"true"/"yes" 为 True，默认值由调用方指定
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
# - EnvOption: 三态布尔（True / False / None），
#   默认值为 None 表示"用户未设置"，便于区分"未配置"与"显式关闭"
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
# - EnvMem   : 解析为内存字节数，支持 "512M"、"1.5G" 等格式
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


class EnvClassSingleton:
    """全局环境变量配置的单例类。

    所有需要从环境变量读取的配置都在这里以类属性的形式声明，
    命名规则：类属性名即环境变量名（去掉 MINISGL_ 前缀）。

    首次实例化时，`__init__` 会遍历所有非下划线开头的类属性，
    对每个 `EnvVar` 调用 `_init`，从 os.environ 读取并解析对应值。
    之后程序中统一通过 `ENV.XXX`（或 `ENV.XXX.value`）访问配置。
    """

    _instance: EnvClassSingleton | None = None

    # ---------- shell 相关配置 ----------
    # shell 模式下单次生成的最大 token 数
    SHELL_MAX_TOKENS = EnvInt(2048)
    # 采样候选数；-1 表示不限制（使用后端默认行为）
    SHELL_TOP_K = EnvInt(-1)
    # 核采样阈值
    SHELL_TOP_P = EnvFloat(1.0)
    # 采样温度，越大越随机
    SHELL_TEMPERATURE = EnvFloat(0.6)

    # ---------- 后端运行时相关配置 ----------
    # 是否强制 FlashInfer 使用 Tensor Core；None 表示未配置，
    # 交由后端根据硬件自动决定
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    # 是否禁用调度与执行的重叠（overlap）调度，默认开启 overlap
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    # PyNcCL 通信缓冲区的最大字节数，默认 1 GiB
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)

    def __new__(cls):
        # 单例实现：始终返回同一个实例，
        # 保证环境变量只被读取和解析一次，全局配置一致
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # 由于 __new__ 保证单例，__init__ 可能被多次调用；
        # 重复调用只是重复执行同样的解析逻辑，结果幂等，安全无害。
        # 遍历类上所有公开属性（下划线开头的是内部成员，跳过），
        # 对每个环境变量包装器执行初始化。
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            # 类型检查：确保公开属性都是环境变量包装器
            assert isinstance(attr_value, BaseEnv)
            # 用 "MINISGL_<属性名>" 作为环境变量名进行读取与解析
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


# 模块级全局配置实例。import 本模块即完成环境变量的读取，
# 其他模块直接 `from minisgl.env import ENV` 后使用 `ENV.XXX.value`
# （或直接在布尔上下文中使用 `ENV.XXX`）。
ENV = EnvClassSingleton()
