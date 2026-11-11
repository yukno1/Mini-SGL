"""通用注册表工具模块。

提供一个轻量级的名称 -> 对象注册器，常用于：
- 注册模型/优化器/调度器等可插拔组件；
- 在 argparse 校验中检查用户传入的名称是否受支持。
"""

from typing import Callable, Generic, Iterable, List, TypeVar

# 泛型类型变量：注册表中存放的对象类型（可以是函数、类等任意对象）
T = TypeVar("T")


class Registry(Generic[T]):
    """按名称注册和查找对象的通用注册表。

    用法示例::

        TOKENIZERS = Registry("tokenizer")

        @TOKENIZERS.register("gpt2")
        def load_gpt2(): ...

        loader = TOKENIZERS["gpt2"]        # 按名称取回对象
        TOKENIZERS.supported_names()       # 返回所有已注册名称
        TOKENIZERS.assert_supported("gpt2")  # 校验名称是否受支持

    泛型参数 T 表示注册对象的类型，方便静态类型检查器推断。
    """

    __slots__ = ("_registry", "_type")

    def __init__(self, type: str):
        """初始化注册表。

        :param type: 注册表的类别名称（如 "optimizer"），仅用于
            在抛出异常时生成更友好的错误信息，不参与逻辑判断。
        """
        # 内部存储：名称 -> 注册对象的映射
        self._registry = {}
        # 类别名称，用于错误信息提示
        self._type = type

    def register(self, name: str) -> Callable[[T], None]:
        """注册装饰器工厂：将指定名称与某个对象绑定。

        用作装饰器::

            @REGISTRY.register("my_name")
            class MyClass: ...

        :param name: 注册名称，必须唯一。
        :return: 一个装饰器函数，被装饰的对象会被存入注册表。
        :raises KeyError: 如果该名称已经被注册过。
        """
        # 先检查名称是否已存在，避免重复注册导致覆盖
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item: T) -> None:
            # 被装饰的对象（类/函数等）以 name 为键存入注册表
            self._registry[name] = item

        return decorator

    def __getitem__(self, name: str) -> T:
        """支持下标语法按名称查找注册的对象，例如 ``REGISTRY["name"]``。

        :param name: 要查找的注册名称。
        :return: 对应的注册对象。
        :raises KeyError: 如果名称不存在，错误信息中带类别名。
        """
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]

    def supported_names(self) -> List[str]:
        """返回当前所有已注册的名称列表（用于错误提示或帮助信息）。"""
        return list(self._registry.keys())

    def assert_supported(self, names: str | Iterable[str]) -> None:
        """断言给定的名称（单个或多个）全部已注册，否则抛出参数错误。

        主要用于 argparse 的 ``type=`` 校验：当用户传入不支持的
        选项值时，能给出"受支持项列表"的清晰错误提示。

        :param names: 单个名称字符串，或名称的可迭代集合。
        :raises argparse.ArgumentTypeError: 存在任何未注册的名称时抛出。
        """
        # 统一处理：单个字符串包装成列表，便于统一遍历
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in self._registry:
                # 延迟导入，避免模块加载时的循环依赖或不必要开销
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {name}. "
                    f"Supported items: {self.supported_names()}"
                )
