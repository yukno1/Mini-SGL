from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """装饰器工厂：控制被装饰函数是否在脚本作为主程序运行时被立即调用。

    参数:
        name: 判断脚本身份用的模块名，通常直接使用默认值 "__main__"。
              被导入（name != "__main__"）时，装饰器只做透传或直接丢弃；
              作为主脚本运行（name == "__main__"）时，装饰器会立即执行被装饰函数。
        discard: 是否"丢弃"被装饰函数，仅在特定分支下生效（见下）。

    各分支行为:
        - name != "__main__"（脚本被 import，如作为模块或启动入口加载）:
            - discard=None（默认 -> False）: 原样返回函数，什么也不做；
            - discard=True: 返回 lambda _: None，即装饰后函数被替换为
              返回 None 的空函数（函数体不会执行）。

        - name == "__main__"（脚本被直接运行）:
            - discard=None（默认 -> True）: 立即调用函数并让该名字变成
              无意义的 True（`(f() or True) and None`），
              避免函数对象在模块命名空间残留；
            - discard=False: 立即调用函数一次，然后返回函数本身
              （`(f() and None) or f`），该名字仍指向可调用的函数。

    典型用法:
        @call_if_main()
        def main():
            ...
        直接 `python xxx.py` 时自动执行 main()，被 import 时则不执行。
    """
    if name != "__main__":
        # 脚本被 import 的情况：默认不丢弃，仅透传
        discard = False if discard is None else discard
        if discard:
            # 装饰后替换为一个永远返回 None 的空函数，函数体永不执行
            return lambda _: None
        else:
            # 原样返回被装饰函数
            return lambda f: f
    else:
        # 脚本作为主程序运行的情况：默认丢弃
        discard = True if discard is None else discard
        if discard:
            # 立即调用 f()，随后整体求值为 None：
            # f() 有返回值 -> or True -> True and None -> None
            # f() 返回 None -> or True -> None
            # 效果：函数执行一次，且模块里该名字不再是函数
            return lambda f: (f() or True) and None
        else:
            # 立即调用 f() 一次，然后仍返回函数本身：
            # f() 返回真值 -> and None -> None -> or f -> f
            # f() 返回 None  -> and None -> None -> or f -> f
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """整除 a / b，要求结果必须是整数（否则断言失败）。

    Args:
        a: 被除数。
        b: 除数。
        allow_replicate: 是否允许"复制"语义（用于 GQA 的 KV 头复制场景）。
                         为 True 且 b > a 时，若 b 能被 a 整除，则返回 1，
                         表示每个 KV 头需要复制到 b / a 个 Q 头，
                         此时没有 Q 头被切分，切分份数为 1。

    Returns:
        a // b（或复制场景下的 1）。

    Raises:
        AssertionError: 不满足整除条件时抛出 。
    """
    if allow_replicate and b > a:
        # KV 头复制场景：Q 头数少于 KV 头数，要求 b 能被 a 整除
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """两整数相除并向上取整（a / b 向上取整）。

    通过 (a + b - 1) // b 实现，例如 div_ceil(7, 3) == 3。

    纯整数运算避免了浮点除法和 math.ceil 的函数调用开销，
    在热点路径（比如推理引擎中频繁计算 block 数、内存对齐）中更快。
    """
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """把 a 向上对齐到 b 的整数倍。

    例如 align_ceil(10, 8) == 16。常用于内存/显存大小、
    块大小等需要按固定粒度对齐的场景。
    """
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """把 a 向下对齐到 b 的整数倍。

    例如 align_down(10, 8) == 8。
    """
    return (a // b) * b


class Unset:
    """表示"未设置"的哨兵类型。

    与 None 不同，Unset 可以区分"值为 None"和"值从未被设置/提供"
    这两种情况，常用于函数默认参数或可选配置项的占位。
    """

    pass


# Unset 的全局唯一实例，作为"未设置"的哨兵值使用：
# 判断时用 `is UNSET`（而不是 ==），保证语义唯一。
UNSET = Unset()
