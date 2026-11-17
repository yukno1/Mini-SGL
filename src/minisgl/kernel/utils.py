"""自定义 CUDA kernel 的编译加载工具。

基于 TVM FFI（``tvm_ffi``）提供的两种编译加载方式：

- :func:`load_aot`: 直接编译 ``csrc/src`` 目录下现成的源文件（AOT 模式），
  适合固定不变的 kernel；
- :func:`load_jit`: 把 ``csrc/jit`` 目录下的源文件 `#include` 进一个
  动态生成的翻译单元，并按需追加 ``TVM_FFI_DLL_EXPORT_TYPED_FUNC``
  导出宏，实现"以模板参数实例化 kernel"的 JIT 编译——同一份
  kernel 源码可以按不同的模板实参编译出多个专属符号。

``*args`` 字符串参数用于构造唯一的模块名（``minisgl__xxx_yyy``），
模块名不同即编译产物不同，可借此实现按配置缓存编译结果。
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

if TYPE_CHECKING:
    from tvm_ffi import Module

# kernel 源码根目录及各默认编译配置
KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]  # 默认头文件搜索路径
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]  # C++ 默认编译选项
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]  # CUDA 默认编译选项
DEFAULT_LDFLAGS = []  # 默认链接选项
# 允许传入 C++ 模板的 Python 标量类型（bool 需转成 true/false）
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


class CppArgList(list[str]):
    """C++ 模板实参列表，额外提供 ``str`` 直接拼成逗号分隔串。"""

    def __str__(self) -> str:
        return ", ".join(self)


class KernelConfig(NamedTuple):
    """kernel 的启动配置，同时作为 JIT 模板实参的一部分。

    :param num_threads: 每个 block 的线程数。
    :param max_occupancy: 期望的最大 occupancy（如 grid 尺寸上限的提示）。
    :param use_pdl: 是否启用 Programmatic Dependent Launch（PDL），
        允许相邻 kernel 重叠执行以隐藏启动延迟。
    """

    num_threads: int
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        """把配置转成 C++ 模板实参字符串，如 ``"256,8,true"``。"""
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


def _make_name(*args: str) -> str:
    """由若干片段拼出唯一模块名，如 ``("gemm", "bf16") -> "minisgl__gemm_bf16"``。

    模块名即编译缓存键：不同模板配置对应不同名字，避免互相覆盖。
    """
    return "minisgl__" + "_".join(str(arg) for arg in args)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    """生成 TVM FFI 的符号导出宏代码行。

    :param tup: ``(导出符号名, kernel 表达式)`` 二元组，例如
        ``("foo_f32", "kernel::foo<float>")``。
    :return: 形如 ``TVM_FFI_DLL_EXPORT_TYPED_FUNC(foo_f32, (kernel::foo<float>));``
        的导出语句，用于把模板实例化的 kernel 暴露为可加载的符号。
    """
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    """把 Python 标量转换为 C++ 模板实参字符串列表。

    - bool -> ``true`` / ``false``（注意需先于 int 判断，bool 是 int 子类）；
    - int / float -> 直接字符串化。

    :param args: 模板实参。
    :return: :class:`CppArgList`。
    :raises TypeError: 出现不支持的类型时抛出。
    """

    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """AOT 模式加载：编译 ``csrc/src`` 下已有的源文件并加载为 TVM FFI 模块。

    :param args: 用于构造唯一模块名的名字片段（也作为缓存键）。
    :param cpp_files: ``csrc/src`` 下的 C++ 源文件名列表。
    :param cuda_files: ``csrc/src`` 下的 CUDA 源文件名列表。
    :param extra_cflags / extra_cuda_cflags / extra_ldflags /
        extra_include_paths: 在默认配置基础上追加的编译/链接选项。
    :param build_directory: 编译产物目录；None 时使用默认缓存目录。
    :return: 编译加载后的 TVM FFI 模块，可直接按符号名调用 kernel。
    """
    from tvm_ffi.cpp import load

    # 各可选参数归一化为空列表
    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # 把相对文件名解析为 csrc/src 下的绝对路径
    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    return load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """JIT 模式加载：动态拼装翻译单元并编译，支持模板实例化导出。

    生成的 C++/CUDA 源由两部分组成：
    1. ``#include`` ``csrc/jit`` 下的对应头文件（kernel 模板实现）；
    2. :func:`_make_wrapper` 生成的导出宏，把"模板 kernel +
       特定实参"实例化为具名符号。

    :param args: 用于构造唯一模块名的名字片段（通常包含模板配置，
        如 ``make_cpp_args`` 生成的实参串，使不同实例化各有缓存）。
    :param cpp_files / cuda_files: ``csrc/jit`` 下要 include 的文件名列表。
    :param cpp_wrappers / cuda_wrappers: ``(导出符号名, kernel 表达式)``
        列表，逐项生成导出宏。
    :param extra_cflags / extra_cuda_cflags / extra_ldflags /
        extra_include_paths: 在默认配置基础上追加的编译/链接选项。
    :param build_directory: 编译产物目录；None 时使用默认缓存目录。
    :return: 编译加载后的 TVM FFI 模块。
    """
    from tvm_ffi.cpp import load_inline

    # 各可选参数归一化为空列表
    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    # 动态生成的 C++ 翻译单元：include 头文件 + 导出宏
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    # 同理生成 CUDA 翻译单元
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
