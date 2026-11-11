import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)


class DisabledTqdm(tqdm):
    """禁用版 tqdm 进度条。

    HuggingFace 下载权重时会为每个分片文件创建一个 tqdm 进度条，
    在服务端/多进程场景下这些进度条会刷屏或产生多余的日志输出。
    本类继承 tqdm 并强制 `disable=True`，使下载过程静默进行；
    同时移除调用方传入的 `name` 参数，避免与内部参数冲突。
    """

    def __init__(self, *args, **kwargs):
        # 移除 snapshot_download 传入的 "name" 参数（本类不需要）
        kwargs.pop("name", None)
        # 强制禁用进度条显示
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """加载指定模型路径的 tokenizer。

    Args:
        model_path: 本地模型目录路径，或 HuggingFace Hub 上的模型 ID
                    （例如 "meta-llama/Llama-3-8B"）。

    Returns:
        加载好的 tokenizer 对象（PreTrainedTokenizerBase 的子类实例）。

    特殊处理:
        部分 Mistral 系列模型不把 chat_template 存在 tokenizer 配置里，
        而是放在仓库中单独的 chat_template.json 文件内。
        因此当加载出的 tokenizer 没有 chat_template 属性时，
        会尝试从 Hub 下载该 JSON 文件并读取其中的 chat_template 字段；
        若文件不存在或下载失败，则静默忽略（此时无聊天模板）。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    # 若 tokenizer 自身没有 chat_template，则尝试从独立 JSON 文件补充
    if not getattr(tokenizer, "chat_template", None):
        try:
            # 从 HuggingFace Hub 下载 chat_template.json（有本地缓存则直接命中）
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                # JSON 结构形如 {"chat_template": "...jinja 模板..."}
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            # 文件不存在 / 模型路径无效等情况：静默跳过，保持无模板状态
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    """带进程级缓存的 HuggingFace 模型配置加载。

    使用 functools.cache 缓存结果：同一个 model_path 在进程内
    只会真正加载一次 AutoConfig，后续调用直接返回缓存对象。
    注意返回的是被缓存的共享实例，外部不应直接修改它，
    这也是上层 `cached_load_hf_config` 要返回副本的原因。
    """
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    """加载模型配置，并返回一个独立的副本。

    Args:
        model_path: 本地模型目录路径或 HuggingFace 模型 ID。

    Returns:
        该模型的 PretrainedConfig 实例（每次调用返回新副本）。

    实现说明:
        1. 先通过带缓存的 `_load_hf_config` 拿到共享的配置对象
           （重复路径只加载一次，开销小）；
        2. 用 `config.to_dict()` 把配置转成字典，再以同类型的
           构造器重新实例化，得到一份深拷贝式的独立配置。
           这样调用方可以安全地修改返回的 config（例如追加
           推理相关字段），而不会污染进程级缓存中的共享对象。
    """
    config = _load_hf_config(model_path)
    # 用同类型构造器从字典重建配置，避免修改被缓存的共享实例
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    """确保模型权重可用，并返回权重的本地路径。

    Args:
        model_path: 本地模型目录路径，或 HuggingFace 模型 ID。

    Returns:
        权重所在的本地目录路径。

    行为说明:
        - 若 model_path 是一个已存在的本地目录，直接认为权重已就绪，
          原样返回该路径，不做任何下载；
        - 否则将其视为 HuggingFace 模型 ID，调用 snapshot_download
          下载仓库快照到本地缓存目录；
          仅下载 "*.safetensors" 格式的权重文件（跳过 .bin 等），
          并使用 DisabledTqdm 静默下载，避免进度条刷屏；
        - 若既不是本地目录、也不是有效的模型 ID（下载失败），
          抛出带原因说明的 ValueError。
    """
    # 本地目录已存在 -> 权重已就绪，无需下载
    if os.path.isdir(model_path):
        return model_path
    try:
        # 视为 HF 模型 ID：下载仓库快照，仅取 safetensors 权重文件
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        # 路径既非本地目录也非有效模型 ID，给出明确报错
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
