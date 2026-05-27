"""
This package provides efficient decoding-time KV cache compression methods.
"""

__version__ = "0.1.0"

def replace_llama(*args, **kwargs):
    from .monkeypatch import replace_llama as _replace_llama

    return _replace_llama(*args, **kwargs)


def replace_qwen3(*args, **kwargs):
    from .monkeypatch import replace_qwen3 as _replace_qwen3

    return _replace_qwen3(*args, **kwargs)


def replace_qwen3moe(*args, **kwargs):
    from .monkeypatch import replace_qwen3moe as _replace_qwen3moe

    return _replace_qwen3moe(*args, **kwargs)


def replace_qwen3_5(*args, **kwargs):
    from .monkeypatch import replace_qwen3_5 as _replace_qwen3_5

    return _replace_qwen3_5(*args, **kwargs)

__all__ = ["replace_llama", "replace_qwen3", "replace_qwen3moe", "replace_qwen3_5"]
