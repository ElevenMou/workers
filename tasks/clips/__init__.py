"""Clip-generation task feature package with lazy task loading."""

from importlib import import_module
from typing import Any

__all__ = ["custom_clip_task", "generate_clip_task"]


def __getattr__(name: str) -> Any:
    if name == "custom_clip_task":
        return import_module("tasks.clips.custom").custom_clip_task
    if name == "generate_clip_task":
        return import_module("tasks.clips.generate").generate_clip_task
    raise AttributeError(f"module 'tasks.clips' has no attribute '{name}'")
