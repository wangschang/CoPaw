# -*- coding: utf-8 -*-
"""Memory management module for CoPaw agents."""

import os

from .agent_md_manager import AgentMdManager
from .hybrid_memory_manager import HybridMemoryManager
from .memory_manager import MemoryManager


def create_memory_manager(working_dir: str) -> MemoryManager:
    """Factory: 根据 USE_HYBRID_MEMORY 环境变量选择 MemoryManager 实现。

    Args:
        working_dir: 工作目录路径

    Returns:
        HybridMemoryManager (USE_HYBRID_MEMORY=true) 或 MemoryManager (默认)
    """
    use_hybrid = (
        os.environ.get("USE_HYBRID_MEMORY", "false").lower() == "true"
    )
    if use_hybrid:
        return HybridMemoryManager(working_dir)
    return MemoryManager(working_dir)


__all__ = [
    "AgentMdManager",
    "MemoryManager",
    "HybridMemoryManager",
    "create_memory_manager",
]
