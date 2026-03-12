# -*- coding: utf-8 -*-
"""Memory management module for CoPaw agents."""

import logging
import os

from .agent_md_manager import AgentMdManager
from .hybrid_memory_manager import HybridMemoryManager
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


def create_memory_manager(working_dir: str) -> MemoryManager:
    """Factory: 根据 USE_HYBRID_MEMORY 环境变量选择 MemoryManager 实现。

    Args:
        working_dir: 工作目录路径

    Returns:
        HybridMemoryManager（默认）或 MemoryManager（USE_HYBRID_MEMORY=false）
    """
    use_hybrid = (
        os.environ.get("USE_HYBRID_MEMORY", "true").lower() != "false"
    )
    if use_hybrid:
        logger.info(
            "create_memory_manager(): using HybridMemoryManager (working_dir=%s, MEM0_ENABLE=%s)",
            working_dir,
            os.environ.get("MEM0_ENABLE", "true"),
        )
        return HybridMemoryManager(working_dir)
    logger.info(
        "create_memory_manager(): using MemoryManager (working_dir=%s)",
        working_dir,
    )
    return MemoryManager(working_dir)


__all__ = [
    "AgentMdManager",
    "MemoryManager",
    "HybridMemoryManager",
    "create_memory_manager",
]
