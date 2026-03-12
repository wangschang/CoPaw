# -*- coding: utf-8 -*-
"""HybridMemoryManager: reme-ai 为主，mem0 为辅的融合记忆管理器。

职责分工：
- reme-ai：负责文件存储、BM25+向量混合检索、上下文压缩（完全不变）
- mem0：负责自动提取对话中的用户偏好/事实，增强 memory_search 结果

改动影响：
- 不修改任何现有文件的核心逻辑
- 默认启用；可通过 USE_HYBRID_MEMORY=false 显式关闭
- mem0 初始化/操作失败不影响主流程，静默降级
"""
import logging
import os
from typing import Optional

from agentscope.message import Msg, TextBlock
from agentscope.tool import ToolResponse

from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# 尝试导入 mem0，不可用时优雅降级
try:
    from mem0 import Memory as Mem0Memory

    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False
    logger.warning(
        "mem0ai package not installed. HybridMemoryManager will run "
        "in reme-only mode. To enable mem0 enhancement: pip install mem0ai"
    )


class HybridMemoryManager(MemoryManager):
    """融合 reme-ai 和 mem0 的混合记忆管理器（Level 1）。

    继承 MemoryManager（reme-ai），在以下环节叠加 mem0：
      1. summary_memory() 触发时：将同批消息同步写入 mem0（提取结构化事实）
      2. add_async_summary_task() 触发时：额外触发 mem0 写入
      3. memory_search() 返回时：将 mem0 检索到的个性化事实追加到结果末尾

    上下文压缩（compact_memory）、文件存储、BM25检索等完全不动。

    mem0 仅支持本地模式（chromadb 向量存储），无需也不可设置 MEM0_API_KEY。
    本地数据文件（.mem0_chroma/、.mem0_history.db）自动生成，已在 .gitignore 中。

    Environment Variables（新增，全部可选）:
        USE_HYBRID_MEMORY: 是否启用混合模式，默认 true
        MEM0_ENABLE:       是否初始化 mem0，默认 true
        MEM0_USER_ID:      mem0 用户隔离 ID，默认 "copaw_default"
        MEM0_SEARCH_LIMIT: mem0 检索返回条数，默认 3

    本地 LLM 环境变量（由 mem0 自动复用）:
        OPENAI_API_KEY / MODEL_API_KEY: LLM API Key
        MODEL_BASE_URL:                 LLM Base URL（可选，支持本地代理）
        MODEL_NAME:                     模型名称，默认 "gpt-4o-mini"
    """

    def __init__(self, working_dir: str):
        # 初始化父类（reme-ai 全部功能不变）
        super().__init__(working_dir)

        self._mem0_working_dir: str = working_dir
        self._mem0: Optional[object] = None
        self._mem0_user_id: str = os.environ.get("MEM0_USER_ID", "copaw_default")
        self._mem0_search_limit: int = int(
            os.environ.get("MEM0_SEARCH_LIMIT", "3")
        )

        mem0_enable = (
            os.environ.get("MEM0_ENABLE", "true").lower() != "false"
        )
        logger.info(
            "HybridMemoryManager: initialized (working_dir=%s, mem0_enable=%s, mem0_available=%s, user_id=%s, search_limit=%s)",
            self._mem0_working_dir,
            mem0_enable,
            _MEM0_AVAILABLE,
            self._mem0_user_id,
            self._mem0_search_limit,
        )
        if mem0_enable and _MEM0_AVAILABLE:
            self._init_mem0()
        elif mem0_enable and not _MEM0_AVAILABLE:
            logger.warning(
                "MEM0_ENABLE is set but mem0ai is not installed. "
                "Running in reme-only mode. "
                "Install with: pip install mem0ai"
            )
        else:
            logger.info(
                "HybridMemoryManager: mem0 integration disabled by MEM0_ENABLE=false; running in reme-only mode"
            )

    @staticmethod
    def _preview_text(content: object, limit: int = 120) -> str:
        """Return a short single-line preview for log output."""
        text = str(content).replace("\n", "\\n")
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _init_mem0(self) -> None:
        """初始化 mem0 本地实例（失败时静默降级）。

        mem0 仅支持本地模式，使用 chromadb 作为向量存储。
        LLM 配置自动复用 CoPaw 已有的环境变量（OPENAI_API_KEY / MODEL_API_KEY、
        MODEL_BASE_URL、MODEL_NAME）。
        """
        try:
            # 优先从 CoPaw 的 ProviderManager 读取大模型配置，如果失败则回退到环境变量
            llm_api_key = ""
            llm_base_url = ""
            llm_model = ""
            try:
                from copaw.providers.provider_manager import ProviderManager
                manager = ProviderManager.get_instance()
                active_llm = manager.get_active_model()
                if active_llm and active_llm.provider_id:
                    provider = manager.get_provider(active_llm.provider_id)
                    if provider:
                        llm_api_key = provider.api_key
                        llm_base_url = provider.base_url
                        llm_model = active_llm.model
            except Exception as e:
                logger.warning("Failed to load LLM config from ProviderManager: %s", e)

            # 环境变量作为后备
            if not llm_api_key:
                llm_api_key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
            if not llm_base_url:
                llm_base_url = os.environ.get("MODEL_BASE_URL", "")
            if not llm_model:
                llm_model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

            mem0_config: dict = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": llm_model,
                        "api_key": llm_api_key,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "api_key": llm_api_key,
                    },
                },
                # 使用 chromadb 本地向量存储，数据目录自动生成
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "copaw_mem0_hybrid",
                        "path": os.path.join(
                            self._mem0_working_dir,
                            ".mem0_chroma",
                        ),
                    },
                },
                "history_db_path": os.path.join(
                    self._mem0_working_dir,
                    ".mem0_history.db",
                ),
            }
            if llm_base_url:
                mem0_config["llm"]["config"]["openai_base_url"] = llm_base_url
                mem0_config["embedder"]["config"]["openai_base_url"] = llm_base_url

            logger.info(
                "HybridMemoryManager: initializing mem0 local store (model=%s, base_url=%s, chroma_path=%s, history_db=%s)",
                llm_model,
                llm_base_url or "<default>",
                mem0_config["vector_store"]["config"]["path"],
                mem0_config["history_db_path"],
            )
            self._mem0 = Mem0Memory.from_config(mem0_config)
            logger.info(
                "HybridMemoryManager: mem0 local initialized "
                "(user_id=%s, model=%s)",
                self._mem0_user_id,
                llm_model,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "HybridMemoryManager: mem0 init failed, falling back to "
                "reme-only mode. Error: %s",
                exc,
            )
            self._mem0 = None

    # ------------------------------------------------------------------ #
    # mem0 内部工具方法
    # ------------------------------------------------------------------ #

    def _mem0_add(self, messages: list) -> None:
        """将消息写入 mem0 提取结构化事实（失败时静默）。"""
        if self._mem0 is None:
            logger.debug(
                "HybridMemoryManager: skipped mem0.add() because mem0 client is unavailable"
            )
            return
        try:
            mem0_msgs = [
                {"role": m.role, "content": str(m.content)}
                for m in messages
                if m.content and m.role in ("user", "assistant")
            ]
            if not mem0_msgs:
                logger.debug(
                    "HybridMemoryManager: skipped mem0.add() because no eligible user/assistant messages were found"
                )
                return
            logger.info(
                "HybridMemoryManager: writing %d messages to mem0 for user_id=%s (first_message=%s)",
                len(mem0_msgs),
                self._mem0_user_id,
                self._preview_text(mem0_msgs[0].get("content", "")),
            )
            self._mem0.add(mem0_msgs, user_id=self._mem0_user_id)
            logger.info(
                "HybridMemoryManager: mem0.add() wrote %d messages for user_id=%s",
                len(mem0_msgs),
                self._mem0_user_id,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "HybridMemoryManager: mem0.add() failed (non-critical): %s",
                exc,
            )

    def _mem0_search(self, query: str) -> list:
        """从 mem0 检索结构化事实（失败时返回空列表）。"""
        if self._mem0 is None:
            logger.debug(
                "HybridMemoryManager: skipped mem0.search() because mem0 client is unavailable"
            )
            return []
        try:
            logger.info(
                "HybridMemoryManager: querying mem0 (user_id=%s, limit=%d, query=%s)",
                self._mem0_user_id,
                self._mem0_search_limit,
                self._preview_text(query),
            )
            results = self._mem0.search(
                query,
                user_id=self._mem0_user_id,
                limit=self._mem0_search_limit,
            )
            memories = (
                results.get("results", [])
                if isinstance(results, dict)
                else []
            )
            facts = [
                item["memory"]
                for item in memories
                if isinstance(item, dict) and item.get("memory")
            ]
            logger.info(
                "HybridMemoryManager: mem0.search() returned %d facts for query=%s",
                len(facts),
                self._preview_text(query),
            )
            return facts
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "HybridMemoryManager: mem0.search() failed (non-critical): %s",
                exc,
            )
            return []

    # ------------------------------------------------------------------ #
    # 重写 MemoryManager 方法（仅叠加 mem0，reme-ai 原有逻辑完全不变）
    # ------------------------------------------------------------------ #

    async def summary_memory(self, messages: list[Msg], **kwargs) -> str:
        """生成摘要（reme-ai）并同步写入 mem0。"""
        # reme-ai 原有逻辑完全不动
        logger.info(
            "HybridMemoryManager: summary_memory() called with %d messages",
            len(messages),
        )
        result = await super().summary_memory(messages=messages, **kwargs)
        # 额外写入 mem0（失败不影响上面的结果）
        self._mem0_add(messages)
        return result

    def add_async_summary_task(
        self, messages: list[Msg], **kwargs
    ) -> None:
        """触发 reme-ai 后台任务，同时写入 mem0。"""
        # reme-ai 原有后台任务
        logger.info(
            "HybridMemoryManager: add_async_summary_task() called with %d messages",
            len(messages),
        )
        super().add_async_summary_task(messages=messages, **kwargs)
        # 额外写入 mem0
        self._mem0_add(messages)

    async def memory_search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.1,
    ) -> "ToolResponse":
        """混合检索：reme-ai 结果（主）+ mem0 个性化事实（追加末尾）。"""
        # reme-ai 原有检索完全不变
        logger.info(
            "HybridMemoryManager: memory_search() started (query=%s, max_results=%d, min_score=%.3f)",
            self._preview_text(query),
            max_results,
            min_score,
        )
        reme_result = await super().memory_search(
            query=query,
            max_results=max_results,
            min_score=min_score,
        )

        # mem0 个性化事实检索
        mem0_facts = self._mem0_search(query)
        if not mem0_facts:
            logger.info(
                "HybridMemoryManager: memory_search() completed without mem0 augmentation"
            )
            return reme_result  # 无增量，直接返回原结果

        # 将 mem0 事实追加为独立 TextBlock
        try:
            facts_text = "\n\n---\n**[mem0 个性化记忆]**\n" + "\n".join(
                f"- {fact}" for fact in mem0_facts
            )
            extra_block = TextBlock(type="text", text=facts_text)
            merged_content = list(reme_result.content or []) + [extra_block]
            logger.info(
                "HybridMemoryManager: memory_search() merged %d mem0 facts into response",
                len(mem0_facts),
            )
            return ToolResponse(content=merged_content)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "HybridMemoryManager: failed to merge mem0 results: %s", exc
            )
            return reme_result
