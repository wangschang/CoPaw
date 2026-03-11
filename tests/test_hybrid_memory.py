# -*- coding: utf-8 -*-
"""Tests for HybridMemoryManager and create_memory_manager factory.

Run with:
    pytest tests/test_hybrid_memory.py -v
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Patch path used to suppress ReMeLight initialisation in all tests.
_PATCH_MM_INIT = "copaw.agents.memory.memory_manager.MemoryManager.__init__"


class TestCreateMemoryManagerFactory:
    """Test the create_memory_manager() factory function."""

    def setup_method(self):
        """Save and remove relevant env vars before each test."""
        self._saved = {
            k: os.environ.pop(k, None)
            for k in ("USE_HYBRID_MEMORY", "MEM0_ENABLE")
        }

    def teardown_method(self):
        """Restore env vars after each test."""
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    @patch(_PATCH_MM_INIT, return_value=None)
    def test_default_mode_returns_memory_manager(self, _mock_init):
        """Test 1: Default mode (no env vars) returns plain MemoryManager."""
        from copaw.agents.memory import (
            HybridMemoryManager,
            MemoryManager,
            create_memory_manager,
        )

        manager = create_memory_manager("/tmp/test_workdir")

        assert isinstance(manager, MemoryManager)
        assert not isinstance(manager, HybridMemoryManager)

    @patch(_PATCH_MM_INIT, return_value=None)
    def test_hybrid_mode_returns_hybrid_memory_manager(self, _mock_init):
        """Test 2: USE_HYBRID_MEMORY=true, MEM0_ENABLE=false returns
        HybridMemoryManager with _mem0 is None."""
        os.environ["USE_HYBRID_MEMORY"] = "true"
        os.environ["MEM0_ENABLE"] = "false"

        from copaw.agents.memory import HybridMemoryManager, create_memory_manager

        manager = create_memory_manager("/tmp/test_workdir")

        assert isinstance(manager, HybridMemoryManager)
        assert manager._mem0 is None


class TestHybridMemoryManagerGracefulDegradation:
    """HybridMemoryManager graceful degradation when mem0 is disabled."""

    def setup_method(self):
        self._saved = {k: os.environ.pop(k, None) for k in ("MEM0_ENABLE",)}

    def teardown_method(self):
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    @patch(_PATCH_MM_INIT, return_value=None)
    def test_mem0_add_empty_does_nothing(self, _mock_init):
        """Test 3a: _mem0_add([]) does nothing without error when _mem0 is None."""
        os.environ["MEM0_ENABLE"] = "false"

        from copaw.agents.memory import HybridMemoryManager

        manager = HybridMemoryManager("/tmp/test_workdir")
        assert manager._mem0 is None

        # Must not raise
        manager._mem0_add([])

    @patch(_PATCH_MM_INIT, return_value=None)
    def test_mem0_search_returns_empty_list(self, _mock_init):
        """Test 3b: _mem0_search("test") returns [] without error when _mem0 is None."""
        os.environ["MEM0_ENABLE"] = "false"

        from copaw.agents.memory import HybridMemoryManager

        manager = HybridMemoryManager("/tmp/test_workdir")
        assert manager._mem0 is None

        result = manager._mem0_search("test")
        assert result == []


class TestHybridMemoryManagerMockMem0Search:
    """HybridMemoryManager with a mocked mem0 client."""

    def setup_method(self):
        self._saved = {k: os.environ.pop(k, None) for k in ("MEM0_ENABLE",)}

    def teardown_method(self):
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    @patch(_PATCH_MM_INIT, return_value=None)
    def test_mock_mem0_search_local_mode(self, _mock_init):
        """Test 4: Mocked _mem0 (local mode) returns expected list of strings."""
        os.environ["MEM0_ENABLE"] = "false"

        from copaw.agents.memory import HybridMemoryManager

        manager = HybridMemoryManager("/tmp/test_workdir")

        # Inject a mock mem0 client
        mock_mem0 = MagicMock()
        # Chinese text is intentional: verifies that Unicode/CJK strings pass
        # through the mem0 result pipeline without corruption.
        mock_mem0.search.return_value = {
            "results": [
                {"memory": "用户偏好使用 pytest"},
                {"memory": "用户喜欢 Python"},
            ]
        }
        manager._mem0 = mock_mem0
        manager._mem0_user_id = "test_user"
        manager._mem0_search_limit = 3

        result = manager._mem0_search("test query")

        assert result == ["用户偏好使用 pytest", "用户喜欢 Python"]
        mock_mem0.search.assert_called_once_with(
            "test query", user_id="test_user", limit=3
        )


class TestHybridMemoryManagerSearchMerge:
    """HybridMemoryManager.memory_search() merges reme-ai and mem0 results."""

    def setup_method(self):
        self._saved = {k: os.environ.pop(k, None) for k in ("MEM0_ENABLE",)}

    def teardown_method(self):
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    async def test_memory_search_merges_reme_and_mem0_results(self):
        """Test 5: memory_search() appends mem0 facts as a TextBlock."""
        os.environ["MEM0_ENABLE"] = "false"

        from agentscope.message import TextBlock
        from agentscope.tool import ToolResponse

        from copaw.agents.memory import HybridMemoryManager, MemoryManager

        with patch(_PATCH_MM_INIT, return_value=None):
            manager = HybridMemoryManager("/tmp/test_workdir")

        # Build a fake reme-ai ToolResponse (text is CJK test fixture data)
        fake_reme_block = TextBlock(type="text", text="reme-ai 记忆内容")
        fake_reme_response = MagicMock(spec=ToolResponse)
        fake_reme_response.content = [fake_reme_block]

        mem0_facts = ["用户偏好: 喜欢简洁回答"]

        with patch.object(
            MemoryManager,
            "memory_search",
            new_callable=AsyncMock,
            return_value=fake_reme_response,
        ):
            manager._mem0_search = MagicMock(return_value=mem0_facts)
            result = await manager.memory_search("test query")

        # The merged result must contain the reme-ai block and the mem0 TextBlock
        assert result is not None
        assert result.content is not None
        assert len(result.content) >= 2

        # First block is unchanged reme-ai content
        assert result.content[0] == fake_reme_block

        # Last block is the injected mem0 TextBlock (TypedDict → dict)
        last_block = result.content[-1]
        assert isinstance(last_block, dict)
        assert last_block.get("type") == "text"
        assert "用户偏好: 喜欢简洁回答" in last_block.get("text", "")
        assert "mem0" in last_block.get("text", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
