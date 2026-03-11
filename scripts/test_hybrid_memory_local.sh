#!/usr/bin/env bash
# =============================================================================
# test_hybrid_memory_local.sh
#
# Local run guide for HybridMemoryManager tests.
#
# mem0 runs in LOCAL-ONLY mode using chromadb for vector storage.
# MEM0_API_KEY is NOT required and NOT supported.
#
# Usage:
#   bash scripts/test_hybrid_memory_local.sh
#
# Prerequisites:
#   - Python ≥3.10
#   - Project installed in editable mode:  pip install -e ".[dev]"
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Check Python version
# -----------------------------------------------------------------------------
echo "=== Checking Python version ==="
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python ${python_version}"

required_major=3
required_minor=10
actual_major=$(echo "${python_version}" | cut -d. -f1)
actual_minor=$(echo "${python_version}" | cut -d. -f2)

if [ "${actual_major}" -lt "${required_major}" ] || \
   { [ "${actual_major}" -eq "${required_major}" ] && [ "${actual_minor}" -lt "${required_minor}" ]; }; then
    echo "ERROR: Python ${required_major}.${required_minor}+ is required (found ${python_version})."
    exit 1
fi
echo "✓ Python version OK"
echo

# -----------------------------------------------------------------------------
# 2. Install mem0ai and chromadb (local vector store backend) if not present
# -----------------------------------------------------------------------------
echo "=== Checking mem0ai and chromadb installation ==="
if python3 -c "import mem0" 2>/dev/null; then
    echo "✓ mem0ai is already installed"
else
    echo "mem0ai not found — installing..."
    pip install "mem0ai>=0.1.0"
    echo "✓ mem0ai installed"
fi

if python3 -c "import chromadb" 2>/dev/null; then
    echo "✓ chromadb is already installed"
else
    echo "chromadb not found — installing..."
    pip install "chromadb>=0.4.0"
    echo "✓ chromadb installed"
fi
echo

# -----------------------------------------------------------------------------
# 3. Set up local environment variables
#
#    Required for running copaw with HybridMemoryManager:
#      USE_HYBRID_MEMORY=true   -- enable HybridMemoryManager
#      OPENAI_API_KEY           -- API key for the local LLM (used by mem0)
#      MODEL_NAME               -- LLM model name (default: gpt-4o-mini)
#      MODEL_BASE_URL           -- optional, for local LLM proxy (e.g. ollama)
#
#    For unit tests only (skips real LLM/vector-DB calls):
#      MEM0_ENABLE=false
#
#    NOTE: MEM0_API_KEY is NOT used. mem0 runs locally only.
#          .mem0_chroma/ and .mem0_history.db are auto-generated locally.
# -----------------------------------------------------------------------------
echo "=== Setting test environment variables ==="
export USE_HYBRID_MEMORY=true
export MEM0_ENABLE=false
echo "  USE_HYBRID_MEMORY=${USE_HYBRID_MEMORY}"
echo "  MEM0_ENABLE=${MEM0_ENABLE}"
echo

# -----------------------------------------------------------------------------
# 4. (Optional) Start copaw main program with local mem0
#    Uncomment and configure the env vars below to launch copaw.
# -----------------------------------------------------------------------------
# export OPENAI_API_KEY=<your-api-key>
# export MODEL_NAME=gpt-4o-mini
# export MODEL_BASE_URL=                  # optional: http://localhost:11434/v1
# export USE_HYBRID_MEMORY=true
# export MEM0_ENABLE=true
# export MEM0_USER_ID=copaw_default       # optional
# export MEM0_SEARCH_LIMIT=3             # optional
# python3 -m copaw
echo "  (To launch copaw with HybridMemoryManager, configure OPENAI_API_KEY,"
echo "   MODEL_NAME, USE_HYBRID_MEMORY=true, MEM0_ENABLE=true, then run:"
echo "   python3 -m copaw)"
echo

# -----------------------------------------------------------------------------
# 5. Run the test file with pytest
# -----------------------------------------------------------------------------
echo "=== Running tests ==="
python3 -m pytest tests/test_hybrid_memory.py -v
echo

echo "Done."
