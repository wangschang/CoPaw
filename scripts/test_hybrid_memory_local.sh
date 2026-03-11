#!/usr/bin/env bash
# =============================================================================
# test_hybrid_memory_local.sh
#
# Local run guide for HybridMemoryManager tests.
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
# 2. Install mem0ai if not present
# -----------------------------------------------------------------------------
echo "=== Checking mem0ai installation ==="
if python3 -c "import mem0" 2>/dev/null; then
    echo "✓ mem0ai is already installed"
else
    echo "mem0ai not found — installing..."
    pip install "mem0ai>=0.1.0"
    echo "✓ mem0ai installed"
fi
echo

# -----------------------------------------------------------------------------
# 3. Set up test environment variables
#    MEM0_ENABLE=false avoids real LLM/vector-DB calls during unit tests.
# -----------------------------------------------------------------------------
echo "=== Setting test environment variables ==="
export USE_HYBRID_MEMORY=true
export MEM0_ENABLE=false
echo "  USE_HYBRID_MEMORY=${USE_HYBRID_MEMORY}"
echo "  MEM0_ENABLE=${MEM0_ENABLE}"
echo

# -----------------------------------------------------------------------------
# 4. Run the test file with pytest
# -----------------------------------------------------------------------------
echo "=== Running tests ==="
python3 -m pytest tests/test_hybrid_memory.py -v
echo

# -----------------------------------------------------------------------------
# 5. How to run with real mem0 cloud
# -----------------------------------------------------------------------------
echo "=== Running with real mem0 cloud (example) ==="
echo ""
echo "  To test against the actual mem0 cloud service, set these env vars:"
echo ""
echo "    export MEM0_API_KEY=<your-mem0-cloud-api-key>"
echo "    export MEM0_USER_ID=my_user           # optional, defaults to copaw_default"
echo "    export MEM0_SEARCH_LIMIT=3            # optional, defaults to 3"
echo "    export USE_HYBRID_MEMORY=true"
echo "    export MEM0_ENABLE=true"
echo ""
echo "  Then run (skipping factory tests that don't need cloud access):"
echo "    pytest tests/test_hybrid_memory.py -v -k 'not (test_default_mode or test_hybrid_mode)'"
echo ""
echo "  Or to run only the factory and graceful-degradation tests (no cloud calls):"
echo "    MEM0_ENABLE=false pytest tests/test_hybrid_memory.py -v"
echo ""
echo "Done."
