#!/usr/bin/env bash
set -euo pipefail

# Autoresearch benchmark for DDTree speculative decoding
# Outputs: METRIC acceptance_length=N  (or METRIC unit_test_passed=1 for fast CI)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== DDTree Autoresearch Benchmark ===" >&2

# Check for GPU
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "No GPU available — running unit tests only" >&2
    
    # Run DDTree unit tests
    .venv/bin/python -m pytest tests/v1/spec_decode/test_ddtree.py -v --tb=short 2>&1 | tee /tmp/autoresearch-output.txt
    
    if grep -q "passed" /tmp/autoresearch-output.txt; then
        echo "METRIC unit_test_passed=1"
        echo "METRIC acceptance_length=0"
    else
        echo "METRIC unit_test_passed=0"
        echo "METRIC acceptance_length=0"
        exit 1
    fi
    exit 0
fi

echo "GPU available" >&2

# Check if DDTreeProposer exists
if [[ ! -f "vllm/v1/spec_decode/ddtree_proposer.py" ]]; then
    echo "METRIC unit_test_passed=0"
    echo "METRIC acceptance_length=0"
    echo "DDTreeProposer not yet implemented" >&2
    exit 1
fi

echo "DDTreeProposer found, running import test..." >&2

# Quick import test
.venv/bin/python -c "
import torch
from vllm.v1.spec_decode.ddtree_proposer import DDTreeProposer
print('DDTreeProposer import: OK')
" 2>&1 | tee /tmp/autoresearch-output.txt

if grep -q "OK" /tmp/autoresearch-output.txt; then
    echo "METRIC import_test=1"
    echo "METRIC acceptance_length=0"
else
    echo "METRIC import_test=0"
    echo "METRIC acceptance_length=0"
    exit 1
fi
