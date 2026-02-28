#!/bin/bash
# Quick-run pipeline validation tests.
# Usage: ./scripts/validate_pipeline.sh
set -e
cd "$(dirname "$0")/.."

VENV="${VENV:-/home/narvis/develop/rendering/.venv-renderer}"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi

echo "=== Running pipeline validation tests ==="
pytest tests/test_pipeline_validation.py -v --tb=long -s 2>&1 | tee validation_report.txt
echo ""
echo "Report saved to validation_report.txt"
