#!/usr/bin/env bash

set -Eeuo pipefail

# Usage:
#   ./myscripts/run_step3.sh [INPUT_DIR] [OUTPUT_DIR]
#
# Examples:
#   ./myscripts/run_step3.sh
#   ./myscripts/run_step3.sh /data/run_eval /data/run_summary
#
# PYTHON_BIN can be used to select a virtual-environment interpreter:
#   PYTHON_BIN=/path/to/python ./myscripts/run_step3.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INPUT_DIR="${1:-${REPO_ROOT}/mydata/test_eval}"
OUTPUT_DIR="${2:-${REPO_ROOT}/mydata/test}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -gt 2 ]]; then
    echo "Usage: $0 [INPUT_DIR] [OUTPUT_DIR]" >&2
    exit 64
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Input directory does not exist: ${INPUT_DIR}" >&2
    exit 66
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/step3-simplify_metrics.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}"
