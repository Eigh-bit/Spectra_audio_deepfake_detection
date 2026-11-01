#!/bin/bash
# Foreground execution; preserve each path argument and the actual Python exit code.
set -euo pipefail
run_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "${SPECTRA_PYTHON:-python3.10}" "$run_dir/model.py" "$@"
