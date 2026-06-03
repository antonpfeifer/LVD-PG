#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -u "$script_dir/get_data_for_PG.py" \
  --max-rows 1000 --max-chunks 1000 --batch-size 32
