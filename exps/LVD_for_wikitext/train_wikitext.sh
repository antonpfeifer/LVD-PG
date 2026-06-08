#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -u "$script_dir/get_data_for_PG.py" \
  --max-rows 20000 --max-chunks 20000 --batch-size 32
