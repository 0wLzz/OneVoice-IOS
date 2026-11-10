#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
uv venv --allow-existing --python 3.12.12 .prepare/venv
uv pip sync --python .prepare/venv/bin/python Tools/requirements.txt
PATH="$PWD/.prepare/venv/bin:$PATH" bash Tools/prepare_native.sh
PYTHONDONTWRITEBYTECODE=1 .prepare/venv/bin/python Tools/prepare_models.py
