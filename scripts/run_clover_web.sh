#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv-clover/bin/python ]]; then
  printf '%s\n' '请先创建环境：python -m venv .venv-clover && .venv-clover/bin/pip install tokenizers' >&2
  exit 1
fi
if [[ ! -x third_party/Strata/build-clover/strata-q36 ]]; then
  printf '%s\n' '没有找到 Strata 可执行文件，请先构建 build-clover/strata-q36。' >&2
  exit 1
fi
exec .venv-clover/bin/python scripts/clover_web.py "$@"
