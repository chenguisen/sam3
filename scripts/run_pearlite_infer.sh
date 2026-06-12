#!/bin/bash
# 珠光体检测启动脚本
# 自动激活项目 venv 并运行推理

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR" || exit 1

# 激活 venv
source "$PROJECT_DIR/.venv/bin/activate"

# 运行 (传递所有命令行参数)
exec python scripts/pearlite_infer.py "$@"
