#!/bin/bash
# 解决 cv2 和 PyQt5 的 Qt 插件冲突
source "$(dirname "$0")/../.venv/bin/activate"
# 移除 cv2 自带的 qt 插件（与 PyQt5 冲突）
mv "$VIRTUAL_ENV/lib/python3.13/site-packages/cv2/qt/plugins" \
   "$VIRTUAL_ENV/lib/python3.13/site-packages/cv2/qt/plugins.bak" 2>/dev/null
exec labelme --labels "object,pearlite" "$@"
