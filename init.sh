#!/usr/bin/env bash
set -euo pipefail

echo "[初始化] 正在准备目录结构..."
mkdir -p /app/data
mkdir -p /app/model
mkdir -p /app/output
mkdir -p /app/temp

BUNDLED_DATA_DIR="/opt/bdc2026-bundle/data"
if [ ! -f /app/data/train.csv ] || [ ! -f /app/data/test.csv ]; then
  echo "[初始化] 检测到挂载数据缺失，正在从镜像内置备份恢复..."
  cp -f "${BUNDLED_DATA_DIR}/train.csv" /app/data/train.csv
  cp -f "${BUNDLED_DATA_DIR}/test.csv" /app/data/test.csv
  echo "[初始化] 已恢复内置 train.csv 和 test.csv。"
fi

echo "[初始化] 目录检查完成。"
