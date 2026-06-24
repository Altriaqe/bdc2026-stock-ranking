#!/usr/bin/env bash
set -euo pipefail

REPRODUCE_FROM_TRAIN="${REPRODUCE_FROM_TRAIN:-0}"

echo "[container] start init/train/test workflow"
bash /app/init.sh

if [ "${REPRODUCE_FROM_TRAIN}" = "1" ]; then
  echo "[container] retrain from local offline data before inference"
  bash /app/train.sh
elif find /app/model -type f -name "xgb_ranker.json" | grep -q .; then
  echo "[container] existing trained model found, skip training and run inference only"
else
  echo "[container] trained model not found, run training first"
  bash /app/train.sh
fi

bash /app/test.sh
echo "[container] workflow completed"
