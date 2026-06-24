# 项目指南

## 当前状态

当前仓库只保留一个提交方案：

- 实验：`xgb_ranker_v3`
- 特征：`alpha_v1`
- 模型标识：`xgb_ranker`
- 提交输出：`002384,1.0`
- 本地分数：`0.230343300111`

## 目录结构

```text
bdc2026-stock-ranking/
├─ code/src/
├─ data/
├─ model/
│  └─ xgb_ranker_v3/
├─ output/
│  └─ result.csv
├─ temp/
│  ├─ self_score.json
│  └─ submission_check.json
├─ docs/
│  ├─ project-guide.md
│  └─ session-operation-summary.md
├─ init.sh
├─ train.sh
├─ test.sh
├─ run.sh
├─ Dockerfile
└─ readme.md
```

## 默认使用方式

```powershell
python code/src/test.py
python code/src/score.py --baseline-result E:\Code\bdc2026-baseline-workspace\output\result.csv
```

说明：

- `test.py` 当前默认就是冲榜版提交态
- 结果文件固定写入 `output/result.csv`
- 不再保留别的方案切换说明

## Docker 提交

```powershell
docker build -t bdc2026:latest .
docker compose up --force-recreate
docker save -o 霹雳.tar bdc2026:latest
```
