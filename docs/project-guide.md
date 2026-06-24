# 项目指南

## 当前状态

当前项目默认主方案已经更新为：

- 实验名：`xgb_ranker_v3`
- 特征预设：`alpha_v1`
- 生产模型：`xgb_ranker`
- 持仓数：`Top1`
- 默认输出：`002384`
- 默认本地分数：`0.230343300111`

旧的 `xgb_ranker_v2` 双模型 Top5 等权方案仍保留在仓库中，方便回溯对比，但不再是默认链路。

## 目录结构

```text
bdc2026-stock-ranking/
├─ code/src/
│  ├─ compliance.py
│  ├─ config.py
│  ├─ featurework.py
│  ├─ score.py
│  ├─ test.py
│  └─ train.py
├─ data/
│  ├─ train.csv
│  ├─ test.csv
│  ├─ stock_data.csv
│  └─ split_train_test.py
├─ model/
│  ├─ xgb_ranker_v2/
│  ├─ xgb_ranker_v3/
│  └─ 其他实验目录
├─ output/
│  └─ result.csv
├─ temp/
│  ├─ self_score.json
│  └─ submission_check.json
├─ docs/
│  ├─ project-guide.md
│  ├─ session-operation-summary.md
│  ├─ 2026-06-23-label-experiment-summary.md
│  └─ 2026-06-24-signal-experiment-summary.md
├─ init.sh
├─ train.sh
├─ test.sh
├─ run.sh
├─ Dockerfile
└─ readme.md
```

## 默认流程

### 1. 训练

```powershell
python code/src/train.py
```

默认会训练 `alpha_v1` 特征下的完整模型族，并把生产配置记录为：

- `selected_models = ["xgb_ranker"]`
- `portfolio_size = 1`

### 2. 推理

```powershell
python code/src/test.py
```

默认会读取 `model/xgb_ranker_v3/model_metadata.json`，并输出单票结果到 `output/result.csv`。

### 3. 评分

```powershell
python code/src/score.py --baseline-result <baseline_result.csv>
```

评分报告输出到 `temp/self_score.json`。

## 特征预设说明

项目目前支持以下特征预设：

- `baseline_v1`：70 维基线量价特征
- `alpha_v1`：81 维，基线 + 11 个价量结构信号，当前默认
- `market_v1`：基线 + 市场状态信号
- `path_v1`：基线 + 路径形状信号
- `path_plus_v2`：路径信号增强版
- `cross_v1`：市场状态 + 路径形状联合
- `full_v1`：全部特征族

其中：

- `market_v1` 在历史与本地实验中明显拖分
- `path_v1 / path_plus_v2` 基本打平
- `alpha_v1` 在 `XGB Top1` 策略下形成当前最优解

## 关键工程约束

- `train.csv` 是训练和正式推理输入
- `test.csv` 只用于本地评分
- 推理输出必须是 `stock_id,weight`
- 权重和不能超过 `1.0`
- 最多输出 5 只股票，当前默认只输出 1 只
- Docker 打包最后再做，不影响当前默认链路
