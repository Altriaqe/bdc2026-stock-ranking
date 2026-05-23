# 项目指南

当前项目 bdc2026-stock-ranking，BDC 2026 股票排序任务。主方案是 XGBoost + LightGBM 双模型集成。

## 目录结构

```text
bdc2026-stock-ranking/
├── code/src/
│   ├── compliance.py    # 结果格式校验
│   ├── config.py        # 所有配置和超参数（含 XGB + LGB）
│   ├── featurework.py   # 数据标准化、特征工程（70 维）、标签构造
│   ├── score.py         # 本地评分（按官方口径）
│   ├── test.py          # 推理入口（双模型集成）
│   └── train.py         # 训练入口（双模型 + 数据扩增）
├── data/
│   ├── run.sh           # 旧版容器入口
│   ├── split_train_test.py
│   ├── stock_data.csv   # 完整历史行情（baostock 抓的）
│   ├── hs300_stock_list.csv
│   ├── train.csv        # 训练 + 推理输入
│   └── test.csv         # 本地评分专用
├── model/xgb_ranker_v2/ # 当前实验
│   ├── xgb_ranker.json
│   ├── lgb_ranker.txt
│   ├── model_metadata.json
│   ├── feature_importance.csv
│   ├── feature_importance_lgb.csv
│   ├── training_summary.json
│   ├── training_report.json
│   └── project_config.json
├── output/result.csv    # 提交结果
├── temp/                # self_score.json + submission_check.json
├── Dockerfile
├── docker-compose.yml
├── run.sh               # 容器入口（根目录，不受 data 挂载影响）
├── init.sh / train.sh / test.sh
├── get_stock_data.py    # 联网抓数据（赛前准备阶段）
└── readme.md
```

## 数据语义

最容易搞混的地方：

- `train.csv` — 预测时点之前你能看到的历史数据。训练用，推理也用（取每只股票最新一天截面预测下周）
- `test.csv` — 未来一周真实数据，只做本地评分。**不要拿来预测**
- `stock_data.csv` — baostock 抓的完整日线，切分前的原始数据

## 完整流程

**1. 抓数据（联网，赛前准备）**

```powershell
python get_stock_data.py --start-date 2024-01-01 --end-date 2026-03-15
```

产出 `data/stock_data.csv` 和 `data/hs300_stock_list.csv`。

**2. 切分训练/测试**

```powershell
python data/split_train_test.py --input data/stock_data.csv
```

产出 `data/train.csv` 和 `data/test.csv`。

**3. 训练**

```powershell
python code/src/train.py
```

读 train.csv，标准化 → 特征工程（70 维）→ 构造标签（T+1 买 T+5 卖）→ 数据扩增（30% 特征加高斯噪声）→ 训 XGBoost Ranker（rank:ndcg）→ 训 LightGBM Ranker（lambdarank）→ 集成验证。模型和报告存到 `model/xgb_ranker_v2/`。

**4. 预测**

```powershell
python code/src/test.py
```

读 train.csv，取每只股票最新截面，两个模型分别打分 → z-score 标准化取平均 → Top 5 等权 → `output/result.csv`。

**5. 本地评分**

```powershell
# 只看自己
python code/src/score.py --score-data data/test.csv

# 跟官方基准比
python code/src/score.py --score-data data/test.csv --baseline-result <官方result.csv>
```

评分报告在 `temp/self_score.json`。

**6. Docker 打包**

```bash
docker build -t bdc2026 .
docker save -o 队伍名.tar bdc2026
```

## 当前模型参数

| 参数 | XGBoost | LightGBM |
| --- | --- | --- |
| 目标函数 | rank:ndcg | lambdarank |
| 评估指标 | ndcg@5 | ndcg@5 |
| 树数量 | 600 | 800 |
| 学习率 | 0.03 | 0.03 |
| 最大深度 | 5 | 6 |
| 叶节点数 | — | 63 |
| 正则化 | alpha=0.05, lambda=1.2 | alpha=0.05, lambda=1.0 |
| 采样 | subsample=0.85, colsample=0.85 | subsample=0.8, colsample=0.8 |
| 特征数 | 70 | 70 |
| 特征窗口 | 3/5/10/20 日 | 3/5/10/20 日 |
| 标签 | 十分位分桶 (0~9) | 十分位分桶 (0~9) |
| 数据扩增 | 30% 特征加高斯噪声 | 30% 特征加高斯噪声（不同种子） |
| 集成方式 | z-score 标准化后取平均 | z-score 标准化后取平均 |

## 当前得分

| 指标 | 值 |
| --- | --- |
| 当前得分 | 0.084 |
| 基准得分 | 0.025 |
| 差值 | +0.059 |
| 排名资格 | 达标 |

选股：600489、600547、300394、000975、002384

## 继续优化的方向

按投入产出比排序：

- 调 LightGBM 超参（目前验证集上 LGB 弱于 XGB，调好了可能反超拉高集成上限）
- featurework.py 加行业相对强弱因子（需要 baostock 拉一次行业分类再报备 MD5）
- 多持有周期集成（T+3 / T+5 / T+10 各训一组再投票）
- 换 CatBoost 做三模型集成
- 滚动训练窗 + 衰减加权
