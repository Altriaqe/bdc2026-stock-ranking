# BDC2026 股票排序方案说明

## 1. 环境配置

本项目按离线复现和 Docker 提交要求组织，核心运行环境如下：

- Python 3.11
- bash
- libgomp1

Python 依赖：

```text
numpy>=1.26,<3.0
pandas>=2.2,<3.0
xgboost>=2.0,<3.0
lightgbm>=4.0,<5.0
scikit-learn>=1.3,<2.0
scipy>=1.11,<2.0
baostock>=0.8.9,<1.0
```

其中 `baostock` 仅用于赛前抓取公开原始数据。正式复现训练与推理阶段不联网，直接使用仓库内固化数据文件。

## 2. 数据说明

当前数据语义已经统一：

- `data/train.csv`：历史可见数据，训练和正式推理都使用它
- `data/test.csv`：未来一周真实数据，只用于本地评分，不参与正式推理
- `data/stock_data.csv`：赛前抓取并切分前的原始历史数据

赛前数据准备流程：

```bash
python get_stock_data.py --start-date 2024-01-01 --end-date 2026-03-15
python data/split_train_test.py --input data/stock_data.csv
```

## 3. 当前默认方案

当前仓库默认主方案已经升级为 `xgb_ranker_v3`，不是旧的双模型 Top5 等权方案。

默认配置如下：

- 特征预设：`alpha_v1`
- 生产模型：`xgb_ranker`
- 持仓数：`Top1`
- 输出权重：单票 `1.0`
- 训练标签：按交易日分组，`T+1` 开盘买入、`T+5` 开盘卖出，未来收益做十分位分桶，标签为 `0~9`

当前默认输出结果：

- 股票：`002384`
- 权重：`1.0`

当前本地评分结果：

- 当前分数：`0.230343300111`
- baseline 分数：`0.025179491217`
- 差值：`+0.205163808894`

## 4. 算法说明

### 4.1 整体思路

本项目将问题建模为按交易日分组的排序任务。每个交易日全部候选股票构成一个 ranking group，模型根据历史量价特征预测未来一周收益更优的股票。

当前默认主线不是“分散持有 5 只”，而是“利用更强的单票置信度做集中配置”：

1. 基于 `train.csv` 构造滚动量价特征和横截面特征
2. 按 `T+1` 买入、`T+5` 卖出定义未来收益
3. 将未来收益按交易日做十分位分桶，形成排序标签
4. 训练 XGBoost Ranker、LightGBM Ranker、HGB 回归器作为实验模型族
5. 默认生产只使用 `alpha_v1` 特征下的 `XGBoost Ranker`
6. 推理时仅输出得分最高的 1 只股票

### 4.2 特征工程

当前特征框架支持多套预设，已接入：

- `baseline_v1`
- `alpha_v1`
- `market_v1`
- `path_v1`
- `path_plus_v2`
- `cross_v1`
- `full_v1`

默认使用的 `alpha_v1` 在 70 维基线特征基础上增加 11 个价量结构信号，共 81 维，主要包括：

- `close_to_vwap`
- `open/close/vwap` 与成交量的 10/20 日滚动相关
- `delta_price_to_ma_10_3`
- 对应的横截面 rank 特征

### 4.3 模型

项目中保留三类模型能力：

1. `XGBoost Ranker`
   - objective: `rank:ndcg`
   - n_estimators: 600
   - learning_rate: 0.03
   - max_depth: 5
2. `LightGBM Ranker`
   - objective: `lambdarank`
   - n_estimators: 800
   - learning_rate: 0.03
   - max_depth: 6
   - num_leaves: 63
3. `HistGradientBoostingRegressor`
   - 作为诊断模型保留，用于对比和策略探索

虽然训练阶段仍会产出三类模型，但默认生产推理只使用 `xgb_ranker`。

### 4.4 数据扩增

训练阶段仍保留轻量特征扰动：

- 约 30% 特征列加入高斯噪声
- 噪声尺度为特征标准差的 `0.01`
- 不同模型使用不同随机种子

## 5. 训练与推理流程

### 5.1 训练

```bash
python code/src/train.py
```

默认行为：

- 实验名：`xgb_ranker_v3`
- 特征预设：`alpha_v1`
- 生产模型：`xgb_ranker`
- 持仓数：`1`

训练输出目录：

- `model/xgb_ranker_v3/xgb_ranker.json`
- `model/xgb_ranker_v3/lgb_ranker.txt`
- `model/xgb_ranker_v3/hgb_regressor.pkl`
- `model/xgb_ranker_v3/model_metadata.json`
- `model/xgb_ranker_v3/training_report.json`
- `model/xgb_ranker_v3/training_summary.json`

### 5.2 推理

```bash
python code/src/test.py
```

默认会读取 `model/xgb_ranker_v3/model_metadata.json`，按照训练时记录的特征列、模型列表和持仓数生成 `output/result.csv`。

### 5.3 本地评分

```bash
python code/src/score.py --baseline-result <baseline_result.csv>
```

评分结果会写入：

- `temp/self_score.json`
- `temp/submission_check.json`

## 6. 合规性说明

本项目当前符合以下工程约束：

- 训练与推理默认使用 `data/train.csv`
- `data/test.csv` 仅用于本地评分
- 输出文件固定为 UTF-8 编码的 `stock_id,weight`
- 最多输出 5 只股票，当前默认只输出 1 只，权重和为 `1.0`
- 整个正式复现链路不依赖联网

## 7. 备注

- Docker 打包入口和离线复现链路已经保留，但最终打包动作暂未在本轮执行
- 后续若需要导出提交镜像，可在当前默认主方案基础上再做一次打包检查
