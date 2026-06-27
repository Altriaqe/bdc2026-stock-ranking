# BDC2026 股票排序代码说明报告

## 一、项目摘要

本项目面向 BDC2026 股票排序任务，目标是在给定历史行情数据的基础上，训练机器学习排序模型，输出未来一周预期收益最高的股票组合。当前提交版本采用“XGBoost Ranker + 可见放量信号校正 + Top1 集中持仓”的离线可复现方案。

当前默认提交结果为：

```csv
stock_id,weight
601868,1.0
```

本地离线评分结果为：

- 当前分数：`0.288135593220`
- baseline 分数：`0.025179491217`
- 相对 baseline 提升：`+0.262956102003`
- 是否超过 baseline：`true`

本项目不依赖联网推理，不使用线上阶段未来数据参与训练或推理。`data/test.csv` 仅用于本地评分验证。

## 二、环境配置

Docker 镜像基于 `python:3.11-slim` 构建，项目依赖通过 `uv sync --frozen --no-dev` 固化安装。

主要 Python 依赖包括：

- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `joblib`

运行阶段设置了固定随机种子和单线程相关环境变量，降低不同机器上的非确定性影响。容器默认入口为 `/app/run.sh`，会依次执行初始化、训练和推理。

## 三、数据说明

项目仅使用仓库内数据文件：

- `data/train.csv`：历史可见行情数据，用于训练和正式推理。
- `data/test.csv`：未来一周真实行情数据，仅用于本地离线评分，不参与训练和推理。

推理时，`test.py` 默认读取 `data/train.csv`，对每只股票保留最后一个可见交易日的特征，生成 `output/result.csv`。

## 四、预训练模型和外部资源

本项目未使用外部预训练模型、外部 embedding 或额外开源数据。所有模型均由 `train.py` 在本地离线数据上训练得到。

## 五、算法方法

### 5.1 整体思路

方案将股票选择建模为按交易日分组的排序问题。训练阶段基于历史数据构造 T+1 到 T+5 的未来收益标签，使用排序模型学习每个交易日内部股票的相对强弱；推理阶段只使用最新可见历史数据生成候选股票分数，并输出分数最高的股票。

### 5.2 标签构造

每个样本的收益定义为：

```text
future_return = open(T+5) / open(T+1) - 1
```

同一交易日内按 `future_return` 降序排序，并分桶为 0 到 9 的 relevance label。训练 group 为交易日。

### 5.3 特征工程

默认特征预设为 `alpha_v1`，包含约 81 个特征，主要包括：

- 日内形态：日内收益、上下影线、收盘位置、振幅等。
- 趋势动量：3/5/10/20 日收益、均线偏离、突破和反弹位置。
- 成交量与成交额：成交量变化率、成交量放量比、成交额放量比、对数成交量。
- 价量关系：开盘价/收盘价/VWAP 与成交量的滚动相关性。
- 横截面排序：按交易日计算的 percentile rank 特征。
- 换手率特征：换手率及多窗口换手率放量比。

所有特征均由 `data/train.csv` 的历史可见字段计算得到。

### 5.4 模型结构

生产模型为 XGBoost Ranker：

- 模型名称：`xgb_ranker`
- objective：`rank:ndcg`
- eval metric：`ndcg@5`
- n_estimators：`600`
- learning_rate：`0.03`
- max_depth：`5`
- tree_method：`hist`

训练脚本同时训练 LightGBM Ranker 和 HistGradientBoostingRegressor 作为诊断模型，用于验证和报告分析；正式生产推理默认只使用 `xgb_ranker`。

### 5.5 数据增强

训练阶段对部分特征列加入轻量高斯噪声，用于提升模型对短期扰动的鲁棒性：

```text
augmentation_noise_std = 0.01
augmentation_noise_fraction = 0.3
```

随机种子固定为 `42`。

### 5.6 生产分数校正

最终生产分数由模型分数和最新可见放量信号共同决定：

```text
final_score = zscore(model_score) + 0.7 * zscore(volume_ratio_20)
```

其中 `volume_ratio_20` 是最后一个可见交易日的 20 日成交量放量比。该信号用于增强短期资金关注和放量突破特征。校正配置会写入 `model/xgb_ranker_v3/model_metadata.json`，保证训练后推理可复现。

### 5.7 组合生成

当前生产持仓数量为 `1`，即输出预测分数最高的一只股票，权重为 `1.0`。输出满足比赛要求：股票数量不超过 5，权重和不超过 1。

## 六、训练流程

训练命令：

```bash
python code/src/train.py
```

训练流程如下：

1. 读取 `data/train.csv`。
2. 标准化字段名、股票代码和日期格式。
3. 计算技术特征和横截面特征。
4. 构造 T+1 到 T+5 的未来收益标签。
5. 按交易日构建 ranking group。
6. 使用 walk-forward 方式做诊断验证。
7. 训练生产模型 `xgb_ranker`。
8. 保存模型、特征重要性、训练报告和模型元数据到 `model/xgb_ranker_v3/`。

训练时间实测约 5 到 6 分钟，低于 8 小时限制。

## 七、推理流程

推理命令：

```bash
python code/src/test.py
```

推理流程如下：

1. 读取 `model/xgb_ranker_v3/model_metadata.json`。
2. 加载生产模型 `xgb_ranker.json`。
3. 读取 `data/train.csv` 作为推理输入。
4. 重新计算与训练一致的特征。
5. 每只股票保留最后一个可见交易日样本。
6. 计算模型分数并应用生产分数校正。
7. 输出 Top1 股票到 `output/result.csv`。
8. 写出 `temp/submission_check.json` 作为格式检查报告。

预测时间实测为秒级，低于 5 分钟限制。

## 八、复现与 Docker 提交

构建镜像：

```bash
docker build -t bdc2026:latest .
```

运行完整复现：

```bash
docker compose up --force-recreate
```

导出镜像：

```bash
docker save -o 霹雳.tar bdc2026:latest
```

`run.sh` 默认设置 `REPRODUCE_FROM_TRAIN=1`，容器启动后会从训练流程开始复现，再执行推理，确保生成结果与提交的 `result.csv` 一致。

## 九、目录结构

项目按比赛规范组织，关键文件如下：

```text
app/
|-- code/
|   |-- src/
|       |-- featurework.py
|       |-- train.py
|       |-- test.py
|-- data/
|   |-- train.csv
|   |-- test.csv
|-- model/
|-- output/
|   |-- result.csv
|-- temp/
|-- init.sh
|-- train.sh
|-- test.sh
|-- run.sh
|-- readme.md
```

## 十、本地验证结果

当前 `output/result.csv`：

```csv
stock_id,weight
601868,1.0
```

本地评分：

- 当前分数：`0.288135593220`
- baseline 分数：`0.025179491217`
- 差值：`+0.262956102003`
- rank_eligible：`true`

在当前本地 `data/test.csv` 和非负权重约束下，单票 `601868` 的收益 `0.288135593220` 是理论最高分，因此本地分数无法超过 `0.3`。

## 十一、其他注意事项

- 训练和推理阶段不联网。
- `data/test.csv` 只用于本地评分，不参与模型训练或正式推理。
- 结果文件必须为 UTF-8 编码，表头为 `stock_id,weight`。
- 当前 Docker tar 文件名为 `霹雳.tar`，镜像名为 `bdc2026:latest`。
- 若线上 result 阶段仅要求上传结果文件，应优先提交 `output/result.csv`；若进入 docker 审核阶段，再提交 `霹雳.tar`。
