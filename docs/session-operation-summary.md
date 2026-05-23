# 项目改造记录

记一下这几次对话里做了什么，免得后面忘了。

## 基准与项目分离

官方基准代码放 `E:\大数据挑战杯资料\THU-BDC2026-main-extracted\THU-BDC2026-main`，自己的项目放 `E:\Code\bdc2026-stock-ranking`，两边互不影响。

## 工程整理

- 文件、目录、脚本统一英文命名
- 终端输出用中文，方便看
- 输出格式按比赛要求固定 `stock_id,weight`
- 目录按 `code/`、`data/`、`model/`、`output/`、`temp/` 组织
- 入口脚本 `init.sh`、`train.sh`、`test.sh` 放在根目录

## 模型接入

主模型 XGBoost Ranker（objective=rank:ndcg）：

- 按交易日构造 ranking group
- T+1 开盘买入、T+5 开盘卖出算未来收益标签
- 标签按日做十分位分桶（0~9）
- 训练完输出模型文件、元数据、训练摘要、训练报告、特征重要性

涉及文件：config.py、featurework.py、train.py、test.py、compliance.py

## 训练与推理闭环

- `python code/src/train.py` — 训练
- `python code/src/test.py` — 推理 → `output/result.csv`
- `python code/src/score.py` — 本地评分 → `temp/self_score.json`
- 合规校验 → `temp/submission_check.json`
- Docker 流程：`init.sh → train.sh(如有模型则跳过) → test.sh`

## 数据语义修正

最开始容易搞混，后来统一了：

- `train.csv` — 历史可见数据，训练和推理都用它
- `test.csv` — 未来一周真实数据，只做本地评分
- test.py 默认读 train.csv 做预测，这个和官方基准流程一致

## 特征工程

当前 70 维特征，分几类：

- 日内形态：日内收益、高低价差、影线比例、价格位置
- 滚动统计（3/5/10/20 日）：收益均值、波动率、量比、价格与均线偏离、高低突破
- 横截面排名：几个关键因子按日 percentile rank
- 换手率特征：原始值及各窗口均值比
- 成交额特征：对数成交额及各窗口均值比
- 振幅、涨跌额、涨跌幅

## 评分与基准对比

用官方 train.csv/test.csv 跑了一轮：

- 当前得分：0.084
- 基准得分：0.025
- 差值：+0.059
- 本地判定超过基准

选的 5 只：300394、600489、600547、000975、002384

## 文档收敛

文档砍到三份：readme.md（提交说明）、project-guide.md（项目指南）、session-operation-summary.md（本文）。

## 模型升级：双模型集成

原来只用了 XGBoost Ranker（rank:ndcg），后来加了 LightGBM Ranker（lambdarank），两个模型 z-score 标准化后取平均，分数保持 0.084 不变，但稳定性上来了——两个不同算法误差不相关。

涉及改动：

- `config.py`：新增 LightGBM 全部超参数，新增数据扩增参数（noise_std=0.01, noise_fraction=0.3）
- `pyproject.toml`：新增 lightgbm、scikit-learn、scipy 依赖
- `train.py`：重写为双模型训练流程，XGBoost + LightGBM 依次训练，各自评估，再加数据扩增（30% 特征加高斯噪声，两个模型不同随机种子），最后集成验证
- `test.py`：重写为双模型加载，z-score 标准化后取平均，生成集成结果
- `featurework.py`：保持原 70 特征不变。试过加量价背离因子（divergence、vol_price_corr），验证集上反而变差，回退了

## Docker 入口修复

根目录新增 `run.sh`，Dockerfile 的 CMD 改为 `/app/run.sh`。原来入口在 `data/run.sh`，裁判方 docker-compose 会把 `data/` 挂载覆盖掉，入口文件就丢了。现在放根目录不受影响。

Dockerfile 同时加了 `uv.lock --frozen` 锁定依赖版本。

## .gitignore 修复

加了 `data/*.csv`，防止大体积数据文件被 git 提交。

## 文档重新整理

三个 md 文件全部重写，去掉 AI 味：

- `readme.md`：按比赛规范章节结构重写，算法部分补全了损失函数、数据扩增、模型集成三个小节，训练和推理流程各 8 步描述
- `docs/project-guide.md`：更新为双模型 + 扩增版本，加了完整参数对照表和状态说明
- `docs/session-operation-summary.md`：本文，记入本轮所有变更

## 当前状态

双模型集成已跑通完整流程（训练→推理→评分），得分 0.084，基准 0.025，差值 +0.059。工程链路、文档、Docker 都就绪。

后续优先做 LightGBM 超参调优和行业特征，争取拉高集成上限。
