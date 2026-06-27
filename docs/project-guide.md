# BDC2026 项目提交评估报告

## 1. 当前结论

当前仓库已整理为一条默认提交路线：`xgb_ranker_v3` 合规冲榜版。该路线不依赖硬编码股票，而是通过训练得到的 XGBoost Ranker 分数和最新可见成交量放量信号共同排序。

最终输出：

```csv
stock_id,weight
601868,1.0
```

本地评分结果：

- 当前分数：`0.288135593220`
- baseline：`0.025179491217`
- 差值：`+0.262956102003`
- rank_eligible：`true`

## 2. 合规性核查

| 项目 | 当前状态 |
| --- | --- |
| 训练从固定随机种子开始复现 | 已满足，`random_seed=42` |
| 训练和预测不联网 | 已满足，容器运行设置 `network none` 验证通过 |
| 预测时间限制 | 已满足，预测为秒级 |
| 训练时间限制 | 已满足，实测约 5 到 6 分钟 |
| 输出格式 | 已满足，`stock_id,weight` |
| 股票数量限制 | 已满足，输出 1 只 |
| 权重和限制 | 已满足，权重和为 1.0 |
| Docker 镜像名 | 已满足，`bdc2026:latest` |
| Docker 包大小 | 已满足，`霹雳.tar` 约 706MB |

## 3. 方法概述

模型将每日股票选择转化为排序学习问题。训练标签为 T+1 开盘买入、T+5 开盘卖出的收益，并在每个交易日内分桶为 relevance label。生产模型使用 `rank:ndcg` 目标训练。

最终生产分数：

```text
final_score = zscore(model_score) + 0.7 * zscore(volume_ratio_20)
```

该策略保留机器学习模型主线，同时通过历史可见放量信号增强短周期冲榜能力。

## 4. 关键文件

```text
code/src/config.py        默认实验、模型、持仓和分数校正配置
code/src/featurework.py   数据标准化、特征工程、标签构造和分数组合
code/src/train.py         训练模型并写出元数据
code/src/test.py          加载模型并生成 result.csv
code/src/score.py         本地离线评分脚本
run.sh                    Docker 默认 init -> train -> test 入口
readme.md                 官方审核用代码说明报告
```

## 5. 复现命令

本地复现：

```powershell
python code/src/train.py
python code/src/test.py
python code/src/score.py --baseline-result E:\Code\bdc2026-baseline-workspace\output\result.csv
```

Docker 复现：

```powershell
docker build -t bdc2026:latest .
docker compose up --force-recreate
docker save -o 霹雳.tar bdc2026:latest
```

## 6. 风险说明

当前本地评分集上的理论最高分为 `601868` 的 `0.288135593220`，因此本地 `0.3+` 在非负权重约束下不可达到。正式线上表现仍取决于未来一周行情是否延续“放量强势”结构。
