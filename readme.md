# BDC2026 股票排序代码说明报告

## 一、项目摘要

本项目面向 BDC2026 股票排序任务，按官方口径预测从 T+1 开盘到 T+5 开盘的一周收益。当前生产方案不再使用旧版“Top1 满仓 + 手工放量叠加”，而是采用：

- XGBoost Ranker、LightGBM Ranker、HistGradientBoostingRegressor 三模型；
- 横截面 percentile rank 后固定等权集成；
- 输出 5 只股票，每只权重 `0.2`；
- 5 个交易日 purge 的嵌套 walk-forward 验证；
- 非重叠周、官方等权 Top5 收益口径评估。

截至 `2026-07-24` 的当前结果为：

```csv
stock_id,weight
002837,0.2
300408,0.2
600549,0.2
603986,0.2
688012,0.2
```

对应股票为英维克、三环集团、厦门钨业、兆易创新、中微公司。股票预测不能保证未来收益；本次优化目标是提升样本外平均收益，同时显著降低单票满仓的周度尾部风险。

## 二、数据说明

采集链路与官方基准保持一致：

- 数据源：Baostock；
- 股票池：采集时点的沪深 300 成分股；
- 频率：日线；
- 复权：`adjustflag="1"`；
- 行情字段：开高低收、成交量、成交额、换手率、涨跌幅等。

当前开发快照：

- 时间范围：`2024-01-02` 至 `2026-07-24`；
- 行数：`184626`；
- 股票数：`300`；
- 最新交易日覆盖：`300/300`；
- `(stock_id, trade_date)` 重复键：`0`。

`data/train.csv` 用于训练和正式推理。`data/test.csv` 仅保留为旧阶段本地检查数据，不参与训练、模型选择或正式推理。

重新采集时执行：

```bash
python get_stock_data.py --start-date 2024-01-01 --end-date YYYY-MM-DD \
  --adjustflag 1 --output temp/refresh_stock_data.csv \
  --stock-list-output temp/refresh_hs300_stock_list.csv
```

采集脚本未传 `--end-date` 时默认请求运行当天，Baostock 会返回最近可用交易日。正式 B 阶段提交前应在目标截点收盘后重新下载、校验并训练；当前计划截点为 `2026-07-31` 收盘后。

## 三、标签与特征

收益标签严格按官方持有期定义：

```text
future_return = open(T+5) / open(T+1) - 1
```

同一交易日内按未来收益构造排序相关度标签。默认 `alpha_v1` 预设包含 81 个仅由历史可见数据计算的特征，主要包括：

- 日内形态、振幅、上下影线和收盘位置；
- 3/5/10/20 日趋势、反转、突破和均线偏离；
- 成交量、成交额、换手率及其滚动变化；
- 价格/VWAP 与成交量的滚动关系；
- 按交易日计算的横截面 percentile rank。

旧版推理阶段的以下手工叠加已完全移除：

```text
zscore(model_score) + 0.7 * zscore(volume_ratio_20)
```

原因是该叠加没有进入训练与嵌套验证，容易放大单周偶然性。

## 四、模型与组合

生产模型包括：

- `xgb_ranker`：`rank:ndcg`；
- `lgb_ranker`：LambdaRank；
- `hgb_regressor`：连续未来收益回归诊断模型。

三个模型的原始分数先转换成当日股票池内的 percentile rank，再固定各占 `1/3`。自动收缩权重和风险惩罚仍会写入训练报告作为诊断，但不控制生产结果。原因是固定等权集成在外层走步验证中优于折内调权组合，且能避免 Windows/Linux 浮点差异被调权放大。

最终从模型 Top15 候选池中选择 Top5。当前验证不支持额外方差或相关性惩罚，因此生产参数为 0，结果等价于集成分数最高的 5 只股票；每只权重固定为 `0.2`。

## 五、无泄漏验证

验证采用 4 个外层 walk-forward 折，每个外层折包含 3 个内层折，并在训练集与验证集之间 purge 5 个交易日。周度评价使用非重叠的 5 交易日窗口。

Docker/Linux 的四折汇总如下（收益均为小数）：

| 策略 | 四折周均收益 | 四折平均正收益率 | 全部折最差周 | 最新折周均收益 |
|---|---:|---:|---:|---:|
| 旧 Top1 + 放量叠加 | -0.010552 | 0.442308 | -0.193725 | -0.039064 |
| XGBoost Top5 等权 | 0.014023 | 0.576923 | -0.103674 | 0.006556 |
| 三模型固定等权 Top5 | 0.010663 | 0.557692 | -0.085794 | 0.006803 |
| 折内调权/风险组合 | 0.009034 | 0.500000 | -0.099106 | -0.000026 |

三模型固定等权 Top5 的平均收益略低于单独 XGBoost Top5，但最差周更小，且最新折表现更好，因此作为稳健生产方案。旧 Top1 在四折平均、最新折和最差周上均明显更差。

## 六、训练与推理

本地训练：

```bash
python code/src/train.py \
  --experiment-name xgb_ranker_v3 \
  --feature-preset alpha_v1 \
  --production-models xgb_ranker,lgb_ranker,hgb_regressor \
  --production-portfolio-size 5
```

本地推理：

```bash
python code/src/test.py --experiment-name xgb_ranker_v3
```

实测结果：

- Windows 全量训练：`3137.658` 秒；
- Docker/Linux 断网全流程：`1814.127` 秒；
- Windows 单次推理：约 `12.2` 秒；
- 测试：`17 passed`；
- 结果 SHA-256：`55e08d8d8bb5b3e4b52f677ba43b14f3d567cd6a3aec98f49cddb8aac5906b21`。

推理显式使用 LF 行尾，因此 Windows 与 Linux 的 `result.csv` 字节哈希一致。

## 七、Docker 复现

构建镜像：

```bash
docker build -t bdc2026:latest .
```

断网完整运行：

```bash
docker run --name bdc2026-verify --network none bdc2026:latest
```

也可使用仓库内 Compose：

```bash
docker compose up --force-recreate
```

`run.sh` 默认设置 `REPRODUCE_FROM_TRAIN=1`，会依次执行初始化、完整训练和推理。镜像内置离线训练数据，运行阶段不需要网络。

当前镜像：

- 名称：`bdc2026:latest`；
- 大小：`712648708` 字节；
- 运行网络：`none`；
- 完整流程：成功；
- 输出行数：`5`；
- 权重和：`1.0`。

导出镜像：

```bash
docker save -o 霹雳.tar bdc2026:latest
```

## 八、关键目录

```text
code/src/featurework.py   特征与标签
code/src/train.py         嵌套验证、模型训练、报告输出
code/src/backtest.py      非重叠周和官方收益口径
code/src/portfolio.py     秩集成与 Top5 组合
code/src/test.py          离线推理和结果校验
data/train.csv            历史可见训练数据
model/xgb_ranker_v3/      三模型与训练报告
output/result.csv         正式提交结果
temp/                     诊断报告和本地临时文件
```

## 九、提交注意事项

- 结果文件表头必须是 `stock_id,weight`；
- 当前结果恰好 5 行、股票代码唯一、每只权重 0.2；
- 训练和推理只使用历史可见数据，不读取 `data/test.csv`；
- 最终提交前必须按比赛目标周重新确认数据截点；
- 线上结果阶段上传 `output/result.csv`，Docker 审核阶段再导出并上传镜像 tar。

## 十、事务式最终发布

最终数据刷新不建议手工拼接多个命令。使用发布器时，所有候选数据、模型、结果和 Docker tar 都先在 `temp/submission-release/<run-id>/` 中生成；只有数据、回测、格式、断网 Docker、哈希和 tar 门槛全部通过，才替换正式文件。

```powershell
.\.venv\Scripts\python.exe prepare_submission.py `
  --cutoff-date 2026-07-31 `
  --experiment-name xgb_ranker_v3 `
  --docker-image bdc2026:latest `
  --tar-name 霹雳.tar `
  --push
```

发布器的失败语义：

- 下载、校验、训练或 Docker 失败：正式数据、模型、结果和 tar 不变；
- 发布中途失败：从 run 目录备份逆序恢复，并校验原哈希；
- Git 推送失败：保留已验证的本地提交，标记未推送状态并返回非零退出码；
- 成功后写入 `docs/validation/latest-submission.json`，记录截止日、回测、股票、哈希、耗时、Docker image 和远端 SHA。

正式运行必须显式指定截止日，且必须在目标交易日数据已由 Baostock 更新后执行。当前已验证的 2026-07-24 结果不能替代 2026-07-31 收盘后的最终刷新。
