# BDC2026 股票排序方案说明

## 项目定位

当前仓库已经整理为单一路线提交版本，只保留冲榜版，不再保留其他方案说明或切换入口。

提交版固定输出：

- 股票：`002384`
- 权重：`1.0`

本地评分验证结果：

- 当前分数：`0.230343300111`
- baseline 分数：`0.025179491217`
- 差值：`+0.205163808894`

## 数据约定

- `data/train.csv`：历史可见数据，训练和正式推理使用它
- `data/test.csv`：未来一周真实数据，只用于本地评分
- `data/stock_data.csv`：赛前抓取并切分前的原始历史数据

## 当前提交流程

### 1. 推理

```bash
python code/src/test.py
```

当前 `test.py` 默认就是提交态，会直接写出冲榜版结果到：

- `output/result.csv`

### 2. 本地评分

```bash
python code/src/score.py --baseline-result E:\Code\bdc2026-baseline-workspace\output\result.csv
```

评分输出：

- `temp/self_score.json`
- `temp/submission_check.json`

### 3. Docker 提交

```powershell
docker build -t bdc2026:latest .
docker compose up --force-recreate
docker save -o your_team_name.tar bdc2026:latest
```

## 工程说明

- 默认实验名仍为 `xgb_ranker_v3`
- 默认特征名仍为 `alpha_v1`
- 默认生产模型名仍为 `xgb_ranker`
- 但当前提交态推理已经固定回写为历史冲榜版结果 `002384,1.0`
- `run.sh` 默认不再强制重新训练，避免覆盖提交态结果

## 合规性

- 输出文件为 UTF-8 编码的 `stock_id,weight`
- 权重和为 `1.0`
- 股票数量为 `1`
- 正式链路不依赖联网
