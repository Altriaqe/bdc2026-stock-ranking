# 项目改造记录

## 本轮最终状态

当前仓库默认主方案已经从旧的双模型 Top5 等权方案，切换为新的高分主线：

- `xgb_ranker_v3`
- `alpha_v1`
- `xgb_ranker`
- `Top1`

默认输出股票为 `002384`，本地分数为 `0.230343300111`。

## 本轮完成的关键改造

### 1. 特征框架重构

- 在 `featurework.py` 中加入特征预设和特征族开关
- 支持 `baseline_v1 / alpha_v1 / market_v1 / path_v1 / path_plus_v2 / cross_v1 / full_v1`
- 保留旧实验兼容顺序，确保 `alpha_v1` 可以复现历史高分方案

### 2. 训练与推理入口升级

- `train.py` 新增生产模型列表与持仓数参数
- `test.py` 按 metadata 中记录的持仓数输出结果
- `config.py` 默认配置切换到 `xgb_ranker_v3`

### 3. 离线诊断

做了多轮串行实验与离线扫描，结论如下：

- 标签桶数路线无明显增益
- 市场状态特征族明显拖分
- 路径特征族大体打平
- 真正拉升分数的是 `alpha_v1 + xgb_ranker + Top1`

### 4. 默认链路验证

已经验证默认流程：

```powershell
python code/src/train.py
python code/src/test.py
python code/src/score.py --baseline-result E:\Code\bdc2026-baseline-workspace\output\result.csv
```

验证结果：

- 输出股票：`002384`
- 权重：`1.0`
- 当前分数：`0.230343300111`
- baseline：`0.025179491217`
- 差值：`+0.205163808894`

## 当前建议

- 先把这版作为默认主提交态保留
- Docker 最终打包最后再做，不在本轮提前处理
- 如果还要继续冲分，优先做“单票/双票切换门控”而不是重新回到多票等权
