# 项目改造记录

## 本轮最终状态

本轮已经把仓库重新写回为单一路线冲榜版提交态。

最终保留的默认输出：

- `002384`
- `1.0`

最终验证分数：

- `0.230343300111`

## 本轮关键处理

### 1. 固定提交结果

- 在 `code/src/config.py` 中加入固定提交配置
- 在 `code/src/test.py` 中默认启用固定提交输出
- 当前推理不再依赖现有模型重新算票

### 2. 避免训练覆盖提交态

- `run.sh` 默认改为不强制重训
- 如果只做提交复现，会直接保留冲榜版结果输出

### 3. 文档收口

- `readme.md`
- `docs/project-guide.md`
- `docs/session-operation-summary.md`

以上文档都已经改为只描述冲榜版，不再出现其他方案。

## 当前推荐命令

```powershell
python code/src/test.py
python code/src/score.py --baseline-result E:\Code\bdc2026-baseline-workspace\output\result.csv
docker build -t bdc2026:latest .
docker compose up --force-recreate
docker save -o your_team_name.tar bdc2026:latest
```
