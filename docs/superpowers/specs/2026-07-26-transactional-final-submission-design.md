# B 阶段事务式最终发布流程设计

## Summary

当前模型、Top5 组合和 Docker 复现已完成截至 2026-07-24 的开发验证，但最终提交仍依赖 2026-07-31 收盘后的数据刷新。现有流程需要人工依次下载、校验、覆盖数据、训练、推理、构建 Docker、导出 tar、提交 Git；其中任何一步失误都可能覆盖有效产物或把未经验证的结果推送到远端。

本设计增加一个跨平台的 Python 事务式发布器。所有候选数据、模型、结果、Docker 镜像和 tar 先在隔离的发布目录中生成并验证；只有全部门槛通过，才发布到正式路径并提交、推送 GitHub。

## Goals

- 用一个明确命令完成最终数据刷新到 GitHub 推送的完整流程。
- 保持官方基准的数据源、字段、日频和 `adjustflag=1` 口径。
- 数据、训练、回测、推理、Docker、tar 和 Git 任一前置门槛失败时，不覆盖当前正式产物。
- 以断网 Docker 完整重训的结果作为正式 `output/result.csv`。
- 生成可机器读取的发布清单，证明数据截点、回测、格式、耗时、哈希和远端提交状态。
- 保留当前稳健策略：三模型固定等权秩集成，5 只股票各 0.2。

## Non-goals

- 不在发布器中改变模型结构、特征或超参数。
- 不引入调度服务、网页界面、数据库或云端训练。
- 不承诺未来收益为正；发布门槛只能证明流程正确、验证无泄漏且历史样本外表现符合要求。
- 不自动猜测官方数据截止日；调用者必须显式传入 `--cutoff-date YYYY-MM-DD`。

## Chosen approach

采用 Python 事务式发布器，而不是手工清单或 PowerShell 串联。

Python 方案跨 Windows 和 Docker/Linux，能够把数据校验、文件哈希、子进程、回滚和单元测试放在同一套接口中。发布器不直接复用业务代码的隐式全局路径；候选产物先写到唯一的 run 目录，正式目录只在发布阶段发生变化。

## Command interface

入口命令：

```bash
python prepare_submission.py \
  --cutoff-date 2026-07-31 \
  --experiment-name xgb_ranker_v3 \
  --docker-image bdc2026:latest \
  --tar-name 霹雳.tar \
  --push
```

必要参数和规则：

- `--cutoff-date` 必填，必须是有效 ISO 日期，且不能晚于运行当天。
- `--experiment-name` 默认 `xgb_ranker_v3`。
- `--docker-image` 默认 `bdc2026:latest`。
- `--tar-name` 默认 `霹雳.tar`，最终路径必须位于工作区根目录。
- `--push` 表示全部门槛通过后提交并推送 `origin/main`；最终正式运行必须带此参数。
- `--keep-run-dir` 仅用于诊断，失败时保留候选发布目录；默认成功后保留清单和日志、清理大型候选副本。

## Architecture

### 1. `prepare_submission.py`

薄入口层，负责参数解析、工作区确认和退出码。它调用 `submission_release.run_release()`，不包含数据或文件操作细节。

### 2. `code/src/submission_release.py`

事务编排器，定义：

```python
@dataclass(frozen=True)
class ReleaseConfig:
    workspace: Path
    cutoff_date: date
    experiment_name: str
    docker_image: str
    tar_name: str
    push: bool
    keep_run_dir: bool

@dataclass(frozen=True)
class ReleaseResult:
    run_id: str
    manifest_path: Path
    result_path: Path
    result_sha256: str
    tar_path: Path
    tar_sha256: str
    commit_sha: str | None
    remote_sha: str | None
```

公开入口为 `run_release(config: ReleaseConfig) -> ReleaseResult`。编排器只通过下列专用模块访问数据校验、发布和命令执行，避免一个超大脚本同时承担所有职责。

### 3. `code/src/release_validation.py`

提供纯函数和小型文件读取函数：

接口为：

- `validate_market_data(path: Path, cutoff_date: date) -> MarketDataReport`
- `validate_result(path: Path) -> SubmissionReport`
- `sha256_file(path: Path) -> str`
- `validate_tar(path: Path, max_bytes: int = 10_000_000_000) -> TarReport`

`MarketDataReport` 至少包含行数、股票数、最早/最晚日期、最新日覆盖数、重复键数、非法 OHLC 行数、关键价格缺失数、停牌字段缺失数和数据文件哈希。

### 4. `code/src/release_publish.py`

负责正式路径的备份、替换和回滚：

```python
@dataclass(frozen=True)
class PublishItem:
    candidate: Path
    destination: Path
```

发布接口为 `publish_with_rollback(items: Sequence[PublishItem], backup_dir: Path) -> None`。所有目标路径必须解析到工作区内。函数先把现有目标复制或移动到 run 目录的 `backup/`，再逐项使用同卷临时文件和 `os.replace()` 发布。任一替换失败，按相反顺序恢复备份。大型目录采用先准备同级候选目录、再重命名的方式；不对工作区根目录执行递归删除。

## Isolated release directory

每次运行创建：

```text
temp/submission-release/<UTC timestamp>-<random suffix>/
├── candidate/
│   ├── data/
│   ├── model/xgb_ranker_v3/
│   ├── output/result.csv
│   ├── temp/
│   └── 霹雳.tar
├── docker-context/
├── docker-artifacts/
├── backup/
├── logs/
└── manifest.json
```

run 目录必须位于工作区内，并通过解析后的绝对路径检查。正式 `data/`、`model/`、`output/` 和 tar 在发布前保持不变。

## Data flow

1. 检查工作区、Git 分支、远端和初始状态。允许由发布器更新的跟踪文件必须干净；发现其他未提交修改立即停止。
2. 调用 `get_stock_data.py`，把行情和成分股名单写入 `candidate/data/`。
3. 校验候选行情；失败则停止，不进入训练。
4. 将完整可见行情同时作为候选 `stock_data.csv` 和 `train.csv`。旧 `test.csv` 仅为兼容 Docker 目录结构，从正式数据目录复制，但不参与训练或推理。
5. 使用 `--train-data` 指向候选训练集，并使用唯一候选实验名训练三个模型。
6. 推理脚本增加显式 `--output-path` 和 `--report-dir`，候选推理不得写正式 `output/` 或 `temp/`。
7. 校验候选本地结果格式，并记录本地预测作为诊断；它不是最终正式结果。
8. 生成独立 Docker 构建上下文，其中包含当前源码、锁文件、候选数据和候选模型。
9. 构建候选镜像，用唯一标签避免覆盖已发布镜像。
10. 使用 `--network none` 在候选容器内完整重训和推理，记录总耗时，要求小于 8 小时。
11. 从容器复制模型、回测报告、训练报告、结果和推理报告到 `docker-artifacts/`。
12. 在同一容器模型上再运行一次断网推理；两个 Docker 结果必须字节哈希一致，单次推理必须小于 5 分钟。
13. 校验 Docker 正式结果，覆盖候选本地结果；Docker 输出是唯一正式提交来源。
14. 从最终候选镜像导出 tar 到 candidate，验证可列出内容且小于 10 GB。
15. 写出完整 `manifest.json`，状态先为 `validated`。
16. 发布候选数据、Docker 模型、Docker 结果、验证清单和 tar；任何文件操作异常触发回滚。
17. 对正式路径重新计算哈希并与清单比对。
18. 更新跟踪的验证清单和 README 当前快照，运行测试与 `git diff --check`。
19. 提交源码管理中的结果、清单和文档；数据、模型和 tar 继续遵循 `.gitignore`。
20. 若启用 `--push`，推送 `origin/main` 并用 `git ls-remote` 验证远端 SHA 等于本地 HEAD。

## Validation gates

### Data gates

- 成分股名单恰好 300 个唯一股票代码。
- 行情股票数恰好 300。
- 最大 `trade_date` 必须等于 `--cutoff-date`；如目标日非交易日，由调用者改用明确的最近交易日，不允许发布器静默回退。
- 最新交易日覆盖 300 只股票。
- `(stock_id, trade_date)` 重复数为 0。
- `stock_id` 必须是 6 位数字。
- `open`、`close`、`high`、`low` 和日期不得缺失。
- 开收盘价必须大于 0，且 `high >= max(open, close, low)`、`low <= min(open, close, high)`。
- 成交量、成交额、换手率和涨跌幅允许停牌行缺失，但必须报告数量。
- 每只股票至少 60 个历史交易日，否则停止。

### Model and backtest gates

- 训练退出码为 0，三个模型文件、元数据和回测报告均存在且可解析。
- 生产配置必须是 `equal_rank_average`，三个权重均为 `1/3`，组合大小为 5，股票权重为 0.2。
- 回测必须包含 4 个 purged outer folds，每折 purge 为 5 个交易日。
- 固定等权三模型 Top5 的四折周均收益必须大于 0。
- 最新外层折周均收益必须大于 0。
- 固定等权三模型 Top5 的全部折最差周必须优于旧 Top1 的全部折最差周。
- 如任何收益门槛失败，流程停止并保留当前已发布模型；不为了通过门槛临时改策略。

### Submission gates

- 结果恰好 5 行、5 个唯一股票代码。
- 所有权重严格为 0.2，权重和为 1.0。
- 股票均在候选沪深 300 名单内，且在截止日有行情行。
- UTF-8、LF 行尾，表头严格为 `stock_id,weight`。

### Runtime and Docker gates

- Docker 构建成功。
- 完整运行显式使用 `--network none`，退出码为 0。
- 完整训练和推理总耗时小于 8 小时。
- 第二次断网推理小于 5 分钟。
- 两次 Docker 推理结果 SHA-256 完全一致。
- Docker 模型元数据和结果中的股票、权重与容器日志一致。
- 导出 tar 可读取且小于 10,000,000,000 字节。

### Git gates

- 发布开始时当前分支为 `main`，`origin` 存在。
- 除发布器允许生成的跟踪文件外，工作树必须干净。
- 提交前测试全部通过，`git diff --check` 为 0。
- 推送后 `origin/main` SHA 必须等于本地 HEAD；否则清单状态为 `committed_not_pushed`，报告可重试命令，不声称远端完成。

## Manifest

跟踪文件 `docs/validation/latest-submission.json` 保存：

- schema 版本、run id、开始/结束时间和截止日；
- 数据来源、复权标记、数据统计和 SHA-256；
- Git 起始 SHA、最终本地 SHA、最终远端 SHA；
- 模型方法、权重、回测四折汇总和门槛判定；
- 最终 5 只股票和权重；
- 本地诊断结果哈希、两次 Docker 结果哈希；
- 训练、推理、Docker build/run 耗时；
- Docker image id、image size、tar size 和 tar SHA-256；
- 发布状态：`validated`、`published`、`committed_not_pushed` 或 `pushed`；
- 所有日志的相对路径。

清单不得包含登录凭证、环境变量值或 Git SSH 私钥信息。

## Error handling and recovery

- 每个外部命令同时记录 stdout、stderr、退出码、开始时间和耗时。
- 数据下载失败或不足 300 只时不生成正式文件。
- 训练或 Docker 超时后终止其精确子进程/容器，不按模糊名称批量清理。
- 候选 Docker 容器和镜像使用 run id 命名；清理只针对这些精确名称。
- 发布前失败无需回滚，因为正式路径从未修改。
- 发布中失败由 `publish_with_rollback()` 恢复所有已替换目标，并校验备份哈希。
- 发布成功但 Git 提交失败时，保留已经验证的正式产物和备份，清单状态保持 `published`，输出恢复或重试说明。
- Git 推送失败不回滚有效本地发布；清单标记 `committed_not_pushed`，下一次仅重试推送和远端 SHA 校验。

## Testing strategy

### Unit tests

- 市场数据校验：正常、股票数不足、日期不符、重复键、OHLC 非法、短历史。
- 结果校验：正常、非 5 行、重复股票、权重错误、错误表头、CRLF。
- tar 校验：正常、小型损坏 tar、超限 tar。
- 路径安全：工作区外目标、符号链接逃逸和过宽目录拒绝。
- 发布事务：第 N 项替换故障时恢复前 N-1 项并保持原哈希。
- manifest 序列化：字段完整、稳定排序、无秘密字段。
- backtest gate：正收益通过、最新折非正失败、尾部风险未改善失败。

### Integration tests

- 使用小型合成 CSV 和假模型命令，验证完整候选流程不写正式目录。
- 注入下载、训练、推理、Docker build/run、tar 和 Git 各阶段失败，验证旧产物哈希不变。
- 成功路径使用假 Docker 命令，验证发布顺序、清单状态和 Git 命令参数。

### Real acceptance test

- 在 2026-07-31 收盘数据可用后运行正式命令。
- 检查 4 折回测门槛、两次 Docker 哈希、tar、测试、提交与远端 SHA。
- 人工只需复核清单、最终 5 股和官方平台上传文件，不需要手工拼接中间步骤。

## Documentation changes

- README 增加一键最终发布命令、失败语义和清单位置。
- `docs/validation/latest-submission.json` 作为当前发布事实来源。
- 每次成功发布额外写出不可变的 `docs/validation/YYYY-MM-DD-b-stage-validation.md`，保留历史对比。

## Open risks

- 2026-07-31 数据在 Baostock 的实际可用时间由外部服务决定；如果当晚尚未更新，严格日期门槛会停止，需要稍后重试。
- Windows 与 Linux 的树模型训练可能产生轻微浮点差异，因此正式结果固定以 Docker/Linux 为准。
- GitHub、Docker registry 元数据和 Baostock 均属于外部状态；流程能安全停止和重试，但无法保证外部服务始终可用。
- 历史回测为正不能保证目标周收益为正，最终组合仍承担市场风险。

## Next skill

设计批准并提交后，使用 `$superpower-writing-plans` 编写逐步实施计划；实施计划通过后再开始编码。
