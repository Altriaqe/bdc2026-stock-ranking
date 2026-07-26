# Transactional Final Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cross-platform transactional publisher that refreshes B-stage data, validates and trains an isolated candidate, reproduces it in offline Docker, publishes only after every gate passes, and optionally commits and pushes `origin/main`.

**Architecture:** Keep the existing downloader, trainer, portfolio, and Docker entrypoint. Add small validation and rollback modules plus a thin `prepare_submission.py` orchestrator. Every run uses `temp/submission-release/<run-id>/`; formal files change only after candidate gates pass.

**Tech Stack:** Python 3.11, pandas, pytest, pathlib, subprocess, Docker CLI, Git CLI, Baostock, existing XGBoost/LightGBM/scikit-learn stack.

## Global Constraints

- Official data remains Baostock daily HS300 with `adjustflag="1"`.
- `--cutoff-date` is required; the publisher never silently substitutes another date.
- Production remains `equal_rank_average`, three models at `1/3`, exactly 5 stocks at `0.2` each.
- Candidate and formal paths must resolve inside the workspace; no broad recursive deletion.
- Full Docker validation runs with `--network none` and retrains from bundled candidate data.
- Any failed gate leaves currently published data, models, result, tar, and remote commit unchanged.

---

### Task 1: Add pure release validators

**Files:**
- Create: `code/src/release_validation.py`
- Create: `tests/test_release_validation.py`

**Interfaces:** `validate_market_data(path: Path, cutoff_date: date) -> MarketDataReport`, `validate_result(path: Path, allowed_stock_ids: set[str] | None = None) -> SubmissionReport`, `validate_tar(path: Path, max_bytes: int = 10_000_000_000) -> TarReport`, `sha256_file(path: Path) -> str`, and `assert_backtest_gates(report: dict[str, object]) -> None`.

- [ ] Write failing tests for valid data, duplicate keys, cutoff mismatch, invalid OHLC, short history, valid/invalid five-row results, CRLF bytes, valid tar, and oversized tar.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_release_validation.py -q`; expect failures for missing functions.
- [ ] Implement frozen report dataclasses. Require exact cutoff, duplicate-free keys, positive prices, valid OHLC, six-digit codes, minimum history, exact result weights, LF bytes, tar readability and size limit. Count suspended volume/amount/turnover/pct missingness without rejecting it.
- [ ] Run the focused tests and `.\.venv\Scripts\python.exe -m pytest -q`; expect all existing and new tests to pass.
- [ ] Commit with `git add code/src/release_validation.py tests/test_release_validation.py; git commit -m "feat: add release validation gates"`.

### Task 2: Add safe publication and rollback

**Files:**
- Create: `code/src/release_publish.py`
- Create: `tests/test_release_publish.py`

**Interfaces:** `PublishItem(candidate: Path, destination: Path)`, `ensure_inside_workspace(workspace: Path, path: Path) -> Path`, `publish_with_rollback(items: Sequence[PublishItem], workspace: Path, backup_dir: Path) -> None`, and `restore_backup(backup_dir: Path, items: Sequence[PublishItem]) -> None`.

- [ ] Write tests proving successful replacement, backup creation, injected second-item failure rollback, hash restoration, and rejection of outside-workspace/symlink-escape paths.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_release_publish.py -q`; expect missing-function failures.
- [ ] Implement resolved-path checks, same-directory temporary copies, `os.replace`, reverse-order restore, and post-restore hash checks. Never delete a workspace root.
- [ ] Run focused and full tests, then commit with `git commit -m "feat: add transactional publication rollback"`.

### Task 3: Make inference outputs relocatable

**Files:**
- Modify: `code/src/test.py` argument parsing and output writes
- Create: `tests/test_inference_paths.py`

**Interface:** Add `--output-path` and `--report-dir`; defaults remain `output/result.csv` and `temp/`.

- [ ] Write a test that runs inference to temporary paths and proves formal output/report files are unchanged.
- [ ] Run the focused test and observe failure.
- [ ] Thread explicit paths through CSV, submission report, portfolio report, and result hash generation; preserve `lineterminator="\n"`.
- [ ] Run focused and full tests, then commit with `git commit -m "feat: support isolated inference outputs"`.

### Task 4: Implement the release orchestrator

**Files:**
- Create: `code/src/submission_release.py`
- Create: `prepare_submission.py`
- Create: `tests/test_submission_release.py`

**Interfaces:** `ReleaseConfig(workspace, cutoff_date, experiment_name, docker_image, tar_name, push, keep_run_dir)`, `ReleaseResult(run_id, manifest_path, result_path, result_sha256, tar_path, tar_sha256, commit_sha, remote_sha)`, and `run_release(config: ReleaseConfig) -> ReleaseResult`.

- [ ] Write seam tests with mocked commands proving data failure prevents training, Docker failure prevents publication, successful fake runs write `validated` before publication and `pushed` after Git verification, and every command records exit code/timing.
- [ ] Run focused tests and observe missing-seam failures.
- [ ] Implement a unique run directory with `candidate/`, `docker-context/`, `docker-artifacts/`, `backup/`, and `logs/`; use exact subprocess argument arrays, captured logs, timeouts, and run-id-specific container/image names.
- [ ] Stage data by invoking `get_stock_data.py` with explicit cutoff/output paths, copying full history to candidate `train.csv`, and validating with Task 1.
- [ ] Train using `train.py --train-data <candidate/train.csv> --experiment-name <candidate-name>`, then run relocatable diagnostic inference and parse the backtest gates.
- [ ] Build an isolated Docker context, run full retraining with `--network none`, copy model/report/result artifacts out, run a second offline inference, require equal result hashes, export/validate tar, and select Docker output as the formal result.
- [ ] Publish candidate data, model, result, tar, manifest and dated validation report through Task 2 only after every gate. Recompute formal hashes afterward.
- [ ] Require `main`, `origin`, and a clean starting worktree; run tests and `git diff --check`; with `--push`, commit intended tracked files, push, and verify `git ls-remote`. Push failure records `committed_not_pushed` and exits nonzero.
- [ ] Run focused and full tests, then commit with `git commit -m "feat: add transactional submission orchestrator"`.

### Task 5: Document and integrate the release command

**Files:**
- Modify: `readme.md`
- Create: `docs/validation/latest-submission.json` on successful release
- Create: `tests/fixtures/release/` synthetic CSVs and fake command scripts

- [ ] Add the exact command `python prepare_submission.py --cutoff-date YYYY-MM-DD --experiment-name xgb_ranker_v3 --docker-image bdc2026:latest --tar-name 霹雳.tar --push`, the rollback guarantee, manifest path, and 7/31 cutoff procedure.
- [ ] Add fixture-based dry-run tests that never claim real Docker/Git success.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest -q` and `git diff --check`, then commit docs and fixtures.

### Task 6: Acceptance and final push

**Files:**
- Generate: `docs/validation/latest-submission.json` and dated validation report only after all gates pass.

- [ ] Run current acceptance: `.\.venv\Scripts\python.exe prepare_submission.py --cutoff-date 2026-07-24 --experiment-name xgb_ranker_v3 --push`.
- [ ] Verify 300-stock data, exact cutoff, four purged outer folds, positive fixed-equal latest/mean return, improved worst week, five 0.2 weights, two Docker hashes, Docker/network/tar limits, and remote SHA.
- [ ] Run the full test suite and `git status --short`; expected tests pass and only ignored runtime artifacts remain.
- [ ] After 2026-07-31 close, rerun the exact command with `--cutoff-date 2026-07-31`; never reuse the 7/24 result if later data is allowed.
- [ ] Verify final result, manifest, tar, and remote SHA after the cutoff rerun.

## Self-review checklist

- [x] Data, rollback, inference, Docker, Git, documentation, and final cutoff requirements map to Tasks 1–6.
- [x] Interfaces are declared before their consumers and no public bypass skips full Docker training.
- [x] Failure semantics preserve old formal artifacts until all candidate gates pass.
- [x] Every step contains a concrete command, expected behavior, or explicit interface.
