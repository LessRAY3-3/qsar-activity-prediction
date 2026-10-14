# Review of draft PR #7, 2026-10-13 (`retrain/time-asof-2026-10-13` @ a8cf496)

PR #7 replaces the time-split labels with the `exact-dated-asof-v1` policy,
retrains 8 RF and 24 GIN time models on three hosts, and adds frozen split
bundles, a restore script and campaign tooling. It builds on c1f0523, which
reached `main` inside PR #4 without review; that commit is covered here too.

## 中文摘要

**结论：方向和实现都站得住，作为草稿可以继续；合并前建议先补几件事。**

做对了的部分（都经过独立复算）：

- **as-of 标签规则正确，没有发现泄漏路径。** 训练标签只用验证截止年之前的
  精确测量，验证标签只用测试截止年之前的，测试标签只用测试截止年及之后的；
  8 套冻结划分全部满足这些时间不变量；42,636 个保留、168 个排除逐 tag 复算一致。
- **结果完全可复现。** 8 个 RF 和 24 个 GIN 的时间预测、16 行 bootstrap 区间、
  README 时间表都从提交的文件复算一致；24 个 GIN 检查点在 CPU 上重放，预测误差
  ≤ 1.9e-6；按 README 在干净检出上 restore → bootstrap → summary，产物逐字节一致；
  a2a 的 RF 时间模型可以逐位重训出来。
- **random/scaffold 没被动过。** 164 个预测文件和所有非 time 指标块与合并前一致。
- hERG 的父结构键修复正确，而且只影响那一个分子。327 个测试在固定版本下全过。

合并前建议处理：

1. **时间切分结果缺常数基线。** RF 在 a2a、egfr_full、herg 上显著差于"直接猜
   训练集均值"；GIN 集成除 herg 外都优于这个基线。a2a 上 GIN 的 +0.389 优势全部
   来自 RF 整体偏低 0.81，RF 的排序其实更好（r 0.57 对 0.45）。README 应写出来。
2. **"通过收敛检查"说得太满。** 24 个时间 GIN 的最佳轮次都在第 2–27 轮；收敛
   标志读的是最后一轮的训练损失，换成最佳检查点后 24 个里有 13 个训练误差 > 0.5，
   a2a 三个种子接近无拟合。
3. **防泄漏的核心规则没有测试。** 把训练截止年改错、让验证标签混进测试期、
   中位数改均值、去掉 nM 过滤，327 个测试都照样通过；`label_id` 版本检查删掉
   也通过（旧预测会被当成新的用）；CI 恢复了冻结划分却没有任何测试去读它。
4. **键不匹配时分子会被静默排除。** 缓存键和原始 SMILES 标准化结果不一致时
   （比如 Kekulé 写法或盐型），分子被当作"无年份"排除，只打印一个汇总数。
5. **RF 时间模型分发。** 你们说的 Release 还没更新：`fetch_artifacts.py` 仍会把
   旧的 RF 时间模型装到标准文件名下，而且只要 26 个模型里缺任意一个，就会悄悄
   覆盖本地已验收的新模型（备份随后删除）。要么更新 Release，要么让 fetch 不安装
   时间模型。README 末尾的 Reproduce 段也还说下载的模型能对上已发布指标。
6. **溯源。** 实际跑训练用的 `run_repair.py`、验收检查、指标合并以及"独立"标签
   审计的代码都不在仓库里；仓库里的审计脚本复用的是被审的同一套函数。

另外，**c1f0523（已在 main）里有一条统计结论只写了一半**：它新增了按骨架整组
重抽样的 bootstrap，README 只报告了 GIN 的 2 格。用到全部 24 个 scaffold 格后，
10 个显著判定里有 7 个变成平局（2/3 个 GNN 胜、5/7 个 RF 胜）。测试集里只有 5–85
个骨架，这种重抽样本身也粗糙，但 scaffold 上"显著"的说法至少要加限定。

我们上一轮（PR #6）的 S 项大多还没处理，可能会单独回复；其中 S1 已经因重训
自然解决：8 个时间格在去掉任意一个 seed 后判定都不变。

主结论"GIN 不胜过调好的 RF"不受影响。

## How this review was done

- Exact-pin Python 3.12 environment. CI steps reproduced on a clean export:
  `ruff` clean, `restore_time_inputs.py` verified 8 splits, `pytest -q` 327
  passed.
- Nine read-only reviewers (label policy, campaign tooling, training code,
  results, reproduction and distribution, docs and earlier items, tests, the
  merged c1f0523, and a new-bug hunt) raised 53 findings.
- Six verifiers tried to refute each with their own code: none refuted, 8
  corrected in detail. A completeness critic added 3. The scaffold-cluster
  result was re-derived by hand for the GIN panel.
- Release assets were not downloaded.

---

## Verified as correct (do not change)

- **Label policy.** `asof_labels` (`scripts/gnn_08_time_splits.py:186-228`):
  train = median pIC50 of exact, nM, positive records in dated documents with
  year < cutoff_valid; valid = year < cutoff_test; test = year ≥ cutoff_test.
  Synthetic edge cases behave as documented (future records ignored, salts
  merged to the parent). In all 8 frozen splits, train years < cutoff_valid ≤
  valid years < cutoff_test ≤ test years, and excluded rows have no label. The
  first-report year uses a superset of the label records, which is
  conservative.
- **Counts.** 42,636 kept and 168 excluded; per-tag train/valid/test/excluded
  match methodology (hERG 4,944 / 1,115 / 2,325 / 19).
- **Results.** RF, per-seed and ensemble GIN time R², the train-only constant,
  and all 16 rows of `summary_time.csv` recompute exactly; the 8 README time
  rows match; RF and every GIN seed share the split's test indices and label
  version. Headline numbers (hERG RF −0.056, GIN −0.065 ± 0.033, ensemble
  −0.034, Δ +0.022 [−0.016, +0.061]; 4 GIN / 2 RF / 2 ties; unweighted means RF
  −0.034, GIN +0.022, ensemble +0.062) are correct. All six significant time
  cells survive Bonferroni and Holm over 8 and every single-seed drop.
- **Reproduction.** README's restore → bootstrap → summary commands on a clean
  export reproduce every artifact byte for byte; `restore_time_inputs.py` is
  idempotent and refuses conflicting splits. All 24 GIN checkpoints replay
  their committed predictions (max 1.9e-6) and validation RMSE; the a2a RF time
  model retrains bit for bit.
- **No collateral change.** All 164 random/scaffold prediction files and every
  non-time metric block are identical to `main`; c1f0523 changed no
  random/scaffold significance or metric files.
- **hERG repair.** The double-standardization bug affects exactly one of
  42,804 cached keys (hERG CHEMBL2035183) under the pinned RDKit; the fix
  restores its 2012 label (7,200 nM) and the regression test catches the old
  code.

---

## Medium

### T1. Scaffold-split significance does not survive scaffold-group resampling (c1f0523, on main)

- **Where**: `README.md:79-86`, `:349`, `:384`, `:404-414`, `:428`,
  `:581-583`, `:603-607`; `docs/methodology.md:353-366`; `docs/interview.md:63-64`;
  `results/significance/scaffold_cluster_sensitivity.csv`.
- **Evidence**: c1f0523 adds `experiments/scaffold_bootstrap.py`, which
  resamples Bemis-Murcko groups of the test molecules, and writes all 24 cells;
  the docs mention only GIN abl1 and mpro. An independent reimplementation
  (B = 10,000, groups recomputed from `parent_smiles`, three-seed ensemble)
  gives:

  | Model / cell | Molecule-level CI | Scaffold-group CI |
  |---|---|---|
  | GIN abl1 | GIN [+0.006, +0.069] | tie [−0.004, +0.252] |
  | GIN mpro | RF [−0.133, −0.018] | tie [−0.174, +0.008] |
  | GINE a2a | RF [−0.121, −0.028] | RF [−0.190, −0.058] (5 groups) |
  | GINE abl1 | GINE [+0.028, +0.092] | GINE [+0.008, +0.336] |
  | GINE hivpr | RF [−0.096, −0.007] | tie [−0.154, +0.059] |
  | GINE mpro | RF [−0.130, −0.012] | tie [−0.169, +0.029] |
  | AFP egfr_full | RF [−0.063, −0.013] | tie [−0.131, +0.044] |
  | AFP herg | AFP [+0.030, +0.092] | tie [−0.059, +0.157] |
  | AFP hivpr | RF [−0.123, −0.003] | tie [−0.198, +0.056] |
  | AFP mpro | RF [−0.189, −0.051] | RF [−0.259, −0.010] |

  7 of 10 significant scaffold verdicts become ties; all 14 ties stay ties.
  Test sets hold only 5-85 scaffold groups, so the group bootstrap is itself
  coarse.
- **Fix**: report all 24 cells and qualify the scaffold tallies ("21 RF / 5 GNN
  wins" and the per-model counts) as not robust to scaffold-group resampling.

### T2. The time-split results need the constant baseline

- **Where**: `README.md:758-831`; `docs/methodology.md` current-results section;
  `experiments/LOG.md:85-86`.
- **Evidence** (train-mean constant, paired bootstrap B = 10,000):
  - RF minus constant: a2a −0.266 [−0.357, −0.180], egfr_full −0.293
    [−0.345, −0.244], herg −0.053 [−0.089, −0.018]; hivpr and mpro not
    significant; abl1, egfr and vegfr2 significantly above.
  - GIN ensemble minus constant: significantly above on every tag except herg
    (−0.031 [−0.066, +0.004]).
  - a2a's +0.389 GIN advantage is entirely the level term (+0.579): RF predicts
    0.81 below the test mean, GIN 0.14 above, while RF correlates better
    (Pearson 0.57 versus 0.45; GIN prediction SD 0.19 of the labels').
    About 56% of egfr_full's +0.565 is level; hivpr, mpro and herg are
    correlation-led.
  - The PR removed the earlier decomposition table without replacing it, and
    the test-fraction caveat (a2a 59%, mpro 51%) is gone.
- **Fix**: add a constant column or sentence, a one-line level note for a2a and
  egfr_full (as arithmetic, not mechanism), and restore the test-fraction
  caveat.

### T3. "Passed convergence checks" overstates

- **Where**: `README.md:792`; `docs/methodology.md:547-549`, `:1115`;
  `scripts/gnn_03_train_gin.py:248-256`, `:360`, `:373`.
- **Evidence**: all 24 time GIN runs reach their best epoch at 2-27 (median
  9.5) of 100. The flag reads the running training loss of the final epoch,
  computed before the best checkpoint is restored. Evaluating the restored
  checkpoints, 13 of 24 have standardized training MSE above 0.5; the three a2a
  seeds sit at 0.75-0.98.
- **Fix**: compute the flag from the restored checkpoint in eval mode, and say
  that the time runs stop very early.

### T4. The leakage fix and its version guard are untested

- **Where**: `scripts/gnn_08_time_splits.py:201`, `:218-222`;
  `scripts/time_labels.py:87`; `.github/workflows/ci.yml:27-30`.
- **Evidence**: each of these mutants leaves all 327 tests passing: train window
  `<= cutoff_v`, train window `< cutoff_t`, valid window `<= cutoff_t`, median
  replaced by mean, nM filter removed, and deleting the `label_id` comparison in
  `require_prediction_version`. With that check removed, the pre-PR a2a
  predictions (all-year labels, same test indices) are accepted. No test reads
  the restored splits; committed summary values replaced with the pre-repair
  hERG values still pass. A 25-line probe with records in both cutoff years, a
  three-value median and one µM row kills every label mutant.
- **Fix**: add that probe; test `require_prediction_version` with a wrong
  `label_id`; recompute `results/time_split/summary.csv` from the restored
  splits in CI.

### T5. A parent-key mismatch silently excludes molecules

- **Where**: `scripts/gnn_08_time_splits.py:181`, `:215`.
- **Evidence**: the fix replaced `_parent_keys(smiles)` with the cached keys as
  they are. In a synthetic run, a molecule cached in Kekulé form (or as a salt)
  no longer joins its raw records and is excluded as undated; the run prints
  only an aggregate exclusion count, and `load_time_split` accepts the split.
  Below 1% mismatches nothing else is printed; at 1-5% the message wrongly says
  "no publication year". Above 5% the run aborts.
- **Fix**: fail closed when any cached key has no raw record, listing the rows.

### T6. The code that produced the committed results is not all in the repo

- **Where**: `logs/time_asof_repair_20261014/`; `results/time_split/campaign_acceptance.json`;
  `scripts/run_time_campaign.py:334-342`.
- **Evidence**: the hERG repair ran through an uncommitted `run_repair.py`; the
  acceptance checker, the merge of `time` into the metric JSONs and the
  "independent" raw-record audit are not committed. The committed
  `experiments/time_label_audit.py` imports `gnn_08`'s own functions, so it is
  not independent. `run_time_campaign.py`'s own metric merge never ran in the
  campaign.
- **Fix**: commit those scripts with their command lines, or say in
  methodology which steps are not reproducible from the repo.

### T7. RF time-model distribution

- **Where**: `scripts/fetch_artifacts.py:90-95`, `:556-590`; `README.md:876-881`;
  the artifacts-v2 release body.
- **Evidence**: artifacts-v2 still ships the eight pre-as-of RF time models as
  `models/rf_time_split_{tag}.joblib` and calls all 26 models refreshed. A
  simulated user with the eight accepted as-of models and one missing random
  model who runs `fetch_artifacts.py --models` gets all eight replaced by the
  legacy ones, exit 0, no warning, and the backups deleted. No code path
  consumes them silently today (uncertainty and interpretability refuse the
  time split), so this is a distribution hazard rather than a wrong number.
  README's closing Reproduce section still says fetched models verify against
  the published metrics.
- **Fix**: republish the release, or stop `fetch_artifacts.py` from installing
  `rf_time_split_*` (move them to `models/historical/`), and correct the release
  body and the Reproduce section.

---

## Low

- `experiments/xgboost_baseline.py:226`, `learning_curve.py:176`,
  `y_randomization.py:154`: still accept `--splits time` and train on the
  all-year labels from the fingerprint cache (a2a: 27 train/valid labels
  differ); reject `time` or load via `load_time_split`.
- A failed `gnn_08` regeneration deletes the restored split and leaves an NA
  marker, after which the trainers exit 0 with "skipped".
- `run_time_campaign.py`: rollback covers Python exceptions only (not SIGTERM
  or SIGKILL); the bundle manifest hash is recorded but never verified;
  `git_info` records the enclosing repo's HEAD for a nested bundle; bundle
  creation includes gitignored `.py` files but reports `dirty=False`.
- `restore_time_inputs.py` writes in place, so an interrupted restore leaves a
  truncated split that later restores reject with a generic message; its
  label-version gate skips the 24 GIN checkpoints.
- The "internal signature" is an unkeyed SHA-256 (integrity, not
  authenticity) that omits `scaffold_smiles`.
- 16 non-equivalent mutants of the tooling survive its tests; trainers are
  stubbed and the checkpoint check mocked. `test_time_campaign_bundle` errors
  when commit signing is enabled. The CHEMBL2035183 regression test silently
  stops testing if RDKit changes the canonical form.
- RF time is a single fit; the hERG repair moved Δ from +0.006 to +0.022 and
  five random states give R² SD 0.006, which the docs do not report.
- `panel_3splits.png` and the console call the random-to-time difference a
  "collapse" although the two use different label policies.
- The out-of-AD aggregate (c1f0523) is a subset R² that partly reflects label
  variance; out-of-AD RMSE is higher in 10 of 13 cells (mean 1.075 versus
  0.746), which states the effect better.
- 46% of a2a's time test labels are three discrete IC50 values from one 2022
  batch; not mentioned.
- Docs: LOG cross-references and legacy numbers, the frozen-package size
  ("2.1 MiB"), `time_split_summary.py`'s a2a test fraction (0.62 versus 0.59),
  and methodology passages written as chat-session narrative ("at the user's
  request").

## Earlier items (findings-2026-10-13b.md, PR #6)

| ID | Status at a8cf496 |
|---|---|
| S1 single-seed on time split | Resolved by the retrain: all 8 time verdicts survive every drop; the README does not yet say the sweep was run. |
| S2 multiple comparisons | Partly: time cells survive Bonferroni(8); no multiplicity note in README. |
| S3 strong wording | Not fixed. |
| S4 Δ conventions | Partly. |
| S5 XGBoost bootstrap | Not fixed. |
| S6 multitask device log | Not fixed. |
| S7 untested 68c736f behaviours | Not fixed. |
| C1 "strengthens the headline" | Not fixed. |
| C2 decomposition caveat | The table was removed rather than caveated (see T2). |

## Limitations

- Raw activity records and release assets were not downloaded, so the label
  audit could not be repeated from raw records, and the artifacts-v2 contents
  were not inspected.
- Only the a2a RF time model was retrained; the other seven are checked
  through their committed predictions.
