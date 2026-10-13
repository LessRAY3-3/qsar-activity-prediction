# Follow-up review, 2026-10-13 (`fix/review-2026-10-11` @ 68c736f)

Commit 68c736f answers `findings-2026-10-13.md` (R1-R5 plus Low, PR #6).
This file is a separate addendum so that the copy of `findings-2026-10-13.md`
already in PR #4 stays byte-identical.

## 中文摘要

**下载脚本这次真正加固了。** 46 个构造的攻击压缩包（符号链接、硬链接、
大小写变体、PAX/GNU 长名、越界路径等）全部被拒，信任锚字节不变；旧版本用户
能自动修复；失败可回滚。R3、R4 已修好。按固定版本跑，205 个测试全过，CI 绿，
README 431 格 0 不一致。

**我们又要更正自己两处（都来自 PR #6）：**

1. **R1 里"这反而加强了主结论"说错了。** "去掉一个 seed"这个检查只扰动 GNN
   一侧：48 格里每一格，去掉一个 seed 都会让 GNN 的集成 Δ 变小（平均 0.008），
   而 RF 的 seed 从未变过。所以 GNN 胜格容易翻、RF 胜格不容易翻，这是检查本身
   决定的，不能当作"RF 优势稳健"的证据。
2. **R2 的分解表少说了一个前提，还有一句说错了。** 离散度和相关性两项怎么分，
   取决于分配约定（a2a 的离散项在不同顺序下从 −0.128 变到 +0.143），只有
   水平项和总量与约定无关；我们写的"只有 egfr 是 RF 更过度离散"不对，
   egfr_full 和 mpro 也是。

**R1–R5 状态：** R3、R4 已修好；R1、R2、R5 部分修好（上面两处是我们的锅，
另有 README 的口径问题）。

**还剩 7 条中等问题（另 2 条是上面的自我更正）：**

1. 单 seed 标准没用到时间切分：hivpr/time 去掉 seed 42 后变平局，但仍算作
   GIN 胜格。
2. 全文没有提多重比较。按面板内 Bonferroni，GIN 面板只剩 3 个 RF 胜、0 个
   GIN 胜；48 格整体 8 个 RF 胜、2 个 GNN 胜（GINE abl1/scaffold、
   AFP herg/scaffold）。时间切分 6 个显著格在 Bonferroni(8) 下都还在。
3. 几处措辞比数字强：egfr/scaffold"真实的差距"、herg/random"明显领先"、
   fairness"两个靶点上明显"，按 README 自己的标准都是平局或条件性。
4. Δ 口径在表格之间来回切换：第 63 行的定义和第 352 行矛盾；"8 个里 7 个
   GIN ≤ RF"只在逐 seed 均值口径下成立，集成口径下 scaffold 是 5/8。
5. XGBoost 其实可以做配对 bootstrap（不需要多个 seed）：结果是
   6 RF / 1 XGB / 9 平，herg/scaffold 上 XGBoost 显著胜（+0.032
   [+0.014, +0.051]），README 却说成"轻微"。
6. 设备说明拿 10 月 4 日刷新前的 multitask 日志当作刷新后 multitask 结果的
   MPS 证据；刷新后那次 multitask 重跑没有提交日志。
7. 68c736f 新增的大部分行为没有测试：35 个变异体里 19 个能躲过全部 205 个
   测试（含白名单边界 `models_evil/`）。

另外一条需要你们动手确认：新的白名单会拒绝几种常见的打包方式（`tar -C dir .`
产生的 `.` 成员、macOS 自动生成的 `._` 附属文件）。仓库里没有 68c736f 对真实
artifacts-v2 跑过 `fetch_artifacts.py --all` 的记录，我们没有下载发布包，
无法确认。建议跑一次并提交日志。

主结论"GIN 不胜过调好的 RF"仍然成立；更正 1、2 只影响怎么表述这个结论。

## How this review was done

- Exact-pin Python 3.12 environment; 68c736f checked out separately; `ruff`
  clean, `pytest -q` 205 passed; CI green.
- Seven read-only reviewers (fetch hardening, R1/R2, R5 and docs, Low code and
  tests, skill folder and notebook, new-bug hunt with regression checks, and a
  statistical-framing audit of the README) raised 68 findings.
- Six verifiers tried to refute each with their own code: none refuted, 8
  corrected in detail. A completeness critic added 5. The one-sidedness of the
  drop-one-seed check was re-derived by hand.
- Nothing was retrained except deterministic XGBoost refits for S5. The
  artifacts-v2 archives were not downloaded.

## Corrections to our own review (findings-2026-10-13.md)

### C1. The drop-one-seed check cannot show that RF wins are robust

R1 said the single-seed sweep "strengthens the headline" because only 1 of 21
RF wins depends on one seed. The check only perturbs the GNN arm, and it is
biased against it:

- For every one of the 48 cells, the mean of the three 2-seed ensemble deltas
  is below the 3-seed delta (median 0.006, mean 0.008, range 0.002-0.029 R²).
  The per-molecule identity is
  mean_k (y − m₋ₖ)² = (y − m)² + mean_k (pₖ − m)² / 4.
- The RF is one `RandomForestRegressor(random_state=42)` fit; its seed is never
  varied.
- Over the 144 drops, 20 of 21 verdict changes move toward RF; no tie becomes a
  GNN win.
- A size-neutral recount (removing the ensemble-size penalty) gives 3 of 5 GNN
  wins, 4 of 21 RF wins and 5 of 22 ties as seed-dependent.

Suggested wording: "The check perturbs only the GNN arm and, averaged over the
three drops, always lowers the ensemble Δ (median 0.006 R²). It identifies GNN
wins that rest on one seed; it is not evidence that RF wins are seed-robust."
README:364-369, methodology:423-425 and LOG.md:1030-1032 adopted our wording.

### C2. The spread/correlation split is not unique

R2 gave a Shapley split of GIN − RF R² into level, spread and correlation.
Only the level term and the total are convention-free:

| Tag | Total | Level | Spread / corr (Shapley) | Spread first | Correlation first |
|---|---|---|---|---|---|
| a2a | +0.326 | +0.580 | +0.008 / −0.263 | −0.128 / −0.127 | +0.143 / −0.398 |
| abl1 | −0.236 | −0.002 | −0.115 / −0.119 | −0.143 / −0.092 | −0.087 / −0.147 |
| egfr_full | +0.592 | +0.392 | +0.032 / +0.168 | +0.039 / +0.161 | +0.025 / +0.175 |
| mpro | +0.084 | +0.025 | +0.027 / +0.032 | +0.033 / +0.026 | +0.021 / +0.038 |

Our clause "only egfr fits less over-dispersion" is false under every split:
egfr_full and mpro show the same reduced RF over-dispersion. The README's
per-tag signs for egfr_full (correlation), herg (spread) and hivpr
(correlation) hold under both sequential orders.

Suggested wording under the table: "Only level and the total are
convention-free; the spread/correlation magnitudes depend on how the cross term
is allocated (a2a's spread term ranges from −0.128 to +0.143). Signs quoted
below hold under both sequential orders."

## Status of the 2026-10-13 items

| ID | Status | Notes |
|---|---|---|
| R1 single-seed standard | **partially fixed** | 48-cell sweep committed and correct (14 cells, matches ours). Not applied to the time split (S1); some wording still universal ("any one seed") or strong (S3); the "strengthens" sentence is our error (C1). |
| R2 decomposition | **partially fixed** | Table added, residual < 1e-15, mechanism labels removed from README. Convention caveat missing; a "compression" label survives in methodology:671-673 (C2). |
| R3 trust anchor | **fixed** | 46 synthetic attack archives all refused with the anchor byte-identical: file and directory symlinks, hardlinks, case and Unicode variants, PAX and GNU long names, out-of-prefix members, `..`, absolute paths, fifos and devices. Post-move anchor re-check works. |
| R4 lock-out | **fixed** | An artifacts-v1-style user is repaired in one `--all` run (11 raw, 26 models, 9 caches); a stale single file is replaced and named; key files checked in full; rollback restores bytes, modes and symlinks. |
| R5 egfr provenance | **partially fixed** | Methodology labels the 9 egfr GIN runs as unlogged with unrecorded device. README's headline table has no caveat; the device paragraph cites the wrong multitask log (S6). |
| Low items | **mostly fixed** | Remaining ones below. |
| Earlier items | **no regression** | GINE 1/7/8, AFP counts, `constant_r2`, XGBoost egfr_full/random 0.6814376116 under 3.4.1, notebook byte-identity, 431 README cells. |

---

## Medium

### S1. The single-seed standard is not applied to the time split

- **Where**: `README.md:352`, `:714`, `:779`; `logs/2026-10-13_single_seed_dependence.log`.
- **Evidence**: GIN hivpr/time is +0.115 [+0.036, +0.196]. Dropping seed 42
  gives +0.041 [−0.046, +0.129], a tie; per-seed Δ are +0.162, −0.014,
  +0.020. The other 7 time cells keep their verdict under every drop. README:352
  says the conditional standard applies to "every verdict in this README", yet
  the time table has no conditional marking and :779 counts hivpr among the
  GIN cells that "hold up".
- **Fix**: extend the sweep to the 8 time cells and mark hivpr/time
  conditional.

### S2. Multiple comparisons are never acknowledged

- **Where**: README, methodology and interview (a grep for multiple,
  Bonferroni, Holm or FDR finds nothing); the one earlier mention was deleted
  in 6c1e3bb.
- **Evidence** (two-sided bootstrap p = 2·min(frac, 1 − frac), from the
  committed summaries):

  | Scope | Uncorrected 95% CI | Bonferroni |
  |---|---|---|
  | GIN panel (16) | 7 RF / 2 GIN | 3 RF / 0 GIN |
  | All 48 GNN cells | 21 RF / 5 GNN | 8 RF / 2 GNN |
  | Time split (8) | 2 RF / 4 GIN | all 6 survive (hivpr only just) |

  The two GNN wins that survive Bonferroni-48 are GINE abl1/scaffold (also
  robust to every seed drop, so README's "genuine" is supported) and AFP
  herg/scaffold.
- **Fix**: one sentence stating that the tallies are uncorrected per-cell 95%
  CIs, with the Bonferroni counts beside them.

### S3. Strong wording on cells the README's own standard calls ties or conditional

- `README.md:277`: egfr/scaffold "real, but closer to the noise floor" uses
  seed std as a test; the ensemble CI is [−0.053, +0.006], a tie, and the cell
  is conditional.
- `README.md:299`: herg/random "clearly leads"; it is conditional and fails
  within-panel Bonferroni.
- `README.md:305`, `:348`, `:503`, `:540`: "when any one training seed is
  dropped" reads as universal; 13 of 14 conditional cells change under only
  one or two of the three drops (GIN herg/random survives the seed-42 drop).
- `README.md:470`: fairness "(i) clearly on two of three targets"; herg's
  +0.033 lift is within its frozen seed std (0.038) and :515 calls vegfr2's
  lift seed noise. `README:462-464` omits frozen GIN herg/scaffold from the
  conditional cells.
- **Fix**: "not significant (conditional)" for egfr/scaffold, "nominally
  leads (conditional)" for herg/random, "at least one seed" with the seed
  named, and "clearly on one of three (vegfr2)" for fairness.

### S4. The Δ convention changes between tables

- **Where**: `README.md:63-64`, `:186-187`, `:301`, `:313-315`, `:352-353`,
  `:714`, `:777`.
- **Evidence**: README:63 defines Δ as the per-seed mean and calls it "the
  effect size every verdict below uses"; :352 says verdicts come from the
  ensemble CI. The panel counts use the per-seed mean; the time header uses
  the ensemble. Recomputed:

  | Split | GIN ≤ RF, per-seed mean | GIN ≤ RF, ensemble |
  |---|---|---|
  | Random | 7 of 8 | 7 of 8 |
  | Scaffold | 7 of 8 | 5 of 8 |

  README:186-187 says "the conclusions hold either way" for a DeepChem-style
  scaffold split that was never run.
- **Fix**: define both conventions once, state each count with its
  convention, and drop or qualify "hold either way".

### S5. XGBoost can be bootstrapped, and it wins one cell significantly

- **Where**: `README.md:573`, `:595-605`; `results/xgboost/`.
- **Evidence**: no per-molecule XGBoost predictions are committed, but the
  fits are deterministic. Refitting with the committed protocol under
  xgboost 3.4.1 reproduces all 16 committed R² to 6 decimals. A paired
  bootstrap against the committed RF predictions (B = 10,000) gives 6 RF wins,
  1 XGBoost win and 9 ties. The win is herg/scaffold +0.032 [+0.014, +0.051],
  which README calls one of "the five exceptions, all modest". README
  attributes the missing uncertainty to bit-identical seeds, but a paired
  test-set bootstrap needs no seed replicates. The sentence "four of the five
  read ≤ +0.010 only after rounding" is wrong: only hivpr/random (+0.0102)
  needs rounding.
- **Fix**: commit per-molecule XGBoost predictions, add the paired bootstrap,
  and name herg/scaffold as a significant XGBoost win. "XGBoost does not beat
  the tuned RF overall" still holds.

### S6. The device paragraph cites the wrong multitask log

- **Where**: `docs/methodology.md:293-297`; `experiments/LOG.md:5-9`.
- **Evidence**: the new text says the refreshed multitask batch ran on the Mac
  under MPS, citing `logs/multitask.log`. That log starts "MULTITASK JOB LAUNCH
  2026-10-04", ran on the M3 Max on the pre-refresh data (egfr_full test 2,700
  versus the refreshed 2,146), and its numbers differ from
  `results/multitask/summary.csv`. The 2026-10-11 multitask rerun has no
  committed log. The `aux_m4_*.log` files print no device. Related: README's
  headline EGFR GNN table still has no caveat that its 9 GIN runs have no
  committed log.
- **Fix**: say the refreshed multitask rerun has no committed log and an
  unrecorded device; call `multitask.log` the superseded pre-refresh run; add
  a provenance footnote under the EGFR table.

### S7. Most behaviours added in 68c736f have no test

- **Evidence**: of 35 non-equivalent mutants of new behaviour, 19 survive the
  full 205-test suite, including: XGBoost `same_pool` ignoring the RF test set,
  the paired_bootstrap summary `n_boot` guard and `boot_seed`, figure placement
  with `--out-dir`, fingerprint_ablation coverage on bits and seeds,
  interpretability default output, panel fallback markers, and the fetch
  allowlist boundary (`models_evil/payload.py` is accepted when the
  `prefix + '/'` check is weakened). Two more are caught only by the notebook
  byte-identity test.
- **Fix**: one behavioural test per surviving mutant; the list is in the
  verifier notes of this review.

---

## Low

### Fetch

- **Unverified against the real release.** The new allowlist refuses archives
  with a `.` root member (`tar -C dir -czf x .`, or `tarfile.add(dir,
  arcname='.')`) and macOS bsdtar's AppleDouble sidecars (`._models`,
  `data/._raw`), which bsdtar emits by default on this macOS version. The real
  archives' member lists are not recorded, and no log shows 68c736f running
  `--all` against them (the end-to-end claim was made with 6c1e3bb). Please run
  it once, commit the log, and record the packaging command (for example
  `COPYFILE_DISABLE=1 tar -czf … <file list>`) in the release body.
- Models and cache install every file under their directory prefix, so
  `processed_cache.tar.gz` could overwrite tracked `data/processed/` files
  (clean CSVs, splits) unverified; install only the key files.
- Rollback covers only `OSError`; a Ctrl-C or a failed restore leaves a
  partial set while the message says files were left unchanged.
  `*.fetch_backup` is not gitignored.
- When the release itself is stale, the remedy text still suggests deleting
  local files; self-repair replaces non-matching local raw files without a
  surviving backup; complete-but-stale models or caches are still skipped.

### Code (latent)

- `experiments/xgboost_baseline.py`: `same_pool` compares training pools only;
  a different test set is not detected when the split file does not partition
  all rows; the RF pool source is not recorded.
- `experiments/time_split_summary.py:35-41`: `constant_r2 < 0` is always true
  for these baselines, so "genuine temporal drift" fires for any negative RF at
  or above the constant; `logs/final_m4_time_split.log` still prints the old
  labels.
- `experiments/multi_target_summary.py:238-257`: still calls any point estimate
  ≥ +0.0005 "a clear lead", including two conditional cells; `afp_panel.py:371`
  says the AFP readout "is real somewhere" for two conditional cells.
- `experiments/paired_bootstrap.py`: summary `boot_seed` is not guarded (the
  docstring says it is); a mid-loop input error still leaves earlier cells
  rewritten; figure placement compares `abspath`, not `realpath`.
- `scripts/02_clean_data.py`: can still write a header-only clean CSV with
  exit 0. `scripts/07_qc_checks.py` re-implements 02's checks and disagrees on
  padded `=` rows; it now skips the BACE ultra-potent audit that methodology
  still quotes.
- `results/interpretability/by_tag/` JSONs for four tags are pre-N10 and differ
  from the `figures/interpretability/by_tag/` copies README cites.

### Docs

- `experiments/LOG.md`: the 2026-10-13 entry describes a mid-batch state; its
  header still says the refresh campaign trained on the CUDA host with
  committed logs; the status pointer orders the two newest entries wrongly;
  committed logs cite LOG.md line numbers that have moved.
- Methodology time-split text: a "compression" label, "in about equal
  measure" for abl1, "the GIN matches y's spread" (0.39 of y's std), "gap
  0.46" (0.45).
- Fairness: the lift is +0.033, not +0.034; multitask verdicts use per-seed
  means (under the ensemble CI three arm-cells are ties); README:289-292
  quotes a seed-42-only chemotype number whose asymmetry flips with another
  seed; bond-feature "help/hurt" is measured against RF, not against the GIN
  without bond features; "any GNN uncertainty claim would be noise" is
  stronger than the data (the GIN ensemble std is weakly but significantly
  informative in some cells).
- `results/_archive-2026-10-09/scaling_ep300/ep300_confirmation.md` now points
  its baselines at refreshed files that do not hold those values.
- The PR description still says 8 of 32 cells got conditional wording and 165
  tests; the reply says 23 new Low-batch tests (the tree has 22);
  `docs/reviews/fixes-2026-10-11.md:56` gives the wrong `test_clean_data`
  count.
- Pre-existing: committed logs contain absolute local paths with usernames.

### Skill folder

- The new README verdict content (conditional column, decomposition table) is
  not machine-checked by `verify_readme_tables.py`, and the scripts that
  generated the two evidence logs are not committed.
- `qc_salts_censoring.py --allow-skip` passes out-of-sync split files as OK;
  it and `chembl_censoring_counts.py` still say they apply 02's relation rule
  "verbatim (no stripping)", but 02 now strips.
- SKILL.md and the release body still describe the pre-R3 fetch.
- No test or CI lint step covers the skill scripts (carried over).

---

## Verified as correct (do not change)

- `fetch_artifacts.py` refuses all 46 synthetic attack archives, repairs an
  artifacts-v1-style tree in one run, re-downloads re-uploaded assets, and
  turns refusals into clean exit-1 messages.
- The manifest header change in 68c736f does not affect verification: the
  parser skips comment lines and all 11 pins are unchanged, so the published
  `raw_data.tar.gz` still matches.
- The 48-cell single-seed sweep, the conditional column and the 14-cell list
  reproduce exactly (bootstrap seeds 42 and 2027).
- The decomposition table reproduces with residual below 1e-15.
- 205 tests pass under the exact pins; README 431 cells, 0 mismatches.
- The copy of `findings-2026-10-13.md` in PR #4 is byte-identical to PR #6.

## Limitations

- The artifacts-v2 archives were not downloaded, so whether the hardened
  fetch accepts them is unverified.
- Nothing was retrained except deterministic XGBoost refits for S5.
