# Independent verification of the QSAR review

## 中文摘要

本报告独立复核 [Claude 的 PR #1](https://github.com/LessRAY3-3/qsar-activity-prediction/pull/1)，核验基线为上游提交 `25d323e82011bef9c92ba3e240e6d627dc1c6572`，核验日期为 2026 年 10 月 10 日。

主要问题成立：删失 IC50 标签被当作精确值、盐型导致母体跨集合重叠、四个补充实验复用 random 超参、时间切分表混用单模型均值与集成指标，以及 y-randomization 对照分数的解释错误。局部 bootstrap 运行覆盖完整汇总表的行为也已复现。

原报告有几处需要纠正：SHAP 背景的前 200 个样本实际覆盖 173 个骨架，并非一两个；`standard_relation == "="` 与 `pchembl_value.notna()` 不等价；建议的 `2*min(p, 1-p)` 公式会在两模型预测完全相同时给出 p=0，不能直接作为修复；盐型重叠最高比例是 ABL1 random 的 2.04%，不是约 1.7%。XGBoost 的确定性已经在原方法文档中说明，无需为了制造 seed 波动而修改算法。

现有证据不能直接推翻 RF 与 GIN 的总体比较，也不能保证修复数据后结论不变。优先工作应是明确标签筛选规则并做敏感性实验。本次提交仅新增报告和证据摘要，不改训练代码、数据、已有实验结果或 Claude 的报告。

## Scope and validation

Prepared by Codex as an independent check of the findings in PR #1, rather than a new model benchmark. Finding IDs below refer to that report. The reviewed desktop copy matched all 907 tracked files at the upstream base commit. Upstream still pointed to that commit when this report was prepared.

- Re-ran the existing suite: **81 passed, 2 warnings**; `ruff check --no-cache scripts experiments tests` passed.
- Downloaded the `artifacts-v1` raw-data release into a temporary directory. The archive matched the release checksum; all ten extracted raw files matched the repository's [raw-data manifest](../../data/raw/sha256sums.txt).
- Independently reconstructed all nine activity clean datasets from those raw CSVs. SMILES order and measurement counts matched; pIC50 values matched numerically.
- Recomputed parent overlap for all nine activity datasets and time-split deltas from saved predictions for all eight panel targets.
- Used small synthetic inputs to exercise the reported missing-split, missing-checkpoint, plotting, and bootstrap edge cases. Partial-run overwrite checks used temporary output directories.

The validation environment was Python **3.11.16**, NumPy **2.4.6**, pandas **3.0.6**, scikit-learn **1.9.1**, torch **2.14.1**, torch-geometric **2.8.0.post1**, RDKit **2026.9.1**, XGBoost **3.2.0**, and pytest **9.1.1**. This was a compatible environment, not the exact Python 3.12 dependency snapshot. The two pytest warnings concerned deprecated `torch.jit.script` and conversion of a gradient-bearing loss tensor to a scalar.

No full model campaign was retrained. The numerical effect of changing labels, molecular standardization, or auxiliary-experiment hyperparameters remains unmeasured. Current API responses were retrieved on the review date; they do not reconstruct every historical activity relation.

The accompanying [evidence summary](evidence/codex-verification-2026-10-10.json) records the parent-overlap counts, cleaning checks, selected API matches, and independently recomputed time metrics. It contains no machine-specific paths and is not a replacement for the original data.

## Findings that affect data and experiment interpretation

### H1 Censored measurements become exact labels

**Confirmed.** Both downloaders omit `standard_relation` from their saved CSV columns: [generic downloader](../../scripts/01e_download_chembl.py#L36) and [incremental EGFR downloader](../../scripts/01d_download_incremental.py#L26). Their queries do not restrict that relation. More precisely, the API returns the field, but the CSV projection discards it.

The [cleaner](../../scripts/02_clean_data.py#L29) retains positive IC50 values and applies the log transform without distinguishing an equality from a bound. Its pChEMBL check drops rows with missing `pchembl_value`, so it cannot detect the lost bound information.

Two current API records match historical raw rows jointly on molecule ID, SMILES, numeric value, assay type, document ID, and assay description:

| Molecule | Activity ID | API relation and IC50 | Committed clean label | Measurements |
|---|---:|---|---:|---:|
| CHEMBL406375 | 2136983 | ≥10000 nM | pIC50 5.0 | 1 |
| CHEMBL3949197 | 16894417 | ≤100 nM | pIC50 7.0 | 1 |

The clean records are in [egfr_full_pic50_clean.csv](../../data/processed/egfr_full_pic50_clean.csv), lines 13445 and 8243 at the review base. The first label represents an upper bound on pIC50, and the second a lower bound, rather than an exact response. Scoring them as exact values changes the regression target.

**The reported 7.8%–29.7% censored-only compound fractions are not fully verified here.** PR #1's `chembl_censoring_counts.py:115` compares current censored counts per SMILES with historical `n_measurements`. A count greater than or equal to the historical count does not establish that every retained historical observation was censored. Equality of total raw row counts does not establish record identity either. The mechanism and concrete retained examples are confirmed; the full historical prevalence needs record-level reconciliation.

**Correct the proposed filter equivalence.** `standard_relation == "="` is not equivalent to `pchembl_value.notna()`. The current EGFR query returned 258 binding and 8 functional records with equality relations but missing pChEMBL. One historical match is activity 32770, CHEMBL306988: exact IC50 500000 nM, `data_validity_comment="Outside typical range"`, and null pChEMBL. Its clean pIC50 is approximately 3.30103. The [official pChEMBL definition](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions) additionally requires acceptable validity metadata.

A suitable first correction is to preserve relation, validity metadata, and activity ID, then define and log the intended inclusion rule. Removing bounds and removing all missing pChEMBL values are different filtering policies. A sensitivity run should rebuild labels and disclose changes in retained molecules and split composition before interpreting any change in model ranking.

### M1 Salt forms create parent overlap across splits

**Confirmed.** The cleaner groups complete SMILES; fingerprints, graphs, and Murcko scaffolds also use the complete structure. Independent normalization with RDKit `LargestFragmentChooser`, followed by the repository's split functions, reproduced the reported counts:

| Tag | Multi-fragment structures | Parents represented more than once | Overlapping test molecules random | Overlapping test molecules scaffold |
|---|---:|---:|---:|---:|
| egfr | 204 | 43 | 15 | 0 |
| egfr_full | 244 | 57 | 13 | 6 |
| herg | 390 | 30 | 16 | 17 |
| vegfr2 | 270 | 64 | 21 | 0 |
| abl1 | 68 | 27 | 11 | 0 |
| mpro | 135 | 8 | 6 | 0 |
| hivpr | 17 | 6 | 4 | 0 |
| a2a | 13 | 1 | 0 | 0 |
| bace | 0 | 0 | 0 | 0 |

Here an overlapping test molecule has the same largest-fragment parent as at least one molecule in the RF training pool. The stored EGFR split arrays matched the independently regenerated test indices and train-plus-validation membership exactly. Other split counts were regenerated from the deterministic source implementation.

The maximum random overlap is **11/540 = 2.04%** for ABL1, so PR #1's “at most approximately 1.7%” should be corrected. In EGFR full and hERG, ring-bearing tosylate fragments can change the complete Murcko scaffold string even when the main molecular fragment is identical.

The overlap is small in sample count, but its effect on headline scores has not been quantified. The assertion that it “will not move the headline numbers” is not established. Molecular standardization should use an explicit policy appropriate to the dataset before deduplication and splitting; largest-fragment matching here is the audit definition, not a claim that it fully resolves charge, tautomer, or stereochemical equivalence.

### M2 Auxiliary scaffold experiments reuse random hyperparameters

**Confirmed with a scope limit.** These scripts select `best_params_random` for both splits:

- [fingerprint_ablation.py](../../experiments/fingerprint_ablation.py#L114)
- [learning_curve.py](../../experiments/learning_curve.py#L76)
- [y_randomization.py](../../experiments/y_randomization.py#L159)
- [interpretability_alignment.py](../../experiments/interpretability_alignment.py#L277)

The random and scaffold grid winners differ for **7 of 9** datasets: a2a, abl1, egfr, herg, hivpr, mpro, and vegfr2. All three grid parameters differ for mpro.

The headline RF trainer performs separate searches for each split. This finding concerns the auxiliary experiments, not a demonstrated parameter error in the main RF panel. Selecting the split-specific winner would improve consistency, but matching parameters alone does not guarantee an identical forest: training membership and row order must also match. For explanations of a saved model's errors, loading that model is preferable to approximating it by refitting.

### M3 XGBoost is deterministic across the tested seeds

**The observation is confirmed; the proposed severity needs qualification.** The committed R², RMSE, and MAE values are identical across the three seeds within each split. A small independent experiment with the same `hist` method and default sampling produced identical tree dumps and predictions for seeds 42, 1, and 2. The entire CSV rows are not byte-identical because seed and training time differ.

The [methodology already describes XGBoost as deterministic across seeds](../methodology.md#L442). A zero standard deviation accurately describes these repeated deterministic fits, but does not estimate uncertainty from changing the data or split. Removing redundant runs or clarifying that limitation is sufficient. Setting `subsample=0.8` and `colsample_bytree=0.8` would define a different model experiment, not a necessary bug fix.

### M4 Time table mixes mean model scores and ensemble scores

**Confirmed.** [time_split_summary.py:41](../../experiments/time_split_summary.py#L41) labels the point estimate as a mean of per-seed R² values, but line 132 takes the delta from the ensemble prediction bootstrap.

Independent recomputation from all eight time prediction sets matched the stored ensemble deltas within `1e-12`. Predictions were cast to float64 before scoring; original JSON per-seed scores can differ at approximately `1e-7` precision. For hERG, rounded to six decimals:

| Quantity | R² or difference |
|---|---:|
| RF | −0.033775 |
| Mean of three individual GIN R² values | −0.076179 |
| Individual-score mean minus RF | −0.042404 |
| R² of the mean GIN prediction | +0.041903 |
| Ensemble minus RF | +0.075677 |

For A2A the analogous differences are +0.158391 and +0.237632. The [README time table](../../README.md#L395) displays individual-score means beside ensemble deltas. The bootstrap computation is internally consistent; the column description is not.

Add an ensemble R² column or clearly distinguish the two quantities. The statement that five time cells significantly favor GIN refers specifically to the three-model ensemble under the existing bootstrap procedure.

### M5 The control gap is incorrectly attributed to skipping GridSearchCV

**Confirmed.** [y_randomization.py:182](../../experiments/y_randomization.py#L182) fits the random-split RF control on **4438** observations. The headline RF training pool contains **4932**, including **494** validation observations excluded from that control. The scaffold counts similarly differ: 4436 versus 4929.

The documented explanation that merely not rerunning GridSearchCV creates the gap is unsupported. Given identical training rows, hyperparameters, and random state, a small independent RF experiment reproduced the grid search's fitted winner exactly by refitting those parameters. Existing learning-curve and ablation results are also consistent with a training-pool explanation, although the exact contribution to the historical score difference was not isolated by a full rerun.

Correct the description of the training pool. If changing the experiment instead, keep real-label and shuffled-label controls on the same pool; changing only the real-label control would create a new mismatch.

## Lower priority findings and corrections

| ID | Verification and appropriate interpretation |
|---|---|
| L1 | Confirmed naming issue. `p_one_sided` is the fraction of bootstrap deltas greater than zero, not a conventional p-value for a GIN win. Current winner labels use the confidence interval, so this naming issue has not inverted them. The suggested replacement formula has a counterexample below. |
| L2 | Confirmed overwrite. Running `--tags egfr --n-boot 300` with both output and figure paths set to a temporary directory changed a copied campaign summary from **32 rows to 4**. For a partial rerun with a different B, merging cells alone could mix B=300 with B=10000. Current per-cell results all use B=10000; output should isolate smoke runs or retain and validate run settings. |
| L3 | Confirmed that `cv=5` creates unshuffled KFold for this regression task. Whether to shuffle, group by scaffold, or split by year is an experimental-design choice; the current behavior alone does not prove erroneous scores or external test leakage. |
| L4 | Confirmed that the RF refitted for SHAP differs in parameters and training membership from the RF whose errors selected the molecules. **The separate claim that the first 200 SHAP background samples contain only one or two scaffold families is false:** they contain **173**, with a maximum family count of **5**, because the validation carve-out shuffles training indices. |
| L5 | Missing-split seed miscounts, NaN round-trip failure, partial-checkpoint failure, sign/CI color disagreement, and hard-coded verdict text were reproduced on minimal inputs. These are latent paths; the inspected committed panel outputs do not trigger them. Details follow. |
| L6 | Python compatibility and dependency declaration concerns are valid. NumPy 2.5.3 requires Python ≥3.12; the fast path does not state this, while conda/CI select 3.12. `requests` and `scipy` are direct imports but transitive dependencies. Some standard-deviation summaries use ddof=0 while plots use ddof=1. This review did not validate the entire pinned CI environment. |
| L7 | The full-precision mpro/scaffold ensemble delta is **−0.0854787091245214**, which formats to **−0.085**, not −0.086. Shuffled GIN random runs include small positive R² values, while their mean is negative. The time-split paragraph highlights two large test fractions without explicitly claiming they are the only fractions above 20%; treating that omission as a definite error is too strong. |

The Python requirement is independently supported by [NumPy 2.5.3 package metadata](https://pypi.org/project/numpy/2.5.3/). Unshuffled folds, missing direct dependency declarations, and different choices of standard-deviation convention warrant clearer documentation or consistency, rather than automatically invalidating current results.

### L1 The suggested p value formula fails on ties

For identical model predictions, the actual bootstrap function returns delta=0, CI=[0,0], and `p_one_sided=0`, because it counts strictly positive differences. Applying the proposed `2*min(p, 1-p)` gives **0**, despite identical predictions. Rename the existing statistic to something such as `frac_boot_gin_better`; a formal hypothesis test requires its own definition rather than a mechanical renaming or formula.

Reproduction from the repository root:

```bash
PYTHONPATH=experiments python - <<'PY'
import numpy as np
from paired_bootstrap import paired_bootstrap

y = np.arange(100, dtype=float)
result = paired_bootstrap(y, y + 0.1, y + 0.1, n_boot=300)
p = result["p_one_sided"]
print(result)
print("proposed replacement:", 2 * min(p, 1 - p))
assert result["delta_r2_point"] == 0
assert result["ci_lo"] == result["ci_hi"] == 0
assert p == 0
PY
```

### L4 The SHAP background counterexample

The [validation carve-out](../../scripts/gnn_01_make_splits.py#L100) already shuffled the indices consumed by the [SHAP background selection](../../experiments/interpretability_alignment.py#L284). The original review appears to conflate the earlier scaffold-sorted RF pool with these subsequent training indices.

```bash
python - <<'PY'
from collections import Counter
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold

split = np.load("data/processed/splits/egfr_scaffold.npz")
smiles = split["smiles"][split["train_idx"][:200]]
counts = Counter(MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles)
print("scaffolds:", len(counts), "largest group:", max(counts.values()))
assert len(counts) == 173
assert max(counts.values()) == 5
PY
```

### L5 Latent robustness cases

- [gnn_04_compare.py:72](../../scripts/gnn_04_compare.py#L72) counts seed files, not contributing values for each split. With three files but only two scaffold entries, the scaffold mean still reports three seeds.
- [multi_target_summary.py:345](../../experiments/multi_target_summary.py#L345) fails a round-trip check for a tag with one missing split because NaNs are not treated as equal. This does not refute the distinct behavior of skipping a wholly absent tag.
- [uncertainty_ad.py:564](../../experiments/uncertainty_ad.py#L564) warns about partial seed availability, then the prediction path at lines 213–216 raises on the first missing checkpoint. The minimal check mocked dataset and model inputs but exercised the real `process_cell` control flow and missing-file check.
- [gine_panel.py:189](../../experiments/gine_panel.py#L189) and [afp_panel.py:175](../../experiments/afp_panel.py#L175) color a negative per-seed point red even when its ensemble CI is entirely positive; their legends interpret red as an RF win. Both functions exhibited this on a synthetic point of −0.01 and CI [+0.01,+0.05]. There were no such sign conflicts in the current stored panel summaries.
- [multi_target_summary.py:247](../../experiments/multi_target_summary.py#L247) printed that no target gives GIN a clear lead even with two synthetic positive cells. [time_split_summary.py:230](../../experiments/time_split_summary.py#L230) likewise embeds campaign-specific conclusions.

All eight targets currently have three GIN metric entries for each of random, scaffold, and time, so the missing-seed scenario does not describe the present campaign.

## Recommended order of work

1. Preserve activity provenance and censoring metadata, define label and parent-standardization policies, and run a controlled sensitivity comparison. Treat the exact historical censored-only proportions as unresolved until records are reconciled.
2. Correct the time-table metric labels and the y-randomization explanation, and align explanation figures with the actual saved RF they describe.
3. Make auxiliary hyperparameter selection explicit per split and protect campaign summaries from partial-run overwrites, including incompatible bootstrap settings.
4. Address latent robustness paths and documentation details. Keep deterministic XGBoost behavior explicit and reject the proposed p-value shortcut.

These changes address verified inconsistencies. Their effect on the overall model ranking must be measured rather than inferred from either review.
