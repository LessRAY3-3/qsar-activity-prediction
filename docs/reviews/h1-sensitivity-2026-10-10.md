# H1 删失值敏感性实验：egfr_full 记录级对账与敏感性设计

## 中文摘要

本实验回应审查报告 [H1](../../.claude/skills/qsar-review/references/findings-2026-10-10.md)（删失 IC50 被当作精确值训练）与[独立核验](codex-verification-2026-10-10.md)提出的两点保留意见（历史删失比例需要记录级对账；`standard_relation == "="` 与 `pchembl_value.notna()` 不等价）。工作分支 `fix/review-2026-10-10`，基线为 HEAD（`94572a6`）。

记录级对账**完全确认**了 PR #1 的量级：历史 raw pull 的 24,560 行在今日 API 上 100% 匹配（0 行丢失、0 行新增），匹配行的删失份额 21.5%（不含空 relation）与 PR #1 的 21.5% (5,283/24,560) 一致；按同样清洗流程（nM、去缺失、正值、SMILES 组中位数，仍不加 relation 过滤）复算，**13,497 个化合物的标签与已提交 clean CSV 零差异**（最大偏差 1.8e-15）。清洗后"仅删失"化合物为 **2,722/13,497 = 20.17%**，与 PR #1 声称的 2,718/13,497 ≈ 20.1% 相差 4 个（口径：`>>`/`~`/空 relation 共 151 条记录的处理边界）。两种过滤策略确认**不等价**：`=` 保留 19,106 行/10,775 化合物，`pchembl_value.notna()` 保留 18,843 行/10,675 化合物，差异全部是一方向——263 行（185 个化合物）relation 为 `=` 但 pChEMBL 为空（典型为 `data_validity_comment="Outside typical range"`，如 activity 32770 / CHEMBL306988，500000 nM）。

若按主策略 `standard_relation == "="` 修复标签：2,722 个仅删失化合物整体消失，另有 **896 个化合物的标签会移动**（|Δ| 均值 0.57、最大 4.30 个对数单位；有符号中位数 +0.195）。敏感性 RF 重跑因与后台任务 CPU 冲突**尚未完成**，修复前后指标对比表见"敏感性实验结果"一节占位，待后续回填。

## 范围与产物

本次只做 egfr_full（CHEMBL203）一个靶点、RF 一个模型族的敏感性实验。不改清洗脚本（02 的修复属 Phase C），实验脚本与中间产物全部放在 `.scratch/h1_sensitivity/`（不进 git），仅本报告与 [scripts/01d_download_incremental.py](../../scripts/01d_download_incremental.py) 的工作区改动属于仓库内容（01d 的 `COLUMNS` 已补齐 `standard_relation`、`data_validity_comment`、`activity_id` 三字段，与 01e 一致，`ruff check` 通过，**未提交**）。

产物清单：

- `data/raw/egfr_full_v2_activities.csv` — 新下载（24,560 行；旧 `egfr_full_activities.csv` 未动）
- `.scratch/h1_sensitivity/reconcile_h1.py` / `reconcile_summary.json` / `label_diff_per_compound.csv` / `censored_only_smiles.csv`（2,722 行）/ `filter_strategy_diff_examples.csv`
- `.scratch/h1_sensitivity/rf_h1_sensitivity.py` / `rf_h1.log` / `rf_h1_sensitivity.json`（后者待生成）

## 方法

### 下载

01e 的 `COLUMNS` 已由主智能体加上 `standard_relation`、`data_validity_comment`、`activity_id`；本实验将 01d 做同样补齐。重下命令：

```bash
.venv/bin/python scripts/01e_download_chembl.py --target-chembl-id CHEMBL203 --tag egfr_full_v2 --page 500
```

结果：B 类 23,615/23,615（100%）、F 类 945/945（100%），共 **24,560 行，与已提交旧版行数完全相同**，无 skipped windows，耗时约 8 分钟。探测阶段的 total_count=25,244 含其他 assay_type（A/T/U 等 684 条），不在 B+F 抓取范围内，新旧两版口径一致。

### 记录级对账

旧 CSV 没有 activity_id/relation，用组合键 **molecule_chembl_id + canonical_smiles + standard_value + assay_type + document_chembl_id** 做**多重集匹配**（键内出现次数对齐，true 1:1 配对），匹配上的行取新 CSV 的 `standard_relation`，即得到历史记录的删失构成。`standard_value` 先做浮点规范化避免 `'500000'` vs `'500000.0'` 失配。

### 清洗复现与仅删失口径

照 [02_clean_data.py](../../scripts/02_clean_data.py) 逐步复现（unit==nM → `standard_value` 数值化并去缺失 → 去空 SMILES → 去非正值 → pIC50 → 按 canonical_smiles 取 median），**仍不加 relation 过滤**，与已提交 `egfr_full_pic50_clean.csv` 逐化合物对比。"仅删失"定义：清洗后存活记录中**没有任何一条** `standard_relation == "="` 的 SMILES（空 relation、`>>`、`~` 均计为非等号）。

### 敏感性 RF 设计（待执行完）

`.scratch/h1_sensitivity/rf_h1_sensitivity.py` 复刻 HEAD 版 [04_train_and_evaluate.py](../../scripts/04_train_and_evaluate.py) 的原始流程（工作区当前副本已含 shuffle 修复，属 L3/Phase C，本次刻意用 HEAD 口径以隔离标签效应）：

- 数据：`egfr_full_fingerprints.npz` 的 X/y（13,497×2048，行序与 clean CSV 一致，已验证 `smiles` 与 `pic50` 逐行相等）；split 索引读自 `data/processed/splits/egfr_full_{random,scaffold}.npz`。
- **训练池 = npz `train_idx ∪ valid_idx`**：已验证与 HEAD 版 04 自己生成的训练索引**逐索引相等**（random 10,797 = `train_test_split(0.8, random_state=42)`，scaffold 10,785 = `scaffold_split`；test 2,700/2,712 亦完全相等）。valid 并入训练池是为了与"修复前"数字可比（HEAD 指标即在此池上训练）。
- 剔除口径：把 2,722 个仅删失 SMILES 的索引从训练池和测试集**同时剔除**（labels 对其余化合物保持不变——仅删失化合物的 y 正是删失本身，其余化合物的 median 标签不受影响）。
- 模型：`RandomForestRegressor(random_state=42)`，同一 `PARAM_GRID`（n_estimators [300,500] × max_depth [None,20,40] × min_samples_split [2,5]，12 组），`GridSearchCV(scoring="r2", cv=KFold(5) 不 shuffle, n_jobs=-1)`。RF 侧 `n_jobs=1`（与 `n_jobs=-1` 结果完全等价，仅消除嵌套超额订阅）。
- "修复前"基准从 `git show HEAD:results/metrics_egfr_full.json` 读取（工作区副本可能已被后台任务改写）。
- 输出只写 `.scratch/h1_sensitivity/`，不碰 `results/` 与 `models/`。

## 对账结果

### 记录级匹配

| 项 | 数值 |
|---|---:|
| 旧 pull 行数 | 24,560 |
| 匹配到新 pull | 24,560（100%） |
| 未匹配 / 新增 | 0 / 0 |
| relation `=` | 19,126 |
| 删失 `<` / `<=` / `>` / `>=` | 2,034 / 87 / 3,154 / 8 |
| 其他非等号 `>>` / `~` | 11 / 1 |
| 空 relation | 139 |
| 删失份额（不含空 relation） | **21.5%**（5,283/24,560，与 PR #1 一致） |
| 删失份额（含空 relation） | 22.1% |

历史数据与今日 API 完全一致，且清洗后标签零差异——PR #1 的"在线测量"可以直接套用到历史 pull 上，不存在"历史记录身份不确定"的残差。

### 仅删失化合物（记录级核实）

- 清洗后化合物：13,497（与已提交 clean CSV 相同）
- **仅删失：2,722 = 20.17%**；PR #1 声称 2,718 ≈ 20.1%（差 4 个，即上述 `>>`/`~`/空 relation 边界）
- 仅删失化合物的最常见标签与 PR #1 一致：pIC50 5.0（596）、7.3（516）、6.0（357）、7.0（294）、4.0（261）——正是 `IC50 > 10000/100000 nM` 与 `IC50 < 50/100 nM` 这类检测限

**结论：PR #1 的 20.1% 在记录级得到核实（20.17%）。**

### 若按 `=` 策略修复，标签会怎么变

| 项 | 数值 |
|---|---:|
| 仅删失化合物（无等号记录，整体消失） | 2,722 |
| 其余标签会移动的化合物 | 896 |
| \|Δ\| 均值 / 最大 | 0.57 / 4.30 |
| 有符号中位数（25%/75% 分位） | +0.195（−0.093 / +0.559） |

即：修复不只是"删掉 20% 的化合物"，还有 896 个（约 8% 的保留化合物）标签会被改写，个别移动超过 4 个对数单位（混合有等号/删失记录的 SMILES，median 被删失端拉偏）。

### 两种过滤策略的差异

| 策略 | 保留行 | 保留化合物 |
|---|---:|---:|
| `standard_relation == "="`（主策略） | 19,106 | 10,775 |
| `pchembl_value.notna()` | 18,843 | 10,675 |

差异**全部是一个方向**：263 行（185 个化合物）relation 为 `=` 但 pChEMBL 为空，典型为 `data_validity_comment="Outside typical range"` 的高 IC50 精确测量（例：activity 32770，CHEMBL306988，500000 nM，见 codex 核验报告）；反向（有 pChEMBL 但非等号）为 **0**。对本数据集 `pchembl_value.notna()` 是 `=` 策略的严格子集。按 `=` 修复后 egfr_full 化合物数 13,497 → 10,775（−20.2%）。

## 敏感性实验结果

**状态：待补（后台任务 CPU 冲突，RF 重跑被推迟，由后续智能体在后台任务全部完成后执行 `.scratch/h1_sensitivity/rf_h1_sensitivity.py` 并回填本节）。**

已确定的设计数字（脚本启动日志）：

| split | 训练池 | 剔除 | 测试集 | 剔除 |
|---|---:|---:|---:|---:|
| random | 10,797 → 8,619 | 2,178 | 2,700 → 2,156 | 544 |
| scaffold | 10,785 → 待回填 | 待回填 | 2,712 → 待回填 | 待回填 |

修复前后指标对比（"修复前" = `git show HEAD:results/metrics_egfr_full.json`；"修复后"待回填）：

| split | 指标 | 修复前（HEAD） | 修复后（剔除仅删失） | Δ |
|---|---|---:|---:|---:|
| random | R² | 0.7592 | 待回填 | 待回填 |
| random | RMSE | 0.7093 | 待回填 | 待回填 |
| random | MAE | 0.4950 | 待回填 | 待回填 |
| scaffold | R² | 0.5926 | 待回填 | 待回填 |
| scaffold | RMSE | 0.8901 | 待回填 | 待回填 |
| scaffold | MAE | 0.6629 | 待回填 | 待回填 |

对比口径说明：修复前后测试集构成不同（random 例：2,700 → 2,156），Δ 同时包含"测试集构成变化"与"训练标签清洗"两个效应；若 Δ 很小，可解释为无论哪个方向都未颠覆排序。

## 对总体结论的初步判断（基于对账数字，非模型重测）

- H1 的机制与量级在记录级被证实：egfr_full 约 1/5 化合物的标签是检测限而非测量值，审查报告的数字不是测量假象。
- **推断**（未测量）：RF 与 GIN 见的是同一批标签，两者相对比较大概率不因 H1 修复而翻盘；但绝对 R²/RMSE、以及建立在标签上的不确定性/AD 校准数字，都应对外报告时注明"目标含约 20% 删失标签"。
- 若按 `=` 策略修复，数据集规模与全部下游统计（README 面板、显著性、时间切分、fairness）都需以 10,775 化合物口径重算——属 Phase C 数据刷新，本次不做。
- 在敏感性数字回填之前，不对"结论是否受影响"下最终判断；尤其 GIN/GINE/AttentiveFP、时间切分与多任务一概未测。

## 局限性

- 只覆盖 egfr_full；其余 6 个受影响靶点（a2a、abl1、herg、mpro、vegfr2、hivpr）未做记录级对账（PR #1 的在线测量仍是最优证据）。
- 敏感性 RF 尚未跑完；本报告的对比表是占位。
- 只测 RF；GNN 家族及所有下游实验（bootstrap 显著性、不确定性/AD、interpretability）未重测。
- 新 pull 是 2026-10-10 的 API 快照；历史记录身份依赖组合键（无 activity_id 留存）。但 100% 行匹配 + 标签 100% 复现使错配风险极低。
- `cv=5` 不 shuffle 是刻意复刻 HEAD 原始流程（L3 已指出该缺陷），本次为隔离标签效应不变更；RF `n_jobs` 取值只影响调度不影响结果。
- "修复后标签变化"统计沿用 02 的 median 聚合口径，未考虑 M1 盐型剥离等其他数据问题；`>>`（11 条）、`~`（1 条）、空 relation（139 条）一律按非等号处理（与 PR #1 口径差 4 个化合物）。
- 敏感性只评估"剔除仅删失化合物"一种较轻的修复格式；"全部标签改用等号记录重算"的更强修复未重训评估（其标签冲击见"若按 `=` 策略修复"一节）。
