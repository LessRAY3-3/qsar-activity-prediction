# README 重写数字底稿（C 阶段 3，2026-10-11，本机 m4）

**用途**：给 README 的最终重写提供"填数即可"的数字来源。本文件**只读汇总**，不改动 `README.md`、`docs/methodology.md`、`results/` 任何文件，未执行任何 git 操作。

**生成时间**：2026-10-11 约 13:15（本机时间）。此后 `results/` 仍在被后台任务写入（见 §0.3），引用本底稿前请复核文件 mtime。

**状态更新（2026-10-13，不改正文原貌）**：上表的 NEW/PENDING/STALE/NEEDS-RECALC
标记冻结在 2026-10-11 13:15，其中多处已被后续批次取代，引用"状态"前以下面为准：

- `results/significance/` 四个 summary 与全部 per-cell csv 已是新 schema
  （`n_boot` + `frac_boot_gin_better`，`p_one_sided` 在 `results/` 0 命中），
  `summary_time.csv` 已含刷新值（herg/time mean3 **+0.032304**），§6 的
  "全部 STALE，待 bootstrap 重跑" 已结（N1/N2 全 16 格 GINE bootstrap）；
- XGBoost 已按同池协议全 8 tag 重跑（16/16 `same_pool: true`，**11/16 ≤ RF**，
  worst `hivpr`/scaffold **−0.080**），§0.2 的 "rf_control NEEDS-RECALC" 已结；
- `vegfr2` RF 已于 2026-10-11 15:31 落盘（0.670/0.585），全表 "PENDING/旧"
  的 vegfr2 RF/panel/time/multitask/xgb 对照列已随之刷新；
- **release 状态**：`artifacts-v2` 已于 2026-10-12 发布（刷新 raw × 10 +
  `document_years.csv` + 26 个 RF 模型），2026-10-13 原位重传补入
  `processed_cache.tar.gz`（P4 快速路径，`--all`/`--cache` 可取）；
  `scripts/fetch_artifacts.py` 指向它，README 快速上手的 stale 警告块已撤
  （当前 `README.md` 中 `stale` 0 命中）；
- **2026-10-13（PR #6 复查 R1-R5 整改批次）**：R1 单 seed 依赖分析见
  `logs/2026-10-13_single_seed_dependence.log`（48 格 14 格条件性：5 个 GNN
  胜中 4 个、21 个 RF 胜中 1 个 = AFP hivpr/scaffold、9 个平局转 RF-significant）；
  R2 时间切分误差分解见 `logs/2026-10-13_time_split_mse_decomposition.log`
  （level/spread/correlation，残差 ≤9e-16，替代机制标签）；批次记录见
  `experiments/LOG.md` 末节 "2026-10-13 PR #6 review-remediation"；
  本文件状态行、`docs/reviews/fixes-2026-10-11.md` 测试数、
  `docs/interview.md` 调参日期（更正为 2026-09-24）同批同步；
- 仍开放（2026-10-13 复核后收窄）：vegfr2 y-randomization / learning
  curve / interpretability（PR #6 仍列其为 pre-refresh）。原先列在此的
  QC `figures/testset_hist_*.png` 与 `permutation_importance_vegfr2.csv`
  已刷新落盘（07 全 8 tag 重跑：7 tag 10-11、bace 10-12、vegfr2 10-13，
  `logs/2026-10-13_art_refresh.log`），7 张 `figures/learning_curve/*.png`
  已于 10-13 重绘为 ddof=0 窄带。

本底稿正文保持生成时原貌；批次级运行记录见 `experiments/LOG.md` **末节**
"2026-10-13 PR #6 review-remediation"（其下为 "2026-10-12
review-remediation campaign (PR #3 follow-up)"；该日志 newest-first，
最新条目在文件底部）。

**统一口径**（与 `experiments/time_split_summary.py`、`scripts/gnn_04_compare.py` 一致）：

- GIN 数值 = 3 seed（seed 42 / 1 / 2）**总体标准差**（`np.std(ddof=0)`），来源 `results/gnn_metrics_{tag}{,_seed1,_seed2}.json`。
- Δ(panel) = **3-seed mean R² − RF R²**（per-seed-mean 口径）。
- Δ(time) **ensemble Δ = R²(3 seed 逐分子平均预测) − R²(RF)**（ensemble 口径），由本底稿用 `results/rf_preds_{tag}_time.npz` + `results/gnn_preds_{tag}_time{,_seed1,_seed2}.npz` **独立复算**；两种 Δ 口径不可混读（见附录 B）。
- 状态标记：**NEW** = 新数据（删失过滤 + 母核标准化）产物；**PENDING** = 该数正在跑/未产出；**STALE** = 旧数据产物，必须重跑；**NEEDS-RECALC** = 派生文件里的 RF 对照列不是当前 RF。

---

## 0. 数据源与状态总览

### 0.1 新数据落盘时间（判定"新/旧"的锚点）

| 产物 | mtime | 说明 |
|---|---|---|
| `data/processed/*_fingerprints.npz` | 2026-10-10 19:51–20:21 | 刷新后的指纹/标签（egfr_full 19:51，a2a/hivpr/mpro 20:21，其余 22:40） |
| `data/processed/splits/*_{random,scaffold}.npz` | 2026-10-10 22:39–22:40；a2a/abl1/egfr_full/hivpr/mpro 于 **10-11 08:48** 重写 | 08:48 重写与 `.scratch/splits_backup/` 中 22:39 版 **md5 完全一致**（egfr_random 除外，见 §7.5） |
| `data/processed/splits/*_time.npz` | 2026-10-10 22:37–22:45 | egfr 22:37，其余 22:44–22:45 |
| `data/raw/document_years.csv` | 2026-10-08 12:13 | 4,147 个 document（4,148 行含表头）——README 中 "4,147 ChEMBL documents" **仍然成立** |

### 0.2 八 tag 产物状态矩阵

| tag | `results/metrics_{tag}.json`（RF） | `results/gnn_metrics_{tag}*.json`（GIN） | `results/xgboost/summary_{tag}.json` | `results/multitask/` | `results/uncertainty/` | `results/significance/` |
|---|---|---|---|---|---|---|
| a2a | NEW（10-11 00:43） | NEW（10-10 23:20） | NEW 数据 10-10 22:56 / **rf_control NEEDS-RECALC** | —（不在 mt 三靶点内） | NEW（n_test=274） | **STALE** |
| abl1 | NEW（10-11 07:35） | NEW（10-10 23:21） | NEW 数据 23:09 / **NEEDS-RECALC** | NEW（10-11 10:47–11:32） | NEW（424） | **STALE** |
| egfr | NEW（10-10 23:46） | NEW（10-11 09:34） | NEW 数据 23:21 / **NEEDS-RECALC** | — | NEW（912） | **STALE** |
| egfr_full | NEW（10-11 07:59） | NEW（10-10 23:25） | NEW 数据 23:27 / **NEEDS-RECALC** | NEW | NEW（2146） | **STALE** |
| herg | NEW（10-11 **11:49**） | NEW（10-10 23:23） | NEW 数据 23:26 / **NEEDS-RECALC** | — | NEW（1681） | **STALE** |
| hivpr | NEW（10-11 00:43） | NEW（10-10 23:20） | NEW 数据 23:03 / **NEEDS-RECALC** | — | NEW（515） | **STALE** |
| mpro | NEW（10-11 00:43） | NEW（10-10 23:21） | NEW 数据 23:17 / **NEEDS-RECALC** | — | NEW（682） | **STALE** |
| vegfr2 | **PENDING / 旧**（10-09 08:50，`04_train_and_evaluate.py` 10-11 10:37 起仍在跑 random） | **NEW**（10-10 23:24，与新 split 索引逐位一致） | 新数据 23:27 / rf_control = 同一份**旧** RF，等新 RF 落地后重算 | st_rf 列 **旧** | **STALE**（n_test=2360/2375 = 旧 test 规模） | **STALE** |

> 注意：任务背景说"7 个 tag 的 RF/GIN/…已是新结果"，实测 **GIN 八个 tag 全部是新数据**（含 vegfr2），只有 **vegfr2 的 RF** 未刷新。

### 0.3 生成本底稿时仍在运行的进程（会影响后续数字）

- `scripts/04_train_and_evaluate.py`（vegfr2 RF，pid 51758，10:37 起，watcher `.scratch/vegfr2_m4_watch.log` 13:11 仍在 `split=random+scaffold`，markers `0000--`：random/scaffold/metrics/time 均未完成）。
- `.scratch/run_aux_exp.sh` 三个辅助实验，tag 顺序 `a2a abl1 egfr egfr_full herg hivpr mpro`（**不含 vegfr2**）：
  - interpretability：13:07 DONE（7/7 tag）
  - learning_curve：13:08 RUNNING `tag=abl1`（a2a 已完成）
  - y_randomization：13:13 RUNNING `tag=abl1`（a2a 已完成）
- 日志：`logs/aux_m4_interpretability.log`、`logs/aux_m4_learning_curve.log`、`logs/aux_m4_y_randomization.log`；状态：`.scratch/aux_m4_status_*.txt`。

### 0.4 一致性校验结果（metrics ↔ preds ↔ splits）

| 校验 | 结果 |
|---|---|
| 7 tag RF `metrics_*.json` random/scaffold/time R²/RMSE vs `rf_preds_*` 复算 | **全部 \|Δ\| < 1e-6**，`test_idx` 与当前 splits **逐位相等** |
| 8 tag GIN `gnn_metrics_*` R² vs `gnn_preds_*` 复算（random/scaffold/time） | **全部 \|Δ\| < 1e-7**，`test_idx`/`y_true` 与 RF preds 一致 |
| vegfr2 RF `metrics_vegfr2.json` vs `rf_preds_vegfr2_*` | **不一致**（见 §7.1） |

---

## 1. 八 tag 主面板（random / scaffold）

**数据源**：RF = `results/metrics_{tag}.json` 的 `random`/`scaffold` 段；GIN = `results/gnn_metrics_{tag}{,_seed1,_seed2}.json` 同名段；n = `data/processed/{tag}_fingerprints.npz` 的 `smiles` 长度（= splits `n_total`）。

### 1.1 random 切分

| tag | n(新) | RF R² | RF RMSE | GIN R² (3-seed) | GIN RMSE (3-seed) | Δ = mean GIN − RF |
|---|---|---|---|---|---|---|
| `a2a` | 1,366 | 0.7100 | 0.6046 | 0.6214 ± 0.0087 | 0.6906 ± 0.0079 | **−0.0885** |
| `abl1` | 2,120 | 0.8222 | 0.6143 | 0.7669 ± 0.0099 | 0.7032 ± 0.0151 | **−0.0552** |
| `egfr` | 4,560 | 0.7214 | 0.7332 | 0.6646 ± 0.0036 | 0.8044 ± 0.0043 | **−0.0568** |
| `egfr_full` | 10,729 | 0.7289 | 0.6953 | 0.6670 ± 0.0060 | 0.7705 ± 0.0069 | **−0.0619** |
| `herg` | 8,403 | 0.6067 | 0.5975 | 0.6187 ± 0.0091 | 0.5882 ± 0.0070 | **+0.0120** |
| `hivpr` | 2,573 | 0.6922 | 0.8852 | 0.6316 ± 0.0033 | 0.9684 ± 0.0043 | **−0.0605** |
| `mpro` | 3,410 | 0.6017 | 0.6674 | 0.5722 ± 0.0149 | 0.6916 ± 0.0120 | **−0.0295** |
| `vegfr2` | 9,643 | **PENDING**（旧 0.7361） | 旧 0.6384 | 0.6055 ± 0.0041 | 0.7090 ± 0.0037 | **PENDING** |

### 1.2 scaffold 切分

| tag | n(新) | RF R² | RF RMSE | GIN R² (3-seed) | GIN RMSE (3-seed) | Δ = mean GIN − RF |
|---|---|---|---|---|---|---|
| `a2a` | 1,366 | 0.7025 | 0.5648 | 0.6539 ± 0.0249 | 0.6087 ± 0.0216 | **−0.0485** |
| `abl1` | 2,120 | 0.7144 | 0.6711 | 0.7367 ± 0.0268 | 0.6434 ± 0.0334 | **+0.0224** |
| `egfr` | 4,560 | 0.6492 | 1.0153 | 0.6124 ± 0.0134 | 1.0670 ± 0.0186 | **−0.0368** |
| `egfr_full` | 10,729 | 0.6081 | 0.8944 | 0.5881 ± 0.0056 | 0.9170 ± 0.0062 | **−0.0201** |
| `herg` | 8,403 | 0.3845 | 0.8882 | 0.3522 ± 0.0377 | 0.9108 ± 0.0264 | **−0.0323** |
| `hivpr` | 2,573 | 0.5380 | 0.9203 | 0.4574 ± 0.0098 | 0.9973 ± 0.0090 | **−0.0807** |
| `mpro` | 3,410 | 0.4332 | 0.6885 | 0.3221 ± 0.0488 | 0.7525 ± 0.0270 | **−0.1111** |
| `vegfr2` | 9,643 | **PENDING**（旧 0.6169） | 旧 0.7700 | 0.5825 ± 0.0025 | 0.6843 ± 0.0020 | **PENDING** |

### 1.3 面板结论（新数字）

- **random**：7/8 个 Δ 为负，唯一正值 `herg` **+0.0120**（旧 README 是 "+0.000 tie"）。
- **scaffold**：7/8 个 Δ 为负，唯一正值 `abl1` **+0.0224**（旧 README 是全部 8 个为负、`abl1` −0.013）。`mpro` 仍最差 **−0.1111**。
- 7 tag（暂不含 vegfr2）random Δ 均值 = **−0.0486**；scaffold Δ 均值 = **−0.0439**。
- 数据规模全线下滑（见附录 A.2）：`herg` 12,019 → 8,403、`egfr` 6,165 → 4,560、`egfr_full` 13,497 → 10,729 等。

### 1.4 该节重写 README 时需要回答的问题

1. `abl1`/scaffold ST-GIN **反超** RF（+0.0224，且 seed 值 0.774/0.712/0.724 有两个高于 RF 0.7144）——"8 个 scaffold Δ 全负"的旧断言**不再成立**，Finding 2 怎么改写？是否要等 bootstrap CI 再决定用词（±std 0.0268 与 Δ 0.0224 同量级）？
2. `herg`/random 由 "tie (+0.000)" 变为 **+0.0120**——旧的"7 负 1 平"要改成"7 负 1 正"吗？0.0120 是否小于 seed 噪声（±0.0091）？
3. 表格的 **n 列**必须换成新值（附录 A.2），`Data` 节的 "Unique compounds" 表（README:90-101）也要同步换；`egfr` 6165→4560 会影响 README:16、107-109 的叙述。
4. `results/multi_target/summary.csv`（README:60-62 声称的表源）**仍是 10-08 旧值**——重写时表源改指向本底稿/`metrics`+`gnn_metrics`，还是先重跑 `multi_target`？
5. `vegfr2` 行在新 RF 落地前**留空还是标 PENDING**？

---

## 2. 时间切分表（time split，8 tag 全部）

**口径**：

- RF time R²/RMSE ← `results/metrics_{tag}.json` 的 `time` 段（已与 `results/rf_preds_{tag}_time.npz` 复算逐位一致）。
- GIN time 3-seed mean±std ← `results/gnn_metrics_{tag}{,_seed1,_seed2}.json` 的 `time` 段（与 `gnn_preds_*_time*.npz` 复算一致）。
- **GIN ens R² = R²(3 个 seed 预测的逐分子平均)**，本底稿复算（与 `experiments/time_split_summary.py` 的 `gin_time_r2` 同源数据，ensemble 口径与 `paired_bootstrap.py` 的 `mean3` 一致）。
- **ensemble Δ = GIN ens R² − RF time R²**（题目指定口径）。**没有 CI**——`results/significance/summary_time.csv` 是旧数据，CI 见 §6 STALE。
- 切分规模/年份 ← `data/processed/splits/{tag}_time.npz`（经 `time_split_summary.time_split_stats()` 的口径读取）。

| tag | n_total | train/valid/test | test frac | test 年份 | RF time R² | RF time RMSE | GIN time 3-seed | GIN ens R² | **ensemble Δ = ens − RF** | Δ(per-seed mean − RF) |
|---|---|---|---|---|---|---|---|---|---|---|
| `a2a` | 1,366 | 421/143/802 | 0.587 | 2022–2025 | **−0.3012** | 1.1890 | −0.0226 ± 0.0594 | +0.0244 | **+0.3256** | +0.2786 |
| `abl1` | 2,120 | 1387/202/531 | 0.250 | 2021–2025 | **+0.5598** | 1.0288 | +0.2712 ± 0.1822 | +0.3240 | **−0.2358** | −0.2886 |
| `egfr` | 4,560 | 2987/340/1233 | 0.270 | 2013–2016 | +0.1230 | 1.1988 | +0.0881 ± 0.0731 | +0.1620 | **+0.0390** | −0.0349 |
| `egfr_full` | 10,729 | 6892/794/3043 | 0.284 | 2021–2025 | **−0.7363** | 1.5774 | −0.1759 ± 0.0522 | −0.1441 | **+0.5923** | +0.5604 |
| `herg` | 8,403 | 4961/1117/2325 | 0.277 | 2018–2024 | −0.0583 | 0.9776 | −0.0530 ± 0.0157 | −0.0260 | **+0.0323** | +0.0053 |
| `hivpr` | 2,573 | 1812/232/529 | 0.206 | 2004–2024 | −0.0271 | 1.7431 | +0.0287 ± 0.0761 | +0.0876 | **+0.1148** | +0.0558 |
| `mpro` | 3,410 | 1017/643/1750 | 0.513 | 2024–2025 | +0.0061 | 1.0509 | +0.0550 ± 0.0193 | +0.0903 | **+0.0842** | +0.0489 |
| `vegfr2` | 9,643 | 6325/1280/2038 | 0.211 | 2018–2025 | **PENDING**（旧 +0.0930 / 1.2292） | 旧 1.2292 | +0.0684 ± 0.0325 | **PENDING** | **PENDING** | PENDING |

- 7 tag（不含 vegfr2）panel 均值：RF time **−0.0620**，GIN ens **+0.0740**。
- 时间切分规模变化：`a2a` test frac 0.623→**0.587**、`mpro` 0.514→**0.513**（仍 >0.5，README 的 "62% / 51%" caveat 要改成 **59% / 51%**）。
- 旧表对照 `results/time_split/summary.csv`（10-09，**STALE**）：`a2a` RF time −0.2552→**−0.3012**，`herg` −0.0338→**−0.0583**，`egfr_full` −0.8909→**−0.7363**，`abl1` +0.2797→**+0.5598**。

### 2.1 herg 核验（题目要求）

| | RF time | GIN time mean | GIN ens | ensemble Δ |
|---|---|---|---|---|
| 旧（`results/time_split/summary.csv` / README:421,431） | −0.0338 | −0.0762 | +0.0419 | **+0.0757 ≈ +0.076** ✓ 与旧 README 一致 |
| **新（本底稿复算）** | **−0.0583** | **−0.0530** | **−0.0260** | **+0.0323** |

新数字下 herg 的"符号反转"故事变弱了：per-seed-mean Δ = **+0.0053**（旧 −0.042），ensemble Δ = **+0.0323**（旧 +0.0757）。两者同号（都是 GIN 略好），旧 README:431-435 "opposite signs from the same three runs" 的**worked example 已失效**，需另找例子（当前最接近的是 `egfr`：per-seed −0.0349 vs ensemble +0.0390，**异号**，可作新例子）。

### 2.2 该节重写 README 时需要回答的问题

1. 时间切分整段的显著性（"5 of 8 significant, all favour the GIN"）**必须等全量 bootstrap**；新点估计下 GIN ens 领先 6/8（`abl1` −0.2358 明显反向），结论框架要不要改？
2. `abl1` time RF **+0.5598**（旧 +0.280）是全表唯一"两模型都可用"的正例，GIN 却显著更低（−0.2358）——这与"GIN 退化更少"的标题**冲突**，标题 `the GIN degrades less, neither model holds up` 还成立吗？
3. `egfr` time 测试年份变成 **2013–2016**（旧 `results/time_split/summary.csv`（10-09，STALE）记为 2014–2023）。已核：`egfr` 与 `egfr_full` 共有 4,560 个 SMILES 的 year 有 4,507 个相同（53 个差 1–11 年，方向为 egfr_full ≤ egfr，属"更多 document 取更早"），年份映射**无错位**；`egfr` 全数据 max year = 2016（旧 6,165 版 max=2017，≥2017 的 41 个化合物已随删失过滤移除）。重写时要说明这个数据事实，别照抄旧年份。
4. `test_frac` 表述（"cumulating to ~20%"）要改成实际的 **0.21–0.28（a2a 0.59 / mpro 0.51）**；`a2a`/`mpro` caveat 数字要更新。
5. §2 表的 CI 列何时能填？依赖 §6 的全量 bootstrap。

---

## 3. XGBoost（`results/xgboost/`）

**数据源**：`results/xgboost/summary_{tag}.json` + `xgboost_{tag}.csv`（8 个 summary 全在，10-10 22:56–23:27，**在数据刷新之后跑，属新数据**；`data/processed` 最晚 22:40，splits 22:39–22:45）。3 seeds = [42, 1, 2]，xgboost 3.4.1。

**RF control 是否按新 RF 重算**：`rf_control` 列读的是"当时"的 `metrics_{tag}.json`。与**当前** metrics 对比：

| tag | split | XGB R² (mean) | XGB RMSE | summary 里的 `rf_control` | 当前 `metrics` R² | control − current | 判定 |
|---|---|---|---|---|---|---|---|
| a2a | random | 0.6883 | 0.6267 | 0.7259 | 0.70995 | +0.0160 | **NEEDS-RECALC** |
| a2a | scaffold | 0.6806 | 0.5852 | 0.6914 | 0.70246 | −0.0110 | **NEEDS-RECALC** |
| abl1 | random | 0.8190 | 0.6198 | 0.7919 | 0.82219 | −0.0303 | **NEEDS-RECALC** |
| abl1 | scaffold | 0.7076 | 0.6789 | 0.7255 | 0.71435 | +0.0111 | **NEEDS-RECALC** |
| egfr | random | 0.7316 | 0.7127 | 0.7411 | 0.72135 | +0.0197 | **NEEDS-RECALC** |
| egfr | scaffold | 0.5172 | 0.9976 | 0.5616 | 0.64920 | −0.0876 | **NEEDS-RECALC**（差 0.088，最大） |
| egfr_full | random | 0.6760 | 0.7601 | 0.7502 | 0.72886 | +0.0213 | **NEEDS-RECALC** |
| egfr_full | scaffold | 0.5704 | 0.9365 | **null** | 0.60815 | n/a（`delta_vs_rf_control` = **NaN**） | **NEEDS-RECALC** |
| herg | random | 0.5700 | 0.6247 | 0.6153 | 0.60669 | +0.0086 | **NEEDS-RECALC** |
| herg | scaffold | 0.3917 | 0.8830 | 0.4296 | 0.38447 | +0.0452 | **NEEDS-RECALC** |
| hivpr | random | 0.6897 | 0.8888 | 0.7354 | 0.69219 | +0.0432 | **NEEDS-RECALC** |
| hivpr | scaffold | 0.4748 | 0.9813 | 0.5522 | 0.53803 | +0.0141 | **NEEDS-RECALC** |
| mpro | random | 0.5982 | 0.6704 | 0.7302 | 0.60167 | +0.1286 | **NEEDS-RECALC**（最大绝对差） |
| mpro | scaffold | 0.4100 | 0.7024 | 0.5087 | 0.43325 | +0.0754 | **NEEDS-RECALC** |
| vegfr2 | random | 0.6175 | 0.6981 | 0.7361 | 0.7361（**旧**） | 0.0000 | 表面 "OK"，实为**两边同旧**，等新 RF 后重算 |
| vegfr2 | scaffold | 0.5897 | 0.6784 | 0.6169 | 0.6169（**旧**） | 0.0000 | 同上 |

**其它必须知道的两点**：

- **3 个 seed 的 XGB 结果完全相同**（`xgboost_a2a.csv` 三行 R²/RMSE/MAE 逐位一样，所有 tag 的 `r2_std = 0.0`）——seed 没有传进 `xgboost.XGBRegressor`（`summary` 的 `xgboost_params` 里无 `random_state`）。因此 "3-seed mean±std" 对 XGB 是**退化的**，README 不应把 XGB 写成"3 次重复"。
- `egfr_full`/scaffold 的 `rf_control` 是 `null`，`delta_vs_rf_control` 为 `NaN`——该格子无法直接引用。

**用新 RF 直接重算的 Δ（XGB − 新 RF，本底稿现算，可直接填）**：

| tag | random Δ | scaffold Δ |
|---|---|---|
| a2a | −0.0217 | −0.0219 |
| abl1 | −0.0031 | −0.0067 |
| egfr | +0.0102 | −0.1320 |
| egfr_full | −0.0529 | −0.0378 |
| herg | −0.0367 | +0.0072 |
| hivpr | −0.0025 | −0.0633 |
| mpro | −0.0035 | −0.0232 |
| vegfr2 | **PENDING**（旧值 −0.1187） | **PENDING**（旧值 −0.0272） |

### 3.1 该节重写 README 时需要回答的问题

1. XGB 是"强基线"还是"又一个被 RF 压住的模型"？新数字下 random 有 7/8 个 XGB ≤ RF（`egfr` 例外 +0.0102），scaffold 有 7/8 ≤ RF（`herg` 例外 +0.0072）——**两处翻转**，与旧叙述是否一致？
2. `egfr`/scaffold 差距 −0.1320（XGB 0.517 vs RF 0.649）是否要单独解释（树模型在 scaffold 外推上的差异）？
3. seed 无效（std=0）这件事要不要在 README 明说？还是先修 `experiments/xgboost_baseline.py` 再重跑？
4. `rf_control` 全线 NEEDS-RECALC：是**重跑 xgboost 脚本**还是只用上面"现算 Δ"表替换？`egfr_full` scaffold 的 null/NaN 怎么补？

---

## 4. 多任务（`results/multitask/summary.csv`，全表）

**数据源**：`results/multitask/summary.csv`（10-11 11:32，**新数据**——`mt_preds_*.npz` 的 per-tag test 长度 2146/2166/424/425/1929/1945 与当前 splits 逐位一致）。3 seeds（`mt_metrics{,_seed1,_seed2}.json`），2 arms（`mt_gin`、`pooled_st`）+ 2 单任务参照（`st_rf`、`st_gin`）。

| task | split | method | R² mean | R² std | n_seeds | Δ vs ST-RF |
|---|---|---|---|---|---|---|
| egfr_full | random | st_rf | 0.728861 | 0.0 | 1 | 0.0 |
| egfr_full | random | st_gin | 0.666990 | 0.005950 | 3 | −0.061872 |
| egfr_full | random | pooled_st | 0.450881 | 0.041380 | 3 | −0.277980 |
| egfr_full | random | mt_gin | 0.491060 | 0.013952 | 3 | −0.237802 |
| egfr_full | scaffold | st_rf | 0.608148 | 0.0 | 1 | 0.0 |
| egfr_full | scaffold | st_gin | 0.588063 | 0.005616 | 3 | −0.020085 |
| egfr_full | scaffold | pooled_st | 0.444317 | 0.006824 | 3 | −0.163831 |
| egfr_full | scaffold | mt_gin | 0.377189 | 0.043796 | 3 | −0.230960 |
| abl1 | random | st_rf | 0.822185 | 0.0 | 1 | 0.0 |
| abl1 | random | st_gin | 0.766937 | 0.009943 | 3 | −0.055248 |
| abl1 | random | pooled_st | 0.673197 | 0.017132 | 3 | −0.148988 |
| abl1 | random | mt_gin | 0.700163 | 0.008867 | 3 | −0.122022 |
| abl1 | scaffold | st_rf | 0.714351 | 0.0 | 1 | 0.0 |
| abl1 | scaffold | **st_gin** | **0.736726** | 0.026801 | 3 | **+0.022375** |
| abl1 | scaffold | pooled_st | 0.669286 | 0.034877 | 3 | −0.045065 |
| abl1 | scaffold | mt_gin | 0.691927 | 0.046424 | 3 | −0.022424 |
| vegfr2 | random | st_rf | 0.736112 | 0.0 | 1 | 0.0（**旧 RF，PENDING**） |
| vegfr2 | random | st_gin | 0.605490 | 0.004086 | 3 | −0.130622（**基于旧 RF，PENDING**） |
| vegfr2 | random | pooled_st | 0.308846 | 0.057050 | 3 | −0.427266（同上） |
| vegfr2 | random | mt_gin | 0.509636 | 0.010070 | 3 | −0.226476（同上） |
| vegfr2 | scaffold | st_rf | 0.616934 | 0.0 | 1 | 0.0（**旧 RF，PENDING**） |
| vegfr2 | scaffold | st_gin | 0.582503 | 0.002450 | 3 | −0.034431（同上） |
| vegfr2 | scaffold | pooled_st | 0.240202 | 0.005896 | 3 | −0.376732（同上） |
| vegfr2 | scaffold | mt_gin | 0.533200 | 0.013996 | 3 | −0.083734（同上） |

**池规模（新，本底稿按 `scripts/gnn_07_multitask.py` 同口径从三个 tag 的 fingerprints 计算）**：

- 总行数 = 10,729 + 2,120 + 9,643 = **22,492**（旧 README:27,992）
- 唯一 SMILES = **21,093**（旧 25,143）
- 出现在 >1 个靶点的唯一 SMILES = **1,193 = 5.66%**（旧 2,525 = 10.0%）；对应行数 2,592 = 11.52%

### 4.1 该节重写 README 时需要回答的问题

1. 标题 "more data does not rescue the GIN" 是否仍成立？新表里 **mt_gin 六格全负**（−0.0224 ~ −0.2378），**pooled_st 六格全负**——主结论**保持**；但 `abl1`/scaffold 出现 ST-GIN 反超 ST-RF（+0.0224），正文"Neither arm beats the single-task RF in any of the 6 cells"仍对，"both are worse than single-task GIN everywhere"也仍对（mt_gin < st_gin 六格）。
2. 池统计三个数（22,492 / 21,093 / 1,193=5.66%）必须替换 README:358-360；"10.0% 跨靶点重叠"这个卖点从 10.0% 掉到 **5.66%**，要不要解释（删失过滤更常发生在共享的多靶点条目上）？
3. `vegfr2` 四行的 Δ 全部依赖旧 ST-RF——**等新 RF 落地后重算这 4 个 delta**（其余 16 行可直接用）。
4. 旧表（README:366-373）与新表差异极大（例：`egfr_full`/random ST-RF 0.759→0.729、pooled 0.535→0.451、mt 0.530→0.491）——引用的 `mt_metrics.json` `_run.outdir` 显示产物来自 `/home/lessray/...`（win4070 远端，经 `.scratch/pull_gpuwave.sh multitask` 拉回），重写时是否要说明远端训练/本地复算的来源？

---

## 5. 可解释性 / 辅助实验

### 5.1 可解释性对齐（`results/interpretability/`）— **部分完成**

**数据源**：`logs/aux_m4_interpretability.log`（13:02:25–13:07:04，7 tag 全跑完，rc=0，**不含 vegfr2 → vegfr2 PENDING**）；产物 `results/interpretability/by_tag/{tag}/{alignment.json,findings.md}`，顶层 `alignment.json`/`findings.md` 是最后一个 tag（mpro）的副本。

| tag | 状态 | 结论（`by_tag/{tag}/findings.md`） |
|---|---|---|
| a2a | 完成但 **0 条记录**（"No molecule completed both methods"，chromone/thienopyrimidine family size=0） | 无数可写 |
| abl1 | 完成但 **0 条记录** | 无数可写 |
| egfr | 完成，5 分子（thienopyrimidine） | GNN top atoms 落在 RF top-bit 覆盖内 **mean 0.35**（0.25–0.57）；GNN 注意力落在 scaffold core **mean 0.48**；RF 复现 bit：1367/491/489（各 5/5） |
| egfr_full | 完成，5 分子（thienopyrimidine） | 覆盖 **mean 0.52**（0.25–0.67）；core **mean 0.21**；RF bit 1581/1452/690（5/5） |
| herg | 完成，5 分子（chromone） | 覆盖 **mean 0.68**（0.50–0.88）；core **mean 0.42**；RF bit 1514/744/1791（各 4/5） |
| hivpr | 完成但 **0 条记录** | 无数可写 |
| mpro | 完成，5 分子（chromone） | 覆盖 **mean 0.66**（0.14–1.00）；core **mean 0.74**；RF bit 1034（5/5）、1840（4/5）、935（3/5） |
| vegfr2 | **PENDING**（辅助实验 driver 不含该 tag） | — |

### 5.2 Y-randomization（`results/y_randomization/`）— **1/8 完成**

**数据源**：`logs/aux_m4_y_randomization.log`、`results/y_randomization/summary_a2a.json`（10-11 13:08，**新**）。

| tag | 状态 | 内容 |
|---|---|---|
| `a2a` | **NEW 完成**（16 行 csv） | random：real RF **0.7037** / real GIN **0.6223**；shuffled RF **−0.2135 ± 0.0390**（−0.2565/−0.2219/−0.1621），shuffled GIN **−0.0859 ± 0.0356**（−0.1187/−0.0365/−0.1027）；scaffold：real RF 0.6997 / real GIN 0.6298，shuffled RF **−0.3209 ± 0.0524**，shuffled GIN **−0.1403 ± 0.0582**。verdict：`ok: all shuffled-label means < 0.2` |
| `abl1` | **RUNNING**（13:13 起） | — |
| egfr / egfr_full / herg / hivpr / mpro | **PENDING**（driver 顺序执行） | — |
| `egfr` 旧文件 | `summary_egfr.json`、`y_randomization_egfr.csv` = **2026-10-03，STALE** | 旧 README:163-175 引用的 egfr 数字（−0.029/+0.002/+0.011、control 0.7414 vs 0.7466）**全部 STALE** |
| `vegfr2` | **PENDING**（driver 不含） | — |

### 5.3 Learning curve（`results/learning_curve/`）— **1/8 完成，且有规模问题**

**数据源**：`logs/aux_m4_learning_curve.log`、`results/learning_curve/summary_a2a.json` / `learning_curve_a2a.csv`（10-11 13:08，**新**）。

| tag | 状态 | 内容 |
|---|---|---|
| `a2a` | **NEW 完成**，但 **只有 n=500 一个点** | sizes 请求 `[500…4400]`，`n≥1000` 全部 `skipped (larger than train pool)`（random pool=982、scaffold pool=960）。random：RF **0.6431±0.0067** / GIN **0.5863±0.0326**；scaffold：RF **0.7230±0.0108** / GIN **0.6518±0.0086**（各 3 seed） |
| `abl1` | **RUNNING**（13:08 起） | — |
| 其余 5 个新数据 tag | **PENDING** | — |
| `egfr` / `egfr_full` 旧文件 | `summary_egfr.json`（10-03）、`summary_egfr_full.json`（10-04）= **STALE**；README:274 "n = 9,717, no crossover" 引用的是旧 egfr_full 曲线 | 旧 `learning_curve_{egfr,egfr_full}.csv` 仍在 |
| `vegfr2` | **PENDING**（driver 不含） | — |

### 5.4 Uncertainty / AD（`results/uncertainty/`）— **7 tag 新 + vegfr2 旧**

**数据源**：`results/uncertainty/summary_{tag}.json` + `summary_{tag}_gine.json`（GIN 家族 16 个 cell；另 16 个 `_gine` 为 GINE ensemble）。本底稿只汇总 `model_kind == "gin"` 的 16 格，其中 **vegfr2 的 n_test = 2360/2375 是旧 test 规模 → STALE**，其余 14 格 n_test 与新 splits 一致 → NEW。

| 指标（14 个新 cell，7 tag × 2 split） | 新值 | 旧 README 值（:453-463） |
|---|---|---|
| RF tree-std Spearman(std, \|err\|) 均值 | **0.345**（0.156–0.490） | ~0.45 |
| GIN 3-seed std Spearman 均值 | **0.147**（0.092–0.231） | ~0.17 |
| RF quintile RMSE 单调 cell | **6 / 14**（abl1-scaf, egfr×2, egfr_full-random, herg-random, hivpr-random） | 12 / 16 |
| GIN 单调 cell | **3 / 14** | — |
| AD 覆盖率（max Tanimoto ≥0.4）均值 | **0.964**（0.930–1.000） | ~97% |
| in-AD RF test R² 均值 | **0.638**（0.392–0.825） | ~0.67 |
| out-AD RF test R² 均值（12 格有值） | **−0.367**，最差 `egfr`/random **−1.451** | ~−0.19，最差 `hivpr`/random −1.36 |
| AD 过滤是否改变 P@100 | **15/16 格完全不变**（唯一变化 `herg`/random 0.65→0.66） | 0.000 in 16/16 |
| 报告阳性集合 {pred≥7} precision：全体 vs std≤中位 | **0.829 → 0.891** | 0.859 → 0.910 |
| 按低 std 排序的 P@100（均值） | **0.579** vs 按 RF 排序 **0.913** | 0.563 vs 0.922 |

（旧 README 的 `herg`/scaffold GIN in/out AD、以及"12/16 单调"这类**点数结论都变了**，方向未变：RF std 可用、GNN std 不可用。）

### 5.5 该节重写 README 时需要回答的问题

1. **可解释性**：8 个 tag 里 3 个（a2a/abl1/hivpr）**一条记录都没跑出来**（family size=0），4 个 tag 有数、vegfr2 PENDING。README 要不要从"举例说明"改成"4/8 tag 可复现、3/8 无合适母核家族"？`chromone`/`thienopyrimidine` 的家族查询是否需要调参？
2. **Y-randomization**：旧 README 用 egfr 具体 seed 值作证据，现在只有 **a2a** 是新数。是等 7 tag 跑完再写，还是先放 a2a？a2a 结论与旧结论同向（shuffled 均值 < 0，real control 0.7037 vs headline 0.7100，差 0.006 —— 与旧 0.7414 vs 0.7466 同量级，可复用"训练池差异"的解释）。
3. **Learning curve**：`a2a` 只剩 **n=500 单点**（train pool 太小），曲线叙事必须换 tag（旧叙事用的 egfr_full n=9,717 是 STALE）；要不要把 LC 的 sizes 降到与新池匹配（如 500/1000/…/pool）重跑？
4. **Uncertainty**：`12/16 单调` → **6/14** 这个弱化要不要如实写？"AD 过滤 P@100 16/16 不变" → 现在是 **15/16 不变**；`vegfr2` 那格是旧数据，等重跑。
5. vegfr2 是否需要单独补一轮辅助实验（`run_aux_exp.sh` 的 TAGS 明确排除了它）？

---

## 6. 显著性（`results/significance/`）— **全部 STALE，待 bootstrap 全量（B=10000）重跑**

**文件清单**（全部标 **STALE**，本底稿不引用其数值作为新结论）：

| 文件 | mtime | 说明 |
|---|---|---|
| `results/significance/summary.csv` | 2026-10-11 10:11 | 8 tag × random/scaffold × {seed42, mean3}，B=10000 |
| `results/significance/summary_time.csv` | 2026-10-11 10:11 | 8 tag × time × {seed42, mean3} |
| `results/significance/summary_gine.csv` | 2026-10-11 10:11 | GINE 面板 |
| `results/significance/summary_attentivefp.csv` | 2026-10-11 10:11 | AttentiveFP 面板 |
| `results/significance/summary_partial_gine.csv` | 2026-10-11 10:11 | 部分 GINE |
| `results/significance/{tag}_{split}[_gine|_attentivefp].csv` × 约 48 个 | 2026-10-08 ~ 10-11 10:11 | per-cell 明细 |

**为什么是 STALE**（mtime 新 ≠ 数据新）：这些表的 RF 参照在 10-11 **11:49**（herg 重跑）与**待跑**（vegfr2）之后就过期了。已核对 `results/gine_panel/summary.csv` / `results/afp_panel/summary.csv`（10-11 10:42，内含同源 bootstrap CI）的 `rf_r2` 列：

| tag | gine/afp `rf_r2`（random） | 当前 `metrics` | 一致性 |
|---|---|---|---|
| a2a | 0.709954 | 0.709954 | ✓ |
| abl1 | 0.822185 | 0.822185 | ✓ |
| egfr | 0.721352 | 0.721352 | ✓ |
| egfr_full | 0.728861 | 0.728861 | ✓ |
| hivpr | 0.692191 | 0.692191 | ✓ |
| mpro | 0.601669 | 0.601669 | ✓ |
| **herg** | **0.615285** | **0.606687** | ✗ **stale**（herg RF 10-11 11:49 才重跑） |
| **vegfr2** | 0.736112 | 0.736112（旧） | ⚠ 双方同旧，**PENDING** |

**旧 README 中依赖这些表、必须重新核对的断言**（全部 STALE，新值待 bootstrap）：

- 16 个 random/scaffold cell 只有 **4 个 CI 排除 0，全是 RF 赢**（`egfr`/random −0.031、`egfr_full`/random −0.033、`mpro`/scaffold −0.085、`vegfr2`/random −0.033）；GIN **0 胜**（README:243-246）。
- seed-42 反例：`abl1`/scaffold +0.030 [0.002, 0.060]、`vegfr2`/scaffold +0.028 [0.008, 0.047]（README:233-235）。**注意**：新 per-seed mean 下 `abl1`/scaffold = +0.0224，seed42 单值 +0.0596（0.7740 − 0.7144），两个口径都变号/变大，结论大概率翻转。
- GINE：**0/10 cell CI>0**，5 个显著偏 RF（README:248-252）。
- AttentiveFP：**1/16 胜 `herg`/scaffold +0.013 [+0.035, +0.092]**，8 个显著偏 RF，random 全负 −0.026~−0.103（README:336-340）。
- time：**5/8 显著且全部偏 GIN**（README:436-437）。
- 另：`results/time_split/summary.csv`（10-09）与 `results/comparison_{a2a,egfr,herg,hivpr,mpro}.csv`（10-07 23:44）、`results/multi_target/summary.csv`（10-08）都是旧值 → **STALE**；`comparison_{abl1,egfr_full,vegfr2}.csv`（10-11 11:32）的 RF 列取自当时 metrics，其中 vegfr2 仍是旧值。

### 6.1 该节重写 README 时需要回答的问题

1. 全量 bootstrap（B=10000，8 tag × 2 split + time + GINE + AFP）什么时候跑？**在跑完之前，README 所有 CI 数字都不能填**。
2. 新 per-seed Δ 已经出现 `abl1`/scaffold +0.0224、`herg`/random +0.0120——"GIN wins 0 cells"大概率不再成立，**Finding 1/2 的措辞要预留改写空间**。
3. herg 的 RF 变了之后，`gine_panel`/`afp_panel`/`significance` 三处 herg 行要**全部重算**（不只 bootstrap）。
4. 是否要在 methodology 里固定"点估计来自 panel 表、显著性来自 CI"的两口径规则（README:262-270 已写，需保留）。

---

## 7. 发现的数字矛盾与风险（重写前必看）

### 7.1 vegfr2 RF：三个文件互相打架（最高优先级）

| 文件 | mtime | random R² | scaffold R² | time test n | 判定 |
|---|---|---|---|---|---|
| `results/metrics_vegfr2.json` | 10-09 08:50 | 0.7361 | 0.6169 | （time n=2583） | **旧数据**（`models/rf_*_vegfr2.joblib` = 10-08） |
| `results/rf_preds_vegfr2_random/scaffold.npz` | 10-11 **11:32:51** | **0.8672** | **0.4147** | — | **混血，不可用**：由 `.scratch/regen_rf_preds_vegfr2.py` 用 **10-08 的旧模型** 打在 **新 split 索引**（n=1929/1945 与当前 splits 逐位相等）上 |
| `results/rf_preds_vegfr2_time.npz` | 10-11 11:32:52 | — | — | **2583（旧）** | 与当前 `vegfr2_time.npz` 的 2038 **对不上**，内容是旧 test 集 |
| `results/comparison_vegfr2.csv` | 10-11 11:32:51 | 0.7361 | 0.6169 | — | RF 列取自旧 metrics，GIN 列取自新 gnn_metrics → **同一行里新旧混用** |

- `regen_rf_preds_vegfr2.py` 自带的自检（`|pred − metrics| < 1e-3`）在这组数据上必然 **FAIL**（0.8672 vs 0.7361，差 0.13）；脚本还含 `assert np.array_equal(rt["test_idx"], ts["test_idx"])`，对 2583 vs 2038 也必然抛异常。
- **结论**：vegfr2 的 RF 数字（含 panel Δ、time Δ、multitask 的 4 个 delta、xgboost 的 rf_control、uncertainty 的 14 格之外那 2 格、gine/afp 的 vegfr2 行）**全部标 PENDING**，等 10:37 起跑的 `04_train_and_evaluate.py` 产出新 `metrics_vegfr2.json` 与新 `rf_preds_*`。**不要**使用 0.8672/0.4147 这两个数。

### 7.2 XGBoost `rf_control` 全线过期

8 个 summary 的 `rf_control` 与当前 `metrics` **无一相同**（最大差 `mpro`/random +0.1286），另有 `egfr_full`/scaffold = `null` → `delta_vs_rf_control = NaN`。**全部 NEEDS-RECALC**（现算 Δ 见 §3 末表）。另外 `a2a` 的 rf_control 0.7259 既不是旧 campaign 值 0.7330 也不是新值 0.7100，说明 summary 读取时 `metrics_a2a.json` 处于**中间状态**（XGB 10-10 22:56 跑，`metrics_a2a.json` 10-11 00:43 才定稿）。

### 7.3 派生表的新旧不一致（同一 README 里可能同时引用到）

| 派生表 | mtime | 问题 |
|---|---|---|
| `results/multi_target/summary.csv` | 10-08 | **全表旧**，README:49-62 面板表的数据源就是它 |
| `results/time_split/summary.csv` | 10-09 | **全表旧**，README:396-410 时间表的数据源 |
| `results/comparison_{a2a,egfr,herg,hivpr,mpro}.csv` | 10-07 23:44 | **全表旧**（RF+GIN 都旧） |
| `results/comparison_{abl1,egfr_full}.csv` | 10-11 11:32 | 新（与当前 metrics 一致） |
| `results/comparison_vegfr2.csv` | 10-11 11:32 | RF 旧 + GIN 新（**行内混用**） |
| `results/gine_panel/summary.csv`、`results/afp_panel/summary.csv` | 10-11 10:42 | 6 tag 新、**herg 旧**、vegfr2 双旧 |
| `results/uncertainty/summary_vegfr2*.json` / `vegfr2_*.csv` | 10-11 12:00 | mtime 新但 **n_test = 2360/2375 是旧 test 规模** |
| `results/error_analysis_vegfr2_{random,scaffold}.csv` | 10-08 | **旧**（其余 7 tag 10-11 12:56–12:57 已刷新） |

### 7.4 `results/error_analysis_*` 与 `scaffold_color_map_*`

7 个 tag 的 error analysis 已在 10-11 12:56–12:57 重生成（vegfr2 仍旧，见 §7.3）。若 README/methodology 要引用 chemotype 级失败分析（README:207-213），需确认所引数字来自 12:56 之后的文件。

### 7.5 `egfr` split 被改过一次

`.scratch/splits_backup/md5_before.txt`（10-11 08:48）记录的 `egfr_random.npz` = `66e9994c…`，与 backup 文件一致；当前文件 = `3c93aece…`（10-11 09:14 由 `.scratch/regen_rf_preds_egfr.py` 确定性重生成）。**当前 `metrics_egfr.json`（10-10 23:46）与当前 split 索引逐位一致**，所以主数字自洽；但 10-11 08:48 那一版 split 与之不同（原因未查）。XGB `egfr` 跑于 10-10 23:21，早于这次改动——严格说 `xgboost/summary_egfr.json` 的测试池**无法从产物侧证明**等于当前 split（其余 7 个 tag 的 split 自 10-10 22:39 后未变，可证）。若要求 bit 级可复现，`egfr` 的 XGB 需要重跑或补一次 n_test 核对。

### 7.6 其它小坑

- **XGB 3 seed 完全相同**（`r2_std = 0.0`）：seed 没传入模型（见 §3）。
- **`a2a` learning curve 只有 1 个点**（n≥1000 全部 skipped，train pool 仅 982/960）。
- **`mt_metrics.json` 的 `_run.outdir` = `/home/lessray/...`**（win4070 远端产物，经 `.scratch/pull_gpuwave.sh multitask` 拉回）——与本机产物混放，来源要写清。
- **`results/_archive-2026-10-09/_diag_tasks1/`**（10-04；2026-10-12 随 re-review 整改从 `results/_diag_tasks1/` 移入归档目录，同批移入的还有 `egfr` tuneA/B 的 metrics/preds/ckpt）与 `results/{afp_panel,fairness,fp_ablation,multi_target}` 等目录含更早的诊断/面板产物，引用前先看 mtime。
- `results/metrics_bace.json`（10-10 12:02）与 `rf_preds_bace_*`（原 10-08；**2026-10-12 重生成**，与 `metrics_bace.json` 复算一致）属 fallback 数据集，不在 8-tag 面板内。

---

## 附录 A

### A.1 GIN 逐 seed R²（`results/gnn_metrics_{tag}{,_seed1,_seed2}.json`，seed 顺序 42 / 1 / 2）

| tag | random | scaffold | time |
|---|---|---|---|
| a2a | 0.6115 / 0.6201 / 0.6326 | 0.6693 / 0.6189 / 0.6737 | −0.0540 / −0.0743 / +0.0606 |
| abl1 | 0.7805 / 0.7635 / 0.7569 | **0.7740** / 0.7120 / 0.7243 | 0.5283 / 0.1273 / 0.1581 |
| egfr | 0.6694 / 0.6608 / 0.6636 | 0.6308 / 0.6074 / 0.5991 | 0.0404 / 0.0325 / 0.1913 |
| egfr_full | 0.6588 / 0.6726 / 0.6696 | 0.5806 / 0.5896 / 0.5941 | −0.1562 / −0.2473 / −0.1242 |
| herg | 0.6078 / 0.6301 / 0.6181 | 0.3633 / 0.3015 / 0.3918 | −0.0531 / −0.0721 / −0.0338 |
| hivpr | 0.6360 / 0.6282 / 0.6307 | 0.4542 / 0.4473 / 0.4707 | 0.1346 / −0.0409 / −0.0075 |
| mpro | 0.5892 / 0.5529 / 0.5745 | 0.2573 / 0.3340 / 0.3751 | 0.0638 / 0.0283 / 0.0729 |
| vegfr2 | 0.6014 / 0.6111 / 0.6040 | 0.5855 / 0.5795 / 0.5825 | 0.0255 / 0.1042 / 0.0754 |

（`abl1`/time seed42 = 0.5283 远高于另两个 seed，std 0.1822——引用该格时要说明离散度。）

### A.2 数据集规模：新 vs 旧 README

| tag | 旧 n（README / `multi_target`） | 新 n | Δ |
|---|---|---|---|
| a2a | 1,746 | **1,366** | −380 |
| abl1 | 2,698 | **2,120** | −578 |
| egfr | 6,165 | **4,560** | −1,605 |
| egfr_full | 13,497 | **10,729** | −2,768 |
| herg | 12,019 | **8,403** | −3,616 |
| hivpr | 2,799 | **2,573** | −226 |
| mpro | 4,424 | **3,410** | −1,014 |
| vegfr2 | 11,797 | **9,643** | −2,154 |
| **合计** | 55,145 | **42,804** | −12,341 |

（2026-10-12 更正：合计行原写 45,145 / 42,796 / −2,349，与本表逐行求和不符；
新值 = 八行直接相加，且新 n 一列等于八个 ChEMBL tag 的
`data/processed/*_pic50_clean.csv` 行数之和，不含 `bace`。）

---

## 附录 B：两种 Δ 口径备忘（重写时别混）

- **panel Δ（§1）** = mean(3 seed R²) − RF R² —— 逐 seed 算 R² 再平均。
- **ensemble Δ（§2）** = R²(mean over seeds of ŷ) − RF R² —— 逐分子平均预测后再算一次 R²；ensemble 通常 ≥ per-seed mean，因此 ensemble Δ ≥ panel 式 Δ。
- 同一组数两种口径可以**异号**：新数据里 `egfr`/time 的 per-seed Δ = −0.0349 而 ensemble Δ = +0.0390（可作 README 的 worked example，替代已失效的 herg 例子）。
- bootstrap CI 永远在 **ensemble** 口径上算（`experiments/paired_bootstrap.py` 的 `mean3`），点估计表永远在 **per-seed mean** 口径上算 —— 这条规则 README:262-270 已写，重写时保留。
