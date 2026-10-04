# ep300 confirmation — egfr_full GIN（100 → 300 epoch 上限）

日期：2026-10-04 ｜ 机器：m3 ｜ 脚本：`scripts/gnn_03_train_gin.py`（未改动）
数据：`data/processed/egfr_full_graphs.npz` + `data/processed/splits/egfr_full_{random,scaffold}.npz`
配置：`QSAR_TAG=egfr_full`，hidden 128 / 4 layers / dropout 0.2 / lr 1e-3 / bs 128 / patience 20，`--epochs 300`
判定阈值：|ΔR2| ≥ 0.01 视为 material
日志：`logs/ep300_confirm.log`（第一轮）、`logs/ep300_repair.log`（补齐轮）

## 6-run 对比表

R2_ep300 / best_epoch_ep300 取自 `results/gnn_metrics_egfr_full{,_seed1,_seed2}_ep300.json`（可由 results/ 复算）；
R2_base / best_epoch_base 取自 `results/gnn_metrics_egfr_full{,_seed1,_seed2}.json`。

| split | seed | R2_base | R2_ep300 | Δ | material? | best_epoch_base | best_epoch_ep300 |
|---|---|---|---|---|---|---|---|
| random | 42 | 0.7006 | 0.7147 | +0.0141 | 是 | 95 | 115 |
| random | 1 | 0.7049 | 0.7040 | −0.0009 | 否 | 93 | 80 |
| random | 2 | 0.7141 | 0.7113 | −0.0028 | 否 | 99 | 105 |
| scaffold | 42 | 0.5600 | 0.5277 | −0.0323 | 是 | 100 | 197 |
| scaffold | 1 | 0.5689 | 0.5186 | −0.0503 | 是 | 98 | 153 |
| scaffold | 2 | 0.5440 | 0.5748 | +0.0308 | 是 | 94 | 127 |

### 3-seed mean

| split | mean R2_base | mean R2_ep300 | Δmean | material? |
|---|---|---|---|---|
| random | 0.7065 | 0.7100 | +0.0035 | 否 |
| scaffold | 0.5576 | 0.5404 | −0.0173 | 是 |

## 重复性：同 seed 同配置两次 ep300 的差异

第一轮 6 次的 seed42/seed1 metrics 被 seed2 覆盖（三轮 `--suffix` 相同 → 同一文件名），数值仅存于日志。
补齐轮按不同 suffix 重跑后，两轮对照如下（seed2 为同一文件，无重跑）：

| split | seed | 第一轮（日志） | 第二轮（文件） | 差 |
|---|---|---|---|---|
| random | 42 | 0.7123 | 0.7147 | +0.0024 |
| random | 1 | 0.7201 | 0.7040 | −0.0161 |
| scaffold | 42 | 0.5473 | 0.5277 | −0.0196 |
| scaffold | 1 | 0.5463 | 0.5186 | −0.0277 |

同 seed 重跑漂移最大 0.0277，远超预设的 0.001 容差，且与 epoch 预算效应同量级。
按第一轮数值，scaffold 3-seed mean Δ = −0.0015（不 material）；按文件数值为 −0.0173（material）。
两者之差来自 run-to-run 非确定性（CUDA scatter/atomics + early-stop 选点放大），不是 epoch 上限本身。

## 结论

1. **逐 seed 双向漂移**：6 个点中 4 个 |Δ| ≥ 0.01，方向不一致（random/42 升、scaffold/42 与 scaffold/1 降、scaffold/2 升），
   random/1 与 random/2 基本不变。没有一致的单向系统偏移。
2. **3-seed mean**：random +0.0035，不 material；scaffold −0.0173，达到 material 阈值，
   但该值落在同 seed 重跑漂移（最大 0.0277）之内，单次补齐轮不足以把 mean 效应与非确定性区分开。
3. **训练预算截断成立**：基线 best_epoch 全部落在 93–100，即全部顶到/逼近 100-epoch 上限；
   放宽到 300 后 best_epoch 为 80–197（6 个中 5 个越过 100），早停 epoch 100–217，
   没有一次跑满 300。说明原 100-epoch 上限对绝大多数 run 是紧约束，大 n 的 GIN 数字确实被预算截断。
4. **但截断解除后的 R2 变化被 run 噪声淹没**：逐 seed 双向、mean 量级与重跑噪声相当。
   若要量化 epoch 预算效应，需要每 seed 多次重复取均值，而非单次对比。
