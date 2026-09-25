# Experiments Log

Running log of the deep-dive experiments. Each entry: what ran, headline
numbers (3-seed mean +/- std unless noted), and the commit holding the
artifacts. Newest first. Maintained by M4 Air; results produced on M3 Max.

## 2026-09-25

### Fingerprint ablation (m4-006) — DELIVERED
- Commit: 862b3f4 (`results: fingerprint ablation r={2,3} bits={1024,2048}`)
- Question: how sensitive is the RF conclusion to the Morgan fingerprint
  parameters? Anchor r=2/2048 (the headline featurization) reproduced
  headline to 0.001 (random) / 0.004 (scaffold), well within 0.01.
- Results (test R2, 3-seed mean +/- std, delta vs anchor):
  - random:   r2/1024 0.7448 (-0.0008) | r2/2048 0.7456 (anchor)
              | r3/1024 0.7454 (-0.0002) | r3/2048 0.7489 (+0.0033)
  - scaffold: r2/1024 0.5468 (-0.0114) | r2/2048 0.5582 (anchor)
              | r3/1024 0.5264 (-0.0318) | r3/2048 0.5567 (-0.0015)
- Conclusions: random split is insensitive to fingerprint config (spread
  0.004). Scaffold exposes 1024-bit as the weak point (r3/1024 -0.032);
  with 2048 bits radius 2 and 3 are equivalent. The headline r=2/2048
  default is sound - and 2048 bits cannot be trimmed away.

### XGBoost baseline (m4-005) — DELIVERED
- Commit: a044dea (`results: xgboost baseline vs RF, 3 seeds`)
- Question: does a stronger gradient-boosted tree beat the RF ceiling on
  the same features (2048-bit Morgan r=2) and same 4438 paired pool?
- Results (test R2, 3 seeds): random XGB 0.7316 vs RF 0.7411 (delta -0.010);
  scaffold XGB 0.5172 vs RF 0.5616 (delta -0.044). XGBoost 3.4.1 hist/CPU
  was bit-identical across seeds (std=0).
- Conclusion: XGBoost with fixed generic hypers does NOT beat the tuned RF;
  it lags clearly on scaffold (novel chemotypes). The RF-vs-GIN story is
  not an artifact of a weak tree baseline.

### Y-randomization (m4-004) — DELIVERED
- Commit: be6d532 (`results: y-randomization, RF + GIN, 3 seeds`)
- Sanity test: train/valid labels permuted (3 seeds), test labels real.
  Controls (real labels, seed 42) must land on known baselines.
- Results (test R2): controls RF 0.7414/0.5598 (random/scaffold),
  GIN 0.7034/0.5244 - all in band. Shuffled: RF -0.177+/-0.028 / -0.152+/-0.012,
  GIN -0.006+/-0.017 / -0.023+/-0.007 - every shuffled mean is NEGATIVE
  (shuffled GIN early-stops by epoch 9).
- Conclusion: the reported signal is real; no leakage. Both models agree.

### Learning curve (m4-002) — DELIVERED
- Commit: 6ea16a6 (`results: learning curve n=500-4400, RF vs GIN, 3 seeds`)
- Question: how do RF / GIN scale with training-set size; is either saturated?
- Protocol: test/val/hyperparameters frozen; only n varies (500..4400,
  9 points); RF and GIN trained on identical molecules per (n, seed).
- Results (test R2, 3-seed mean +/- std):
  - random:   RF 0.541+/-0.007 (n=500) -> 0.741+/-0.000 (n=4400);
              GIN 0.400+/-0.035 -> 0.701+/-0.011
  - scaffold: RF 0.438+/-0.018 -> 0.562+/-0.002;
              GIN 0.389+/-0.014 -> 0.526+/-0.020
- Conclusions: RF saturates early (flat from ~n=2000); GIN still climbing
  at n=4400; scaffold gap narrows to 0.036, random gap 0.040. More data
  likely helps the GIN more than the RF.
- Note: curve RF tops slightly below headline (0.741 vs 0.747) by design -
  the paired pool is the 4438-molecule GNN train pool; headline RF trained
  on the full 4932.

### Batch-1 scripts ready (task books m4-004/005/006 posted)
- Commit: 82bebd0 - y_randomization.py, xgboost_baseline.py,
  fingerprint_ablation.py (+ xgboost==3.4.1 pin, libomp)
- Smoke-tested on M4 Air (incl. one dataset-y alignment bug caught and
  fixed before dispatch). M3 Max executing; results pending.

## 2026-09-24

### Reference baseline (main @ 8807296)
- EGFR, 6165 unique compounds; test = 20%; splits persisted under
  data/processed/splits/.
- RF (Morgan r=2/2048, grid-tuned): random R2=0.747, scaffold R2=0.562.
- GIN (MoleculeNet-style, seed 42): random 0.691 / scaffold 0.544;
  3-seed means 0.690+/-0.001 / 0.526+/-0.016.
- Headline finding: on this dataset the GIN does not beat fingerprint+RF;
  both lose ~0.2 R2 from random to scaffold split.
