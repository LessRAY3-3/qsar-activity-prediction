# Experiments Log

Running log of the deep-dive experiments. Each entry: what ran, headline
numbers (3-seed mean +/- std unless noted), and the commit holding the
artifacts. Newest first. Maintained by M4 Air; results produced on M3 Max.

## 2026-09-25

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
