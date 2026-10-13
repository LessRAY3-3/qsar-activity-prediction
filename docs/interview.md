# Interview preparation notes

This file is interview-prep material distilled from this project's
results. It is deliberately kept **separate from the technical
documentation** (the top-level `README.md` and
[`docs/methodology.md`](methodology.md)) — use it to rehearse answers,
not as a source of record; every claim below is backed by the tables and
figures in the README and the run ledger in `experiments/LOG.md`.
Numbers are the refreshed (2026-10-11 refresh / 2026-10-12 re-run)
values; anything marked **pre-refresh** is from the 6,165-compound `egfr`
pull that the refresh replaced and should be quoted only as history.

1. **Why pIC50** - order-of-magnitude target standardisation.
2. **Why scaffold split** - random splits leak analogues across
   train/test and inflate scores; scaffold splits emulate new-chemotype
   prediction. This project's EGFR numbers show the effect directly
   (refreshed: 0.7214 random vs 0.6492 scaffold; **pre-refresh**: 0.747
   vs 0.562).
3. **Honest importance** - MDI is biased under correlated features; all
   substructure claims were validated with permutation importance on the
   held-out set before being made.
4. **Industry use** - first stage of virtual screening: in silico triage of
   large libraries before experimental validation, cutting wet-lab cost by
   orders of magnitude.
5. **Controlled GNN vs RF comparison** - same molecules, splits verified
   bit-identical (saved RF models reproduce the original metrics to 7
   decimals on the regenerated indices). Result: the GIN does *not* beat
   fingerprints here - on the refreshed `egfr` pool (3,648 training
   molecules for the headline RF, 3,283 for the GNN train pool; the
   pre-refresh run had ~4.4k), message passing buys nothing over
   circular substructure counts, and both give up accuracy on novel
   scaffolds (refreshed RF 0.7214 -> 0.6492, GIN 0.665 -> 0.612;
   pre-refresh the drop was ~0.2 R2, 0.747 -> 0.562).
6. **Negative results reported** - a scheduler/dropout tuning pass
   (**pre-refresh**: the archived 2026-10-09 tuning run, artifacts under
   `results/_archive-2026-10-09/`; see `docs/methodology.md`,
   "Hyperparameters: an honest negative result") improved validation but
   degraded test (validation-set selection noise); the simplest recipe
   was kept. GIN scaffold-split R2 varies +-0.02
   across seeds - single-seed GNN comparisons are meaningless.
7. **Failure-mode analysis** - the two models' 50 worst molecules overlap
   ~9x more than chance (refreshed: 25 random / 25 scaffold shared of 50,
   vs 2.7 expected; per-molecule |error| Spearman 0.50 random / 0.59
   scaffold; **pre-refresh**: 25 / 21, Spearman ~0.52): many failures are
   molecule-intrinsic (assay noise - censored records are filtered out of
   the labels now), yet specific chemotypes still fail asymmetrically
   (thienopyrimidines hurt RF more: 47 scaffold-test molecules at mean
   |error| 1.32 vs the GIN's 1.02; the chromone series that hurt the GIN
   **pre-refresh** - n=37 at 1.12 vs 0.64 - is gone from the refreshed
   test set, its query matches 0 of 912) - fingerprints and graphs
   extrapolate differently.
