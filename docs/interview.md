# Interview preparation notes

This file is interview-prep material distilled from this project's
results. It is deliberately kept **separate from the technical
documentation** (the top-level `README.md` and
[`docs/methodology.md`](methodology.md)) — use it to rehearse answers,
not as a source of record; every claim below is backed by the tables and
figures in the README and the run ledger in `experiments/LOG.md`.

1. **Why pIC50** - order-of-magnitude target standardisation.
2. **Why scaffold split** - random splits leak analogues across
   train/test and inflate scores; scaffold splits emulate new-chemotype
   prediction. This project's EGFR numbers show the effect directly
   (0.747 random vs 0.562 scaffold).
3. **Honest importance** - MDI is biased under correlated features; all
   substructure claims were validated with permutation importance on the
   held-out set before being made.
4. **Industry use** - first stage of virtual screening: in silico triage of
   large libraries before experimental validation, cutting wet-lab cost by
   orders of magnitude.
5. **Controlled GNN vs RF comparison** - same molecules, splits verified
   bit-identical (saved RF models reproduce the original metrics to 7
   decimals on the regenerated indices). Result: the GIN does *not* beat
   fingerprints here - on ~4.4k training molecules, message passing buys
   nothing over circular substructure counts, and both drop ~0.2 R2 on
   novel scaffolds.
6. **Negative results reported** - a scheduler/dropout tuning pass
   improved validation but degraded test (validation-set selection noise);
   the simplest recipe was kept. GIN scaffold-split R2 varies +-0.02
   across seeds - single-seed GNN comparisons are meaningless.
7. **Failure-mode analysis** - the two models' 50 worst molecules overlap
   ~10-12x more than chance (25 random / 21 scaffold shared, vs ~2 expected;
   Spearman ~0.52): many failures are
   molecule-intrinsic (assay censoring/noise), yet specific chemotypes
   fail asymmetrically (chromones hurt the GIN more, thienopyrimidines
   hurt RF more) - fingerprints and graphs extrapolate differently.
