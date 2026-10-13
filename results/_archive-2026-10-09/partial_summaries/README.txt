partial_summaries/ -- what these four files are (written 2026-10-13, PR #5)
==========================================================================

The directory lives under results/_archive-2026-10-09/, but NONE of the
four files is pre-refresh (2026-10-09-vintage) material.  They are
partial-run outputs of experiments/paired_bootstrap.py (the guard that
keeps a run which does not cover all 8 tags away from the campaign
summary/figure), parked here on 2026-10-12 12:37 together with the rest
of the archive sweep.

  file                                  last written        vintage
  ------------------------------------  ------------------  ------------------
  summary_partial_attentivefp.csv       2026-10-12 09:40   2026-10-12 partial
  ci_panel_partial_attentivefp.png      2026-10-12 09:40   2026-10-12 partial
  summary_partial_gine.csv              2026-10-11 19:48*  2026-10-11 partial
  ci_panel_partial_gine.png             2026-10-11 19:48*  2026-10-11 partial
  (* mtime = last checkout; experiments/LOG.md:40 dates
     summary_partial_gine.csv to 2026-10-11 17:16, during the refresh
     campaign itself, i.e. post-refresh numbers.)

So: two of the four are 2026-10-12 partial-run products, two are
2026-10-11 ones -- all four are post-refresh.  The folder name is an
archive label, not a date stamp for its contents.

Contents
--------
* summary_partial_attentivefp.csv - 6 rows (egfr random, egfr scaffold,
  abl1 scaffold) x 2 pairs (seed42, mean3), n_boot 10000.  The two
  abl1/scaffold rows are the 2026-10-12 N3 update and are identical to
  the current campaign file results/significance/summary_attentivefp.csv.
  The four egfr rows are OLDER than the campaign file and disagree with
  it by up to 0.024 in d_r2 (e.g. egfr/scaffold mean3 -0.0289 here vs
  -0.0051 in the campaign summary) -- they are leftovers of an earlier
  partial egfr run, not the values any current document should cite.
* ci_panel_partial_attentivefp.png - figure rendered from that 6-row
  summary during the 2026-10-12 09:40 partial run.
* summary_partial_gine.csv - 20 rows (a2a, egfr, egfr_full, hivpr, mpro)
  x 2 pairs; every row is equal to the matching row of the current
  campaign file results/significance/summary_gine.csv (deterministic
  bootstrap, B=10000, seed 42, same inputs).
* ci_panel_partial_gine.png - figure rendered from that 20-row summary.

Why they are here
-----------------
They are superseded by the campaign files in results/significance/
(summary_attentivefp.csv, summary_gine.csv) and by
figures/significance/ci_panel_with_{afp,gine}.png, which is why they were
moved out of results/significance/ rather than deleted: they document what
the partial-run guard produced.  Nothing in this folder should be quoted
as a result; read the campaign summaries instead.

See also: /tmp/artifact_moves.md (PR #5 move log), and the run records
logs/2026-10-12_gine_full_panel_bootstrap.log and
logs/2026-10-12_summary_attentivefp_abl1_update.log.
