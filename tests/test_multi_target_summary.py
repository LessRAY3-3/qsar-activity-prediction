"""multi_target_summary: verdicts are labelled point estimates, no CI.

print_conclusion must never present a point-estimate lead as a
significant one (6c1e3bb, PR #5 Low "clear lead without a CI"): the
verdict line carries the "(point estimates, no CI)" qualifier, and the
single-seed (seed-42) reference column is reported as a reference only,
"not used for any verdict". Independent of the notebook byte-identity
test (which fails on ANY script edit; these pin the wording contract).
"""
def _row(tag, d_random, d_scaffold, seed42_random, seed42_scaffold):
    rf_r, rf_s = 0.60, 0.55
    return {
        "tag": tag, "family": "kinase", "n_molecules": 100,
        "rf_r2_random": rf_r, "gin_r2_mean_random": rf_r + d_random,
        "gin_r2_std_random": 0.01, "d_r2_random": d_random,
        "rf_r2_scaffold": rf_s, "gin_r2_mean_scaffold": rf_s + d_scaffold,
        "gin_r2_std_scaffold": 0.02, "d_r2_scaffold": d_scaffold,
        "d_r2_seed42_random": seed42_random,
        "d_r2_seed42_scaffold": seed42_scaffold,
    }


def test_verdict_line_carries_point_estimate_no_ci_qualifier(load_experiment,
                                                             capsys):
    """Mutant killed: a verdict without the no-CI qualifier.

    A +0.100 lead here has no bootstrap behind it; the console verdict
    must say "(point estimates, no CI)" so it cannot be read as one of
    the paired-bootstrap significant cells.
    """
    mod = load_experiment("multi_target_summary")
    rows = [
        _row("t1", 0.10, 0.07, 0.09, 0.05),
        _row("t2", -0.02, 0.00, -0.01, 0.03),
        _row("t3", -0.05, -0.04, -0.06, -0.03),
    ]
    mod.print_conclusion(rows)
    out = capsys.readouterr().out

    assert "point estimates, no CI" in out
    # the lead itself is still reported, under the no-CI label
    assert "clear lead on 2 cell(s): t1/random, t1/scaffold" in out
    # the seed-42 column would add a scaffold win for t2 -- reference only
    assert "not used for any verdict" in out
    assert "t2" in out


def test_seed42_reference_absent_when_column_disagrees_nothing(
        load_experiment, capsys
):
    """No seed-42 scaffold win -> no reference line is printed at all."""
    mod = load_experiment("multi_target_summary")
    rows = [_row("t1", 0.10, 0.07, -0.01, -0.02)]
    mod.print_conclusion(rows)
    out = capsys.readouterr().out
    assert "not used for any verdict" not in out
    assert "point estimates, no CI" in out
