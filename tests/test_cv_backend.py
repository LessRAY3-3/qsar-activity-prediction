"""L3 shuffled CV and the QSAR_CV_BACKEND switch (scripts/04_train_and_evaluate.py).

Review L3: the grid-search CV must be KFold(5, shuffle=True, random_state=42).
Scaffold/time train pools are ordered by scaffold group or SMILES, so an
unshuffled KFold would put whole chemotype blocks into single folds.

The tests capture the cv object actually handed to GridSearchCV and check
(1) the shuffle/seed constants, and (2) that the fold assignment differs from
the unshuffled one -- an unshuffled mutant produces exactly the contiguous
fold sets and must fail.

QSAR_CV_BACKEND selects the joblib backend wrapping GridSearchCV.fit:
unset/empty -> joblib default (loky), ``loky_nommap`` -> loky without
memmapping (max_nbytes=None), any other value -> that backend passed to
joblib.parallel_backend.  An unknown value must raise a clear ValueError
BEFORE any fitting happens.
"""
import numpy as np
import pytest
from sklearn.model_selection import KFold

N = 60
TRAIN_N = 48
FIG_NAME = "pred_vs_actual_random_synth.png"


def _prepare(mod, monkeypatch, tmp_path):
    """Shrink the grid to one config, spy on GridSearchCV/parallel_backend.

    Returns (captured, run): captured holds the cv object handed to
    GridSearchCV, every parallel_backend() call, and the joblib backend
    active inside each fit; run() executes one run_split on tiny data.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 10))
    y = X[:, 0] * 2.0 + rng.normal(0.0, 0.5, N)
    smiles = [f"mol_{i:03d}" for i in range(N)]
    train_idx = np.arange(TRAIN_N)
    test_idx = np.arange(TRAIN_N, N)
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    # one grid config -> the 5-fold CV still runs, but finishes in ms
    monkeypatch.setattr(
        mod, "PARAM_GRID",
        {"n_estimators": [5], "max_depth": [None], "min_samples_split": [2]},
    )

    captured = {"cv": None, "pb_calls": [], "fit_backends": []}

    real_gs = mod.GridSearchCV

    def gs_factory(*args, **kwargs):
        captured["cv"] = kwargs.get("cv")
        kwargs["n_jobs"] = 1                 # keep the spy run single-process
        gs = real_gs(*args, **kwargs)
        real_fit = gs.fit

        def fit_spy(X_, y_=None, **fit_kwargs):
            from joblib.parallel import get_active_backend
            backend, _n_jobs = get_active_backend()
            captured["fit_backends"].append(type(backend).__name__)
            return real_fit(X_, y_, **fit_kwargs)

        gs.fit = fit_spy
        return gs

    monkeypatch.setattr(mod, "GridSearchCV", gs_factory)

    real_pb = mod.parallel_backend

    def pb_spy(*args, **kwargs):
        captured["pb_calls"].append((args, kwargs))
        return real_pb(*args, **kwargs)

    monkeypatch.setattr(mod, "parallel_backend", pb_spy)

    def run():
        return mod.run_split(X, y, smiles, train_idx, test_idx, "random",
                             str(fig_dir / FIG_NAME), "Synthetic kinase")

    return captured, run


def test_grid_cv_is_kfold_five_shuffled_with_seed_42(load_script, monkeypatch,
                                                     tmp_path, capsys):
    mod = load_script("04_train_and_evaluate")
    # production constants the docstring/LOG.md promise (L3)
    assert mod.CV_FOLDS == 5
    assert mod.RANDOM_STATE == 42

    captured, run = _prepare(mod, monkeypatch, tmp_path)
    run()
    capsys.readouterr()                       # drop GridSearchCV chatter

    cv = captured["cv"]
    assert isinstance(cv, KFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42

    dummy = np.zeros((N, 1))
    got = [set(te) for _, te in cv.split(dummy)]
    want_shuffled = [set(te) for _, te in
                     KFold(5, shuffle=True, random_state=42).split(dummy)]
    want_unshuffled = [set(te) for _, te in KFold(5, shuffle=False).split(dummy)]
    assert got == want_shuffled
    # a shuffle=False mutant regresses to contiguous folds -> must fail here
    assert got != want_unshuffled


def test_default_env_runs_fit_under_joblib_default_backend(
        load_script, monkeypatch, tmp_path, capsys):
    mod = load_script("04_train_and_evaluate")
    monkeypatch.delenv("QSAR_CV_BACKEND", raising=False)

    captured, run = _prepare(mod, monkeypatch, tmp_path)
    run()
    capsys.readouterr()

    # no backend context requested -> joblib's default loky stays active
    assert captured["pb_calls"] == []
    assert captured["fit_backends"] == ["LokyBackend"]


def test_env_threading_switches_cv_backend(load_script, monkeypatch,
                                           tmp_path, capsys):
    mod = load_script("04_train_and_evaluate")
    monkeypatch.setenv("QSAR_CV_BACKEND", "threading")

    captured, run = _prepare(mod, monkeypatch, tmp_path)
    run()
    capsys.readouterr()

    assert captured["pb_calls"] == [(("threading",), {})]
    # the context really wraps GridSearchCV.fit, not just the call site
    assert captured["fit_backends"] == ["ThreadingBackend"]


def test_env_loky_nommap_uses_loky_without_memmap(load_script, monkeypatch,
                                                  tmp_path, capsys):
    mod = load_script("04_train_and_evaluate")
    monkeypatch.setenv("QSAR_CV_BACKEND", "loky_nommap")

    captured, run = _prepare(mod, monkeypatch, tmp_path)
    run()
    capsys.readouterr()

    assert captured["pb_calls"] == [(("loky",), {"max_nbytes": None})]
    assert captured["fit_backends"] == ["LokyBackend"]


def test_env_invalid_backend_raises_clear_error_before_fit(
        load_script, monkeypatch, tmp_path, capsys):
    mod = load_script("04_train_and_evaluate")
    monkeypatch.setenv("QSAR_CV_BACKEND", "not_a_backend")

    captured, run = _prepare(mod, monkeypatch, tmp_path)
    with pytest.raises(ValueError) as excinfo:
        run()
    capsys.readouterr()

    msg = str(excinfo.value)
    assert "not_a_backend" in msg          # names the offending value
    assert "expected one of" in msg        # and the legal options
    # the error fires before any fit: nothing fitted, no figure written
    assert captured["pb_calls"] == [(("not_a_backend",), {})]
    assert captured["fit_backends"] == []
    assert not (tmp_path / "figures" / FIG_NAME).exists()
