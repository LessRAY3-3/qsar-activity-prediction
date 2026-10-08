"""Invariants of qsar_common.scaffold_split that the whole pipeline relies on."""
import numpy as np
from sklearn.model_selection import train_test_split

from qsar_common import RANDOM_STATE, murcko_scaffold_list, scaffold_split


def test_train_test_disjoint_and_exhaustive(smiles_mixed):
    n = len(smiles_mixed)
    train, test = scaffold_split(smiles_mixed, test_size=0.2)
    assert len(train) > 0 and len(test) > 0
    assert set(train) & set(test) == set()          # disjoint
    assert sorted(np.concatenate([train, test])) == list(range(n))  # exhaustive


def test_deterministic_on_same_input(smiles_mixed):
    train1, test1 = scaffold_split(smiles_mixed, test_size=0.2)
    train2, test2 = scaffold_split(smiles_mixed, test_size=0.2)
    assert np.array_equal(train1, train2)
    assert np.array_equal(test1, test2)


def test_test_size_is_approximately_20_percent(smiles_mixed):
    n = len(smiles_mixed)
    train, test = scaffold_split(smiles_mixed, test_size=0.2)
    n_test = int(n * 0.2)
    # whole scaffolds are assigned at once: the boundary group may overshoot
    # but the loop guarantees at least n_test test molecules.
    assert len(test) >= n_test
    scaffold_sizes = np.unique(murcko_scaffold_list(smiles_mixed), return_counts=True)[1]
    assert len(test) <= n_test + int(scaffold_sizes.max())
    frac = len(test) / n
    assert 0.15 <= frac <= 0.35, f"test fraction {frac:.3f} not ~0.2"


def test_invalid_smiles_gets_its_own_scaffold(smiles_mixed, invalid_smiles):
    idx = smiles_mixed.index(invalid_smiles)
    scaffolds = murcko_scaffold_list(smiles_mixed)
    # fallback contract: an unparseable SMILES is its own scaffold...
    assert scaffolds[idx] == invalid_smiles
    # ...and no other molecule can collide with that pseudo-scaffold
    for j, scaf in enumerate(scaffolds):
        if j != idx:
            assert scaf != invalid_smiles
    # the split still covers the invalid entry exactly once
    train, test = scaffold_split(smiles_mixed, test_size=0.2)
    assert sorted(np.concatenate([train, test])) == list(range(len(smiles_mixed)))


def test_random_split_matches_sklearn_with_same_random_state():
    """RANDOM_STATE pins the sklearn contract: identical seed => identical shuffle."""
    assert RANDOM_STATE == 42
    n = 31
    mine = train_test_split(np.arange(n), test_size=0.2, random_state=RANDOM_STATE)
    theirs = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])
    # pinned reference: sklearn's seed-42 test indices must not drift silently
    assert mine[1].tolist() == [27, 15, 23, 17, 8, 9, 29]
