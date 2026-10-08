"""Shared fixtures: scripts on sys.path + a small mixed SMILES panel.

``scripts/`` is located via ``__file__`` (never cwd), so importing
``qsar_common`` / the numbered pipeline scripts works from any working
directory.  The numbered scripts (``02_clean_data.py`` ...) are not
importable as package modules (digits + no package), so tests load them
by file path through the ``load_script`` fixture.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
from rdkit import RDLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RDLogger.DisableLog("rdApp.*")  # the invalid-SMILES fixture would spam parse errors

INVALID_SMILES = "INVALID_SMILES_XYZ"

# 30 real drug-like molecules with deliberately diverse Bemis-Murcko scaffolds
# (benzene, pyridine, fused rings, saturated heterocycles, acyclic), plus one
# invalid SMILES at the end so fallback paths (scaffold = raw string) run too.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",             # aspirin            (benzene)
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",      # caffeine           (purine)
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",        # ibuprofen          (benzene)
    "CC(=O)Nc1ccc(O)cc1",                # paracetamol        (benzene)
    "c1ccc2c(c1)ccc1ccccc12",            # phenanthrene       (fused benzene)
    "CN1CCC[C@H]1c1cccnc1",              # nicotine           (pyridine+pyrrolidine)
    "CCO",                               # ethanol            (acyclic)
    "CC(C)O",                            # isopropanol        (acyclic)
    "CCCCO",                             # n-butanol          (acyclic)
    "CC(=O)Oc1ccccc1C(=O)OCC",           # aspirin ethyl ester(benzene)
    "Clc1ccccc1Cl",                      # 1,2-dichlorobenzene(benzene)
    "Oc1ccccc1O",                        # catechol           (benzene)
    "CCN(CC)CC",                         # triethylamine      (acyclic)
    "c1ccncc1",                          # pyridine
    "c1ccoc1",                           # furan
    "c1ccsc1",                           # thiophene
    "c1cnc[nH]1",                        # imidazole
    "c1cn[nH]c1",                        # pyrazole
    "C1CCCCC1",                          # cyclohexane
    "C1CCNCC1",                          # piperidine
    "C1CCOCC1",                          # tetrahydropyran
    "C1CCSCC1",                          # thiane
    "c1ccc2[nH]ccc2c1",                  # indole
    "c1ccc2ncccc2c1",                    # quinoline
    "c1ccc2cnccc2c1",                    # isoquinoline
    "c1ccc2occc2c1",                     # benzofuran
    "c1ccc2[nH]cnc2c1",                  # benzimidazole
    "O=C1CCCCC1",                        # cyclohexanone
    "O=C(c1ccccc1)Nc1ccccc1",            # benzanilide        (benzene)
    "CC(C)NCC(O)c1ccc(O)c(CO)c1",        # salbutamol         (benzene)
    INVALID_SMILES,                       # invalid -> fallback path
]


def _load_by_path(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def load_script():
    """Load ``scripts/<name>.py`` by file path (script names start with digits)."""

    def _load(name: str):
        return _load_by_path(SCRIPTS_DIR / f"{name}.py", f"qsar_under_test_{name}")

    return _load


@pytest.fixture(scope="session")
def load_experiment():
    """Load ``experiments/<name>.py`` by file path."""

    def _load(name: str):
        return _load_by_path(EXPERIMENTS_DIR / f"{name}.py", f"qsar_experiment_{name}")

    return _load


@pytest.fixture
def smiles_mixed():
    """~30 real molecules + 1 invalid SMILES (invalid entry is last)."""
    return list(SMILES)


@pytest.fixture
def valid_smiles():
    """Only the parseable entries, order preserved."""
    return [s for s in SMILES if s != INVALID_SMILES]


@pytest.fixture
def invalid_smiles():
    return INVALID_SMILES
