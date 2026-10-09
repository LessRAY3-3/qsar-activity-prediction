"""qsar_common.select_device: cuda > mps > cpu, with mocked availability.

No GPU needed: torch.cuda.is_available / torch.backends.mps.is_available
are monkeypatched, so every branch runs on any machine.
"""
import torch

from qsar_common import select_device  # conftest puts scripts/ on sys.path


def test_cuda_wins_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device() == "cuda"


def test_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device() == "mps"


def test_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device() == "cpu"
