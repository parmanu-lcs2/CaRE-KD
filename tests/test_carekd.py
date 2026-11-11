"""Fast, model-free unit tests for the CaRE-KD method components.

Run with:  pytest tests/test_carekd.py
These check the maths of CaRE-Divergence and Revival on small tensors; no model
download or network access is required.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carekd.losses import care_divergence, confidence
from carekd.revival import BALDActiveLearning, RunningPercentile, revival_reject, tau_at


def test_confidence_bounds_and_extremes():
    peaked = torch.zeros(1, 1, 100)
    peaked[..., 0] = 50.0                     # almost one-hot -> confident
    uniform = torch.zeros(1, 1, 100)          # uniform -> unconfident
    assert confidence(peaked).item() > 0.95
    assert confidence(uniform).item() < 0.05
    c = confidence(torch.randn(4, 7, 100))
    assert torch.all((c >= 0) & (c <= 1))


def test_care_divergence_zero_when_identical():
    logits = torch.randn(2, 5, 64)
    labels = torch.ones(2, 5, dtype=torch.long)       # all valid
    loss, _ = care_divergence(logits, logits.clone(), labels, gate_mode="hard")
    assert loss.item() < 1e-4                          # q == p -> no divergence


def test_care_divergence_nonnegative_and_gate():
    student = torch.randn(3, 6, 128)
    teacher = torch.randn(3, 6, 128)
    labels = torch.ones(3, 6, dtype=torch.long)
    for mode in ("hard", "soft"):
        loss, diag = care_divergence(student, teacher, labels, gate_mode=mode)
        assert loss.item() >= 0.0
        assert 0.0 <= diag["gate_rate"] <= 1.0


def test_bald_zero_when_deterministic():
    bald = BALDActiveLearning(num_samples=4)
    probs = torch.softmax(torch.randn(1, 2, 5, 20), dim=-1)   # (N=1 repeated) -> no disagreement
    identical = probs.repeat(4, 1, 1, 1)
    assert bald.compute_from_tensors(identical).abs().max().item() < 1e-5
    varied = torch.softmax(torch.randn(4, 2, 5, 20), dim=-1)  # disagreement -> positive BALD
    assert bald.compute_from_tensors(varied).mean().item() > 0.0


def test_running_percentile_tracks_median():
    rp = RunningPercentile(percentile=0.5, step=1.0)
    for v in torch.linspace(0, 100, 500):
        rp.push(v.item())
    assert 30.0 < rp.get_percentile() < 70.0


def test_revival_reject_logic():
    tpct, spct = RunningPercentile(0.5), RunningPercentile(0.5)
    for _ in range(50):                                        # warm up both percentiles ~0.5
        revival_reject(torch.tensor([0.5]), torch.tensor([0.5]), tpct, spct)
    # teacher uncertain (high), student confident (low) -> reject
    assert revival_reject(torch.tensor([5.0]), torch.tensor([0.0]), tpct, spct) is True
    # teacher confident (low), student uncertain (high) -> keep
    assert revival_reject(torch.tensor([0.0]), torch.tensor([5.0]), tpct, spct) is False


def test_tau_schedule():
    assert tau_at(0, 10, 0.6, "constant") == 0.6
    assert abs(tau_at(0, 10, 0.6, "increasing")) < 1e-6        # starts at 0
    assert tau_at(9, 10, 0.6, "increasing") > tau_at(3, 10, 0.6, "increasing")   # monotone up
    assert tau_at(0, 10, 0.6, "decreasing") > tau_at(9, 10, 0.6, "decreasing")   # monotone down


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
