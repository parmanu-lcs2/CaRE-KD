"""Revival: the sequence-level epistemic rejection mechanism of CaRE-KD.

Revival estimates epistemic uncertainty with the BALD score (via MC dropout,
``BALDActiveLearning``) and rejects a whole batch when the teacher is uncertain
*relative to* a confident student:

    I_BALD = H[E_w p(y|x,w)] - E_w[ H(p(y|x,w)) ]                      (Eq. 4)
    reject  if  mean(I_T) > Q_{1-tau}(I_T)  and  mean(I_S) < Q_tau(I_S)   (Eq. 5)

``Q`` is tracked online per model with ``RunningPercentile``; ``tau`` (the target
skip percentage) is annealed over training by ``tau_at`` (Appendix D). This code
is the implementation used in the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------- BALD
class BALDActiveLearning:
    """Monte-Carlo-dropout BALD scorer (Eq. 4), mean-pooled to one score/sequence."""

    def __init__(self, num_samples: int = 3, dropout_prob: float = 0.1):
        self.num_samples = num_samples
        self.dropout_prob = dropout_prob

    def _enable_dropout(self, model: nn.Module):
        """Force dropout on for the stochastic passes (works even if shipped off)."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = self.dropout_prob
        model.train()

    def _disable_dropout(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0

    def compute_from_tensors(self, prediction: torch.Tensor) -> torch.Tensor:
        """BALD from stacked MC predictions ``(N, B, S, V)`` -> ``(B,)``."""
        mean_prediction = prediction.mean(dim=0)                        # E_w p
        inter_1 = mean_prediction * torch.log(mean_prediction)
        entropy_mean = -torch.where(
            torch.isinf(inter_1) | torch.isnan(inter_1), torch.tensor(0.0), inter_1
        ).sum(dim=-1)                                                   # H[E_w p]
        inter_2 = prediction * torch.log(prediction)
        entropy_individual = -torch.where(
            torch.isinf(inter_2) | torch.isnan(inter_2), torch.tensor(0.0), inter_2
        ).sum(dim=-1)
        mean_entropy = entropy_individual.mean(dim=0)                   # E_w H(p)
        return (entropy_mean - mean_entropy).mean(dim=-1)              # mean-pool over S -> (B,)

    @torch.no_grad()
    def compute_bald_score(self, model: nn.Module, batch: dict, toggle_dropout: bool = True) -> torch.Tensor:
        """Per-sequence BALD score for ``batch`` (keys: input_ids, attention_mask)."""
        if toggle_dropout:
            self._enable_dropout(model)
        predictions = []
        for _ in range(self.num_samples):
            logits = model(**batch).logits
            predictions.append(torch.softmax(logits, dim=-1))
        if toggle_dropout:
            self._disable_dropout(model)
        return self.compute_from_tensors(torch.stack(predictions))


# ------------------------------------------------------ running tau-quantile Q
class RunningPercentile:
    """Online estimate of a percentile (Frugal-style streaming quantile)."""

    def __init__(self, percentile: float = 0.5, step: float = 0.1):
        self.step = step
        self.step_up = 1.0 - percentile
        self.step_down = percentile
        self.x = None

    def push(self, observation) -> None:
        observation = float(observation)
        if self.x is None:
            self.x = observation
            return
        if self.x > observation:
            self.x -= self.step * self.step_up
        elif self.x < observation:
            self.x += self.step * self.step_down
        if abs(observation - self.x) < self.step:
            self.step /= 2.0

    def get_percentile(self):
        return self.x


def revival_reject(bald_teacher: torch.Tensor, bald_student: torch.Tensor,
                   teacher_pct: "RunningPercentile", student_pct: "RunningPercentile") -> bool:
    """Batch-level rejection decision of Eq. (5).

    Fires when the teacher is uncertain (mean BALD above its running percentile)
    while the student is confident (mean BALD below its own). Returns ``True``
    when the batch should be skipped (its loss zeroed, Eq. 6).
    """
    bt, bs = bald_teacher.mean(), bald_student.mean()
    student_pct.push(bs)
    teacher_pct.push(bt)
    qt, qs = teacher_pct.get_percentile(), student_pct.get_percentile()
    if qt is None or qs is None:
        return False
    return bool(bt > qt and bs < qs)


# --------------------------------------------------------------- tau schedules
def _exp_incr(epoch: float, T: float, tau: float, k: float = 5.0) -> float:
    e = max(0.0, min(float(epoch), float(T)))
    return float(tau * (1.0 - math.exp(-k * e / T)) / (1.0 - math.exp(-k)))


def _exp_decr(epoch: float, T: float, tau: float, a: float = 5.0) -> float:
    e = max(0.0, min(float(epoch), float(T)))
    if a < 1e-8:
        return float(tau * (1.0 - e / T))
    return float(tau * (math.exp(-a * e / T) - math.exp(-a)) / (1.0 - math.exp(-a)))


def tau_at(epoch: int, total_epochs: int, tau: float, schedule: str = "increasing") -> float:
    """Target skip percentage ``tau`` at ``epoch`` (Appendix D).

    ``"increasing"`` ramps selectivity up over training (best in the paper);
    ``"decreasing"`` relaxes it; ``"constant"`` keeps ``tau`` fixed.
    """
    if total_epochs <= 1 or schedule == "constant":
        return tau
    if schedule == "increasing":
        return _exp_incr(epoch, total_epochs, tau)
    if schedule == "decreasing":
        return _exp_decr(epoch, total_epochs, tau)
    raise ValueError(f"schedule must be constant/increasing/decreasing, got {schedule!r}")
