"""CaRE-Divergence: the token-level objective of CaRE-KD.

This is the implementation used in the paper (``teacher_aware_biKL_kd``), exposed
under the paper name ``care_divergence``. With p = teacher and q = student
next-token distributions:

    c(.) = 1 - H(.) / log|V|                                         (Eq. 1)
    g_hard = 1[c(p) >= c(q)],  g_soft = sigma(k (c(p) - c(q) - m))    (Eq. 3)
    L_CARE = g . KL(p||q) + (1 - g) . KL(q||p)                        (Eq. 2)

When the teacher is confident (g -> 1) the loss is mean-seeking Forward KL;
when the teacher is uncertain (g -> 0) it is mode-seeking Reverse KL.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def confidence(logits: torch.Tensor, T_for_conf: float = 1.0) -> torch.Tensor:
    """Normalized-entropy confidence c(.) of Eq. (1), per token, in [0, 1].

        c = 1 - H(p) / log|V|,   p = softmax(logits / T_for_conf)
    """
    _, _, V = logits.shape
    logp = F.log_softmax(logits.float() / T_for_conf, dim=-1)
    H = -(logp.exp() * logp).sum(dim=-1)          # [B, S]
    return (1.0 - H / math.log(V)).clamp_(0.0, 1.0)


def care_divergence(
    student_logits: torch.Tensor,   # [B, S, V] student q
    teacher_logits: torch.Tensor,   # [B, S, V] teacher p (same vocabulary)
    labels: torch.Tensor,           # [B, S], -100 marks ignored (prompt / pad) tokens
    *,
    temperature: float = 3.0,       # KD temperature T
    gate_mode: str = "hard",        # "hard" (highest peak) or "soft" (more robust)
    margin: float = 0.05,           # margin m of the soft gate
    soft_gate_slope: float = 8.0,   # slope k of the soft gate
    T_for_conf: float = 1.0,        # temperature for the confidence measurement
    detach_teacher: bool = True,    # keep gradients out of the (frozen) teacher
):
    """Token-level CaRE-Divergence loss of Eq. (2), averaged over valid tokens.

    Returns:
        (loss, diagnostics) -- ``loss`` is a scalar; ``diagnostics`` logs the
        gate rate and mean teacher/student confidence.
    """
    B, S, V = student_logits.shape
    dtype = student_logits.dtype
    valid = (labels != -100).float()                                   # [B, S]

    # --- confidence gate g (Eq. 3) ---
    c_t = confidence(teacher_logits, T_for_conf)
    c_s = confidence(student_logits, T_for_conf)
    if gate_mode == "hard":
        g = (c_t >= c_s).float()
    elif gate_mode == "soft":
        g = torch.sigmoid(soft_gate_slope * (c_t - c_s - margin)).float()
    else:
        raise ValueError(f"gate_mode must be 'hard' or 'soft', got {gate_mode!r}")
    g = g * valid                                                      # only valid tokens

    # --- forward / reverse KL at temperature T ---
    T = float(temperature)
    s_logp = F.log_softmax(student_logits.float() / T, dim=-1)
    t_logp = F.log_softmax(
        (teacher_logits.float().detach() if detach_teacher else teacher_logits.float()) / T,
        dim=-1,
    ).clamp_min(-30.0)                                                 # floor to avoid log(0)
    s_p, t_p = s_logp.exp(), t_logp.exp()

    forward_kl = (t_p * (t_logp - s_logp)).sum(dim=-1)                 # KL(p||q), mean-seeking
    reverse_kl = (s_p * (s_logp - t_logp)).sum(dim=-1)                 # KL(q||p), mode-seeking
    care_tok = g * forward_kl + (1.0 - g) * reverse_kl                 # Eq. (2), [B, S]
    care_tok = (T * T) * care_tok                                      # scale by T^2

    loss = (care_tok * valid).sum() / valid.sum().clamp_min(1e-9)
    diag = {
        "gate_rate": float(g.sum().item() / (valid.sum().item() + 1e-9)),
        "mean_ct": float((c_t * valid).sum().item() / (valid.sum().item() + 1e-9)),
        "mean_cs": float((c_s * valid).sum().item() / (valid.sum().item() + 1e-9)),
    }
    return loss.to(dtype), diag
