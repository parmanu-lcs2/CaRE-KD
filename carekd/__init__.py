"""CaRE-KD: Confidence-Aware Selective Distillation for LLMs.

Public API:
    confidence, care_divergence                     -- CaRE-Divergence (token level, Eq. 1-3)
    BALDActiveLearning, RunningPercentile,
    revival_reject, tau_at                           -- Revival (sequence level, Eq. 4-5)
    CaREKDConfig, CaREKDTrainer                       -- the full CaRE-KD objective (Eq. 6)
"""

from .losses import care_divergence, confidence
from .revival import BALDActiveLearning, RunningPercentile, revival_reject, tau_at
from .trainer import CaREKDConfig, CaREKDTrainer

__version__ = "0.1.0"

__all__ = [
    "confidence",
    "care_divergence",
    "BALDActiveLearning",
    "RunningPercentile",
    "revival_reject",
    "tau_at",
    "CaREKDConfig",
    "CaREKDTrainer",
]
