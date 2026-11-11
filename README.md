# CaRE-KD: Confidence-Aware Selective Distillation for LLMs

Official, minimal implementation of **CaRE-KD** from
*Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models* (NeurIPS 2026).

Standard knowledge distillation assumes the teacher is a reliable oracle and forces the
student to match it everywhere. LLM teachers are frequently uncertain or hallucinate,
so this erodes a student's already-correct priors. CaRE-KD instead decides **how** and
**when** to learn from the teacher, with two components:

- **CaRE-Divergence** (token level) — a confidence gate switches the objective between
  Forward KL (teacher confident) and Reverse KL (teacher uncertain).
- **Revival** (sequence level) — an epistemic (BALD) rejection rule that skips a batch
  when the teacher is more uncertain than a confident student.

This repo contains only the method and a minimal training loop. The teacher and student
must **share a tokenizer / vocabulary** (same-family, white-box distillation, as in the paper).

## Method

With `p` = teacher and `q` = student next-token distributions:

```
c(.)   = 1 - H(.) / log|V|                                         (Eq. 1)   confidence
g_hard = 1[c(p) >= c(q)],   g_soft = sigma(k (c(p) - c(q) - m))     (Eq. 3)   gate
L_CARE = g . KL(p||q) + (1 - g) . KL(q||p)                         (Eq. 2)   CaRE-Divergence

I_BALD = H[E_w p(y|x,w)] - E_w[ H(p(y|x,w)) ]                       (Eq. 4)   epistemic uncertainty
reject if mean(I_T) > Q_{1-tau}(I_T) and mean(I_S) < Q_tau(I_S)     (Eq. 5)   Revival mask
L_CaRE-KD = (1 - M_Revival) . L_CARE                               (Eq. 6)   final objective
```

| File | Contents |
|------|----------|
| `carekd/losses.py`  | `confidence` (Eq. 1), `care_divergence` (Eq. 2-3) |
| `carekd/revival.py` | `BALDActiveLearning` (Eq. 4), `RunningPercentile` / `tau_at` (schedules), `revival_reject` (Eq. 5) |
| `carekd/trainer.py` | `CaREKDConfig`, `CaREKDTrainer` — assembles the objective (Eq. 6), SGO / non-SGO, LoRA, ROUGE-L checkpointing |
| `carekd/data.py`    | instruction-following data (Dolly-15k) with a response-only loss mask |
| `train.py`          | CLI to distill any student from any teacher |
| `tests/`            | instruction-following training script + model-free unit tests |

## Install

```bash
pip install -r requirements.txt
# or:  pip install -e .
```

## Quickstart

Smoke run (tiny GPT-2, a few seconds, CPU):

```bash
python tests/train_instruction_following.py
```

Distill any student from any teacher on Dolly-15k:

```bash
python train.py \
    --teacher gpt2-xl --student gpt2 \
    --gate hard --tau 0.5 --schedule increasing --sgo \
    --output_dir runs/gpt2xl_to_gpt2
```

Unit tests for the method maths (no downloads):

```bash
pytest tests/test_carekd.py
```

## Key hyperparameters (paper naming)

| Flag | Paper | Default | Notes |
|------|-------|---------|-------|
| `--gate` | gate `g` | `hard` | `hard` = highest peak, `soft` = more robust |
| `--margin` | `m` | `0.05` | soft-gate margin |
| `--temperature` | `T` | `3.0` | KD temperature |
| `--tau` | skip % `tau` | `0.5` | Revival percentile cutoff |
| `--schedule` | `tau` schedule | `increasing` | `constant` / `increasing` / `decreasing` |
| `--mc_samples` | `N` | `3` | MC-dropout passes for BALD |
| `--kd_ratio` | — | `0.5` | distil vs. CE weight (`1.0` = pure Eq. 6) |
| `--sgo` | SGO | off | student-generated (on-policy); paper's main setting |

Ablation settings `--no_revival` (CaRE-Divergence only) and `--no_lora` (full fine-tuning) are also available.

## Citation

```bibtex
@inproceedings{sengupta2026carekd,
  title     = {Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models},
  author    = {Sengupta, Ayan and Seth, Vaibhav and Chakraborty, Tanmoy},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).
