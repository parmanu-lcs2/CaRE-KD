"""CaRE-KD trainer: assembles CaRE-Divergence and Revival into one objective.

Per batch (teacher frozen, student optionally LoRA-adapted):

    distil_loss = L_CARE(p, q)                              CaRE-Divergence (Eq. 2)
    loss        = (1 - kd_ratio) . CE(q, y) + kd_ratio . distil_loss
    loss        = 0   if Revival rejects the batch           (Eq. 5-6)

Teacher-forcing (default) and student-generated-output (SGO, on-policy) training
are both supported. ``tau`` (skip percentage) is annealed per epoch (Appendix D).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import care_divergence
from .revival import BALDActiveLearning, RunningPercentile, revival_reject, tau_at


@dataclass
class CaREKDConfig:
    # CaRE-Divergence (Eq. 2-3)
    gate: str = "hard"              # "hard" (highest peak) or "soft" (more robust)
    margin: float = 0.05            # margin m of the soft gate
    temperature: float = 3.0        # KD temperature T
    kd_ratio: float = 0.5           # weight of distil vs. CE-to-reference (1.0 = pure Eq. 6)
    # Revival (Eq. 4-5)
    use_revival: bool = True
    mc_samples: int = 3             # N MC-dropout passes for BALD
    dropout_p: float = 0.1          # dropout forced during the BALD passes
    tau: float = 0.5                # target skip percentage (percentile cutoff)
    schedule: str = "increasing"    # tau schedule: constant/increasing/decreasing
    # optimization
    sgo: bool = False               # student-generated outputs (paper's main setting)
    lr: float = 5e-5
    epochs: int = 20
    batch_size: int = 8
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    max_new_tokens: int = 256
    gen_temperature: float = 1.0
    use_lora: bool = True
    lora_rank: int = 16


class CaREKDTrainer:
    def __init__(self, student, teacher, tokenizer, config: CaREKDConfig, device=None):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.teacher = teacher.to(self.device).eval().requires_grad_(False)
        self.student = self._maybe_lora(student).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad], lr=config.lr
        )
        self.bald = BALDActiveLearning(config.mc_samples, config.dropout_p)

    def _maybe_lora(self, student):
        if not self.config.use_lora:
            return student
        from peft import LoraConfig, get_peft_model

        return get_peft_model(
            student,
            LoraConfig(r=self.config.lora_rank, lora_alpha=2 * self.config.lora_rank,
                       lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
        )

    # ----------------------------------------------------------------- loss
    def care_kd_loss(self, batch, teacher_pct, student_pct):
        """Final CaRE-KD objective (Eq. 6) for one batch; returns (loss, stats)."""
        cfg = self.config
        if cfg.sgo:
            input_ids, attention_mask, loss_mask = self._sgo_rollout(batch)
        else:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)

        labels = input_ids.masked_fill(loss_mask == 0, -100)
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        self.student.train()        # consistent dropout for the gradient forward (BALD toggles it)
        student_logits = self.student(input_ids=input_ids, attention_mask=attention_mask).logits

        # next-token alignment: logits at position t score token t+1
        s, t, y = student_logits[:, :-1], teacher_logits[:, :-1], labels[:, 1:]
        distil_loss, diag = care_divergence(
            s, t, y, temperature=cfg.temperature, gate_mode=cfg.gate, margin=cfg.margin
        )
        lm_loss = F.cross_entropy(s.float().reshape(-1, s.size(-1)), y.reshape(-1), ignore_index=-100)
        loss = (1.0 - cfg.kd_ratio) * lm_loss + cfg.kd_ratio * distil_loss

        reject = False
        if cfg.use_revival:
            model_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            bald_student = self.bald.compute_bald_score(self.student, model_batch)
            bald_teacher = self.bald.compute_bald_score(self.teacher, model_batch)
            reject = revival_reject(bald_teacher, bald_student, teacher_pct, student_pct)  # Eq. (5)
            if reject:
                loss = loss * 0.0                                                          # Eq. (6)
        return loss, {"loss": float(loss.item()), "reject": int(reject), **diag}

    # ------------------------------------------------------------- SGO rollout
    @torch.no_grad()
    def _sgo_rollout(self, batch):
        """Sample on-policy responses from the student; both models score them."""
        cfg = self.config
        prompt_ids = batch["prompt_ids"].to(self.device)
        prompt_mask = batch["prompt_attention_mask"].to(self.device)
        width = prompt_ids.size(1)
        gen = self.student.generate(
            input_ids=prompt_ids, attention_mask=prompt_mask, max_new_tokens=cfg.max_new_tokens,
            do_sample=True, temperature=cfg.gen_temperature, top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        attention_mask = (gen != self.tokenizer.pad_token_id).long()
        loss_mask = torch.zeros_like(gen)
        response = gen[:, width:]
        is_eos = response == self.tokenizer.eos_token_id
        after_eos = is_eos.cumsum(dim=1) - is_eos.long()          # 1 only strictly after first eos
        loss_mask[:, width:] = ((response != self.tokenizer.pad_token_id) & (after_eos == 0)).long()
        return gen, attention_mask, loss_mask

    # ------------------------------------------------------------------- train
    def train(self, train_dataset, collate_fn, val_dataset=None, output_dir=None):
        cfg = self.config
        loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        best_rouge = -1.0

        for epoch in range(cfg.epochs):
            tau_e = tau_at(epoch, cfg.epochs, cfg.tau, cfg.schedule)
            # fresh running percentiles each epoch (teacher high, student low side)
            teacher_pct = RunningPercentile(percentile=1.0 - tau_e)
            student_pct = RunningPercentile(percentile=tau_e)

            self.student.train()
            self.optimizer.zero_grad()
            skipped = 0
            bar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs} (tau={tau_e:.2f})")
            for i, batch in enumerate(bar):
                loss, stats = self.care_kd_loss(batch, teacher_pct, student_pct)
                skipped += stats["reject"]
                (loss / cfg.grad_accum).backward()
                if (i + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                bar.set_postfix(loss=f"{stats['loss']:.3f}", gate=f"{stats['gate_rate']:.2f}", skip=skipped)

            if val_dataset is not None:
                rouge = self.evaluate_rouge_l(val_dataset, collate_fn)
                print(f"[epoch {epoch + 1}] validation ROUGE-L = {rouge:.4f}  (skipped {skipped} batches)")
                if output_dir and rouge > best_rouge:        # checkpoint by ROUGE-L, as in the paper
                    best_rouge = rouge
                    self.save(output_dir)
                    print(f"  -> new best, saved to {output_dir}")
        if output_dir and val_dataset is None:
            self.save(output_dir)

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_rouge_l(self, dataset, collate_fn, max_batches=None):
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=collate_fn)
        self.student.eval()
        scores = []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            prompt_ids = batch["prompt_ids"].to(self.device)
            gen = self.student.generate(
                input_ids=prompt_ids, attention_mask=batch["prompt_attention_mask"].to(self.device),
                max_new_tokens=self.config.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            preds = self.tokenizer.batch_decode(gen[:, prompt_ids.size(1):], skip_special_tokens=True)
            for pred, ref in zip(preds, batch["references"]):
                scores.append(scorer.score(ref, pred)["rougeL"].fmeasure)
        return sum(scores) / max(len(scores), 1)

    def save(self, output_dir):
        self.student.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
