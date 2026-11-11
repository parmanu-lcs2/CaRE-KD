"""Instruction-following data for CaRE-KD (Dolly-15k by default).

Each example is tokenized as ``prompt + response`` with a ``loss_mask`` that is
1 on the response tokens only, so both the CaRE-Divergence loss and the Revival
BALD score are computed over the response region. The prompt template follows
Gu et al. (MiniLLM).

Teacher and student must share a vocabulary (white-box, same-family distillation
as in the paper), so a single tokenizer is used for both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n{context}### Response:\n"
)


def build_prompt(instruction: str, context: str = "") -> str:
    context_block = f"### Input:\n{context}\n\n" if context else ""
    return PROMPT_TEMPLATE.format(instruction=instruction, context=context_block)


class InstructionDataset(Dataset):
    """Tokenized instruction/response pairs with a response-only loss mask."""

    def __init__(self, examples, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = build_prompt(ex["instruction"], ex.get("context", "") or "")
        response = ex["response"].strip() + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        # loss mask: 0 over the prompt, 1 over the response (clipped to max_length)
        loss_mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
            "prompt_ids": torch.tensor(prompt_ids[: self.max_length], dtype=torch.long),
            "reference": ex["response"].strip(),
        }


@dataclass
class Collator:
    """Left-pads prompts (for generation) and right-pads full sequences."""

    pad_token_id: int

    def __call__(self, batch):
        def pad(seqs, value, left=False):
            width = max(len(s) for s in seqs)
            out = torch.full((len(seqs), width), value, dtype=torch.long)
            for i, s in enumerate(seqs):
                if left:
                    out[i, width - len(s):] = s
                else:
                    out[i, : len(s)] = s
            return out

        input_ids = pad([b["input_ids"] for b in batch], self.pad_token_id)
        loss_mask = pad([b["loss_mask"] for b in batch], 0)
        prompt_ids = pad([b["prompt_ids"] for b in batch], self.pad_token_id, left=True)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
            "loss_mask": loss_mask,
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": (prompt_ids != self.pad_token_id).long(),
            "references": [b["reference"] for b in batch],
        }


def load_dolly(split: str = "train", max_samples: int | None = None):
    """Load Dolly-15k as a list of ``{instruction, context, response}`` dicts."""
    from datasets import load_dataset

    data = load_dataset("databricks/databricks-dolly-15k", split="train")
    # deterministic train/validation split (Dolly ships only a 'train' split)
    data = data.train_test_split(test_size=0.05, seed=42)
    data = data["train"] if split == "train" else data["test"]
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
    return [
        {"instruction": r["instruction"], "context": r["context"], "response": r["response"]}
        for r in data
    ]
