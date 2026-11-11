"""Train a student from a teacher with CaRE-KD on an instruction-following task.

This is the minimal end-to-end example (and smoke test) for the instruction-
following setting of the paper. It defaults to tiny GPT-2 models and a small
Dolly-15k subset so it runs in seconds on CPU; point ``--teacher``/``--student``
at a real pair (e.g. ``gpt2-xl`` -> ``gpt2``) to reproduce the paper.

    python tests/train_instruction_following.py                 # fast smoke run
    python tests/train_instruction_following.py \
        --teacher gpt2-xl --student gpt2 --epochs 10 --sgo       # paper-style run
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carekd import CaREKDConfig, CaREKDTrainer
from carekd.data import Collator, InstructionDataset, load_dolly

# Tiny offline fallback so the smoke test needs no network for the dataset.
TOY_EXAMPLES = [
    {"instruction": "Give three tips for staying healthy.", "context": "",
     "response": "Eat a balanced diet, exercise regularly, and get enough sleep."},
    {"instruction": "What is the capital of France?", "context": "",
     "response": "The capital of France is Paris."},
    {"instruction": "Summarize the sentence.", "context": "The cat sat on the warm windowsill in the sun.",
     "response": "A cat sat in the sun on a windowsill."},
    {"instruction": "Convert 10 kilometers to miles.", "context": "",
     "response": "10 kilometers is about 6.21 miles."},
] * 8


def load_examples(split, n):
    try:
        return load_dolly(split, n)
    except Exception as e:                      # offline / no datasets -> toy set
        print(f"[warn] falling back to toy examples ({type(e).__name__}: {e})")
        return TOY_EXAMPLES[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="sshleifer/tiny-gpt2")
    ap.add_argument("--student", default="sshleifer/tiny-gpt2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--train_samples", type=int, default=32)
    ap.add_argument("--val_samples", type=int, default=8)
    ap.add_argument("--gate", choices=["hard", "soft"], default="hard")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--schedule", default="increasing")
    ap.add_argument("--sgo", action="store_true")
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--output_dir", default="runs/smoke")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float32)
    assert teacher.config.vocab_size == student.config.vocab_size, "shared vocabulary required"

    config = CaREKDConfig(
        gate=args.gate, tau=args.tau, schedule=args.schedule, sgo=args.sgo,
        epochs=args.epochs, batch_size=args.batch_size, mc_samples=3,
        use_lora=not args.no_lora, max_new_tokens=32,
    )
    train_set = InstructionDataset(load_examples("train", args.train_samples), tokenizer, max_length=128)
    val_set = InstructionDataset(load_examples("validation", args.val_samples), tokenizer, max_length=128)
    collate = Collator(tokenizer.pad_token_id)

    trainer = CaREKDTrainer(student, teacher, tokenizer, config)
    trainer.train(train_set, collate, val_dataset=val_set, output_dir=args.output_dir)
    print("CaRE-KD instruction-following training finished.")


if __name__ == "__main__":
    main()
