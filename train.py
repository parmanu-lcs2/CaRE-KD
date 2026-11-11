"""Train any student from any teacher with CaRE-KD.

Example
-------
    python train.py \
        --teacher gpt2-xl --student gpt2 \
        --gate hard --tau 0.5 --schedule increasing --sgo \
        --output_dir runs/gpt2xl_to_gpt2

Teacher and student must share a tokenizer / vocabulary (same-family, white-box
distillation, as in the paper).
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from carekd import CaREKDConfig, CaREKDTrainer
from carekd.data import Collator, InstructionDataset, load_dolly


def parse_args():
    ap = argparse.ArgumentParser(description="CaRE-KD training")
    # models / data
    ap.add_argument("--teacher", required=True, help="teacher model name or path")
    ap.add_argument("--student", required=True, help="student model name or path")
    ap.add_argument("--output_dir", default="runs/care_kd")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=500)
    # CaRE-Divergence
    ap.add_argument("--gate", choices=["soft", "hard"], default="hard")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=3.0, help="KD temperature T")
    ap.add_argument("--kd_ratio", type=float, default=0.5, help="distil vs. CE weight (1.0 = pure Eq. 6)")
    # Revival
    ap.add_argument("--no_revival", action="store_true", help="disable Revival (CaRE-Divergence only)")
    ap.add_argument("--mc_samples", type=int, default=3)
    ap.add_argument("--dropout_p", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5, help="target skip percentage")
    ap.add_argument("--schedule", choices=["constant", "increasing", "decreasing"], default="increasing")
    # optimization
    ap.add_argument("--sgo", action="store_true", help="student-generated outputs (paper's main setting)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=16)
    return ap.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype)
    assert teacher.config.vocab_size == student.config.vocab_size, (
        "CaRE-Divergence requires a shared vocabulary between teacher and student."
    )

    config = CaREKDConfig(
        gate=args.gate, margin=args.margin, temperature=args.temperature, kd_ratio=args.kd_ratio,
        use_revival=not args.no_revival, mc_samples=args.mc_samples, dropout_p=args.dropout_p,
        tau=args.tau, schedule=args.schedule, sgo=args.sgo, lr=args.lr,
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        use_lora=not args.no_lora, lora_rank=args.lora_rank,
    )

    train_set = InstructionDataset(load_dolly("train", args.max_train_samples), tokenizer, args.max_length)
    val_set = InstructionDataset(load_dolly("validation", args.max_val_samples), tokenizer, args.max_length)
    collate = Collator(tokenizer.pad_token_id)

    trainer = CaREKDTrainer(student, teacher, tokenizer, config)
    trainer.train(train_set, collate, val_dataset=val_set, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
