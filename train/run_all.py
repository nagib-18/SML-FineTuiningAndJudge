"""
Train coder, judge and self-critique one after another.

Usage: python train/run_all.py --tasks coder selfcritique judge --data data/final_python --lang python
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Must be set before torch starts: reduces memory fragmentation on the GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = [str(ROOT / d) for d in ["data", "train", "eval"]]
import sft_utils as S  # noqa: E402

# Same settings as the notebook.
CONFIG = dict(
    model="Qwen/Qwen3.5-0.8B",
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    # effective batch 16; batch 4 needs an A100 because of the 248k vocabulary
    learning_rate=2e-4, epochs=2, batch_size=1, grad_accum=16,
    max_len=4096, warmup_ratio=0.03, eval_steps=200, seed=42,
)
EVAL_STEPS_PER_TASK = {"judge": 400}   # judge val set is large: validate half as often


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(name: str, dtype):
    import transformers as tf
    for cls_name in ["AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"]:
        cls = getattr(tf, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(name, torch_dtype=dtype, device_map="cuda")
            log(f"loaded with {cls_name}")
            return model
        except (ValueError, KeyError) as err:
            log(f"{cls_name} failed: {str(err)[:120]}")
    raise RuntimeError("could not load the model; upgrade transformers")


def train_one(task: str, data_dir: Path, lang: str, smoke: bool) -> dict:
    c = CONFIG
    run_name = f"{lang}_{task}" + ("_smoke" if smoke else "")
    out = ROOT / "checkpoints" / run_name
    if (out / "merged" / "run_config.json").exists():
        log(f"{run_name}: merged model already exists, skipping")
        return {"status": "skipped", "dir": str(out)}

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    torch.manual_seed(c["seed"])
    train_rows = S.load_jsonl(data_dir / f"{task}_train.jsonl", limit=200 if smoke else None)
    val_rows = S.load_jsonl(data_dir / f"{task}_val.jsonl", limit=50 if smoke else None)
    log(f"{run_name}: train {len(train_rows)}, val {len(val_rows)}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(c["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(c["model"], dtype)

    targets = S.find_lora_targets(model)
    log(f"LoRA targets: {targets}")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=c["lora_r"], lora_alpha=c["lora_alpha"],
                                             lora_dropout=c["lora_dropout"], target_modules=targets,
                                             task_type="CAUSAL_LM"))

    train_ds, dropped_train = S.build_dataset(tokenizer, train_rows, c["max_len"])
    val_ds, dropped_val = S.build_dataset(tokenizer, val_rows, c["max_len"])
    log(f"dropped as too long: train {dropped_train}, val {dropped_val}")
    for kind in sorted({r["kind"] for r in train_rows}):
        first = next(r for r in train_rows if r["kind"] == kind)
        feats = S.build_features(tokenizer, first, c["max_len"])
        assert tokenizer.decode(feats["input_ids"]).count("<think>") <= 1, "think block twice"
        log(f"masking check {kind}:\n" + S.describe_masking(tokenizer, feats))

    eval_steps = 10 if smoke else EVAL_STEPS_PER_TASK.get(task, c["eval_steps"])
    use_bf16 = dtype == torch.bfloat16
    steps = len(train_ds) // (c["batch_size"] * c["grad_accum"]) * c["epochs"]
    warmup_steps = max(1, int(c["warmup_ratio"] * (30 if smoke else steps)))
    settings = dict(
        output_dir=str(out), num_train_epochs=c["epochs"], max_steps=30 if smoke else -1,
        per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=c["batch_size"],
        gradient_accumulation_steps=c["grad_accum"], learning_rate=c["learning_rate"],
        lr_scheduler_type="cosine", warmup_steps=warmup_steps, weight_decay=0.01,
        bf16=use_bf16, fp16=not use_bf16, logging_steps=10,
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        group_by_length=True, remove_unused_columns=False, report_to="none",
        prediction_loss_only=True,   # validation needs only the loss, not all 248k-wide logits
        seed=c["seed"], data_seed=c["seed"], disable_tqdm=False,
    )
    # transformers 5 removed some arguments (e.g. warmup_ratio). Keep only supported ones.
    import inspect
    known = set(inspect.signature(TrainingArguments.__init__).parameters)
    skipped = sorted(k for k in settings if k not in known)
    if skipped:
        log(f"not supported by this transformers version, skipped: {skipped}")
    args = TrainingArguments(**{k: v for k, v in settings.items() if k in known})
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=S.PadCollator(tokenizer.pad_token_id))
    log(f"{run_name}: ~{steps} optimizer steps planned, eval every {eval_steps}")
    t0 = time.time()
    resume = any(out.glob("checkpoint-*"))       # continue after an interruption
    trainer.train(resume_from_checkpoint=True if resume else None)
    minutes = (time.time() - t0) / 60

    history = trainer.state.log_history
    (out / "log_history.json").write_text(json.dumps(history, indent=1))
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    best = min(ev, key=lambda x: x[1]) if ev else (None, None)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
        plt.figure()
        if tr:
            plt.plot(*zip(*tr), label="train loss")
        if ev:
            plt.plot(*zip(*ev), "o-", label="validation loss")
        plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title(run_name)
        plt.savefig(out / "loss.png", dpi=120); plt.close()
    except Exception as err:  # the plot is a convenience, never a reason to fail
        log(f"plot skipped: {err}")

    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged")
    tokenizer.save_pretrained(out / "merged")
    run_config = {"task": task, "lang": lang, "smoke": smoke, **c, "eval_steps": eval_steps,
                  "train_examples": len(train_ds), "dropped_train": dropped_train,
                  "best_eval_loss": best[1], "best_step": best[0], "train_minutes": round(minutes, 1)}
    (out / "merged" / "run_config.json").write_text(json.dumps(run_config, indent=1))
    log(f"{run_name}: done in {minutes:.0f} min, best val loss {best[1]} at step {best[0]}")

    del trainer, model, merged
    gc.collect()
    torch.cuda.empty_cache()
    return {"status": "done", "dir": str(out), "minutes": round(minutes, 1), "best_eval_loss": best[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["coder", "selfcritique", "judge"],
                    choices=["coder", "judge", "selfcritique"])
    ap.add_argument("--data", default="data/final_python")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--smoke", action="store_true", help="30 steps per task, to test the runner")
    ap.add_argument("--batch-size", type=int, default=CONFIG["batch_size"],
                    help="examples per step. 1 fits an L4; 4 needs an A100/H100 (more memory, much faster)")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="default: 16 / batch-size, so the effective batch stays 16")
    args = ap.parse_args()
    CONFIG["batch_size"] = args.batch_size
    CONFIG["grad_accum"] = args.grad_accum or max(1, 16 // args.batch_size)
    log(f"batch {CONFIG['batch_size']} x accumulation {CONFIG['grad_accum']} "
        f"= effective batch {CONFIG['batch_size'] * CONFIG['grad_accum']}")

    status_path = ROOT / "checkpoints" / "run_all_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    data_dir = ROOT / args.data

    for task in args.tasks:
        key = f"{args.lang}_{task}" + ("_smoke" if args.smoke else "")
        status[key] = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M")}
        status_path.write_text(json.dumps(status, indent=1))
        try:
            status[key] = train_one(task, data_dir, args.lang, args.smoke)
        except Exception as err:
            log(f"{key} FAILED: {err}")
            status[key] = {"status": "failed", "error": str(err)[:500],
                           "traceback": traceback.format_exc()[-2000:]}
        status[key]["finished"] = time.strftime("%Y-%m-%d %H:%M")
        status_path.write_text(json.dumps(status, indent=1))

    log("all tasks processed:\n" + json.dumps({k: v["status"] for k, v in status.items()}, indent=1))
    # non-zero exit if a task in this run failed
    if any(status[f"{args.lang}_{t}" + ("_smoke" if args.smoke else "")]["status"] == "failed"
           for t in args.tasks):
        sys.exit(1)


if __name__ == "__main__":
    main()
