"""
Evaluate all models on HumanEval+/MBPP+ and the judge test set.

Usage: python eval/run_eval.py [--quick] [--only base python_coder]
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path[:0] = [str(ROOT / d) for d in ["data", "train", "eval"]]
import eval_utils as E  # noqa: E402
import prompts as P  # noqa: E402

CK = ROOT / "checkpoints"
BASE = "Qwen/Qwen3.5-0.8B"
MAX_NEW_TOKENS = {False: 1024, True: 4096}
JUDGE_MAX_NEW_TOKENS = 300
JUDGE_SAMPLE = 600
SEED = 0
DATASETS = ["humaneval", "mbpp"]
FORMAT_JUDGE_PROMPT = P.SYSTEM_JUDGE_FORMAT

# (run name, model, thinking modes, run judge?), cheap runs first
PLAN = [
    ("base",                BASE,                                  [],      True),
    ("python_judge",        CK / "python_judge/merged",            [],      True),
    ("base",                BASE,                                  [False], False),
    ("python_coder",        CK / "python_coder/merged",            [False], False),
    ("all_coder",           CK / "all_coder/merged",               [False], False),
    ("python_selfcritique", CK / "python_selfcritique/merged",     [True],  False),
    ("all_selfcritique",    CK / "all_selfcritique/merged",        [True],  False),
    ("base",                BASE,                                  [True],  False),
    ("all_judge",           CK / "all_judge/merged",               [],      True),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_ready(model) -> bool:
    if model == BASE:
        return True
    model = Path(model)
    # run_config.json is written last, so its presence means training finished
    weights = list(model.glob("*.safetensors"))
    return (model / "run_config.json").exists() and bool(weights) and all(
        w.stat().st_size > 1_000_000 for w in weights)


def load_model(name, dtype):
    import transformers as tf
    for cls_name in ["AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"]:
        cls = getattr(tf, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(str(name), torch_dtype=dtype, device_map="cuda")
            log(f"loaded with {cls_name}")
            return model
        except (ValueError, KeyError) as err:
            log(f"{cls_name} failed: {str(err)[:120]}")
    raise RuntimeError("could not load the model; upgrade transformers")


def codegen_done(results: Path, tag: str, n_problems: int) -> bool:
    f = results / f"{tag}_summary.json"
    return f.exists() and json.loads(f.read_text()).get("problems") == n_problems


def run_codegen(model, tokenizer, model_name, results, dataset, thinking, args):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    try:
        from evalplus.sanitize import sanitize
    except ImportError:
        sanitize = None
    problems = {"humaneval": get_human_eval_plus, "mbpp": get_mbpp_plus}[dataset]()
    tag = f"{dataset}_{'think' if thinking else 'nothink'}"
    task_ids = sorted(problems)[:5] if args.quick else sorted(problems)
    if codegen_done(results, tag, len(task_ids)):
        log(f"  {tag}: already done, skipping")
        return "skipped"

    log(f"  {tag}: generating {len(task_ids)} problems")
    convs = [E.codegen_messages(problems[t]["prompt"]) for t in task_ids]
    gens = E.generate(model, tokenizer, convs, enable_thinking=thinking,
                      max_new_tokens=MAX_NEW_TOKENS[thinking], batch_size=args.batch_size, seed=SEED)
    samples, unfinished = [], 0
    for t, g in zip(task_ids, gens):
        answer, finished = E.strip_thinking(g["text"], thinking)
        unfinished += not finished
        code_text = E.extract_code(answer)
        if sanitize:
            try:
                code_text = sanitize(code_text, problems[t]["entry_point"])
            except Exception:
                pass
        samples.append({"task_id": t, "solution": code_text})
    sample_path = results / f"{tag}_samples.jsonl"
    sample_path.write_text("\n".join(json.dumps(s) for s in E.fill_missing_problems(samples, sorted(problems))))
    (results / f"{tag}_raw.jsonl").write_text("\n".join(
        json.dumps({"task_id": t, **g}) for t, g in zip(task_ids, gens)))

    for old in glob.glob(str(results / f"{tag}_samples*eval_results.json")) + glob.glob(str(results / "eval_results.json")):
        os.remove(old)
    log(f"  {tag}: running EvalPlus tests")
    subprocess.run(["evalplus.evaluate", "--dataset", dataset, "--samples", str(sample_path)],
                   check=True, input="Y\n", text=True, stdout=subprocess.DEVNULL)
    found = (sorted(glob.glob(str(results / f"{tag}_samples*eval_results.json")))
             or sorted(glob.glob(str(results / "eval_results.json"))))
    assert found, f"no EvalPlus results file in {results}"
    per_task = E.parse_evalplus_results(found[-1])
    per_task = {t: per_task[t] for t in task_ids if t in per_task}

    rate, lo, hi = E.bootstrap_ci([v["plus"] for v in per_task.values()], seed=SEED)
    stats = {
        "pass@1_plus": rate, "ci95": [lo, hi],
        "pass@1_base": sum(v["base"] for v in per_task.values()) / len(per_task),
        "problems": len(per_task), "thinking_unfinished": unfinished,
        "mean_new_tokens": statistics.mean(g["new_tokens"] for g in gens),
        "mean_seconds": statistics.mean(g["seconds"] for g in gens),
        "decoding": E.DECODING["thinking" if thinking else "greedy"],
        "max_new_tokens": MAX_NEW_TOKENS[thinking], "model": str(model_name),
        "batch_size": args.batch_size,
    }
    (results / f"{tag}_per_task.json").write_text(json.dumps(per_task, indent=1))
    (results / f"{tag}_summary.json").write_text(json.dumps(stats, indent=1))
    log(f"  {tag}: pass@1+ {rate:.1%} [{lo:.1%}-{hi:.1%}]  base tests {stats['pass@1_base']:.1%}  "
        f"tokens {stats['mean_new_tokens']:.0f}  unfinished thinking {unfinished}")
    return "done"


def run_judge(model, tokenizer, model_name, results, args):
    from sft_utils import load_jsonl
    judge_test = load_jsonl(ROOT / args.judge_data / "judge_test.jsonl")
    sample = random.Random(SEED).sample(judge_test, min(JUDGE_SAMPLE, len(judge_test)))
    if args.quick:
        sample = sample[:10]
    true = [ex["rating"] for ex in sample]
    out = {}
    for variant in ["promptnew", "promptold"]:
        tag = f"judge_{variant}"
        f = results / f"{tag}_metrics.json"
        if f.exists() and json.loads(f.read_text()).get("n") == len(sample):
            log(f"  {tag}: already done, skipping")
            out[tag] = "skipped"
            continue
        convs = []
        for ex in sample:
            msgs = [dict(m) for m in ex["messages"][:-1]]          # system + user, no answer
            assert msgs[0]["role"] == "system"
            if variant == "promptnew":
                msgs[0]["content"] = FORMAT_JUDGE_PROMPT
            convs.append(msgs)
        system_used = convs[0][0]["content"]
        log(f"  {tag}: {len(sample)} items")
        gens = E.generate(model, tokenizer, convs, enable_thinking=False,
                          max_new_tokens=JUDGE_MAX_NEW_TOKENS, batch_size=args.batch_size, seed=SEED)
        pred = [P.parse_judge_output(g["text"].replace("**", ""))[0] for g in gens]
        metrics = E.judge_metrics(true, pred)
        metrics.update({"model": str(model_name), "system_prompt": system_used,
                        "test_file": f"{args.judge_data}/judge_test.jsonl",
                        "mean_new_tokens": statistics.mean(g["new_tokens"] for g in gens),
                        "mean_seconds": statistics.mean(g["seconds"] for g in gens)})
        f.write_text(json.dumps(metrics, indent=1))
        (results / f"{tag}_raw.jsonl").write_text("\n".join(
            json.dumps({"task_id": ex["task_id"], "true": t, "pred": p, "text": g["text"]})
            for ex, t, p, g in zip(sample, true, pred, gens)))
        log(f"  {tag}: parse {metrics['parse_rate']:.0%}  acc {metrics.get('accuracy', 0):.1%}  "
            f"qwk {metrics.get('qwk', float('nan')):.3f}  (majority baseline "
            f"{metrics['majority_baseline_accuracy']:.1%})")
        out[tag] = "done"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="+", help="run names to include, e.g. base python_coder")
    ap.add_argument("--quick", action="store_true",
                    help="5 problems / 10 judge items per run, into results/<run>_quick/")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--batch-size", type=int, default=64, help="64 fits an L4 for a 0.8B model")
    ap.add_argument("--judge-data", default="data/final_python")
    ap.add_argument("--think-tokens", type=int, default=MAX_NEW_TOKENS[True],
                    help="token limit with thinking on; a non-default value writes to "
                         "results/<run>_think<N>/ (sensitivity check, e.g. 8192)")
    args = ap.parse_args()
    suffix = "_quick" if args.quick else ""
    if args.think_tokens != MAX_NEW_TOKENS[True]:
        MAX_NEW_TOKENS[True] = args.think_tokens
        suffix += f"_think{args.think_tokens}"

    plan = [p for p in PLAN if not args.only or p[0] in args.only]
    status_path = ROOT / "results" / f"eval_status{suffix}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    for name, model_name, modes, judge in plan:
        what = ([f"codegen {'think' if m else 'nothink'}" for m in modes] + (["judge"] if judge else []))
        log(f"{name}: {', '.join(what)}  | model {'READY' if model_ready(model_name) else 'NOT READY'}")
    if args.dry_run:
        return

    import torch
    from transformers import AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log(f"GPU {torch.cuda.get_device_name(0)}, batch {args.batch_size}")

    for name, model_name, modes, judge in plan:
        key = f"{name}:" + ",".join(["think" if m else "nothink" for m in modes] + (["judge"] if judge else []))
        if not model_ready(model_name):
            log(f"{key}: model not ready yet ({model_name}), skipping")
            status[key] = {"status": "not_ready"}
            status_path.write_text(json.dumps(status, indent=1))
            continue
        results = ROOT / "results" / (name + suffix)
        results.mkdir(parents=True, exist_ok=True)
        status[key] = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M")}
        status_path.write_text(json.dumps(status, indent=1))
        t0 = time.time()
        model = None
        try:
            log(f"=== {key} ===")
            tokenizer = AutoTokenizer.from_pretrained(str(model_name))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = load_model(model_name, dtype)
            model.eval()
            parts = {}
            for dataset in DATASETS:
                for thinking in modes:
                    parts[f"{dataset}_{'think' if thinking else 'nothink'}"] = run_codegen(
                        model, tokenizer, model_name, results, dataset, thinking, args)
            if judge:
                parts.update(run_judge(model, tokenizer, model_name, results, args))
            status[key] = {"status": "done", "parts": parts}
        except Exception as err:
            log(f"{key} FAILED: {err}")
            status[key] = {"status": "failed", "error": str(err)[:500],
                           "traceback": traceback.format_exc()[-2000:]}
        status[key]["minutes"] = round((time.time() - t0) / 60, 1)
        status[key]["finished"] = time.strftime("%Y-%m-%d %H:%M")
        status_path.write_text(json.dumps(status, indent=1))
        log(f"{key}: {status[key]['status']} in {status[key]['minutes']} min")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    log("ALL DONE:\n" + json.dumps({k: v["status"] for k, v in status.items()}, indent=1))


if __name__ == "__main__":
    main()
