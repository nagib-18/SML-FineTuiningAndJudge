"""
Setup 1: coder writes, judge rates, coder revises until the rating is 4/5 (max 3 revisions).

Usage: python eval/run_loop.py [--quick] [--only base_coder+base_judge]
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
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
from run_eval import BASE, CK, load_model, log, model_ready  # noqa: E402

THRESHOLD = 4
MAX_REVISIONS = 3                  # at most 4 answers per problem
DIMENSION = "instruction-following"  # judge dimension closest to "does it do what was asked"
CODER_TOKENS = 1024
JUDGE_TOKENS = 300
DATASETS = ["humaneval", "mbpp"]

# base judge gets the format prompt, trained judges get their training prompt
FORMAT_JUDGE_PROMPT = P.SYSTEM_JUDGE_FORMAT

# (name, coder, judge)
CONDITIONS = [
    ("base_coder+base_judge", BASE, BASE),
    ("base_coder+all_judge", BASE, CK / "all_judge/merged"),
    ("python_coder+python_judge", CK / "python_coder/merged", CK / "python_judge/merged"),
    ("all_coder+all_judge", CK / "all_coder/merged", CK / "all_judge/merged"),
]


def training_judge_prompt() -> str:
    with open(ROOT / "data/final_python/judge_test.jsonl") as f:
        return json.loads(f.readline())["messages"][0]["content"]


def load(name, dtype):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(name))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(name, dtype)
    model.eval()
    return model, tok


def run_evalplus(results: Path, tag: str, dataset: str, solutions: dict, all_ids: list) -> dict:
    samples = [{"task_id": t, "solution": s} for t, s in solutions.items()]
    path = results / f"{tag}_samples.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in E.fill_missing_problems(samples, all_ids)))
    for old in glob.glob(str(results / f"{tag}_samples*eval_results.json")):
        os.remove(old)
    subprocess.run(["evalplus.evaluate", "--dataset", dataset, "--samples", str(path)],
                   check=True, input="Y\n", text=True, stdout=subprocess.DEVNULL)
    found = sorted(glob.glob(str(results / f"{tag}_samples*eval_results.json")))
    assert found, f"no EvalPlus results for {tag}"
    per_task = E.parse_evalplus_results(found[-1])
    per_task = {t: per_task[t] for t in solutions}
    (results / f"{tag}_per_task.json").write_text(json.dumps(per_task, indent=1))
    return per_task


def to_solution(text: str, entry_point: str, sanitize) -> str:
    answer, _ = E.strip_thinking(text, False)
    code = E.extract_code(answer)
    if sanitize:
        try:
            code = sanitize(code, entry_point)
        except Exception:
            pass
    return code


def run_condition(name, coder_name, judge_name, args, dtype):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    try:
        from evalplus.sanitize import sanitize
    except ImportError:
        sanitize = None
    results = ROOT / "results" / ("loop_" + name + ("_quick" if args.quick else ""))
    results.mkdir(parents=True, exist_ok=True)
    if (results / "loop_summary.json").exists():
        log(f"{name}: already done, skipping")
        return "skipped"

    coder, ctok = load(coder_name, dtype)
    if judge_name == coder_name:
        judge, jtok = coder, ctok
    else:
        judge, jtok = load(judge_name, dtype)
    judge_system = FORMAT_JUDGE_PROMPT if judge_name == BASE else training_judge_prompt()

    summary = {"condition": name, "coder": str(coder_name), "judge": str(judge_name),
               "threshold": THRESHOLD, "max_revisions": MAX_REVISIONS, "dimension": DIMENSION,
               "judge_system_prompt": judge_system, "decoding": "greedy, thinking off", "datasets": {}}
    for dataset in DATASETS:
        problems = {"humaneval": get_human_eval_plus, "mbpp": get_mbpp_plus}[dataset]()
        ids = sorted(problems)[:5] if args.quick else sorted(problems)
        tasks = {t: E.codegen_messages(problems[t]["prompt"])[1]["content"] for t in ids}
        history = {t: [] for t in ids}       # list of {text, rating, rationale, judge_text}
        active = list(ids)
        t0 = time.time()
        for rnd in range(MAX_REVISIONS + 1):
            # coder
            convs = []
            for t in active:
                if rnd == 0:
                    user = tasks[t]
                else:
                    prev = history[t][-1]
                    user = P.revision_prompt(tasks[t], E.strip_thinking(prev["text"], False)[0],
                                             prev["rationale"] or "The review could not be read; check the answer carefully.",
                                             DIMENSION, prev["rating"] or 1)
                convs.append([{"role": "system", "content": P.SYSTEM_CODER}, {"role": "user", "content": user}])
            gens = E.generate(coder, ctok, convs, enable_thinking=False, max_new_tokens=CODER_TOKENS,
                              batch_size=args.batch_size, seed=rnd)
            # judge
            jconvs = [[{"role": "system", "content": judge_system},
                       {"role": "user", "content": P.judge_prompt(tasks[t], E.strip_thinking(g["text"], False)[0], DIMENSION)}]
                      for t, g in zip(active, gens)]
            jgens = E.generate(judge, jtok, jconvs, enable_thinking=False, max_new_tokens=JUDGE_TOKENS,
                               batch_size=args.batch_size, seed=rnd)
            for t, g, j in zip(active, gens, jgens):
                rating, rationale = P.parse_judge_output(j["text"].replace("**", ""))
                history[t].append({"round": rnd, "text": g["text"], "new_tokens": g["new_tokens"],
                                   "rating": rating, "rationale": rationale, "judge_text": j["text"]})
            active = [t for t in active if (history[t][-1]["rating"] or 0) < THRESHOLD]
            log(f"  {name} {dataset} round {rnd}: {len(active)} below {THRESHOLD}/5")
            if not active:
                break
        minutes = (time.time() - t0) / 60
        (results / f"{dataset}_history.jsonl").write_text("\n".join(
            json.dumps({"task_id": t, "rounds": h}) for t, h in history.items()))

        def pick(h, which):
            if which == "draft":
                return h[0]
            if which == "last":
                return h[-1]
            best = max(r["rating"] or 0 for r in h)
            return [r for r in h if (r["rating"] or 0) == best][-1]

        passes = {}
        for which in ["draft", "last", "best"]:
            sols = {t: to_solution(pick(history[t], which)["text"], problems[t]["entry_point"], sanitize) for t in ids}
            pt = run_evalplus(results, f"{dataset}_{which}", dataset, sols, sorted(problems))
            passes[which] = {t: v["plus"] for t, v in pt.items()}

        rounds_used = [len(h) - 1 for h in history.values()]
        draft_r = [(history[t][0]["rating"], passes["draft"][t]) for t in ids]
        accepted = [p for r, p in draft_r if (r or 0) >= THRESHOLD]
        rejected = [p for r, p in draft_r if (r or 0) < THRESHOLD]
        ds = {
            "problems": len(ids), "minutes": round(minutes, 1),
            "pass@1_plus": {w: sum(p.values()) / len(p) for w, p in passes.items()},
            "ci95": {w: E.bootstrap_ci(list(p.values()), seed=0)[1:] for w, p in passes.items()},
            "draft_vs_last": E.mcnemar_exact(passes["draft"], passes["last"]),
            "draft_vs_best": E.mcnemar_exact(passes["draft"], passes["best"]),
            "revisions_per_problem": statistics.mean(rounds_used),
            "rounds_histogram": {k: rounds_used.count(k) for k in range(MAX_REVISIONS + 1)},
            "reached_threshold": sum((h[-1]["rating"] or 0) >= THRESHOLD for h in history.values()) / len(ids),
            "judge_unreadable": sum(r["rating"] is None for h in history.values() for r in h)
                                / sum(len(h) for h in history.values()),
            # Can the judge be trusted? Pass rate of round-0 code it accepted vs rejected.
            "draft_pass_rate_if_judge_accepted": sum(accepted) / len(accepted) if accepted else None,
            "draft_pass_rate_if_judge_rejected": sum(rejected) / len(rejected) if rejected else None,
            "drafts_accepted": len(accepted),
            "changed_by_loop": {"fixed": sum(passes["last"][t] and not passes["draft"][t] for t in ids),
                                "broken": sum(passes["draft"][t] and not passes["last"][t] for t in ids)},
        }
        summary["datasets"][dataset] = ds
        log(f"  {name} {dataset}: draft {ds['pass@1_plus']['draft']:.1%} -> last {ds['pass@1_plus']['last']:.1%} "
            f"(best {ds['pass@1_plus']['best']:.1%}), p={ds['draft_vs_last']['p_value']:.3f}, "
            f"judge-accepted drafts pass {ds['draft_pass_rate_if_judge_accepted']} vs rejected {ds['draft_pass_rate_if_judge_rejected']}")
    (results / "loop_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    del coder, judge
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="+", help="condition names to run")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    plan = [c for c in CONDITIONS if not args.only or c[0] in args.only]
    for name, coder, judge in plan:
        log(f"{name}: coder {'READY' if model_ready(coder) else 'NOT READY'}, judge {'READY' if model_ready(judge) else 'NOT READY'}")
    if args.dry_run:
        return
    import torch
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    status_path = ROOT / "results" / ("loop_status_quick.json" if args.quick else "loop_status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for name, coder, judge in plan:
        if not (model_ready(coder) and model_ready(judge)):
            status[name] = {"status": "not_ready"}
            continue
        t0 = time.time()
        try:
            log(f"=== {name} ===")
            status[name] = {"status": run_condition(name, coder, judge, args, dtype)}
        except Exception as err:
            log(f"{name} FAILED: {err}")
            status[name] = {"status": "failed", "error": str(err)[:500], "traceback": traceback.format_exc()[-2000:]}
            gc.collect()
            torch.cuda.empty_cache()
        status[name]["minutes"] = round((time.time() - t0) / 60, 1)
        status_path.write_text(json.dumps(status, indent=1))
    log("ALL DONE:\n" + json.dumps({k: v["status"] for k, v in status.items()}, indent=1))


if __name__ == "__main__":
    main()
