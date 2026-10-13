"""
Make the result tables (CIs, McNemar tests, failure reasons) from results/.

Usage: python eval/compare.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "eval")]
import eval_utils as E  # noqa: E402
import prompts as P  # noqa: E402

DATASETS = ["humaneval", "mbpp"]
# (label, reference model, compared model)
COMPARISONS = [
    ("python_coder vs base (thinking off)", ("base", "nothink"), ("python_coder", "nothink")),
    ("all_coder vs base (thinking off)", ("base", "nothink"), ("all_coder", "nothink")),
    ("all_coder vs python_coder", ("python_coder", "nothink"), ("all_coder", "nothink")),
    ("python_selfcritique vs base (thinking on)", ("base", "think"), ("python_selfcritique", "think")),
    ("all_selfcritique vs base (thinking on)", ("base", "think"), ("all_selfcritique", "think")),
    ("python_selfcritique vs base (thinking off)", ("base", "nothink"), ("python_selfcritique", "think")),
    ("all_selfcritique vs base (thinking off)", ("base", "nothink"), ("all_selfcritique", "think")),
    ("all_selfcritique vs python_selfcritique", ("python_selfcritique", "think"), ("all_selfcritique", "think")),
]
JUDGE_ORDER = ["base", "python_judge", "all_judge"]


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


# Failure analysis
def repeats(text: str, min_len: int = 20, times: int = 4) -> bool:
    """True if some non-trivial line occurs `times` or more: the model is looping."""
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= min_len]
    return bool(lines) and Counter(lines).most_common(1)[0][1] >= times


def failure_reason(raw: dict, thinking: bool, passed: bool, entry_point: str | None) -> str:
    if passed:
        return "passed"
    answer, finished = E.strip_thinking(raw["text"], thinking)
    if not finished:
        return "thinking unfinished"
    if "```" not in answer and "def " not in answer:
        return "no code"
    code = E.extract_code(answer)
    try:
        ast.parse(code)
    except SyntaxError:
        return "code does not parse"
    if entry_point and not re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code):
        return "function missing/renamed"
    return "wrong result"


def normalized(code: str) -> str:
    """Code without comments, docstrings and formatting, to spot cosmetic-only revisions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return re.sub(r"\s+", "", code)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def revision_stats(history_path: Path) -> dict:
    rows = load_jsonl(history_path)
    rev = [r for r in rows if len(r["rounds"]) > 1]
    code = lambda t: E.extract_code(E.strip_thinking(t, False)[0])
    first = [code(r["rounds"][0]["text"]) for r in rev]
    last = [code(r["rounds"][-1]["text"]) for r in rev]
    return {"revised": len(rev),
            "code_changed": sum(a != b for a, b in zip(first, last)),
            "logic_changed": sum(normalized(a) != normalized(b) for a, b in zip(first, last))}


def entry_points(dataset: str) -> dict:
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        problems = {"humaneval": get_human_eval_plus, "mbpp": get_mbpp_plus}[dataset]()
        return {t: p["entry_point"] for t, p in problems.items()}
    except Exception:
        return {}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    res = Path(args.results)
    runs = sorted(d.name for d in res.iterdir() if d.is_dir() and not d.name.endswith("_quick"))
    out = {"codegen": {}, "comparisons": [], "judge": {}, "failures": {}}
    md = ["# Results summary", ""]

    # code benchmarks
    per_task = {}
    md += ["## Code correctness (pass@1, HumanEval+ / MBPP+ extra tests)", "",
           "| Model | Mode | Benchmark | pass@1+ | 95% CI | pass@1 (base tests) | unfinished thinking | mean tokens |",
           "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        for ds in DATASETS:
            for mode in ["nothink", "think"]:
                s = load_json(res / run / f"{ds}_{mode}_summary.json")
                pt = load_json(res / run / f"{ds}_{mode}_per_task.json")
                if not s or not pt:
                    continue
                per_task[(run, mode, ds)] = {t: v["plus"] for t, v in pt.items()}
                out["codegen"][f"{run}/{ds}_{mode}"] = s
                md.append(f"| {run} | {mode} | {ds} | {pct(s['pass@1_plus'])} | "
                          f"{pct(s['ci95'][0])}–{pct(s['ci95'][1])} | {pct(s['pass@1_base'])} | "
                          f"{s['thinking_unfinished']}/{s['problems']} | {s['mean_new_tokens']:.0f} |")

    # paired tests
    tests = {}
    for label, (ra, ma), (rb, mb) in COMPARISONS:
        for ds in DATASETS:
            a, b = per_task.get((ra, ma, ds)), per_task.get((rb, mb, ds))
            if a is None or b is None:
                continue
            r = E.mcnemar_exact(a, b)
            r.update(label=label, dataset=ds,
                     rate_a=sum(a.values()) / len(a), rate_b=sum(b.values()) / len(b))
            tests[f"{label} | {ds}"] = r
    adjusted = E.holm({k: v["p_value"] for k, v in tests.items()}) if tests else {}
    md += ["", "## Paired comparisons (exact McNemar, Holm-corrected over all rows)", "",
           "| Comparison | Benchmark | A → B | fixed | broken | p | p (Holm) |",
           "|---|---|---|---|---|---|---|"]
    for k, r in tests.items():
        r["p_holm"] = adjusted[k]
        out["comparisons"].append(r)
        md.append(f"| {r['label']} | {r['dataset']} | {pct(r['rate_a'])} → {pct(r['rate_b'])} | "
                  f"{r['fixed']} | {r['broken']} | {r['p_value']:.4f} | {r['p_holm']:.4f} |")

    # failure analysis
    md += ["", "## Why answers failed", "",
           "Share of all problems per outcome. *Looping* = a line repeated 4+ times (degenerate repetition).", "",
           "| Model | Mode | Benchmark | passed | wrong result | function missing | does not parse | no code | thinking unfinished | looping |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    eps = {ds: entry_points(ds) for ds in DATASETS}
    for (run, mode, ds), passes in sorted(per_task.items()):
        raw = {r["task_id"]: r for r in load_jsonl(res / run / f"{ds}_{mode}_raw.jsonl")}
        if not raw:
            continue
        reasons = Counter(failure_reason(raw[t], mode == "think", passes[t], eps[ds].get(t))
                          for t in passes if t in raw)
        loops = sum(repeats(raw[t]["text"]) for t in passes if t in raw)
        n = sum(reasons.values())
        out["failures"][f"{run}/{ds}_{mode}"] = {**reasons, "looping": loops, "n": n}
        cols = ["passed", "wrong result", "function missing/renamed", "code does not parse",
                "no code", "thinking unfinished"]
        md.append(f"| {run} | {mode} | {ds} | " + " | ".join(pct(reasons[c] / n) for c in cols)
                  + f" | {pct(loops / n)} |")

    # judges
    md += ["", "## Judges (re-parsed with the current parser)", "",
           "| Judge | Prompt | readable | accuracy | 95% CI | within ±1 | MAE | QWK | Spearman | majority baseline | predicted 1/2/3/4/5 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    correct = {}
    judges = [r for r in JUDGE_ORDER if r in runs] + [r for r in runs if r.endswith("judge") and r not in JUDGE_ORDER]
    for run in judges:
        seen_prompts = set()
        for raw_path in sorted((res / run).glob("judge_prompt*_raw.jsonl")):
            rows = load_jsonl(raw_path)
            if not rows:
                continue
            variant = raw_path.name.removesuffix("_raw.jsonl")
            meta = load_json(res / run / f"{variant}_metrics.json") or {}
            if meta.get("system_prompt") in seen_prompts:
                continue          # same prompt already reported for this judge
            seen_prompts.add(meta.get("system_prompt"))
            true = [r["true"] for r in rows]
            pred = [P.parse_judge_output(r["text"].replace("**", ""))[0] for r in rows]
            m = E.judge_metrics(true, pred)
            # accuracy over ALL items (unreadable = wrong), so readability problems are not hidden
            ok = [p == t for t, p in zip(true, pred)]
            acc, lo, hi = E.bootstrap_ci(ok)
            m.update(accuracy_all_items=acc, ci95=[lo, hi], system_prompt=meta.get("system_prompt"))
            out["judge"][f"{run}/{variant}"] = m
            correct[(run, variant)] = {f"{r['task_id']}#{i}": c for i, (r, c) in enumerate(zip(rows, ok))}
            dist = m.get("predicted_distribution", {})
            prompt = "states format" if "Reply in exactly this format" in (meta.get("system_prompt") or "") else "training prompt"
            md.append(f"| {run} | {prompt} | {pct(m['parse_rate'])} | {pct(acc)} | {pct(lo)}–{pct(hi)} | "
                      f"{pct(m.get('within_1'))} | {m.get('mae', float('nan')):.2f} | {m.get('qwk', float('nan')):.3f} | "
                      f"{m.get('spearman', float('nan')):.3f} | {pct(m['majority_baseline_accuracy'])} | "
                      + "/".join(str(dist.get(k, 0)) for k in range(1, 6)) + " |")
    md += ["", "Accuracy counts unreadable answers as wrong. QWK: 0 = chance, 1 = perfect agreement.", ""]
    base_keys = [k for k in correct if k[0] == "base"]
    if base_keys:
        md += ["| Judge vs base judge (same items) | fixed | broken | p |", "|---|---|---|---|"]
        for key in correct:
            if key[0] == "base":
                continue
            ref = next((b for b in base_keys if b[1] == key[1]), base_keys[0])
            r = E.mcnemar_exact(correct[ref], correct[key])
            out.setdefault("judge_tests", {})[f"{key[0]}/{key[1]} vs base/{ref[1]}"] = r
            md.append(f"| {key[0]} ({key[1]}) vs base ({ref[1]}) | {r['fixed']} | {r['broken']} | {r['p_value']:.4f} |")

    # setup 1: coder + judge loop
    loops = [d for d in sorted(res.glob("loop_*")) if d.is_dir() and not d.name.endswith("_quick")
             and (d / "loop_summary.json").exists()]
    if loops:
        md += ["", "## Setup 1: coder + judge revise loop (threshold 4/5, max 3 revisions)", "",
               "| Condition | Benchmark | draft | last | best | fixed / broken (draft→last) | p | revised / code changed / logic changed | reached 4/5 | judge unreadable | drafts accepted: pass rate | rejected: pass rate |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in loops:
            s = load_json(d / "loop_summary.json")
            out.setdefault("loop", {})[s["condition"]] = s
            for ds, r in s["datasets"].items():
                p1 = r["pass@1_plus"]
                rs = revision_stats(d / f"{ds}_history.jsonl")
                r["revision_stats"] = rs
                md.append(f"| {s['condition']} | {ds} | {pct(p1['draft'])} | {pct(p1['last'])} | {pct(p1['best'])} | "
                          f"{r['changed_by_loop']['fixed']} / {r['changed_by_loop']['broken']} | "
                          f"{r['draft_vs_last']['p_value']:.4f} | {rs['revised']} / {rs['code_changed']} / {rs['logic_changed']} | "
                          f"{pct(r['reached_threshold'])} | {pct(r['judge_unreadable'])} | "
                          f"{pct(r['draft_pass_rate_if_judge_accepted'])} ({r['drafts_accepted']}) | "
                          f"{pct(r['draft_pass_rate_if_judge_rejected'])} |")
        md += ["", "Logic changed = the code differs after removing comments, docstrings and formatting. "
               "If the judge were useful, drafts it accepts would pass far more often than drafts it rejects.", ""]

    (res / "summary.md").write_text("\n".join(md) + "\n")
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(md))
    return out


if __name__ == "__main__":
    main()
