"""Helpers for evaluation: generation, code extraction, statistics."""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import prompts as P  # noqa: E402

# greedy without thinking; with thinking Qwen's recommended sampling (greedy makes it repeat)
DECODING = {
    "greedy": dict(do_sample=False),
    "thinking": dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20),
}


# Prompts for code benchmarks
def codegen_messages(problem_prompt: str) -> list[dict]:
    """Chat prompt for one HumanEval+/MBPP+ problem, with the same system prompt as training."""
    user = ("Complete the following Python function. Return the complete function, including "
            "the signature and any imports, in one ```python code block.\n\n"
            f"```python\n{problem_prompt.rstrip()}\n```")
    return [{"role": "system", "content": P.SYSTEM_CODER}, {"role": "user", "content": user}]


def strip_thinking(text: str, thinking: bool = False) -> tuple[str, bool]:
    """Remove the thinking block. Returns (answer, thinking_finished)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip(), True
    if thinking or "<think>" in text:
        return "", False
    return text.strip(), True


CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_code(answer: str) -> str:
    """The longest fenced code block, else the raw answer (EvalPlus sanitizes it further)."""
    blocks = CODE_BLOCK_RE.findall(answer)
    return max(blocks, key=len).strip() if blocks else answer.strip()


END_MARKERS = ("<|im_end|>", "<|endoftext|>")


def clean_generation(text: str, tokenizer=None) -> str:
    """Cut at the first end-of-turn marker and drop padding tokens."""
    markers = list(END_MARKERS)
    if tokenizer is not None and tokenizer.eos_token:
        markers.append(tokenizer.eos_token)
    cut = min((text.find(m) for m in markers if m in text), default=len(text))
    text = text[:cut]
    if tokenizer is not None and tokenizer.pad_token:
        text = text.replace(tokenizer.pad_token, "")
    return text.strip()


def fill_missing_problems(samples: list[dict], all_task_ids: list[str]) -> list[dict]:
    """Give missing problems an empty solution, EvalPlus needs all of them."""
    have = {s["task_id"] for s in samples}
    return samples + [{"task_id": t, "solution": ""} for t in all_task_ids if t not in have]


# Generation
def generate(model, tokenizer, conversations: list[list[dict]], enable_thinking: bool,
             max_new_tokens: int, batch_size: int = 16, seed: int = 0) -> list[dict]:
    """Batched generation. Returns per conversation: text, new token count, seconds."""
    import torch
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"   # left padding: all prompts end at the same position
    decoding = DECODING["thinking" if enable_thinking else "greedy"]
    results = []
    for start in range(0, len(conversations), batch_size):
        chunk = conversations[start:start + batch_size]
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=enable_thinking) for c in chunk]
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id, **decoding)
        seconds = (time.time() - t0) / len(chunk)
        new = out[:, batch["input_ids"].shape[1]:]
        for row in new:
            n_tokens = int((row != tokenizer.pad_token_id).sum())
            # keep special tokens, otherwise the <think> tags get removed too
            text = clean_generation(tokenizer.decode(row, skip_special_tokens=False), tokenizer)
            results.append({"text": text, "new_tokens": n_tokens, "seconds": seconds})
        print(f"  generated {min(start + batch_size, len(conversations))}/{len(conversations)}",
              end="\r")
    print()
    return results


# EvalPlus results
def parse_evalplus_results(path: str | Path) -> dict[str, dict[str, bool]]:
    """{task_id: {"base": passed original tests, "plus": passed original + extra tests}}."""
    data = json.loads(Path(path).read_text())
    out = {}
    for task_id, runs in data["eval"].items():
        run = runs[0] if isinstance(runs, list) else runs
        out[task_id] = {"base": run.get("base_status") == "pass",
                        "plus": run.get("base_status") == "pass" and run.get("plus_status") == "pass"}
    return out


# Statistics
def bootstrap_ci(passes: list[bool], n_resamples: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """(pass rate, lower, upper): resample problems with replacement."""
    rng = random.Random(seed)
    n = len(passes)
    rate = sum(passes) / n
    stats = sorted(sum(passes[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo = stats[int((1 - level) / 2 * n_resamples)]
    hi = stats[int((1 + level) / 2 * n_resamples) - 1]
    return rate, lo, hi


def mcnemar_exact(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """Paired comparison of two models on the same problems."""
    common = sorted(set(a) & set(b))
    fixed = sum(1 for t in common if b[t] and not a[t])
    broken = sum(1 for t in common if a[t] and not b[t])
    n = fixed + broken
    if n == 0:
        p = 1.0
    else:
        k = min(fixed, broken)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"problems": len(common), "fixed": fixed, "broken": broken, "p_value": p}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for several comparisons at once."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else float("nan")


def quadratic_weighted_kappa(true: list[int], pred: list[int], k: int = 5) -> float:
    """Agreement on a 1..k scale; a 1-vs-5 disagreement costs much more than 4-vs-5."""
    n = len(true)
    obs = [[0] * k for _ in range(k)]
    for t, p in zip(true, pred):
        obs[t - 1][p - 1] += 1
    hist_t = [sum(row) for row in obs]
    hist_p = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * obs[i][j]
            den += w * hist_t[i] * hist_p[j] / n
    return 1 - num / den if den else float("nan")


def judge_metrics(true: list[int], pred: list[int | None]) -> dict:
    """Judge metrics; unreadable outputs only count in parse_rate."""
    pairs = [(t, p) for t, p in zip(true, pred) if p is not None]
    majority = max(set(true), key=true.count)
    out = {"n": len(true), "parse_rate": len(pairs) / len(true),
           "majority_baseline_accuracy": true.count(majority) / len(true),
           "majority_rating": majority}
    if pairs:
        t, p = zip(*pairs)
        out.update({
            "n_parsed": len(pairs),
            "accuracy": sum(a == b for a, b in pairs) / len(pairs),
            "within_1": sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs),
            "mae": sum(abs(a - b) for a, b in pairs) / len(pairs),
            "qwk": quadratic_weighted_kappa(list(t), list(p)),
            "spearman": spearman(list(t), list(p)),
            "predicted_distribution": {r: list(p).count(r) for r in range(1, 6)},
        })
    return out
