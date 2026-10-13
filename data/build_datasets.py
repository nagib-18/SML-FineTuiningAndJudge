"""
Build the coder, judge and self-critique training sets from data/processed.

Usage: python data/build_datasets.py [--language python] [--out data/final_python]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts as P  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["train", "val", "test"]
CHARS_PER_TOKEN = 3.5
TEMPLATE_OVERHEAD_TOKENS = 120   # system prompt, role markers, fixed prompt wording


def est_tokens(*texts: str) -> int:
    return int(sum(len(t or "") for t in texts) / CHARS_PER_TOKEN) + TEMPLATE_OVERHEAD_TOKENS


def example(task_id: str, kind: str, system: str, user: str, assistant: str, **meta) -> dict:
    return {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}],
        "task_id": task_id,
        "kind": kind,
        "est_tokens": est_tokens(system, user, assistant),
        **meta,
    }


def pick(rng: np.random.Generator, rows: list[dict]) -> dict:
    """Deterministic random choice of one row (ties broken by the seeded generator)."""
    return rows[int(rng.integers(len(rows)))]


def group_tasks(pairs: pd.DataFrame) -> list[tuple[str, list[dict]]]:
    """[(task_id, rows)] sorted by task_id; rows keep file order."""
    groups: dict[str, list[dict]] = {}
    for row in pairs.to_dict("records"):
        groups.setdefault(row["task_id"], []).append(row)
    return sorted(groups.items())


def drafts_for(rows: list[dict], best_rating: int, min_gap: int) -> list[dict]:
    """Weaker answers (at least `min_gap` below the best) that have a critique to learn from."""
    return [r for r in rows if r["rating"] <= best_rating - min_gap and r["rationale"]]


# Setup 1: coder
def build_coder(pairs: pd.DataFrame, rng, min_rating: int, min_gap: int,
                revision_share: float) -> list[dict]:
    fresh, revisions = [], []
    for task_id, rows in group_tasks(pairs):
        best_rating = max(r["rating"] for r in rows)
        if best_rating < min_rating:
            continue  # no answer good enough to imitate
        best = pick(rng, [r for r in rows if r["rating"] == best_rating])
        fresh.append(example(task_id, "fresh", P.SYSTEM_CODER, P.coder_prompt(best["instruction"]),
                             P.with_think(None, best["response"]),
                             dimension=best["preference"], target_rating=int(best_rating)))

        weak_rows = drafts_for(rows, best_rating, min_gap)
        if weak_rows:
            weak = pick(rng, weak_rows)
            user = P.revision_prompt(best["instruction"], weak["response"], weak["rationale"],
                                     best["preference"], int(weak["rating"]))
            revisions.append(example(task_id, "revision", P.SYSTEM_CODER, user,
                                     P.with_think(None, best["response"]),
                                     dimension=best["preference"], target_rating=int(best_rating),
                                     draft_rating=int(weak["rating"])))

    # Mix: revision examples make up `revision_share` of the coder set (if enough exist).
    wanted = int(round(revision_share / (1 - revision_share) * len(fresh))) if revision_share < 1 else len(revisions)
    if len(revisions) > wanted:
        keep = sorted(rng.choice(len(revisions), size=wanted, replace=False))
        revisions = [revisions[i] for i in keep]
    return fresh + revisions


# Setup 1: judge
def build_judge(pairs: pd.DataFrame) -> list[dict]:
    out = []
    rows = sorted(pairs.to_dict("records"), key=lambda r: (r["task_id"], r["model"]))
    for row in rows:
        if not row["rationale"]:
            continue
        out.append(example(row["task_id"], "judge", P.SYSTEM_JUDGE,
                           P.judge_prompt(row["instruction"], row["response"], row["preference"]),
                           P.with_think(None, P.judge_target(row["rationale"], row["rating"])),
                           dimension=row["preference"], rating=int(row["rating"]),
                           model=row["model"]))
    return out


def rebalance_ratings(examples: list[dict], max_share: float, rng) -> tuple[list[dict], dict]:
    """Downsample over-represented ratings until no rating exceeds `max_share` of the set."""
    by_rating = {r: [e for e in examples if e["rating"] == r] for r in range(1, 6)}
    original = {r: len(v) for r, v in by_rating.items()}
    present = sum(1 for n in original.values() if n)
    if present and max_share < 1 / present:
        raise ValueError(f"--judge-max-rating-share {max_share} is impossible with {present} "
                         f"ratings present; the minimum is {1 / present:.2f}")
    # every rating keeps min(count, C), with C as large as the cap allows
    def ok(cap):
        kept_total = sum(min(n, cap) for n in original.values())
        return kept_total > 0 and min(max(original.values()), cap) <= max_share * kept_total
    lo, hi = 1, max(original.values(), default=0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
    counts = {r: min(n, lo) for r, n in original.items()}
    kept = []
    for r, items in by_rating.items():
        if counts[r] < len(items):
            idx = sorted(rng.choice(len(items), size=counts[r], replace=False))
            items = [items[i] for i in idx]
        kept.extend(items)
    kept.sort(key=lambda e: (e["task_id"], e["model"]))
    return kept, counts


# Setup 2: self-critique
def build_selfcritique(pairs: pd.DataFrame, rng, min_rating: int, min_gap: int,
                       max_drafts: int) -> list[dict]:
    out = []
    for task_id, rows in group_tasks(pairs):
        best_rating = max(r["rating"] for r in rows)
        if best_rating < min_rating:
            continue
        best = pick(rng, [r for r in rows if r["rating"] == best_rating])
        weak_rows = drafts_for(rows, best_rating, min_gap)
        if not weak_rows:
            continue
        n = min(max_drafts, len(weak_rows))
        for i in sorted(rng.choice(len(weak_rows), size=n, replace=False)):
            weak = weak_rows[i]
            thinking = P.self_critique_thinking(weak["response"], best["preference"],
                                                int(weak["rating"]), weak["rationale"])
            out.append(example(task_id, "self_critique", P.SYSTEM_CODER,
                               P.coder_prompt(best["instruction"]),
                               P.with_think(thinking, best["response"]),
                               dimension=best["preference"], target_rating=int(best_rating),
                               draft_rating=int(weak["rating"])))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def render_preview(sets: dict[str, list[dict]]) -> str:
    lines = ["# Training examples preview", "",
             "One example of every kind from the train split, as the model will see it "
             "(before the chat template is applied). Long fields are cut at 700 characters.", ""]
    seen = set()
    for name, rows in sets.items():
        for ex in rows:
            if ex["kind"] in seen:
                continue
            seen.add(ex["kind"])
            lines += [f"## {name}: {ex['kind']} ({ex['task_id']}, ~{ex['est_tokens']} tokens)", ""]
            for msg in ex["messages"]:
                text = msg["content"]
                text = text if len(text) <= 700 else text[:700] + "\n[... cut ...]"
                # 4-backtick fence: the answers contain their own ``` code blocks
                lines += [f"**{msg['role']}**", "", "````text", text, "````", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pairs-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "final"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-coder-rating", type=int, default=4,
                    help="best answer must be rated at least this to be a training target")
    ap.add_argument("--min-gap", type=int, default=2,
                    help="a draft must be rated this many points below the best answer")
    ap.add_argument("--revision-share", type=float, default=0.3,
                    help="share of revision examples in the coder set")
    ap.add_argument("--max-drafts-per-task", type=int, default=2, help="setup 2 examples per task")
    ap.add_argument("--max-seq-tokens", type=int, default=4096,
                    help="drop examples estimated longer than this")
    ap.add_argument("--judge-max-rating-share", type=float, default=None,
                    help="cap any single rating at this share of judge TRAIN data, e.g. 0.35")
    ap.add_argument("--language", default=None, help="keep only this language, e.g. python")
    args = ap.parse_args(argv)

    pairs_dir, out = Path(args.pairs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits_path = pairs_dir / "splits.json"
    splits = json.loads(splits_path.read_text())
    rng = np.random.default_rng(args.seed)

    manifest = {"config": vars(args),
                "splits_json_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
                "sets": {}, "dropped_over_max_tokens": {}, "judge_train_rebalanced_counts": None}
    all_sets: dict[str, dict[str, list[dict]]] = {"coder": {}, "judge": {}, "selfcritique": {}}

    for split in SPLITS:
        pairs = pd.read_json(pairs_dir / f"pairs_{split}.jsonl", lines=True)
        pairs["rationale"] = pairs["rationale"].fillna("").astype(str)
        if args.language:
            pairs = pairs[pairs["language"] == args.language]
        built = {
            "coder": build_coder(pairs, rng, args.min_coder_rating, args.min_gap, args.revision_share),
            "judge": build_judge(pairs),
            "selfcritique": build_selfcritique(pairs, rng, args.min_coder_rating, args.min_gap,
                                               args.max_drafts_per_task),
        }
        if split == "train" and args.judge_max_rating_share:
            built["judge"], counts = rebalance_ratings(built["judge"], args.judge_max_rating_share, rng)
            manifest["judge_train_rebalanced_counts"] = counts

        for name, rows in built.items():
            kept = [r for r in rows if r["est_tokens"] <= args.max_seq_tokens]
            manifest["dropped_over_max_tokens"][f"{name}_{split}"] = len(rows) - len(kept)
            # Leakage guard: every example must come from a task assigned to this split.
            wrong = [r["task_id"] for r in kept if splits.get(r["task_id"]) != split]
            assert not wrong, f"{name}_{split}: {len(wrong)} examples from other splits, e.g. {wrong[:3]}"
            all_sets[name][split] = kept
            write_jsonl(out / f"{name}_{split}.jsonl", kept)
            info = {"examples": len(kept), "tasks": len({r["task_id"] for r in kept}),
                    "kinds": dict(Counter(r["kind"] for r in kept))}
            if name == "judge":
                info["ratings"] = dict(sorted(Counter(r["rating"] for r in kept).items()))
            tok = [r["est_tokens"] for r in kept] or [0]
            info["est_tokens_p50_p95_max"] = [int(np.percentile(tok, 50)), int(np.percentile(tok, 95)),
                                              int(max(tok))]
            manifest["sets"][f"{name}_{split}"] = info

    # setup 2 and the coder revisions should come from the same tasks
    manifest["note"] = ("Coder revision examples and setup 2 examples use the same draft/critique "
                        "material, so differences between setups come from the architecture.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "preview.md").write_text(render_preview({n: s["train"] for n, s in all_sets.items()}))

    print(f"{'set':<20}{'examples':>10}{'tasks':>8}   kinds")
    for key, info in manifest["sets"].items():
        print(f"{key:<20}{info['examples']:>10}{info['tasks']:>8}   {info['kinds']}")
    print(f"\nDropped for length: {sum(manifest['dropped_over_max_tokens'].values())}")
    print(f"Wrote {out.resolve()}  (read preview.md before training)")
    return manifest


if __name__ == "__main__":
    main()
