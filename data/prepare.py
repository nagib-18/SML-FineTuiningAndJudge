"""
Clean CodeUltraFeedback and make the task-level train/val/test split.

Usage: python data/prepare.py [--input FILE] [--tokenizer Qwen/Qwen3.5-0.8B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "coseal/CodeUltraFeedback"
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_RESPONSE_CHARS = 20          # shorter responses carry no usable signal
CHARS_PER_TOKEN_ESTIMATE = 3.5   # rough ratio for code when no tokenizer is given

# Language detection
FENCE_ALIASES = {
    "py": "python", "python": "python", "python3": "python",
    "js": "javascript", "javascript": "javascript", "node": "javascript", "jsx": "javascript",
    "ts": "typescript", "typescript": "typescript", "tsx": "typescript",
    "java": "java", "c": "c", "cpp": "cpp", "c++": "cpp", "cc": "cpp",
    "cs": "csharp", "csharp": "csharp", "c#": "csharp",
    "go": "go", "golang": "go", "rust": "rust", "rs": "rust",
    "lua": "lua", "ruby": "ruby", "rb": "ruby", "php": "php",
    "swift": "swift", "kotlin": "kotlin", "kt": "kotlin", "scala": "scala",
    "sql": "sql", "mysql": "sql", "postgresql": "sql", "sqlite": "sql",
    "r": "r", "perl": "perl", "haskell": "haskell", "hs": "haskell",
    "bash": "shell", "sh": "shell", "shell": "shell", "zsh": "shell", "powershell": "shell",
    "html": "html", "css": "css", "dart": "dart", "julia": "julia", "matlab": "matlab",
}
# Instruction keywords, used only when responses contain no tagged code fences.
INSTRUCTION_PATTERNS = [
    ("python", r"\bpython\b"),
    ("javascript", r"\bjavascript\b|\bnode\.?js\b|\breact\b"),
    ("typescript", r"\btypescript\b"),
    ("java", r"\bjava\b(?!\s*script)"),
    ("cpp", r"c\+\+"),
    ("csharp", r"c#|\.net\b"),
    ("go", r"\bgolang\b|\bin go\b"),
    ("rust", r"\brust\b"),
    ("lua", r"\blua\b"),
    ("ruby", r"\bruby\b"),
    ("php", r"\bphp\b"),
    ("swift", r"\bswift\b"),
    ("kotlin", r"\bkotlin\b"),
    ("sql", r"\bsql\b|\bquery\b.*\btable\b"),
    ("html", r"\bhtml\b|\bcss\b|\bsvg\b"),
    ("shell", r"\bbash\b|\bshell script\b"),
]
FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9+#]+)")


def detect_language(instruction: str, responses: list[str]) -> str:
    """Majority code-fence tag across the task's responses; fall back to instruction keywords."""
    tags = Counter()
    for text in responses:
        for raw in FENCE_RE.findall(text or ""):
            lang = FENCE_ALIASES.get(raw.lower())
            if lang:
                tags[lang] += 1
    if tags:
        return tags.most_common(1)[0][0]
    lowered = (instruction or "").lower()
    for lang, pattern in INSTRUCTION_PATTERNS:
        if re.search(pattern, lowered):
            return lang
    return "unknown"


# Loading
def load_raw(input_path: str | None) -> pd.DataFrame:
    if input_path is None:
        from datasets import load_dataset  # imported lazily: only needed for download
        return load_dataset(DATASET_ID, split="train").to_pandas()
    path = Path(input_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _as_list(value) -> list:
    """Parquet gives numpy arrays of dicts, jsonl gives lists. Normalise to list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value)
    return list(value)


# Explode tasks into (task, response) pairs
def parse_rating(raw) -> int | None:
    """Ratings are stored as strings ("3"), sometimes "N/A" or junk. Valid: integers 1-5."""
    if raw is None:
        return None
    match = re.fullmatch(r"\s*([1-5])(?:\.0)?\s*(?:/\s*5)?\s*", str(raw))
    return int(match.group(1)) if match else None


def normalise_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def explode_pairs(raw: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """One row per (task, response). Annotations are matched to responses BY MODEL NAME."""
    drops = Counter()
    rows = []
    for idx, task in raw.iterrows():
        task_id = f"cuf-{idx:05d}"
        instruction = task["instruction"]
        responses = _as_list(task["responses"])
        annotations = _as_list(task["annotations"])

        ann_by_model = {}
        for ann in annotations:
            model = ann.get("model")
            if model in ann_by_model:
                drops["duplicate_annotation_for_model"] += 1
                continue
            ann_by_model[model] = ann

        language = detect_language(instruction, [r.get("response", "") for r in responses])

        for resp in responses:
            model = resp.get("model")
            text = resp.get("response") or ""
            ann = ann_by_model.get(model)
            if ann is None:
                drops["no_annotation_for_response"] += 1
                continue
            rating = parse_rating(ann.get("rating"))
            if rating is None:
                drops["rating_missing_or_invalid"] += 1
                continue
            if len(text.strip()) < MIN_RESPONSE_CHARS:
                drops["response_empty_or_too_short"] += 1
                continue
            rows.append({
                "task_id": task_id,
                "instruction": instruction,
                "preference": task["preference"],
                "language": language,
                "model": model,
                "response": text,
                "rating": rating,
                "rationale": (ann.get("rationale") or "").strip(),
            })
    return pd.DataFrame(rows), drops


# Cleaning
def count_tokens(texts: pd.Series, tokenizer_name: str | None) -> tuple[pd.Series, str]:
    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        counts = [len(ids) for ids in tok(list(texts), add_special_tokens=False)["input_ids"]]
        return pd.Series(counts, index=texts.index), f"exact ({tokenizer_name})"
    est = (texts.str.len() / CHARS_PER_TOKEN_ESTIMATE).round().astype(int)
    return est, f"estimated (chars / {CHARS_PER_TOKEN_ESTIMATE})"


def clean(pairs: pd.DataFrame, drops: Counter, max_response_tokens: int,
          tokenizer_name: str | None, python_only: bool) -> tuple[pd.DataFrame, str]:
    df = pairs.copy()

    # Same response text submitted twice for one task: keep one, it adds no information.
    before = len(df)
    df = df.drop_duplicates(subset=["task_id", "response"])
    drops["duplicate_response_within_task"] += before - len(df)

    # Token lengths: needed to choose max sequence length for training.
    df["instruction_tokens"], token_method = count_tokens(df["instruction"], tokenizer_name)
    df["response_tokens"], _ = count_tokens(df["response"], tokenizer_name)
    too_long = df["response_tokens"] > max_response_tokens
    drops[f"response_over_{max_response_tokens}_tokens"] += int(too_long.sum())
    df = df[~too_long]

    if python_only:
        not_python = df["language"] != "python"
        drops["not_python (--python-only)"] += int(not_python.sum())
        df = df[~not_python]

    # Tasks left with fewer than 2 rated responses can't show a quality contrast.
    per_task = df.groupby("task_id")["response"].transform("size")
    drops["task_left_with_under_2_responses"] += int((per_task < 2).sum())
    df = df[per_task >= 2]

    # group identical instructions so copies can't end up in different splits
    df["group_id"] = df["instruction"].map(
        lambda s: hashlib.sha1(normalise_instruction(s).encode()).hexdigest()[:12])

    # Per-task summary columns used by the later training-set builders.
    df["task_max_rating"] = df.groupby("task_id")["rating"].transform("max")
    df["task_min_rating"] = df.groupby("task_id")["rating"].transform("min")
    df["is_best_in_task"] = df["rating"] == df["task_max_rating"]
    return df.reset_index(drop=True), token_method


# Split by task group, stratified by dimension
def split_by_group(df: pd.DataFrame, seed: int) -> dict[str, str]:
    """Assign every instruction group to exactly one split."""
    rng = np.random.default_rng(seed)
    group_dim = df.groupby("group_id")["preference"].first()
    group_split = {}
    for dim in sorted(group_dim.unique()):
        groups = np.array(sorted(group_dim[group_dim == dim].index))
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(SPLIT_FRACTIONS["train"] * n))
        n_val = int(round(SPLIT_FRACTIONS["val"] * n))
        for i, g in enumerate(groups):
            group_split[g] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    task_group = df.groupby("task_id")["group_id"].first()
    return {task: group_split[g] for task, g in task_group.items()}


def verify_split(df: pd.DataFrame) -> None:
    """Fail loudly if any instruction group or task appears in more than one split."""
    per_group = df.groupby("group_id")["split"].nunique()
    leaking = per_group[per_group > 1]
    assert leaking.empty, f"{len(leaking)} instruction groups appear in more than one split"
    per_task = df.groupby("task_id")["split"].nunique()
    assert (per_task == 1).all(), "a task has responses in more than one split"


# Report
def _pct_table(series: pd.Series) -> str:
    counts = series.value_counts().sort_index()
    total = counts.sum()
    return " · ".join(f"{k}: {v} ({v / total:.0%})" for k, v in counts.items())


def build_report(raw_rows: int, df: pd.DataFrame, drops: Counter, token_method: str,
                 args: argparse.Namespace) -> str:
    tasks = df.groupby("task_id").first()
    lines = [
        "# CodeUltraFeedback: cleaning and split report", "",
        f"- Source: `{args.input or DATASET_ID}`",
        f"- Seed: {args.seed} · Python only: {args.python_only} · "
        f"Max response tokens: {args.max_response_tokens}",
        f"- Token counts: {token_method}", "",
        "## Size", "",
        "| | Tasks | Pairs (task, response) |", "|---|---:|---:|",
        f"| Raw | {raw_rows} | {raw_rows * 4} (nominal) |",
        f"| After cleaning | {df['task_id'].nunique()} | {len(df)} |",
    ]
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        lines.append(f"| {split} | {part['task_id'].nunique()} | {len(part)} |")

    lines += ["", "## Dropped", "", "| Reason | Pairs |", "|---|---:|"]
    lines += [f"| {reason} | {n} |" for reason, n in sorted(drops.items()) if n] or ["| none | 0 |"]

    dup_groups = df.groupby("group_id")["task_id"].nunique()
    lines += ["", f"Instruction groups with more than one task (kept together in one split): "
                  f"{int((dup_groups > 1).sum())}"]

    lines += ["", "## Rating distribution", "", "| Split | Ratings 1–5 |", "|---|---|"]
    for split in ["train", "val", "test"]:
        lines.append(f"| {split} | {_pct_table(df.loc[df['split'] == split, 'rating'])} |")

    lines += ["", "| Dimension | Pairs | Mean rating | Ratings 1–5 |", "|---|---:|---:|---|"]
    for dim, part in df.groupby("preference"):
        lines.append(f"| {dim} | {len(part)} | {part['rating'].mean():.2f} | "
                     f"{_pct_table(part['rating'])} |")

    lines += ["", "## Languages (per task)", "", "| Language | Tasks | Share |", "|---|---:|---:|"]
    lang = tasks["language"].value_counts()
    for name, n in lang.head(15).items():
        lines.append(f"| {name} | {n} | {n / len(tasks):.1%} |")

    q = df["response_tokens"].quantile([0.5, 0.9, 0.95, 0.99]).round().astype(int)
    qi = df["instruction_tokens"].quantile([0.5, 0.95]).round().astype(int)
    lines += [
        "", "## Lengths (tokens)", "",
        f"- Response: median {q[0.5]}, p90 {q[0.9]}, p95 {q[0.95]}, p99 {q[0.99]}",
        f"- Instruction: median {qi[0.5]}, p95 {qi[0.95]}",
        f"- Setup 2 example ≈ instruction + draft + rationale + best answer: "
        f"p95 roughly {qi[0.95] + 2 * q[0.95] + 100} tokens before chat-template overhead",
    ]

    best = tasks["task_max_rating"]
    gap = tasks["task_max_rating"] - tasks["task_min_rating"]
    train_tasks = tasks[tasks["split"] == "train"]
    lines += [
        "", "## Usable training material (train split)", "",
        f"- Tasks whose best answer is rated ≥4 (coder targets): "
        f"{int((train_tasks['task_max_rating'] >= 4).sum())} of {len(train_tasks)}",
        f"- Tasks with a rating gap ≥2 (draft→best pairs for revision and setup 2): "
        f"{int(((train_tasks['task_max_rating'] - train_tasks['task_min_rating']) >= 2).sum())}",
        f"- Judge examples (all pairs): {int((df['split'] == 'train').sum())}",
        "", f"All splits: best ≥4 in {int((best >= 4).sum())} tasks, gap ≥2 in {int((gap >= 2).sum())}.",
        "", "## Decisions this report should inform", "",
        "1. Python only? Compare the Python share above with what training needs.",
        "2. Rebalance the judge data? If one rating dominates, the judge can score well by always predicting it.",
        "3. Max sequence length: cover ~95% of setup 2 examples (see lengths).",
        "4. Not done here: correctness check of coder targets (needs an external judge). "
        "Ratings score one dimension, not whether the code works.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", help="local parquet/jsonl; omit to download from Hugging Face")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-response-tokens", type=int, default=3000)
    parser.add_argument("--tokenizer", help="HF tokenizer for exact counts, e.g. Qwen/Qwen3.5-0.8B")
    parser.add_argument("--python-only", action="store_true")
    args = parser.parse_args(argv)

    raw = load_raw(args.input)
    print(f"Loaded {len(raw)} tasks")
    pairs, drops = explode_pairs(raw)
    df, token_method = clean(pairs, drops, args.max_response_tokens, args.tokenizer, args.python_only)

    splits = split_by_group(df, args.seed)
    df["split"] = df["task_id"].map(splits)
    verify_split(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "pairs_all.jsonl", orient="records", lines=True, force_ascii=False)
    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_json(out / f"pairs_{split}.jsonl", orient="records",
                                         lines=True, force_ascii=False)
    (out / "splits.json").write_text(json.dumps(splits, indent=0, sort_keys=True))
    report = build_report(len(raw), df, drops, token_method, args)
    (out / "data_report.md").write_text(report)

    print(report)
    print(f"Wrote outputs to {out.resolve()}")
    return df


if __name__ == "__main__":
    main()
