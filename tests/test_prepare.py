"""Tests for data/prepare.py on a small synthetic dataset. Run: python tests/test_prepare.py"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import prepare  # noqa: E402

DIMENSIONS = ["instruction-following", "complexity", "readability", "style", "explanation"]
MODELS = ["gpt-4", "gpt-3.5-turbo", "codellama-7b-instruct", "wizardlm-33b"]


def make_task(i, preference, instruction=None, lang="python"):
    instruction = instruction or f"Task {i}: write a {lang} function number {i}."
    responses = [{"model": m, "response": f"```{lang}\n# answer {i} by {m}\nreturn {k}\n```"}
                 for k, m in enumerate(MODELS)]
    ratings = {m: str(1 + (i + k) % 5) for k, m in enumerate(MODELS)}
    # Annotations in REVERSED order: matching by position would mis-assign every rating.
    annotations = [{"model": m, "rating": ratings[m], "rationale": f"rationale for {m} on {i}"}
                   for m in reversed(MODELS)]
    return {"instruction": instruction, "preference": preference, "models": MODELS,
            "responses": responses, "annotations": annotations}, ratings


def build_fixture():
    tasks, expected = [], {}
    for i in range(200):
        task, ratings = make_task(i, DIMENSIONS[i % 5], lang="python" if i % 3 else "lua")
        tasks.append(task)
        expected[f"cuf-{i:05d}"] = ratings

    # 200: invalid ratings ("N/A", junk) for two models
    t, _ = make_task(200, "style")
    t["annotations"][0]["rating"] = "N/A"
    t["annotations"][1]["rating"] = "excellent"
    tasks.append(t)
    # 201: one response has no annotation at all
    t, _ = make_task(201, "style")
    t["annotations"] = t["annotations"][1:]
    tasks.append(t)
    # 202: empty response
    t, _ = make_task(202, "readability")
    t["responses"][2]["response"] = "   "
    tasks.append(t)
    # 203: two models gave identical text
    t, _ = make_task(203, "readability")
    t["responses"][1]["response"] = t["responses"][0]["response"]
    tasks.append(t)
    # 204 + 205: same instruction, different preference and whitespace -> one group
    t, _ = make_task(204, "complexity", instruction="Sort a list   in Python.")
    tasks.append(t)
    t, _ = make_task(205, "style", instruction="sort a list in python.  ")
    tasks.append(t)
    # 206: an extremely long response
    t, _ = make_task(206, "explanation")
    t["responses"][0]["response"] = "```python\n" + "x = 1\n" * 4000 + "```"
    tasks.append(t)
    # 207: translation task, Python in the instruction, Lua in the answers
    t, _ = make_task(207, "style", instruction="Can you show me the Lua way to do this? import heapq",
                     lang="lua")
    t["responses"][0]["response"] = "No code fence here, just prose about the approach."
    tasks.append(t)
    return tasks, expected


def run_prepare(tmp, extra_args=()):
    tasks, expected = build_fixture()
    src = Path(tmp) / "fixture.jsonl"
    src.write_text("\n".join(json.dumps(t) for t in tasks))
    out = Path(tmp) / "out"
    df = prepare.main(["--input", str(src), "--out", str(out), *extra_args])
    return df, expected, out


def test_ratings_matched_by_model_name_not_position():
    with tempfile.TemporaryDirectory() as tmp:
        df, expected, _ = run_prepare(tmp)
        checked = 0
        for row in df.itertuples():
            if row.task_id in expected:
                assert row.rating == int(expected[row.task_id][row.model]), row
                checked += 1
        assert checked > 700


def test_invalid_rows_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        df, _, _ = run_prepare(tmp)
        t200 = df[df.task_id == "cuf-00200"]
        assert set(t200.model) == {"gpt-3.5-turbo", "gpt-4"}  # N/A and junk removed
        assert len(df[df.task_id == "cuf-00201"]) == 3         # unannotated response removed
        assert len(df[df.task_id == "cuf-00202"]) == 3         # empty response removed
        assert len(df[df.task_id == "cuf-00203"]) == 3         # duplicate text removed
        assert len(df[df.task_id == "cuf-00206"]) == 3         # over-long response removed
        assert df.rating.between(1, 5).all() and df.rating.dtype.kind == "i"


def test_duplicate_instructions_share_a_split():
    with tempfile.TemporaryDirectory() as tmp:
        df, _, _ = run_prepare(tmp)
        pair = df[df.task_id.isin(["cuf-00204", "cuf-00205"])]
        assert pair.group_id.nunique() == 1
        assert pair.split.nunique() == 1


def test_no_leakage_and_proportions():
    with tempfile.TemporaryDirectory() as tmp:
        df, _, out = run_prepare(tmp)
        splits = json.loads((out / "splits.json").read_text())
        tasks = {s: {t for t, v in splits.items() if v == s} for s in ["train", "val", "test"]}
        assert not (tasks["train"] & tasks["val"] or tasks["train"] & tasks["test"]
                    or tasks["val"] & tasks["test"])
        share = {s: len(v) / len(splits) for s, v in tasks.items()}
        assert 0.66 < share["train"] < 0.74 and 0.11 < share["val"] < 0.19, share
        # each dimension present in every split
        for split in ["train", "val", "test"]:
            assert df[df.split == split].preference.nunique() == 5


def test_split_is_deterministic():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _, _, out_a = run_prepare(a)
        _, _, out_b = run_prepare(b)
        assert (out_a / "splits.json").read_text() == (out_b / "splits.json").read_text()


def test_language_detection():
    assert prepare.detect_language("Write a JS thing", ["```js\nx\n```"]) == "javascript"
    assert prepare.detect_language("Lua way? import heapq",
                                   ["```lua\na\n```", "```lua\nb\n```", "```python\nc\n```"]) == "lua"
    assert prepare.detect_language("Do it in Java please", ["no fences"]) == "java"
    assert prepare.detect_language("Explain JavaScript closures", ["prose"]) == "javascript"
    assert prepare.detect_language("Something vague", ["prose"]) == "unknown"
    with tempfile.TemporaryDirectory() as tmp:
        df, _, _ = run_prepare(tmp)
        assert set(df[df.task_id == "cuf-00207"].language) == {"lua"}


def test_python_only_flag():
    with tempfile.TemporaryDirectory() as tmp:
        df, _, _ = run_prepare(tmp, ["--python-only"])
        assert set(df.language) == {"python"}


def test_rating_parser():
    assert prepare.parse_rating("3") == 3
    assert prepare.parse_rating(" 5 ") == 5
    assert prepare.parse_rating("4/5") == 4
    assert prepare.parse_rating("4.0") == 4
    for bad in ["N/A", "", None, "0", "6", "excellent", "3.5"]:
        assert prepare.parse_rating(bad) is None, bad


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed")
