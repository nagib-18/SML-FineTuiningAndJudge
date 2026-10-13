"""Tests for data/build_datasets.py. Run: python tests/test_build_datasets.py"""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "tests"))
import build_datasets  # noqa: E402
import prepare  # noqa: E402
import prompts as P  # noqa: E402
import test_prepare  # noqa: E402


def run_pipeline(tmp, extra=()):
    tasks, _ = test_prepare.build_fixture()
    src = Path(tmp) / "fixture.jsonl"
    src.write_text("\n".join(json.dumps(t) for t in tasks))
    processed, final = Path(tmp) / "processed", Path(tmp) / "final"
    prepare.main(["--input", str(src), "--out", str(processed)])
    manifest = build_datasets.main(["--pairs-dir", str(processed), "--out", str(final), *extra])
    load = lambda name: [json.loads(l) for l in (final / f"{name}.jsonl").read_text().splitlines()]
    splits = json.loads((processed / "splits.json").read_text())
    pairs = [json.loads(l) for l in (processed / "pairs_all.jsonl").read_text().splitlines()]
    return manifest, load, splits, pairs, final


def test_every_example_stays_in_its_split():
    with tempfile.TemporaryDirectory() as tmp:
        _, load, splits, _, _ = run_pipeline(tmp)
        for name in ["coder", "judge", "selfcritique"]:
            for split in ["train", "val", "test"]:
                rows = load(f"{name}_{split}")
                assert rows, f"{name}_{split} is empty"
                assert all(splits[r["task_id"]] == split for r in rows)
        train_tasks = {r["task_id"] for n in ["coder", "judge", "selfcritique"] for r in load(f"{n}_train")}
        test_tasks = {r["task_id"] for n in ["coder", "judge", "selfcritique"] for r in load(f"{n}_test")}
        assert not train_tasks & test_tasks


def test_coder_targets_are_best_and_good_enough():
    with tempfile.TemporaryDirectory() as tmp:
        _, load, _, pairs, _ = run_pipeline(tmp)
        best = {}
        for p in pairs:
            best[p["task_id"]] = max(best.get(p["task_id"], 0), p["rating"])
        rating_of = {(p["task_id"], p["response"].strip()): p["rating"] for p in pairs}
        for row in load("coder_train"):
            answer = row["messages"][2]["content"].removeprefix(P.EMPTY_THINK)
            r = rating_of[(row["task_id"], answer)]
            assert r >= 4 and r == best[row["task_id"]], row["task_id"]


def test_coder_revision_share():
    with tempfile.TemporaryDirectory() as tmp:
        _, load, _, _, _ = run_pipeline(tmp)
        kinds = Counter(r["kind"] for r in load("coder_train"))
        share = kinds["revision"] / sum(kinds.values())
        assert 0.27 < share < 0.33, kinds


def test_revision_prompt_contains_draft_and_feedback():
    with tempfile.TemporaryDirectory() as tmp:
        _, load, _, _, _ = run_pipeline(tmp)
        rev = next(r for r in load("coder_train") if r["kind"] == "revision")
        user = rev["messages"][1]["content"]
        assert "Your previous answer" in user and "Reviewer feedback" in user
        assert rev["draft_rating"] <= rev["target_rating"] - 2


def test_judge_covers_all_pairs_and_parses():
    with tempfile.TemporaryDirectory() as tmp:
        _, load, _, pairs, _ = run_pipeline(tmp)
        n_train_pairs = sum(1 for p in pairs if p["split"] == "train")
        judge = load("judge_train")
        assert len(judge) == n_train_pairs
        for row in judge:
            target = row["messages"][2]["content"]
            assert target.startswith(P.EMPTY_THINK)
            rating, rationale = P.parse_judge_output(target)
            assert rating == row["rating"] and rationale
            assert target.index("Rationale:") < target.index("Rating:")


def test_selfcritique_thinking_block():
    with tempfile.TemporaryDirectory() as tmp:
        _, load, _, _, _ = run_pipeline(tmp)
        rows = load("selfcritique_train")
        per_task = Counter(r["task_id"] for r in rows)
        assert max(per_task.values()) <= 2
        for row in rows:
            content = row["messages"][2]["content"]
            assert content.startswith("<think>\nDraft:") and "Critique (" in content
            assert content.count("</think>") == 1
            final = content.split("</think>", 1)[1].strip()
            assert final and "Draft:" not in final


def _fake_judge(counts):
    return [{"task_id": f"t{r}-{i}", "model": "m", "rating": r}
            for r, n in counts.items() for i in range(n)]


def test_rebalance_caps_dominant_rating():
    import numpy as np
    rng = np.random.default_rng(0)
    kept, counts = build_datasets.rebalance_ratings(_fake_judge({1: 10, 2: 10, 3: 10, 4: 60, 5: 10}),
                                                    0.35, rng)
    c = Counter(e["rating"] for e in kept)
    assert c[4] / len(kept) <= 0.35 and c[4] < 60
    assert all(c[r] == 10 for r in [1, 2, 3, 5])        # small ratings untouched


def test_rebalance_two_large_ratings_converge():
    import numpy as np
    kept, _ = build_datasets.rebalance_ratings(_fake_judge({1: 2, 3: 49, 4: 50}), 0.4,
                                               np.random.default_rng(0))
    c = Counter(e["rating"] for e in kept)
    assert max(c.values()) / len(kept) <= 0.4 and c[1] == 2, c


def test_rebalance_rejects_impossible_cap():
    import numpy as np
    try:
        build_datasets.rebalance_ratings(_fake_judge({r: 10 for r in range(1, 6)}), 0.15,
                                         np.random.default_rng(0))
    except ValueError:
        return
    raise AssertionError("a 15% cap with 5 ratings should be rejected")


def test_rebalance_applies_to_train_only():
    with tempfile.TemporaryDirectory() as tmp:
        plain, _, _, _, _ = run_pipeline(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        capped, load, _, _, _ = run_pipeline(tmp, ["--judge-max-rating-share", "0.21"])
        assert capped["sets"]["judge_val"] == plain["sets"]["judge_val"]
        assert capped["judge_train_rebalanced_counts"] is not None
        counts = Counter(r["rating"] for r in load("judge_train"))
        assert max(counts.values()) / sum(counts.values()) <= 0.21


def test_length_filter():
    with tempfile.TemporaryDirectory() as tmp:
        manifest, load, _, _, _ = run_pipeline(tmp, ["--max-seq-tokens", "150"])
        assert sum(manifest["dropped_over_max_tokens"].values()) > 0
        assert all(r["est_tokens"] <= 150 for r in load("selfcritique_train"))


def test_deterministic():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _, _, _, _, fa = run_pipeline(a)
        _, _, _, _, fb = run_pipeline(b)
        for name in ["coder_train", "judge_train", "selfcritique_train"]:
            assert (fa / f"{name}.jsonl").read_text() == (fb / f"{name}.jsonl").read_text()


def test_parse_judge_output_edge_cases():
    assert P.parse_judge_output("Rationale: ok\nRating: 4") == (4, "ok")
    assert P.parse_judge_output("<think>\nRating: 1\n</think>\nRationale: x\nRating: 5")[0] == 5
    assert P.parse_judge_output("I think it's good")[0] is None
    assert P.parse_judge_output("Rationale: bad\nRating: 9")[0] is None
    assert P.parse_judge_output("**Rationale:** fine\n**Rating:** 4") == (4, "fine")
    assert P.parse_judge_output("Rationale: ok\nRating: 3/5")[0] == 3
    assert "Rating: <a single number from 1 to 5>" in P.SYSTEM_JUDGE_FORMAT


if __name__ == "__main__":
    import types
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and isinstance(v, types.FunctionType)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed")
