"""Tests for sft_utils and eval_utils with a fake tokenizer. Run: python tests/test_train_eval_utils.py"""

import json
import math
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ["train", "eval", "data"]:
    sys.path.insert(0, str(ROOT / sub))
import eval_utils as E  # noqa: E402
import prompts as P  # noqa: E402
import sft_utils as S  # noqa: E402

SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


class FakeTokenizer:
    """Characters are tokens; special tokens are single tokens."""

    def __init__(self, opens_think_when_on=False):
        self.opens_think_when_on = opens_think_when_on
        self.eos_token = "<|endoftext|>"
        self.pad_token_id = 0

    def get_vocab(self):
        return {s: 1000 + i for i, s in enumerate(SPECIAL)}

    def _encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for j, s in enumerate(SPECIAL):
                if text.startswith(s, i):
                    ids.append(1000 + j)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encode(text)}

    def decode(self, ids):
        return "".join(SPECIAL[i - 1000] if i >= 1000 else chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if not enable_thinking:
                text += "<think>\n\n</think>\n\n"
            elif self.opens_think_when_on:
                text += "<think>\n"
        return text


def coder_example(answer="def f():\n    return 1"):
    return {"kind": "fresh", "messages": [
        {"role": "system", "content": P.SYSTEM_CODER},
        {"role": "user", "content": "Write f."},
        {"role": "assistant", "content": P.with_think(None, answer)}]}


def selfcritique_example():
    return {"kind": "self_critique", "messages": [
        {"role": "system", "content": P.SYSTEM_CODER},
        {"role": "user", "content": "Write f."},
        {"role": "assistant", "content": P.with_think("Draft:\nx\n\nCritique (style, 2/5): bad",
                                                      "def f():\n    return 1")}]}


def target_text(tok, feats):
    return tok.decode([i for i, l in zip(feats["input_ids"], feats["labels"]) if l != S.IGNORE_INDEX])


def prompt_text(tok, feats):
    return tok.decode([i for i, l in zip(feats["input_ids"], feats["labels"]) if l == S.IGNORE_INDEX])


# training features
def test_non_thinking_target_has_no_duplicate_think_block():
    tok = FakeTokenizer()
    feats = S.build_features(tok, coder_example(), max_len=10_000)
    target = target_text(tok, feats)
    full = tok.decode(feats["input_ids"])
    assert target == "def f():\n    return 1<|im_end|>", repr(target)
    assert full.count("<think>") == 1                   # the template's, not a second one
    assert prompt_text(tok, feats).endswith("<think>\n\n</think>\n\n")


def test_training_prompt_equals_inference_prompt():
    tok = FakeTokenizer()
    ex = coder_example()
    feats = S.build_features(tok, ex, max_len=10_000)
    inference_prompt = tok.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True,
                                               enable_thinking=False)
    assert prompt_text(tok, feats) == inference_prompt


def test_thinking_target_keeps_whole_reasoning():
    tok = FakeTokenizer()
    feats = S.build_features(tok, selfcritique_example(), max_len=10_000)
    target = target_text(tok, feats)
    assert target.startswith("<think>\nDraft:") and target.endswith("return 1<|im_end|>")
    assert "Critique (style, 2/5)" in target          # the critique is learned, not masked


def test_thinking_template_that_opens_think_itself():
    # This is what the real Qwen3.5 chat template does (checked on the Hub).
    tok = FakeTokenizer(opens_think_when_on=True)
    feats = S.build_features(tok, selfcritique_example(), max_len=10_000)
    full = tok.decode(feats["input_ids"])
    assert full.count("<think>") == 1
    assert target_text(tok, feats).startswith("Draft:")


def test_long_examples_dropped_not_truncated():
    tok = FakeTokenizer()
    feats, dropped = S.build_dataset(tok, [coder_example(), coder_example("x" * 5000)], max_len=400)
    assert dropped == 1 and len(feats) == 1


def test_split_overlap_leaves_plain_text_alone():
    assert S.split_overlap("...\n", "\nhello") == "\nhello"
    assert S.split_overlap("abc<think>\n", "<think>\nhi") == "hi"


def test_collator_pads_and_masks():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("    (torch not installed here; collator checked in the notebook)")
        return
    batch = S.PadCollator(0)([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
                              {"input_ids": [5], "labels": [5]}])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [5, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_find_lora_targets_skips_vision_and_head():
    Linear = type("Linear", (), {})
    Linear4bit = type("Linear4bit", (), {})
    Conv = type("Conv2d", (), {})
    modules = [("model.language_model.layers.0.self_attn.q_proj", Linear()),
               ("model.language_model.layers.0.linear_attn.in_proj_qkvz", Linear4bit()),
               ("model.language_model.layers.0.mlp.down_proj", Linear()),
               ("model.visual.blocks.0.attn.qkv", Linear()),
               ("model.visual.patch_embed.proj", Conv()),
               ("lm_head", Linear())]
    model = types.SimpleNamespace(named_modules=lambda: modules)
    assert S.find_lora_targets(model) == ["down_proj", "in_proj_qkvz", "q_proj"]


# evaluation helpers
def test_strip_thinking_and_extract_code():
    assert E.strip_thinking("<think>\nhm\n</think>\n\nanswer") == ("answer", True)
    assert E.strip_thinking("<think>\nran out of tokens") == ("", False)
    assert E.strip_thinking("plain") == ("plain", True)
    # Qwen3.5 thinking mode: "<think>\n" is in the prompt, the output only closes it
    assert E.strip_thinking("plan the loop\n</think>\n\nanswer", thinking=True) == ("answer", True)
    # ran out of tokens while thinking: no closing tag, and no opening tag either
    assert E.strip_thinking("still reasoning about the loop", thinking=True) == ("", False)
    text = "Here:\n```python\ndef f():\n    return 1\n```\nand\n```\nx\n```"
    assert E.extract_code(text) == "def f():\n    return 1"
    assert E.extract_code("def g(): pass") == "def g(): pass"


def test_clean_generation_keeps_think_tags():
    tok = types.SimpleNamespace(eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    raw = "<think>\nplan\n</think>\n\nanswer<|im_end|><|endoftext|><|endoftext|>"
    cleaned = E.clean_generation(raw, tok)
    assert cleaned == "<think>\nplan\n</think>\n\nanswer"
    assert E.strip_thinking(cleaned) == ("answer", True)


def test_fill_missing_problems():
    full = E.fill_missing_problems([{"task_id": "B", "solution": "x"}], ["A", "B", "C"])
    assert sorted(s["task_id"] for s in full) == ["A", "B", "C"]
    assert {s["task_id"]: s["solution"] for s in full} == {"A": "", "B": "x", "C": ""}


def test_codegen_prompt_uses_training_system_prompt():
    msgs = E.codegen_messages("def f(x):\n    \"\"\"doc\"\"\"\n")
    assert msgs[0]["content"] == P.SYSTEM_CODER and "```python" in msgs[1]["content"]


def test_parse_evalplus_results():
    data = {"eval": {"HumanEval/0": [{"base_status": "pass", "plus_status": "fail"}],
                     "HumanEval/1": [{"base_status": "pass", "plus_status": "pass"}],
                     "HumanEval/2": [{"base_status": "fail", "plus_status": "pass"}]}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
    r = E.parse_evalplus_results(f.name)
    assert r["HumanEval/0"] == {"base": True, "plus": False}
    assert r["HumanEval/1"] == {"base": True, "plus": True}
    assert r["HumanEval/2"] == {"base": False, "plus": False}   # plus requires base too


def test_mcnemar_matches_hand_calculation():
    a = {f"t{i}": i < 57 for i in range(164)}                   # base solves t0..t56
    b = dict(a)
    for i in range(7):
        b[f"t{i}"] = False                                       # 7 broken
    for i in range(57, 71):
        b[f"t{i}"] = True                                        # 14 fixed
    r = E.mcnemar_exact(a, b)
    assert (r["fixed"], r["broken"]) == (14, 7)
    expected = 2 * sum(math.comb(21, i) for i in range(8)) / 2 ** 21
    assert abs(r["p_value"] - expected) < 1e-12 and 0.18 < r["p_value"] < 0.2


def test_holm():
    adj = E.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_bootstrap_ci_contains_rate():
    passes = [True] * 57 + [False] * 107
    rate, lo, hi = E.bootstrap_ci(passes, seed=1)
    assert abs(rate - 57 / 164) < 1e-12 and lo < rate < hi and 0.25 < lo and hi < 0.45


def test_judge_metrics():
    assert E.quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert E.quadratic_weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0
    assert abs(E.spearman([1, 2, 3], [10, 20, 30]) - 1) < 1e-12
    m = E.judge_metrics([4, 4, 3, 2, 5], [4, None, 3, 3, 5])
    assert m["parse_rate"] == 0.8 and m["accuracy"] == 0.75 and m["mae"] == 0.25
    assert m["majority_rating"] == 4 and m["majority_baseline_accuracy"] == 0.4


def test_always_predicting_majority_has_zero_kappa():
    true = [1] * 5 + [2] * 18 + [3] * 33 + [4] * 34 + [5] * 10
    m = E.judge_metrics(true, [4] * len(true))
    assert m["accuracy"] == 0.34 and abs(m["qwk"]) < 1e-12   # good accuracy, no skill


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and isinstance(v, types.FunctionType)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed")
