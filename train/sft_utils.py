"""Helpers for LoRA fine-tuning: tokenizing, loss masking, padding."""

from __future__ import annotations

import json
from pathlib import Path

IGNORE_INDEX = -100   # label value the loss function skips
THINK_OPEN = "<think>"


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def end_of_turn_token(tokenizer) -> str:
    """Qwen chat models end every turn with <|im_end|>; fall back to the eos token."""
    vocab = tokenizer.get_vocab()
    return "<|im_end|>" if "<|im_end|>" in vocab else tokenizer.eos_token


def render_prompt(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    """Chat-template text for everything before the assistant's answer."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=enable_thinking)


def split_overlap(prompt_text: str, assistant_content: str) -> str:
    """Remove the part of the answer the chat template already wrote into the prompt."""
    if not assistant_content.startswith(THINK_OPEN):
        return assistant_content
    for k in range(len(assistant_content), 0, -1):
        if prompt_text.endswith(assistant_content[:k]):
            return assistant_content[k:]
    return assistant_content


def uses_thinking(example: dict) -> bool:
    """Self-critique examples train the thinking mode; coder and judge train with it off."""
    return example.get("kind") == "self_critique"


def build_features(tokenizer, example: dict, max_len: int) -> dict | None:
    """Token ids and labels for one chat example, or None if it is longer than max_len."""
    messages = example["messages"]
    assert messages[-1]["role"] == "assistant", "last message must be the assistant target"
    prompt_text = render_prompt(tokenizer, messages[:-1], uses_thinking(example))
    target_text = split_overlap(prompt_text, messages[-1]["content"]) + end_of_turn_token(tokenizer)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(target_ids) > max_len:
        return None
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def build_dataset(tokenizer, examples: list[dict], max_len: int) -> tuple[list[dict], int]:
    features, dropped = [], 0
    for ex in examples:
        f = build_features(tokenizer, ex, max_len)
        if f is None:
            dropped += 1
        else:
            features.append(f)
    return features, dropped


class PadCollator:
    """Right-pads a batch. Padding positions get attention 0 and label -100."""

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad] * (width - n))
            labels.append(b["labels"] + [IGNORE_INDEX] * (width - n))
            mask.append([1] * n + [0] * (width - n))
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


EXCLUDE_FROM_LORA = ("vision", "visual", "lm_head", "embed", "merger", "mm_projector", "patch")


def find_lora_targets(model) -> list[str]:
    """Names of the language model's linear layers, to attach LoRA adapters to."""
    targets = set()
    for name, module in model.named_modules():
        if "Linear" not in type(module).__name__:
            continue
        if any(word in name.lower() for word in EXCLUDE_FROM_LORA):
            continue
        targets.add(name.split(".")[-1])
    return sorted(targets)


def describe_masking(tokenizer, features: dict, width: int = 60) -> str:
    """Human-readable view of which text the loss is computed on, for the notebook check."""
    ids, labels = features["input_ids"], features["labels"]
    first = next(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
    prompt_tail = tokenizer.decode(ids[max(0, first - 15):first])
    target_head = tokenizer.decode(ids[first:first + 40])
    target_tail = tokenizer.decode(ids[-10:])
    return (f"prompt tokens (no loss): {first}, target tokens (loss): {len(ids) - first}\n"
            f"...end of prompt : {prompt_tail!r}\n"
            f"start of target  : {target_head[:width]!r}\n"
            f"end of target    : {target_tail!r}")
