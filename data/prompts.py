"""Prompt templates used in training and evaluation."""

from __future__ import annotations

import re

# Short rubric per dimension, matched by keyword.
DIMENSION_RUBRICS = {
    "instruction": "Instruction following: does the answer do exactly what the task asks, "
                   "including every stated requirement and constraint?",
    "explanation": "Code explanation: are the code and its key steps explained clearly enough "
                   "for the reader to understand and use it?",
    "complexity": "Complexity and efficiency: is the solution efficient in time and memory, "
                  "without unnecessary work?",
    "readability": "Readability: is the code easy to read, with clear names, structure and "
                   "comments where they help?",
    "style": "Coding style: does the code follow the idioms and conventions of its language?",
}
GENERIC_RUBRIC = "Overall quality of the code for this task."

SYSTEM_CODER = "You are a careful programming assistant. Answer with working code and a brief explanation."
# The judges were trained with this prompt.
SYSTEM_JUDGE = ("You are a strict code reviewer. Rate the answer on the given dimension from 1 (poor) "
                "to 5 (excellent). First write a short rationale, then the rating.")
# Only for the untrained base model as judge, it often ignores the format otherwise.
SYSTEM_JUDGE_FORMAT = ("You are a strict code reviewer. Rate the answer on the given dimension from 1 (poor) "
                       "to 5 (excellent). Reply in exactly this format and nothing else:\n"
                       "Rationale: <one to three sentences>\n"
                       "Rating: <a single number from 1 to 5>")

EMPTY_THINK = "<think>\n\n</think>\n\n"


def rubric_for(dimension: str) -> str:
    key = (dimension or "").lower()
    for name, text in DIMENSION_RUBRICS.items():
        if name in key:
            return text
    return GENERIC_RUBRIC


def with_think(thinking: str | None, answer: str) -> str:
    """Assistant content with an explicit think block (empty when thinking is off)."""
    if not thinking:
        return EMPTY_THINK + answer.strip()
    return f"<think>\n{thinking.strip()}\n</think>\n\n{answer.strip()}"


# Coder
def coder_prompt(task: str) -> str:
    return task.strip()


def revision_prompt(task: str, previous_answer: str, feedback: str, dimension: str,
                    rating: int) -> str:
    """Prompt for the coder's second turn in the setup 1 loop (and its training examples)."""
    return (f"{task.strip()}\n\n"
            f"--- Your previous answer ---\n{previous_answer.strip()}\n\n"
            f"--- Reviewer feedback ({dimension}, rated {rating}/5) ---\n{feedback.strip()}\n\n"
            f"Write an improved answer that fixes the issues above.")


# Judge
def judge_prompt(task: str, answer: str, dimension: str) -> str:
    return (f"Dimension: {dimension}\nRubric: {rubric_for(dimension)}\n\n"
            f"--- Task ---\n{task.strip()}\n\n"
            f"--- Answer to rate ---\n{answer.strip()}")


def judge_target(rationale: str, rating: int) -> str:
    # Rationale first: the model reasons before it commits to a number.
    return f"Rationale: {rationale.strip()}\nRating: {rating}"


JUDGE_RATING_RE = re.compile(r"Rating:\s*([1-5])\b")
JUDGE_RATIONALE_RE = re.compile(r"Rationale:\s*(.*?)\s*(?:\n\s*Rating:|$)", re.S)


def parse_judge_output(text: str) -> tuple[int | None, str]:
    """Return (rating or None, rationale). A None rating counts as a parse failure."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("**", "").replace("__", "")   # tolerate Markdown bold: **Rating:** 4
    ratings = JUDGE_RATING_RE.findall(text)
    rationale = JUDGE_RATIONALE_RE.search(text)
    return (int(ratings[-1]) if ratings else None,
            rationale.group(1).strip() if rationale else "")


# Setup 2: self-critique inside thinking
def self_critique_thinking(draft: str, dimension: str, rating: int, rationale: str) -> str:
    return (f"Draft:\n{draft.strip()}\n\n"
            f"Critique ({dimension}, {rating}/5): {rationale.strip()}\n\n"
            f"Plan: write a new answer that fixes these issues.")
