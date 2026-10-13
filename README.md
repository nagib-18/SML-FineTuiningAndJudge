# Small coder + judge models on CodeUltraFeedback

Proof of concept for the trustworthy LLM project. I fine-tuned Qwen3.5-0.8B with LoRA on
CodeUltraFeedback and compared two setups:

1. a coder model plus a separate judge model; the coder revises until the judge gives 4/5
   (max 3 revisions)
2. one self-critique model that writes a draft, critiques it and fixes it in its thinking

Each setup was trained on Python-only data and on all languages. Everything was evaluated
before and after training on HumanEval+ and MBPP+ (EvalPlus) and on a held-out judge test set.
Full tables are in `results/summary.md`.

## Results in short

Fine-tuning did not make the coder better. The base model with thinking off scored 28.0% on
HumanEval+ and 33.3% on MBPP+; the Python coder dropped to 21.3% / 23.8% and the all-languages
coder to 23.2% / 26.2% (the MBPP+ drop is significant, McNemar p < 0.01). Part of the MBPP+ drop
is the trained models renaming the requested function. The training data is rated on style and
explanation, not on whether the code runs, which is the most likely reason.

The self-critique models (setup 2) beat the base model with thinking on (22.0% vs 15.9% on
HumanEval+), but only because the base model runs out of tokens while thinking. Compared with the
base model without thinking they are worse, and they often repeat the same sentences until the
token limit.

The judges did improve: accuracy against the dataset ratings went from 20.7% (base) to 35.2%
(Python judge) and 38.2% (all-languages judge), QWK from 0.15 to 0.26. That is still only a few
points above always guessing the most common rating (33.2%), and the trained judges mostly answer 4.

In the coder + judge loop (setup 1) the coder changed its code in many revisions, but pass@1
barely moved (at most +1.2 points, never significant). The judges are not good at telling working
code from broken code, so their feedback does not lead to fixes.

## Running it

    pip install -r requirements.txt
    python data/prepare.py
    python data/build_datasets.py --language python --out data/final_python
    python data/build_datasets.py --out data/final_all
    python train/run_all.py --data data/final_python --lang python
    python train/run_all.py --data data/final_all --lang all --batch-size 4
    python eval/run_eval.py
    python eval/run_loop.py
    python eval/compare.py

Training and evaluation need a GPU; I used Colab (A100 for training, L4 for evaluation).
The notebooks in `notebooks/` do the same steps one model at a time. The datasets and trained
models are not in the repo because of their size; the scripts rebuild the data.

Tests (no GPU needed): `python tests/test_prepare.py` and the other two files in `tests/`.

## Layout

    data/      cleaning, split, training sets, prompts
    train/     LoRA training
    eval/      evaluation, setup 1 loop, result tables
    notebooks/ Colab notebooks
    results/   summary tables
