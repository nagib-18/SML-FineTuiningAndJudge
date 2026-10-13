# Results summary

## Code correctness (pass@1, HumanEval+ / MBPP+ extra tests)

| Model | Mode | Benchmark | pass@1+ | 95% CI | pass@1 (base tests) | unfinished thinking | mean tokens |
|---|---|---|---|---|---|---|---|
| all_coder | nothink | humaneval | 23.2% | 16.5%–29.9% | 26.2% | 0/164 | 191 |
| all_coder | nothink | mbpp | 26.2% | 21.7%–30.4% | 31.2% | 0/378 | 227 |
| all_selfcritique | think | humaneval | 18.9% | 12.8%–25.0% | 23.2% | 10/164 | 1297 |
| all_selfcritique | think | mbpp | 18.8% | 15.1%–22.8% | 22.8% | 67/378 | 1675 |
| base | nothink | humaneval | 28.0% | 22.0%–34.8% | 32.3% | 0/164 | 267 |
| base | think | humaneval | 15.9% | 11.0%–21.3% | 17.1% | 99/164 | 2943 |
| base | nothink | mbpp | 33.3% | 28.8%–38.1% | 42.1% | 0/378 | 275 |
| base | think | mbpp | 16.1% | 12.7%–19.8% | 19.6% | 199/378 | 2813 |
| python_coder | nothink | humaneval | 21.3% | 15.2%–27.4% | 25.6% | 0/164 | 247 |
| python_coder | nothink | mbpp | 23.8% | 19.8%–28.6% | 30.4% | 0/378 | 172 |
| python_selfcritique | think | humaneval | 22.0% | 15.9%–28.0% | 25.6% | 11/164 | 920 |
| python_selfcritique | think | mbpp | 20.4% | 16.4%–24.3% | 26.5% | 34/378 | 819 |

## Paired comparisons (exact McNemar, Holm-corrected over all rows)

| Comparison | Benchmark | A → B | fixed | broken | p | p (Holm) |
|---|---|---|---|---|---|---|
| python_coder vs base (thinking off) | humaneval | 28.0% → 21.3% | 6 | 17 | 0.0347 | 0.3816 |
| python_coder vs base (thinking off) | mbpp | 33.3% → 23.8% | 23 | 59 | 0.0001 | 0.0012 |
| all_coder vs base (thinking off) | humaneval | 28.0% → 23.2% | 7 | 15 | 0.1338 | 0.9470 |
| all_coder vs base (thinking off) | mbpp | 33.3% → 26.2% | 28 | 55 | 0.0040 | 0.0525 |
| all_coder vs python_coder | humaneval | 21.3% → 23.2% | 8 | 5 | 0.5811 | 1.0000 |
| all_coder vs python_coder | mbpp | 23.8% → 26.2% | 29 | 20 | 0.2529 | 1.0000 |
| python_selfcritique vs base (thinking on) | humaneval | 15.9% → 22.0% | 21 | 11 | 0.1102 | 0.9470 |
| python_selfcritique vs base (thinking on) | mbpp | 16.1% → 20.4% | 51 | 35 | 0.1052 | 0.9470 |
| all_selfcritique vs base (thinking on) | humaneval | 15.9% → 18.9% | 18 | 13 | 0.4731 | 1.0000 |
| all_selfcritique vs base (thinking on) | mbpp | 16.1% → 18.8% | 46 | 36 | 0.3203 | 1.0000 |
| python_selfcritique vs base (thinking off) | humaneval | 28.0% → 22.0% | 8 | 18 | 0.0755 | 0.7552 |
| python_selfcritique vs base (thinking off) | mbpp | 33.3% → 20.4% | 24 | 73 | 0.0000 | 0.0000 |
| all_selfcritique vs base (thinking off) | humaneval | 28.0% → 18.9% | 6 | 21 | 0.0059 | 0.0711 |
| all_selfcritique vs base (thinking off) | mbpp | 33.3% → 18.8% | 19 | 74 | 0.0000 | 0.0000 |
| all_selfcritique vs python_selfcritique | humaneval | 22.0% → 18.9% | 6 | 11 | 0.3323 | 1.0000 |
| all_selfcritique vs python_selfcritique | mbpp | 20.4% → 18.8% | 34 | 40 | 0.5614 | 1.0000 |

## Why answers failed

Share of all problems per outcome. *Looping* = a line repeated 4+ times (degenerate repetition).

| Model | Mode | Benchmark | passed | wrong result | function missing | does not parse | no code | thinking unfinished | looping |
|---|---|---|---|---|---|---|---|---|---|
| all_coder | nothink | humaneval | 23.2% | 75.0% | 0.0% | 1.8% | 0.0% | 0.0% | 1.2% |
| all_coder | nothink | mbpp | 26.2% | 50.5% | 22.0% | 1.3% | 0.0% | 0.0% | 3.4% |
| all_selfcritique | think | humaneval | 18.9% | 70.1% | 1.2% | 3.7% | 0.0% | 6.1% | 9.8% |
| all_selfcritique | think | mbpp | 18.8% | 39.2% | 22.0% | 2.4% | 0.0% | 17.7% | 18.5% |
| base | nothink | humaneval | 28.0% | 66.5% | 0.0% | 5.5% | 0.0% | 0.0% | 3.7% |
| base | nothink | mbpp | 33.3% | 48.7% | 9.5% | 8.5% | 0.0% | 0.0% | 2.4% |
| base | think | humaneval | 15.9% | 23.8% | 0.0% | 0.0% | 0.0% | 60.4% | 58.5% |
| base | think | mbpp | 16.1% | 19.3% | 9.0% | 2.9% | 0.0% | 52.6% | 71.7% |
| python_coder | nothink | humaneval | 21.3% | 75.6% | 0.0% | 3.0% | 0.0% | 0.0% | 0.6% |
| python_coder | nothink | mbpp | 23.8% | 59.5% | 15.6% | 1.1% | 0.0% | 0.0% | 2.4% |
| python_selfcritique | think | humaneval | 22.0% | 68.3% | 0.0% | 3.0% | 0.0% | 6.7% | 5.5% |
| python_selfcritique | think | mbpp | 20.4% | 47.4% | 22.0% | 1.3% | 0.0% | 9.0% | 9.5% |

## Judges (re-parsed with the current parser)

| Judge | Prompt | readable | accuracy | 95% CI | within ±1 | MAE | QWK | Spearman | majority baseline | predicted 1/2/3/4/5 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | states format | 98.7% | 20.7% | 17.5%–24.0% | 57.6% | 1.41 | 0.152 | 0.165 | 33.2% | 130/101/84/50/227 |
| base | training prompt | 60.8% | 9.2% | 7.0%–11.5% | 53.2% | 1.51 | 0.085 | 0.101 | 33.2% | 54/45/27/56/183 |
| python_judge | states format | 100.0% | 35.2% | 31.3%–38.5% | 80.3% | 0.88 | 0.184 | 0.179 | 33.2% | 12/87/156/341/4 |
| python_judge | training prompt | 100.0% | 34.3% | 30.7%–37.8% | 78.5% | 0.90 | 0.177 | 0.166 | 33.2% | 15/88/151/341/5 |
| all_judge | states format | 100.0% | 38.2% | 34.5%–41.8% | 82.2% | 0.83 | 0.261 | 0.262 | 33.2% | 17/60/169/328/26 |
| all_judge | training prompt | 100.0% | 36.2% | 32.2%–39.8% | 81.2% | 0.86 | 0.237 | 0.252 | 33.2% | 15/52/192/312/29 |

Accuracy counts unreadable answers as wrong. QWK: 0 = chance, 1 = perfect agreement.

| Judge vs base judge (same items) | fixed | broken | p |
|---|---|---|---|
| python_judge (judge_promptnew) vs base (judge_promptnew) | 186 | 99 | 0.0000 |
| python_judge (judge_promptold) vs base (judge_promptold) | 196 | 45 | 0.0000 |
| all_judge (judge_promptnew) vs base (judge_promptnew) | 194 | 89 | 0.0000 |
| all_judge (judge_promptold) vs base (judge_promptold) | 202 | 40 | 0.0000 |

## Setup 1: coder + judge revise loop (threshold 4/5, max 3 revisions)

| Condition | Benchmark | draft | last | best | fixed / broken (draft→last) | p | revised / code changed / logic changed | reached 4/5 | judge unreadable | drafts accepted: pass rate | rejected: pass rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_coder+all_judge | humaneval | 23.8% | 25.0% | 25.0% | 2 / 0 | 0.5000 | 57 / 33 / 29 | 74.4% | 0.0% | 25.2% (107) | 21.1% |
| all_coder+all_judge | mbpp | 26.2% | 27.2% | 27.2% | 4 / 0 | 0.1250 | 138 / 69 / 61 | 79.4% | 0.0% | 27.5% (240) | 23.9% |
| base_coder+all_judge | humaneval | 27.4% | 27.4% | 27.4% | 0 / 0 | 1.0000 | 38 / 16 / 10 | 84.1% | 0.0% | 27.8% (126) | 26.3% |
| base_coder+all_judge | mbpp | 32.8% | 32.3% | 32.3% | 0 / 2 | 0.5000 | 81 / 56 / 41 | 82.5% | 0.0% | 35.7% (297) | 22.2% |
| base_coder+base_judge | humaneval | 27.4% | 27.4% | 27.4% | 0 / 0 | 1.0000 | 35 / 28 / 24 | 89.0% | 3.3% | 31.8% (129) | 11.4% |
| base_coder+base_judge | mbpp | 32.8% | 32.8% | 32.8% | 2 / 2 | 1.0000 | 92 / 70 / 65 | 90.7% | 0.5% | 36.7% (286) | 20.7% |
| python_coder+python_judge | humaneval | 20.1% | 20.1% | 20.1% | 0 / 0 | 1.0000 | 57 / 21 / 18 | 81.7% | 0.0% | 19.6% (107) | 21.1% |
| python_coder+python_judge | mbpp | 23.5% | 23.8% | 23.8% | 1 / 0 | 1.0000 | 150 / 60 / 43 | 74.3% | 0.5% | 28.5% (228) | 16.0% |

Logic changed = the code differs after removing comments, docstrings and formatting. If the judge were useful, drafts it accepts would pass far more often than drafts it rejects.

