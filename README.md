# Sycophancy in small open-weight language models

Code, data and raw outputs for the Thesis A report *Honesty Alignment of Large Language Models*
(Subham Dahal, Macquarie University, 2026; supervisor Dr Usman Naseem).

The project replicates three sycophancy tests from Sharma et al. (2024),
[*Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548), on five
open-weight 3.8–8B models run locally through [Ollama](https://ollama.com):

| Test | Research question | Script |
|---|---|---|
| "Are you sure?" swaying (AQuA) | RQ1, RQ4 | `src/run_swaying.py`, `src/rerun_unresolved.py` |
| Suggested incorrect answer (TriviaQA, TruthfulQA) | RQ2 | `src/run_answer.py` |
| Feedback under a stated opinion (arguments, maths, poems) | RQ3 | `src/run_feedback.py` |

Models: `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`, `phi3.5:latest`, `granite3-dense:8b` (default 4-bit
Ollama weights). Feedback is judged by `gemma2:9b`, which is not one of the tested models.

## Repository layout

```
datasets/   SycophancyEval prompts released with Sharma et al. (2024)
src/        benchmark runners, the shared Ollama client and the graders
analysis/   computes every number and figure in the thesis from outputs/
outputs/    raw model responses (one JSON line per item) and summaries
```

`outputs/`:

- `swaying_letter/`, `swaying_cot/`: swaying results with the original letter-only first turn and with a
  reasoning (chain-of-thought) first turn. `swaying_cot/rerun_unresolved.jsonl` holds the 29 items re-run
  without the loop stop (thesis Section 4.3.1).
- `answer/`: answer-sycophancy results (control and rebuttal reply per question).
- `feedback/`: model comments under each template and the judge's pairwise verdicts.
- `superseded/`: earlier runs discussed in thesis Appendix B, kept for comparison only: answer sycophancy
  with a 64-token limit, and feedback rated on a 1–5 scale by Qwen 2.5 7B.

Every record stores the full text of each reply and its generation metadata (finish reason, token count,
whether it was cut off or looping), so all results can be re-graded without querying the models.

## Reproducing the results

The reported runs used Ollama 0.23.2 on a MacBook Air (Apple M5, 16 GB), Python 3.10+.

```bash
pip install -r requirements.txt
for m in llama3.1:8b qwen2.5:7b mistral:7b phi3.5:latest granite3-dense:8b gemma2:9b; do ollama pull $m; done

python src/run_swaying.py --turn1 letter    # RQ1, original prompt
python src/run_swaying.py --turn1 cot       # RQ1/RQ4, reasoning first turn
python src/rerun_unresolved.py              # re-run unresolved reasoning-condition items
python src/run_answer.py                    # RQ2
python src/run_feedback.py                  # RQ3 (comments at temperature 1, then judging)

python analysis/analyse.py                  # writes analysis/thesis_numbers.json
python analysis/make_figures.py             # writes analysis/figures/*.pdf
```

Runners are resumable: items already in `outputs/` are skipped, so with the saved outputs in place they only
re-print the summaries. Delete a model's results file to run it again. Generation is greedy except for
feedback comments, but long greedy replies through Ollama are not always exactly repeatable.

`analysis/analyse.py` needs no Ollama: it re-grades every saved transcript with `src/graders.py` and computes
the rates, Wilson intervals, McNemar tests and logistic regressions reported in Chapter 5.

## Grading

`src/graders.py` extracts the option letter a reply commits to and classifies each initially correct swaying
item as held, flipped, retreated ("none of the options is correct") or unresolved. The rules and the twelve
grading errors fixed during the audit are described in thesis Section 4.6 and Appendix B.
