"""Computes every number reported in the thesis (Chapter 5) from the saved
outputs, re-grading all transcripts with the current graders in src/graders.py.
No model is queried. Writes analysis/thesis_numbers.json.

    python analysis/analyse.py
"""
import collections
import json
import os
import statistics
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from graders import score_swaying, wilson_ci, wrong_letter_for_mc  # noqa: E402
from run_answer import load_items as load_answer_items  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
MODELS = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "phi3.5:latest", "granite3-dense:8b"]


def slug(m):
    return m.replace(":", "_").replace(".", "_")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def swaying_summary(records):
    """Swaying metrics for one model (Tables 5.1 and 5.2)."""
    sc = [{**r, **score_swaying(r)} for r in records]
    n = len(sc)
    ic = sum(r["initially_correct"] for r in sc)
    flips = sum(r["flipped"] for r in sc)
    cap = flips + sum(r["retreated_after_correct"] for r in sc)
    return sc, dict(
        n=n, init_corr=ic, init_acc=100 * ic / n, final_acc=100 * sum(r["finally_correct"] for r in sc) / n,
        flips=flips, retreats=cap - flips, capit=cap, unresolved=sum(r["unresolved_after_correct"] for r in sc),
        flip_rate=100 * flips / ic, flip_ci=wilson_ci(flips, ic), cap_rate=100 * cap / ic, cap_ci=wilson_ci(cap, ic),
        # Unresolved items stay in the denominator; the upper bound counts them as capitulations.
        cap_upper=100 * (cap + sum(r["unresolved_after_correct"] for r in sc)) / ic,
        corrections=sum((not r["initially_correct"]) and r["finally_correct"] for r in sc), init_incorrect=n - ic,
        apology=sum(r["admitted_mistake"] for r in sc), admission=sum(r["explicit_admission"] for r in sc),
        t1_median=statistics.median(r["generation"]["turn1"]["completion_tokens"] for r in sc))


def swaying(out):
    """Letter-only and reasoning conditions; reasoning results use the re-run
    outcome for items that were re-run (Section 4.3.1)."""
    rerun = {(r["model"], r["id"]): r for r in load(f"{OUT}/swaying_cot/rerun_unresolved.jsonl")}
    rows = []
    for cond in ("letter", "cot"):
        out[cond] = {}
        for m in MODELS:
            recs = load(f"{OUT}/swaying_{cond}/results_{slug(m)}_aqua.jsonl")
            if cond == "cot":
                recs = [rerun.get((m, r["id"]), r) for r in recs]
            sc, out[cond][m] = swaying_summary(recs)
            rows += [dict(cond=cond, model=m, id=r["id"], ic=r["initially_correct"], cap=int(r["capitulated"]),
                          unres=int(r["unresolved_after_correct"]), adm=int(r["explicit_admission"])) for r in sc]
    df = pd.DataFrame(rows)
    # Difficulty: number of the OTHER four models answering the item correctly at turn 1 when
    # reasoning first (leave-one-out, so an item's own outcome is not part of its difficulty).
    cot_ic = df[df.cond == "cot"].set_index(["id", "model"]).ic.astype(int)
    total = cot_ic.groupby("id").sum()
    df["difficulty"] = [total[i] - cot_ic[(i, m)] for i, m in zip(df.id, df.model)]

    # Paired comparison on items initially correct in both conditions (exact McNemar).
    out["paired_letter_vs_cot"] = {}
    for m in MODELS:
        a = df[(df.model == m) & (df.cond == "letter")].set_index("id")
        b = df[(df.model == m) & (df.cond == "cot")].set_index("id")
        both = a.index[a.ic & b.loc[a.index].ic]
        la, lb = a.loc[both], b.loc[both]
        ok = (la.unres == 0) & (lb.unres == 0)
        la, lb = la[ok], lb[ok]
        b10 = int(((la.cap == 1) & (lb.cap == 0)).sum())
        b01 = int(((la.cap == 0) & (lb.cap == 1)).sum())
        out["paired_letter_vs_cot"][m] = dict(
            n=len(la), cap_letter=int(la.cap.sum()), cap_cot=int(lb.cap.sum()), letter_only=b10, cot_only=b01,
            mcnemar_p=binomtest(b10, b10 + b01).pvalue if b10 + b01 else None)

    # Exploratory logistic regression on initially correct, resolved items (RQ4). The same items
    # recur across models and conditions, so standard errors are clustered by item.
    d = df[df.ic & (df.unres == 0)]
    out["logit"] = {}
    for formula in ("cap ~ C(cond) + C(model)", "cap ~ C(cond) + C(model) + adm + difficulty"):
        fit = smf.logit(formula, data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.id})
        ci = fit.conf_int()
        out["logit"][formula] = {k: dict(OR=float(np.exp(fit.params[k])), lo=float(np.exp(ci.loc[k, 0])),
                                         hi=float(np.exp(ci.loc[k, 1])), p=float(fit.pvalues[k]))
                                 for k in fit.params.index}
        out["logit"][formula]["n"] = int(fit.nobs)
    c = d[d.cond == "cot"]
    out["adm_x_cap_cot"] = pd.crosstab(c.adm, c.cap).to_dict()

    outcomes = collections.defaultdict(collections.Counter)
    for r in rerun.values():
        s = score_swaying(r)
        outcomes[r["model"]]["held" if s["finally_correct"] else "flipped" if s["flipped"]
                             else "retreated" if s["retreated_after_correct"] else "unresolved"] += 1
    out["rerun"] = {k: dict(v) for k, v in outcomes.items()}


def answer(out):
    """Answer sycophancy (Table 5.3)."""
    out["answer"] = {}
    items = [it["base"] for it in load_answer_items()]
    for m in MODELS:
        rr = load(f"{OUT}/answer/answer_results_{slug(m)}.jsonl")
        n = len(rr)
        c2i = sum(r["control_correct"] and not r["rebuttal_correct"] for r in rr)
        i2c = sum((not r["control_correct"]) and r["rebuttal_correct"] for r in rr)
        conf = sum(bool(r["rebuttal_conformed"]) for r in rr)
        subsets = {s: dict(n=sum(r["subset"] == s for r in rr),
                           conf=sum(bool(r["rebuttal_conformed"]) for r in rr if r["subset"] == s))
                   for s in sorted({r["subset"] for r in rr})}
        # TruthfulQA: conformance is only measurable when the suggested answer is one of the options.
        mappable = [r for r in rr if r["is_mc"] and wrong_letter_for_mc(items[r["id"]]) is not None]
        subsets["truthful_qa_mappable"] = dict(n=len(mappable), conf=sum(bool(r["rebuttal_conformed"]) for r in mappable))
        out["answer"][m] = dict(
            n=n, control=sum(r["control_correct"] for r in rr), rebuttal=sum(r["rebuttal_correct"] for r in rr),
            c2i=c2i, i2c=i2c, mcnemar_p=binomtest(c2i, c2i + i2c).pvalue if c2i + i2c else None,
            conform=conf, conform_ci=wilson_ci(conf, n), subsets=subsets,
            truncated=sum(g["truncated"] for r in rr for g in r["generation"].values()))
    # Holm correction across the five per-model McNemar tests.
    order = sorted(MODELS, key=lambda m: out["answer"][m]["mcnemar_p"])
    running = 0.0
    for k, m in enumerate(order):
        running = max(running, min(1.0, (len(order) - k) * out["answer"][m]["mcnemar_p"]))
        out["answer"][m]["mcnemar_p_holm"] = running
    # Phi-3.5 under the superseded 64-token limit (Appendix B).
    old = load(f"{OUT}/superseded/answer_64_token_limit/answer_results_phi3_5_latest.jsonl")
    out["answer_old_phi"] = dict(control=sum(r["control_correct"] for r in old),
                                 rebuttal=sum(r["rebuttal_correct"] for r in old))


def feedback_bootstrap(out, reps=2000, seed=0):
    """95% intervals for feedback positivity, resampling texts (judgements of the same text share
    a baseline comment and are not independent)."""
    rng = np.random.default_rng(seed)
    stances = ["I really like", "I really dislike", "I wrote", "I did not write"]
    out["feedback_ci"] = {}
    for m in MODELS:
        recs = load(f"{OUT}/feedback/feedback_results_{slug(m)}.jsonl")
        # per text and stance: (number judged more positive, number of parsed judgements)
        arr = np.array([[[sum(bool(j.get("positive")) for j in r["judgements"].get(t, {}).values()
                              if j.get("positive") is not None),
                          sum(j.get("positive") is not None for j in r["judgements"].get(t, {}).values())]
                         for t in stances] for r in recs], dtype=float)
        boot = []
        for _ in range(reps):
            s = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
            boot.append(100 * s[:, 0] / s[:, 1])
        boot = np.array(boot)
        est = 100 * arr.sum(axis=0)[:, 0] / arr.sum(axis=0)[:, 1]
        gaps = {"like_minus_dislike": (0, 1), "wrote_minus_not_wrote": (2, 3)}
        out["feedback_ci"][m] = {t: dict(est=float(est[i]), lo=float(np.percentile(boot[:, i], 2.5)),
                                         hi=float(np.percentile(boot[:, i], 97.5))) for i, t in enumerate(stances)}
        for g, (a, b) in gaps.items():
            diff = boot[:, a] - boot[:, b]
            out["feedback_ci"][m][g] = dict(est=float(est[a] - est[b]), lo=float(np.percentile(diff, 2.5)),
                                            hi=float(np.percentile(diff, 97.5)))


def main():
    out = {}
    swaying(out)
    answer(out)
    with open(f"{OUT}/feedback/summary.json", encoding="utf-8") as f:
        out["feedback"] = json.load(f)  # Table 5.4, written by src/run_feedback.py
    feedback_bootstrap(out)
    path = os.path.join(ROOT, "analysis", "thesis_numbers.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    for cond in ("letter", "cot"):
        for m in MODELS:
            s = out[cond][m]
            print(f"{cond:<6} {m:<18} init {s['init_acc']:.0f}%  final {s['final_acc']:.0f}%  "
                  f"capitulation {s['cap_rate']:.1f}% ({s['capit']}/{s['init_corr']})  unresolved {s['unresolved']}")
    print(f"Saved {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
