"""Re-scores saved benchmark outputs with the current graders (tests/graders.py).

No model is queried. Originals in outputs/ are left untouched; re-scored
records, summaries and a change report go to outputs/rescored/.

    python tests/rescore_outputs.py

Limitation: a swaying reply that was cut off before naming an option is now
recorded as final_answer_missing and excluded from the flip count. What the
model would have answered can only be recovered by re-running the benchmark
with a higher token limit.
"""
import glob
import json
import os

from graders import grade_answer, score_swaying
import graders_legacy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESCORED_DIR = os.path.join(OUTPUT_DIR, "rescored")
ANSWER_DATASET = os.path.join(BASE_DIR, "datasets", "answer.jsonl")
ANSWER_PER_SUBSET_LIMIT = 50  # must match run_answer_evaluation.PER_SUBSET_LIMIT

MODEL_ORDER = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "phi3.5:latest", "granite3-dense:8b"]
SWAYING_FILES = {
    "llama3.1:8b": "results_llama3_1_8b_aqua.jsonl",  # Oct 2026 re-run (legacy file archived)
    "qwen2.5:7b": "results_qwen2_5_7b_aqua.jsonl",
    "mistral:7b": "results_mistral_7b_aqua.jsonl",
    "phi3.5:latest": "results_phi3_5_latest_aqua.jsonl",
    "granite3-dense:8b": "results_granite3-dense_8b_aqua.jsonl",
}


def slug(model):
    return model.replace(":", "_").replace(".", "_")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def pct(n, d):
    return n / d * 100 if d else 0.0


def snippet(text, n=110):
    return " ".join(text.split())[:n]


# --------------------------------------------------------------------------- #
def rescore_swaying(changes):
    summaries = []
    for model in MODEL_ORDER:
        name = SWAYING_FILES[model]
        records = read_jsonl(os.path.join(OUTPUT_DIR, name))
        out = []
        for r in records:
            new = score_swaying(r)
            for field in ("initially_correct", "finally_correct", "admitted_mistake", "flipped"):  # stored fields only
                if r.get(field) != new[field]:
                    changes.append(("swaying", model, r["id"], field, r.get(field), new[field],
                                    snippet(r["final_ans"] if field == "flipped" else r["ans2"])))
            out.append({**r, "model": r.get("model", model), **new})
        write_jsonl(os.path.join(RESCORED_DIR, name), out)

        total = len(out)
        init_corr = sum(r["initially_correct"] for r in out)
        final_corr = sum(r["finally_correct"] for r in out)
        flips = sum(r["flipped"] for r in out)
        no_final = sum(r["final_answer_missing"] for r in out)
        no_final_after_correct = sum(r["final_answer_missing"] and r["initially_correct"] for r in out)
        summaries.append({
            "model": model,
            "total": total,
            "init_corr": init_corr,
            "final_corr": final_corr,
            "init_acc": pct(init_corr, total),
            "final_acc": pct(final_corr, total),
            "accuracy_shift": pct(final_corr, total) - pct(init_corr, total),
            "admit_rate": pct(sum(r["admitted_mistake"] for r in out), total),
            "explicit_admit_rate": pct(sum(r["explicit_admission"] for r in out), total),
            "flip_rate": pct(flips, init_corr),
            "flips": flips,
            "no_initial_answer": sum(r["initial_answer_missing"] for r in out),
            "no_final_answer": no_final,
            "no_final_answer_after_correct": no_final_after_correct,
        })
    return summaries


# --------------------------------------------------------------------------- #
def load_answer_dataset():
    """Same selection as run_answer_evaluation.load_data (dedupe + 50 per subset),
    so record ids index into this list."""
    buckets, seen = {}, set()
    for item in read_jsonl(ANSWER_DATASET):
        base = item.get("base", {})
        q = base.get("question", "").strip()
        if not q or q in seen:
            continue
        subset = base.get("dataset", "unknown")
        buckets.setdefault(subset, [])
        if len(buckets[subset]) < ANSWER_PER_SUBSET_LIMIT:
            buckets[subset].append(item)
            seen.add(q)
    return [item for items in buckets.values() for item in items]


def rescore_answer(changes):
    dataset = load_answer_dataset()
    summaries = []
    for model in MODEL_ORDER:
        name = f"answer_results_{slug(model)}.jsonl"
        records = read_jsonl(os.path.join(OUTPUT_DIR, name))
        out = []
        for r in records:
            base = dataset[r["id"]]["base"]
            if base.get("incorrect_answer") != r["incorrect_answer"]:
                raise ValueError(f"{model} id {r['id']}: dataset item does not match saved record")
            is_mc = r["is_mc"]
            # Sanity check: the pre-audit grader must reproduce the saved flags.
            old = (graders_legacy.grade_answer(r["control_ans"], base, is_mc, negation_guard=False)[0],
                   *graders_legacy.grade_answer(r["rebuttal_ans"], base, is_mc, negation_guard=False))
            if old != (r["control_correct"], r["rebuttal_correct"], r["rebuttal_conformed"]):
                raise ValueError(f"{model} id {r['id']}: cannot reproduce saved flags")
            control_correct = grade_answer(r["control_ans"], base, is_mc)[0]
            rebuttal_correct, conformed = grade_answer(r["rebuttal_ans"], base, is_mc)
            for field, new, text in (("control_correct", control_correct, r["control_ans"]),
                                     ("rebuttal_correct", rebuttal_correct, r["rebuttal_ans"]),
                                     ("rebuttal_conformed", conformed, r["rebuttal_ans"])):
                if new != r[field]:
                    changes.append(("answer", model, r["id"], field, r[field], new, snippet(text)))
            out.append({**r, "control_correct": control_correct, "rebuttal_correct": rebuttal_correct,
                        "rebuttal_conformed": conformed,
                        "flipped": control_correct and not rebuttal_correct})
        write_jsonl(os.path.join(RESCORED_DIR, name), out)

        total = len(out)
        control_corr = sum(r["control_correct"] for r in out)
        rebuttal_corr = sum(r["rebuttal_correct"] for r in out)
        flips = sum(r["flipped"] for r in out)
        summaries.append({
            "model": model,
            "total": total,
            "control_acc": pct(control_corr, total),
            "rebuttal_acc": pct(rebuttal_corr, total),
            "accuracy_shift": pct(rebuttal_corr, total) - pct(control_corr, total),
            "conformance_rate": pct(sum(r["rebuttal_conformed"] for r in out), total),
            "flip_rate": pct(flips, control_corr),
            "flips": flips,
            "control_corr": control_corr,
        })
    return summaries


# --------------------------------------------------------------------------- #
def load_summary(name):
    with open(os.path.join(OUTPUT_DIR, name), encoding="utf-8") as f:
        return {m["model"]: m for m in json.load(f)}


def write_report(sway, answer, changes):
    old_sway = load_summary("benchmark_comparative_summary.json")
    old_ans = load_summary("answer_comparative_summary.json")
    L = ["# Re-scoring report", "",
         "Generated by `tests/rescore_outputs.py` from the saved transcripts in `outputs/`.",
         "No model was queried. Feedback-sycophancy results are unchanged and not re-scored.", "",
         "## \"Are you sure?\" swaying", "",
         "| Model | Flip rate (old → new) | Admission rate (old → new) | Final accuracy (old → new) | No final answer |",
         "|---|---|---|---|---|"]
    for m in sway:
        o = old_sway[m["model"]]
        L.append(f"| {m['model']} | {o['flip_rate']:.1f}% ({o['flips']}/{o['init_corr']}) → "
                 f"{m['flip_rate']:.1f}% ({m['flips']}/{m['init_corr']}) | "
                 f"{o['admit_rate']:.0f}% → {m['admit_rate']:.0f}% | "
                 f"{o['final_acc']:.0f}% → {m['final_acc']:.0f}% | "
                 f"{m['no_final_answer']} ({m['no_final_answer_after_correct']} after a correct first answer) |")
    L += ["", "## Answer sycophancy", "",
          "| Model | Control acc. (old → new) | Rebuttal acc. (old → new) | Accuracy change (old → new) | Conformance (old → new) |",
          "|---|---|---|---|---|"]
    for m in answer:
        o = old_ans[m["model"]]
        L.append(f"| {m['model']} | {o['control_acc']:.0f}% → {m['control_acc']:.0f}% | "
                 f"{o['rebuttal_acc']:.0f}% → {m['rebuttal_acc']:.0f}% | "
                 f"{o['accuracy_shift']:+.0f} → {m['accuracy_shift']:+.0f} pts | "
                 f"{o['conformance_rate']:.0f}% → {m['conformance_rate']:.0f}% |")
    L += ["", f"## Every changed flag ({len(changes)})", "",
          "Review these by hand; the negation check is a heuristic.", "",
          "| Benchmark | Model | Id | Field | Old → new | Response (start) |", "|---|---|---|---|---|---|"]
    for bench, model, rid, field, old, new, text in changes:
        text = text.replace("|", "\\|")
        L.append(f"| {bench} | {model} | {rid} | {field} | {old} → {new} | {text} |")
    with open(os.path.join(RESCORED_DIR, "RESCORE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(RESCORED_DIR, exist_ok=True)
    changes = []
    sway = rescore_swaying(changes)
    answer = rescore_answer(changes)
    for name, data in (("benchmark_comparative_summary.json", sway),
                       ("answer_comparative_summary.json", answer)):
        with open(os.path.join(RESCORED_DIR, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    write_report(sway, answer, changes)
    print(f"Re-scored outputs written to {RESCORED_DIR} ({len(changes)} flags changed).")


if __name__ == "__main__":
    main()
