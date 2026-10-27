"""Re-runs swaying items left unresolved after a correct first answer
(thesis Section 4.3.1).

generate() stops a reply that hits its token limit while repeating itself.
After "Are you sure?", Llama 3.1 often re-works the problem at length, and the
loop check can stop such replies before they reach an answer. This script keeps
the stored turn-1 answer, regenerates turns 2 and 3 with the loop stop switched
off, and writes the new records to rerun_unresolved.jsonl next to the original
results (which are not changed). analysis/analyse.py uses the re-run outcome in
place of the original for these items.

    python src/rerun_unresolved.py              # reasoning (CoT) condition
"""
import argparse
import json
import os

from graders import score_swaying
from ollama_client import generate
from run_swaying import (CHALLENGE, FINAL_ASK, FINAL_TURN_MAX_TOKENS, MODELS, TOKEN_CEILING, TURN2_MAX_TOKENS,
                         gen_meta, load_items, output_dir, read_jsonl, results_path, turn1_prompt)


def main():
    ap = argparse.ArgumentParser(description="Re-run unresolved swaying items without the loop stop")
    ap.add_argument("--turn1", choices=["letter", "cot"], default="cot")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    args = ap.parse_args()

    out_path = os.path.join(output_dir(args.turn1), "rerun_unresolved.jsonl")
    done = {(r["model"], r["id"]) for r in read_jsonl(out_path)}
    prompts = {i: turn1_prompt(item, args.turn1) for i, item in load_items()}

    with open(out_path, "a", encoding="utf-8") as out:
        for model in args.models:
            todo = [r for r in read_jsonl(results_path(model, args.turn1))
                    if score_swaying(r)["unresolved_after_correct"] and (model, r["id"]) not in done]
            print(f"[{model}] {len(todo)} unresolved items to re-run")
            for r in todo:
                history = [{"role": "user", "content": prompts[r["id"]]},
                           {"role": "assistant", "content": r["ans1"]},
                           {"role": "user", "content": CHALLENGE}]
                r2 = generate(model, history, TURN2_MAX_TOKENS, ceiling=TOKEN_CEILING, stop_on_loop=False)
                history += [{"role": "assistant", "content": r2["text"]}, {"role": "user", "content": FINAL_ASK}]
                r3 = generate(model, history, FINAL_TURN_MAX_TOKENS, ceiling=TOKEN_CEILING, stop_on_loop=False)
                new = {**{k: r[k] for k in ("id", "model", "ground_truth", "ans1", "is_mc")},
                       "ans2": r2["text"], "final_ans": r3["text"], "rerun": True,
                       "generation": {"turn1": r["generation"]["turn1"], "turn2": gen_meta(r2), "turn3": gen_meta(r3)}}
                new.update(score_swaying(new))
                out.write(json.dumps(new) + "\n")
                out.flush()

    # Outcome of every re-run item, re-graded with the current graders.
    summary = {}
    for r in read_jsonl(out_path):
        s = score_swaying(r)
        outcome = ("held" if s["finally_correct"] else "flipped" if s["flipped"]
                   else "retreated" if s["retreated_after_correct"] else "unresolved")
        summary.setdefault(r["model"], {}).setdefault(outcome, 0)
        summary[r["model"]][outcome] += 1
    print(json.dumps(summary, indent=2))
    with open(os.path.join(output_dir(args.turn1), "rerun_unresolved_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
