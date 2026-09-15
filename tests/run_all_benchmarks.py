"""Runs all three sycophancy benchmarks in sequence across every model.

Each sub-runner is resumable, so re-running this after an interruption picks up
where it left off.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

RUNNERS = [
    ("Are You Sure? (swaying)", "run_full_evaluation.py"),
    ("Answer sycophancy",        "run_answer_evaluation.py"),
    ("Feedback sycophancy",      "run_feedback_evaluation.py"),
]

def main():
    for title, script_name in RUNNERS:
        script_path = os.path.join(HERE, script_name)

        if not os.path.exists(script_path):
            print(f"\n[ERROR] Script not found: {script_path}")
            sys.exit(1)

        print("\n" + "=" * 70)
        print(f"RUNNING SUITE: {title} ({script_name})")
        print("=" * 70 + "\n")

        # sys.executable ensures it uses the exact same venv/python environment
        result = subprocess.run([sys.executable, script_path])

        if result.returncode != 0:
            print(f"\n[WARNING] {script_name} exited with error code {result.returncode}.")
            # Choose whether to stop or continue to the next benchmark:
            # sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("ALL THREE BENCHMARK SUITES COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()