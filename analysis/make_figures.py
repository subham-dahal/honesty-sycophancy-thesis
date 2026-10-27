"""Draws the three thesis figures (Figures 5.1-5.3) from thesis_numbers.json
into analysis/figures/.

    python analysis/analyse.py && python analysis/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
MODELS = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "phi3.5:latest", "granite3-dense:8b"]
LABELS = {"llama3.1:8b": "Llama 3.1 8B", "qwen2.5:7b": "Qwen 2.5 7B", "mistral:7b": "Mistral 7B",
          "phi3.5:latest": "Phi-3.5", "granite3-dense:8b": "Granite 3 8B"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})


def setup(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([LABELS[m] for m in reversed(MODELS)])


def legend(ax, items, **kw):
    handles = [Line2D([], [], marker=mk, ls=ls, color=c, label=lab) for mk, ls, c, lab in items]
    ax.legend(handles=handles, loc="lower center", frameon=False, **kw)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name))


def answer_figure(d):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for i, m in enumerate(MODELS):
        y, c, r = len(MODELS) - 1 - i, d["answer"][m]["control"], d["answer"][m]["rebuttal"]
        ax.plot([r, c], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter(c, y, s=60, color=BLUE, marker="o", zorder=3, edgecolor="white", lw=1.5)
        ax.scatter(r, y, s=60, color=ORANGE, marker="D", zorder=3, edgecolor="white", lw=1.5)
        ax.text(c + 1.2, y, f"{c}%", va="center", fontsize=8.5, color=INK2)
        ax.text(r - 1.2, y, f"{r}% ({r - c:+d})", va="center", ha="right", fontsize=8.5, color=INK2)
    setup(ax)
    ax.set_xlim(25, 75)
    ax.set_xlabel("Accuracy (%)")
    legend(ax, [("o", "", BLUE, "No hint (control)"), ("D", "", ORANGE, "User suggests a wrong answer")],
           bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5)
    save(fig, "answer_accuracy.pdf")


def swaying_figure(d):
    fig, ax = plt.subplots(figsize=(6, 3.0))
    for i, m in enumerate(MODELS):
        y = len(MODELS) - 1 - i
        for cond, colour, marker, offset in (("letter", BLUE, "o", 0.15), ("cot", ORANGE, "D", -0.15)):
            s = d[cond][m]
            lo, hi = s["cap_ci"]
            ax.plot([lo, hi], [y + offset] * 2, color=colour, lw=1.5, zorder=2)
            ax.scatter(s["cap_rate"], y + offset, s=50, color=colour, marker=marker, zorder=3,
                       edgecolor="white", lw=1.2)
            ax.text(hi + 1.5, y + offset, f"{s['cap_rate']:.0f}% ({s['capit']}/{s['init_corr']})",
                    va="center", fontsize=8, color=INK2)
    setup(ax)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Capitulation rate (% of initially correct answers, 95% CI)")
    legend(ax, [("o", "-", BLUE, "Letter-only first turn"), ("D", "-", ORANGE, "Reasoning first turn")],
           bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5)
    save(fig, "swaying_capitulation.pdf")


def feedback_figure(d):
    fb = {m["model"]: m["positivity"]["all"] for m in d["feedback"]}
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.3), sharey=True)
    panels = (("I really like", "I really dislike", "Stated opinion"),
              ("I wrote", "I did not write", "Stated authorship"))
    for ax, (t1, t2, title) in zip(axs, panels):
        for i, m in enumerate(MODELS):
            y, v1, v2 = len(MODELS) - 1 - i, fb[m][t1]["positivity"], fb[m][t2]["positivity"]
            ax.plot([v2, v1], [y, y], color=GRID, lw=2, zorder=1)
            ax.scatter(v1, y, s=55, color=BLUE, marker="o", zorder=3, edgecolor="white", lw=1.2)
            ax.scatter(v2, y, s=55, color=ORANGE, marker="D", zorder=3, edgecolor="white", lw=1.2)
        ax.axvline(50, color=INK2, lw=0.8, ls=(0, (3, 3)))  # 50% = no effect
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=10, color=INK, pad=40)
        ax.set_xlabel("Feedback positivity (%)", fontsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        legend(ax, [("o", "", BLUE, f'"{t1}"'), ("D", "", ORANGE, f'"{t2}"')],
               bbox_to_anchor=(0.5, 1.08), ncol=1, fontsize=8)
    axs[0].set_yticks(range(len(MODELS)))
    axs[0].set_yticklabels([LABELS[m] for m in reversed(MODELS)])
    save(fig, "feedback_positivity.pdf")


def main():
    with open(os.path.join(HERE, "thesis_numbers.json"), encoding="utf-8") as f:
        d = json.load(f)
    os.makedirs(FIG_DIR, exist_ok=True)
    answer_figure(d)
    swaying_figure(d)
    feedback_figure(d)
    print(f"Figures saved to {os.path.relpath(FIG_DIR)}")


if __name__ == "__main__":
    main()
