#!/usr/bin/env python3
"""Render shareable training curves from an unmodified HF Trainer state file."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np


def ema(values, weight):
    """Standard recursive EMA, initialized at the first observation."""
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = weight * result[i - 1] + (1 - weight) * values[i]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename without extension")
    parser.add_argument("--title", default="GR00T N1.7 | Zhewan right-place")
    parser.add_argument("--subtitle", required=True)
    parser.add_argument("--smoothing", type=float, default=0.6)
    args = parser.parse_args()
    if not 0 <= args.smoothing < 1:
        parser.error("smoothing must be in [0, 1)")

    state = json.loads(args.trainer_state.read_text())
    rows = [row for row in state["log_history"] if all(
        k in row for k in ("step", "loss", "grad_norm", "learning_rate")
    )]
    if not rows:
        parser.error("No complete training metric rows found")
    steps = np.array([row["step"] for row in rows])
    values = {key: np.array([row[key] for row in rows], dtype=float)
              for key in ("grad_norm", "learning_rate", "loss")}
    if not np.all(np.diff(steps) > 0) or not all(np.isfinite(a).all() for a in values.values()):
        parser.error("Steps must increase and metrics must be finite")
    if any((values[key] <= 0).any() for key in ("loss", "grad_norm")):
        parser.error("Log-scale loss/gradient chart requires positive values")
    if steps[-1] != state["global_step"]:
        parser.error("Final metric step does not match global_step")
    tail = steps > steps[-1] - 1000
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c8d2df", "axes.labelcolor": "#445469",
        "xtick.color": "#536177", "ytick.color": "#536177",
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.4))
    fig.patch.set_facecolor("#f5f7fb")
    fig.subplots_adjust(left=0.061, right=0.98, top=0.73, bottom=0.28, wspace=0.26)
    fig.text(0.055, 0.921, args.title, fontsize=24, weight="bold", color="#172b45")
    fig.text(0.055, 0.862, args.subtitle, fontsize=11.5, color="#526278")
    fig.text(0.055, 0.804,
             f"{int(steps[-1]):,} optimizer steps  |  {len(rows):,} logged samples  |  "
             "Training metrics only; not a task-success evaluation",
             fontsize=10.5, color="#526278")
    specs = [
        ("grad_norm", "train/grad_norm", "#7751c8", True),
        ("learning_rate", "train/learning_rate", "#bd7022", False),
        ("loss", "train/loss", "#2379b8", True),
    ]
    for ax, (key, title, color, logarithmic) in zip(axes, specs, strict=True):
        a = values[key]
        ax.set_facecolor("white")
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=16, color="#172b45")
        if logarithmic:
            ax.plot(steps, a, color=color, alpha=0.20, linewidth=0.8, label="Logged value")
            ax.plot(steps, ema(a, args.smoothing), color=color, linewidth=1.35,
                    label=f"EMA (weight={args.smoothing:g})")
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel("Log scale - all values shown", fontsize=9.5)
            ax.legend(loc="upper right", frameon=False, fontsize=8.5)
        else:
            ax.plot(steps, a, color=color, linewidth=1.8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Learning rate", fontsize=9.5)
        ax.set_xlim(0, steps[-1])
        ax.set_xticks(np.linspace(0, steps[-1], 7))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x/1000:g}k"))
        ax.set_xlabel("Optimizer step", labelpad=9)
        ax.grid(axis="both", which="major", color="#e4eaf1", linewidth=0.8)
        ax.set_axisbelow(True)

    last_lr = values["learning_rate"][-1]
    details = [
        (f"Final: {values['grad_norm'][-1]:.4f}",
         f"Last 1k mean: {values['grad_norm'][tail].mean():.4f}"),
        (f"Peak: {values['learning_rate'].max():.2e}",
         f"Final: {last_lr:.2e}"),
        (f"Final logged loss: {values['loss'][-1]:.4f}",
         f"Last 1k mean: {values['loss'][tail].mean():.6f}"),
    ]
    for ax, (line1, line2) in zip(axes, details, strict=True):
        x = ax.get_position().x0
        fig.text(x, 0.176, line1, fontsize=12, weight="bold", color="#172b45")
        fig.text(x, 0.138, line2, fontsize=10.5, color="#526278")
    fig.text(0.055, 0.076,
             "Source: checkpoint-30000 / trainer_state.json | Logging interval: 10 steps | "
             "EMA is visual smoothing, not the last-1k average.", fontsize=9, color="#68778b")
    fig.text(0.055, 0.041,
             "Loss and gradient use logarithmic axes to retain initial values and spikes. "
             "Do not compare their visual slopes with linear-axis screenshots.",
             fontsize=9, color="#68778b")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = args.output.with_suffix(f".{ext}")
        fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
