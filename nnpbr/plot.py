"""Plot recorded batch losses on the optimizer's continuous update timeline."""

import argparse
import json
from pathlib import Path
import re

from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter


def draw_loss(ax, history):
    for phase, label, color in (
        (1, "Warmup", "#2474ae"),
        (2, "BC6 training", "#ba6316"),
        (3, "Decoder tuning", "#23805e"),
    ):
        points = [(h["global_step"], h["mse"]) for h in history if h["phase"] == phase]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, color=color, linewidth=0.8, marker=".", markersize=2, label=label)
            if phase > 1:
                ax.axvline(x[0], color=color, linestyle=":", alpha=0.55)
    losses = [h["mse"] for h in history if "mse" in h]
    if losses and min(losses) > 0:
        ax.set_yscale("log")
    ax.set(xlabel="Optimizer update across all phases", ylabel="Batch mean squared error")
    ax.xaxis.set_major_formatter(EngFormatter(sep=""))
    ax.grid(True, alpha=0.2)
    if losses:
        ax.legend()


def save_loss(run, history):
    path = run / "analysis/training_loss.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    ax = fig.subplots()
    draw_loss(ax, history)
    ax.set_title("Training loss by optimizer update")
    fig.savefig(path)
    return path


def read_history(run):
    history_file = run / "training_history.json"
    if history_file.is_file():
        return json.loads(history_file.read_text())
    history = []
    lengths = {}
    for phase, step, length, mse in re.findall(
        r"^\[phase (\d+) (\d+)/(\d+)\] mse=([\d.eE+-]+)$",
        (run / "train.log").read_text(), re.MULTILINE
    ):
        phase = int(phase)
        lengths[phase] = int(length)
        offset = sum(n for p, n in lengths.items() if p < phase)
        history.append({"phase": phase, "global_step": offset + int(step), "mse": float(mse)})
    if not history:
        raise ValueError("No logged training updates found")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run
    print(save_loss(run, read_history(run)))
