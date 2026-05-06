#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir, tags):
    acc = EventAccumulator(str(logdir), size_guidance={"scalars": 1_000_000})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    curves = {}
    for tag in tags:
        if tag not in available:
            continue
        # Keep the last value if a step appears more than once.
        by_step = {}
        for item in acc.Scalars(tag):
            by_step[int(item.step)] = float(item.value)
        curves[tag] = sorted(by_step.items())
    return curves


def moving_average(points, window):
    if window <= 1 or len(points) <= 1:
        return points
    out = []
    values = []
    running = 0.0
    for step, value in points:
        values.append(value)
        running += value
        if len(values) > window:
            running -= values.pop(0)
        out.append((step, running / len(values)))
    return out


def write_csv(curves, output_csv):
    steps = sorted({step for points in curves.values() for step, _ in points})
    by_tag = {tag: dict(points) for tag, points in curves.items()}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *curves.keys()])
        for step in steps:
            writer.writerow([step, *[by_tag[tag].get(step, "") for tag in curves.keys()]])


def plot_curves(curves, output_png, smooth_window):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    n = len(curves)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(4, 3.5 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (tag, points) in zip(axes, curves.items()):
        xs = [s for s, _ in points]
        ys = [v for _, v in points]
        ax.plot(xs, ys, alpha=0.28, linewidth=1.0, label=f"{tag} raw")
        if smooth_window > 1:
            smoothed = moving_average(points, smooth_window)
            ax.plot(
                [s for s, _ in smoothed],
                [v for _, v in smoothed],
                linewidth=2.0,
                label=f"{tag} ma{smooth_window}",
            )
        ax.set_ylabel(tag)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def print_summary(curves, smooth_window):
    for tag, points in curves.items():
        if not points:
            continue
        print(f"\n[{tag}]")
        print(f"points: {len(points)}")
        print(f"first: step={points[0][0]} value={points[0][1]:.6g}")
        print(f"last : step={points[-1][0]} value={points[-1][1]:.6g}")
        tail = points[-10:]
        print("tail :", ", ".join(f"{s}:{v:.6g}" for s, v in tail))
        if len(points) >= smooth_window and smooth_window > 1:
            smoothed = moving_average(points, smooth_window)
            print(
                f"ma{smooth_window}: first={smoothed[smooth_window - 1][1]:.6g} "
                f"last={smoothed[-1][1]:.6g}"
            )


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard scalar curves.")
    parser.add_argument("--logdir", default="logs/vosr_text_0.5B_40k")
    parser.add_argument("--out", default="training_curve.png")
    parser.add_argument("--csv", default="training_curve.csv")
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--tags", nargs="+", default=["loss", "v_loss", "lr"])
    args = parser.parse_args()

    curves = load_scalars(Path(args.logdir), args.tags)
    if not curves:
        raise SystemExit(f"No matching scalar tags found in {args.logdir}.")

    write_csv(curves, Path(args.csv))
    plot_curves(curves, Path(args.out), args.smooth)
    print_summary(curves, args.smooth)
    print(f"\nWrote: {args.out}")
    print(f"Wrote: {args.csv}")


if __name__ == "__main__":
    main()
