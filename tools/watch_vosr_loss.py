#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCALAR_TAGS = ("loss", "v_loss", "lr", "eval/lpips", "eval/musiq")


def find_event_files(exp_dir: Path):
    return sorted(glob.glob(str(exp_dir / "**" / "events.out.tfevents.*"), recursive=True))


def read_scalars(event_files):
    rows = []
    latest = {}
    for event_file in event_files:
        acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception:
            continue
        available = set(acc.Tags().get("scalars", []))
        for tag in SCALAR_TAGS:
            if tag not in available:
                continue
            for event in acc.Scalars(tag):
                row = {
                    "step": event.step,
                    "tag": tag,
                    "value": event.value,
                    "wall_time": event.wall_time,
                    "event_file": event_file,
                }
                rows.append(row)
                if tag not in latest or event.step >= latest[tag]["step"]:
                    latest[tag] = row
    rows.sort(key=lambda r: (r["step"], r["tag"], r["event_file"]))
    return rows, latest


def write_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "tag", "value", "wall_time", "event_file"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(exp_dir: Path, event_files, latest, summary_path: Path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"exp_dir: {exp_dir}",
        f"updated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"event_files: {len(event_files)}",
        "",
        "latest:",
    ]
    if not latest:
        lines.append("  no scalar events found yet")
    else:
        for tag in SCALAR_TAGS:
            row = latest.get(tag)
            if row:
                lines.append(f"  {tag}: step={row['step']} value={row['value']:.8g}")
    summary_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, help="Experiment directory under exp_vosr.")
    parser.add_argument("--interval", type=int, default=900, help="Polling interval in seconds.")
    parser.add_argument("--out-dir", default=None, help="Directory for loss CSV and summary.")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit.")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "loss_monitor"
    csv_path = out_dir / "scalars.csv"
    summary_path = out_dir / "summary.txt"

    while True:
        event_files = find_event_files(exp_dir)
        rows, latest = read_scalars(event_files)
        write_csv(rows, csv_path)
        write_summary(exp_dir, event_files, latest, summary_path)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] wrote {len(rows)} scalar rows to {csv_path}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
