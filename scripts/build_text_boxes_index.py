"""Build a slim text-box index for GT-box OCR-REPA (experiment 2a-GT).

Reads the full restoration_dataset.json (bbox + text + polygon per text instance,
~468 MB) and writes a compact pickle keyed by image stem:

    {stem: {"boxes": [[x1, y1, x2, y2], ...], "texts": ["CHO", ...]}}

Polygons are dropped (they dominate the JSON size and 2a only needs axis-aligned
boxes). The resulting pickle is small enough to load once per dataloader worker.

Usage:
    python scripts/build_text_boxes_index.py \
        --src /data/ywk/dataset/restoration_dataset.json \
        --out preset/text_boxes_index.pkl
"""

import argparse
import json
import os
import pickle
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/ywk/dataset/restoration_dataset.json")
    ap.add_argument("--out", default="preset/text_boxes_index.pkl")
    args = ap.parse_args()

    t = time.time()
    with open(args.src, "r") as f:
        data = json.load(f)
    print(f"loaded {len(data)} images in {time.time() - t:.1f}s")

    index = {}
    n_boxes = 0
    n_with_text = 0
    for stem, sub in data.items():
        boxes, texts = [], []
        # Every image has a single sub-key ("0"); iterate defensively anyway.
        for _, entry in (sub.items() if isinstance(sub, dict) else []):
            for inst in entry.get("text_instances", []):
                bb = inst.get("bbox")
                if not bb or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in bb)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                texts.append(inst.get("text", "") or "")
        if boxes:
            index[stem] = {"boxes": boxes, "texts": texts}
            n_boxes += len(boxes)
            n_with_text += sum(1 for t_ in texts if t_)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(args.out) / 1e6
    print(
        f"wrote {len(index)} images, {n_boxes} boxes "
        f"({n_with_text} with text) -> {args.out} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
