"""Preprocess imitation learning recordings into X/Y samples.

Creates `preprocessed_data.json` with structure:
{
    "X": [ [...], ... ],
    "Y": [ [...], ... ]
}

Each input sample uses a 10-frame window:
- For each of 10 frames: 6 ticks + 6 velocities + 6 signed errors vs target
- Plus 6 target ticks appended at the end

Total input width: 10 * (6 + 6 + 6) + 6 = 186

Output Y is the delta positions between next frame and current frame (6 values).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


WINDOW_SIZE = 10


def _ensure_len(arr: List[float], n: int) -> List[float]:
    if arr is None:
        return [0.0] * n
    if len(arr) >= n:
        return arr[:n]
    return list(arr) + [0.0] * (n - len(arr))


def process(input_path: str = "imitation_learning_recordings.json",
            output_path: str = "preprocessed_data.json") -> dict:
    inp = Path(input_path)
    out = Path(output_path)
    # read with utf-8-sig to gracefully handle files that start with a BOM
    with inp.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    X = []
    Y = []

    for session in data:
        frames = session.get("frames", [])
        target = session.get("target", {})
        target_pos = target.get("positions")
        if not target_pos:
            # skip sessions without a known target
            continue

        n = len(frames)
        target_pos_6 = _ensure_len(target_pos, 6)
        # Use sliding windows [t, t+WINDOW_SIZE-1], predict delta at t+WINDOW_SIZE.
        for t in range(0, n - WINDOW_SIZE):
            sample: List[float] = []

            for k in range(t, t + WINDOW_SIZE):
                frame = frames[k]
                curr_pos = _ensure_len(frame.get("positions"), 6)
                curr_vel = _ensure_len(frame.get("velocities"), 6)
                sample.extend(curr_pos)
                sample.extend(curr_vel)
                sample.extend([curr_pos[i] - target_pos_6[i] for i in range(6)])

            sample.extend(target_pos_6)

            # predict one-step-ahead delta after the window end
            last_pos = _ensure_len(frames[t + WINDOW_SIZE - 1].get("positions"), 6)
            next_pos = _ensure_len(frames[t + WINDOW_SIZE].get("positions"), 6)
            delta = [next_pos[i] - last_pos[i] for i in range(6)]

            X.append(sample)
            Y.append(delta)

    out.write_text(json.dumps({"X": X, "Y": Y}, indent=2))
    return {"X_len": len(X), "Y_len": len(Y), "out": str(out)}


def _cli():
    p = argparse.ArgumentParser(description="Preprocess imitation learning recordings")
    p.add_argument("--input", "-i", default="imitation_learning_recordings.json")
    p.add_argument("--output", "-o", default="preprocessed_data.json")
    args = p.parse_args()
    res = process(args.input, args.output)
    print(f"Wrote {res['out']}: X={res['X_len']} Y={res['Y_len']}")


if __name__ == "__main__":
    _cli()
