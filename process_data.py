"""Preprocess imitation learning recordings into X/Y samples.

Creates `preprocessed_data.json` with structure:
{
        "X": [ [...], ... ],
        "Y": [ [...], ... ],
        "X_mean": [...],
        "X_std": [...],
        "Y_mean": [...],
        "Y_std": [...]
}

Shoulder-pan-only setup:
- X has 3 values per sample:
    1) current shoulder_pan angle
    2) target shoulder_pan angle
    3) error = target - current
- Y has 1 value per sample:
    - shoulder_pan delta to next frame

Data is normalized using z-score normalization (standardization):
normalized = (value - mean) / std_dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


JOINT_INDICES = [1]
INPUT_WIDTH = 3
OUTPUT_WIDTH = 1


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

        target_pos_6 = _ensure_len(target_pos, 6)
        # Use the second joint (shoulder_lift) only
        target_vals = [float(target_pos_6[i]) for i in JOINT_INDICES]
        n = len(frames)
        # Use each frame as a separate sample, predict delta to the next frame.
        for t in range(0, n - 1):
            frame = frames[t]
            curr_pos = _ensure_len(frame.get("positions"), 6)
            # Build features (current, target, error) for the selected joint
            sample: List[float] = []
            for i, tgt in zip(JOINT_INDICES, target_vals):
                cur = float(curr_pos[i])
                sample.append(cur)
                sample.append(float(tgt))
                sample.append(float(tgt - cur))
            if len(sample) != INPUT_WIDTH:
                raise ValueError(
                    f"Built sample width {len(sample)} does not match expected input width {INPUT_WIDTH}"
                )

            # Predict one-step-ahead shoulder-pan delta from the current frame.
            next_pos = _ensure_len(frames[t + 1].get("positions"), 6)
            # Build delta output for the selected joint
            delta: List[float] = []
            for i in JOINT_INDICES:
                delta.append(float(next_pos[i] - curr_pos[i]))
            if len(delta) != OUTPUT_WIDTH:
                raise ValueError(
                    f"Built label width {len(delta)} does not match expected output width {OUTPUT_WIDTH}"
                )

            X.append(sample)
            Y.append(delta)

    # Normalize X and Y using z-score normalization (standardization)
    if not X:
        raise ValueError("No valid training samples were produced from the input recordings")

    X_arr = np.array(X, dtype=np.float32)
    Y_arr = np.array(Y, dtype=np.float32)

    X_mean = X_arr.mean(axis=0)
    X_std = X_arr.std(axis=0)
    # Avoid division by zero: add a small epsilon
    X_std = np.where(X_std == 0, 1.0, X_std)
    X_normalized = (X_arr - X_mean) / X_std

    Y_mean = Y_arr.mean(axis=0)
    Y_std = Y_arr.std(axis=0)
    # Avoid division by zero: add a small epsilon
    Y_std = np.where(Y_std == 0, 1.0, Y_std)
    Y_normalized = (Y_arr - Y_mean) / Y_std

    out.write_text(json.dumps({
        "X": X_normalized.tolist(),
        "Y": Y_normalized.tolist(),
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "Y_mean": Y_mean.tolist(),
        "Y_std": Y_std.tolist()
    }, indent=2))
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
