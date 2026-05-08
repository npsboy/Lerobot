"""Preprocess imitation learning recordings into X/Y samples.

Creates `preprocessed_data.json` with structure:
{
  "X": [ [...], ... ],
  "Y": [ [...], ... ]
}

Input sample format (for `past_frames=10`):
- past 10 frames each: 6 positions then 6 velocities (flattened oldest->newest)
- followed by 6 target positions

Output Y is the delta positions between next frame and current frame (6 values).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _ensure_len(arr: List[float], n: int) -> List[float]:
    if arr is None:
        return [0.0] * n
    if len(arr) >= n:
        return arr[:n]
    return list(arr) + [0.0] * (n - len(arr))


def process(input_path: str = "imitation_learning_recordings.json",
            output_path: str = "preprocessed_data.json",
            past_frames: int = 10) -> dict:
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
        # for each timestep t where we have past_frames up to t and a next frame (t+1)
        for t in range(past_frames - 1, n - 1):
            indices = list(range(t - (past_frames - 1), t + 1))
            sample: List[float] = []

            ok = True
            for idx in indices:
                f = frames[idx]
                pos = _ensure_len(f.get("positions"), 6)
                vel = _ensure_len(f.get("velocities"), 6)
                sample.extend(pos)
                sample.extend(vel)

            # append target positions (session-level)
            sample.extend(_ensure_len(target_pos, 6))

            # compute delta: next_positions - current_positions
            next_pos = _ensure_len(frames[t + 1].get("positions"), 6)
            curr_pos = _ensure_len(frames[t].get("positions"), 6)
            delta = [next_pos[i] - curr_pos[i] for i in range(6)]

            X.append(sample)
            Y.append(delta)

    out.write_text(json.dumps({"X": X, "Y": Y}, indent=2))
    return {"X_len": len(X), "Y_len": len(Y), "out": str(out)}


def _cli():
    p = argparse.ArgumentParser(description="Preprocess imitation learning recordings")
    p.add_argument("--input", "-i", default="imitation_learning_recordings.json")
    p.add_argument("--output", "-o", default="preprocessed_data.json")
    p.add_argument("--past-frames", "-k", type=int, default=10)
    args = p.parse_args()
    res = process(args.input, args.output, args.past_frames)
    print(f"Wrote {res['out']}: X={res['X_len']} Y={res['Y_len']}")


if __name__ == "__main__":
    _cli()
