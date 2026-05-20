"""Run the trained model to drive the follower arm slowly."""
from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
from typing import List

import torch
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from Keyboard_controll import read_all_joint_angles, set_joint_angles
from train import load_model

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

TARGET_STOP_TOLERANCE = 30
JOINT_INDICES = [1]


def _dict_to_list(pos_dict: dict[str, int]) -> List[int]:
    return [int(pos_dict.get(joint, 0)) for joint in JOINT_ORDER]


def _list_to_dict(pos_list: List[float]) -> dict[str, int]:
    return {joint: int(round(pos_list[i])) for i, joint in enumerate(JOINT_ORDER)}


def _build_sample(current_pos: List[int], target_pos: List[int], expected_input_dim: int) -> List[float]:
    # Supports 3-feature input (1 joint x 3 features)
    if expected_input_dim != 3:
        raise ValueError(f"Unsupported model input_dim={expected_input_dim}. Expected 3.")
    sample: List[float] = []
    for i in JOINT_INDICES:
        cur = float(current_pos[i])
        tgt = float(target_pos[i])
        sample.append(cur)
        sample.append(tgt)
        sample.append(tgt - cur)
    return sample


def _normalize_sample(sample: List[float], mean: List[float] | None, std: List[float] | None) -> List[float]:
    if mean is None or std is None:
        return sample
    if len(mean) != len(sample) or len(std) != len(sample):
        return sample
    normalized: List[float] = []
    for i in range(len(sample)):
        denom = std[i] if std[i] != 0 else 1.0
        normalized.append((sample[i] - mean[i]) / denom)
    return normalized


def _predict_delta(
    model: torch.nn.Module,
    current_pos: List[int],
    target_pos: List[int],
    expected_input_dim: int,
    X_mean: List[float] | None = None,
    X_std: List[float] | None = None,
    Y_mean: List[float] | None = None,
    Y_std: List[float] | None = None,
) -> List[float]:
    sample = _build_sample(current_pos, target_pos, expected_input_dim)
    sample = _normalize_sample(sample, X_mean, X_std)
    tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor).squeeze(0)

    # Denormalize output if statistics are available
    if Y_mean is not None and Y_std is not None:
        Y_mean_tensor = torch.tensor(Y_mean, dtype=torch.float32)
        Y_std_tensor = torch.tensor(Y_std, dtype=torch.float32)
        output = output * Y_std_tensor + Y_mean_tensor

    vals = output.squeeze().tolist()
    # `tolist()` returns a float for single-element tensors, handle both cases
    if isinstance(vals, (float, int)):
        return [float(vals)]
    return [float(x) for x in vals]


def _connect_robot(port: str, robot_id: str, *, disable_torque: bool) -> SO101Follower:
    config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    if disable_torque:
        bus = getattr(robot, "bus", robot)
        if hasattr(bus, "disable_torque"):
            try:
                bus.disable_torque()
            except Exception:
                pass
    return robot


def _load_calibration(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"Warning: calibration file not found: {path}")
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        print(f"Failed to read calibration file {path}: {exc}")
        return {}


def _clamp_and_log_targets(targets: dict[str, int], calib: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for joint, val in targets.items():
        orig = int(round(val))
        if joint in calib and isinstance(calib[joint], dict):
            lo = int(calib[joint].get("range_min", -10**9))
            hi = int(calib[joint].get("range_max", 10**9))
            clamped = max(lo, min(hi, orig))
            if clamped != orig:
                print(f"Clamped {joint}: requested {orig} -> {clamped}")
            out[joint] = int(clamped)
        else:
            out[joint] = orig
    return out


def _shoulder_pan_within_tolerance(predicted: List[int], target: List[int], tolerance: int) -> bool:
    # Return True when all monitored joints are within tolerance of the target
    for i in JOINT_INDICES:
        if abs(predicted[i] - target[i]) > tolerance:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained model to drive follower arm")
    parser.add_argument("--model", type=str, default="my_model.pth", help="Path to model checkpoint")
    parser.add_argument("--leader-port", type=str, default="COM7")
    parser.add_argument("--follower-port", type=str, default="COM6")
    parser.add_argument("--calib", type=str, default=r"C:\Users\Tusha\.cache\huggingface\lerobot\calibration\robots\so_follower\my_follower_arm.json", help="Path to follower calibration JSON")
    parser.add_argument("--loop-hz", type=float, default=2.0, help="Hz for slow follower updates")
    parser.add_argument("--max-relative-target", type=float, default=20.0, help="Max per-step ticks change")
    args = parser.parse_args()

    print(f"Connecting to leader on {args.leader_port}...")
    try:
        leader = _connect_robot(args.leader_port, "leader_arm", disable_torque=True)
    except Exception as exc:
        print(f"Failed to connect to leader: {exc}")
        return

    try:
        input("Move leader to target position, then press Enter...")
        target_pos = _dict_to_list(read_all_joint_angles(leader))
        print(f"Target ticks recorded: {target_pos}")
    finally:
        try:
            bus = getattr(leader, "bus", leader)
            if hasattr(bus, "enable_torque"):
                bus.enable_torque()
        except Exception:
            pass
        leader.disconnect()

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    model.eval()
    expected_input_dim = model[0].in_features
    expected_output_dim = model[2].out_features
    if expected_input_dim != 3 or expected_output_dim != 1:
        print(
            f"Unsupported model dimensions: input_dim={expected_input_dim}, output_dim={expected_output_dim}. Expected 3->1."
        )
        return
    
    # Load normalization statistics from model metadata
    checkpoint = torch.load(args.model, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    X_mean = metadata.get("X_mean")
    X_std = metadata.get("X_std")
    Y_mean = metadata.get("Y_mean")
    Y_std = metadata.get("Y_std")
    
    print(
        f"Model dimensions: input_dim={expected_input_dim}, output_dim={expected_output_dim}"
    )

    print(f"Connecting to follower on {args.follower_port}...")
    try:
        follower = _connect_robot(args.follower_port, "follower_arm", disable_torque=False)
    except Exception as exc:
        print(f"Failed to connect to follower: {exc}")
        return

    # Load follower calibration for strict clamping
    calib = _load_calibration(args.calib)

    loop_period = 1.0 / float(args.loop_hz)
    # Residual accumulator to resolve integer truncation for small deltas.
    residual = [0.0 for _ in JOINT_INDICES]

    try:
        print("Sending first model-driven command (slow mode)...")
        current_pos = _dict_to_list(read_all_joint_angles(follower))
        deltas = _predict_delta(model, current_pos, target_pos, expected_input_dim, X_mean, X_std, Y_mean, Y_std)
        next_pos = list(current_pos)
        for idx_i, joint_idx in enumerate(JOINT_INDICES):
            desired = current_pos[joint_idx] + deltas[idx_i] + residual[idx_i]
            desired_int = int(round(desired))
            residual[idx_i] = desired - desired_int
            next_pos[joint_idx] = desired_int

        if _shoulder_pan_within_tolerance(next_pos, target_pos, TARGET_STOP_TOLERANCE):
            print(
                f"Predicted joints are within +/- {TARGET_STOP_TOLERANCE} ticks of target. Stopping."
            )
            return
            
        try:
            tgt = _list_to_dict(next_pos)
            clamped = _clamp_and_log_targets(tgt, calib)
            print(f"Sending positions (max_rel={args.max_relative_target}): {clamped}")
            set_joint_angles(
                follower,
                clamped,
                max_relative_target=args.max_relative_target,
            )
        except Exception as exc:
            print(f"Failed to send first target: {exc}")
            return
        time.sleep(loop_period)

        print("Running continuous slow updates. Press Ctrl+C to stop.")
        while True:
            pos = _dict_to_list(read_all_joint_angles(follower))
            deltas = _predict_delta(model, pos, target_pos, expected_input_dim, X_mean, X_std, Y_mean, Y_std)
            next_pos = list(pos)
            for idx_i, joint_idx in enumerate(JOINT_INDICES):
                desired = pos[joint_idx] + deltas[idx_i] + residual[idx_i]
                desired_int = int(round(desired))
                residual[idx_i] = desired - desired_int
                next_pos[joint_idx] = desired_int

            if _shoulder_pan_within_tolerance(next_pos, target_pos, TARGET_STOP_TOLERANCE):
                print(
                    f"Predicted joints are within +/- {TARGET_STOP_TOLERANCE} ticks of target. Stopping."
                )
                break
                
            try:
                tgt = _list_to_dict(next_pos)
                clamped = _clamp_and_log_targets(tgt, calib)
                print(f"Sending positions (max_rel={args.max_relative_target}): {clamped}")
                set_joint_angles(
                    follower,
                    clamped,
                    max_relative_target=args.max_relative_target,
                )
            except Exception as exc:
                print(f"Failed to send target: {exc}")
                break

            time.sleep(loop_period)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
