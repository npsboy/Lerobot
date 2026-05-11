"""Run the trained model to drive the follower arm slowly."""
from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
from typing import Any, List

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

TARGET_STOP_TOLERANCE = 10


def _dict_to_list(pos_dict: dict[str, int]) -> List[int]:
    return [int(pos_dict.get(joint, 0)) for joint in JOINT_ORDER]


def _list_to_dict(pos_list: List[float]) -> dict[str, int]:
    return {joint: int(round(pos_list[i])) for i, joint in enumerate(JOINT_ORDER)}


def _compute_velocities(prev: List[int] | None, curr: List[int], dt: float) -> List[float]:
    if prev is None or dt <= 0:
        return [0.0] * len(curr)
    return [(curr[i] - prev[i]) / dt for i in range(len(curr))]


def _capture_frames(robot: Any, num_frames: int, hz: float) -> List[dict[str, List[float]]]:
    frames: List[dict[str, List[float]]] = []
    period = 1.0 / float(hz)
    prev_pos: List[int] | None = None
    for _ in range(num_frames):
        t0 = time.time()
        pos_dict = read_all_joint_angles(robot)
        pos_list = _dict_to_list(pos_dict)
        vel_list = _compute_velocities(prev_pos, pos_list, period)
        frames.append({"positions": pos_list, "velocities": vel_list})
        prev_pos = pos_list
        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))
    return frames


def _build_sample(frames: List[dict[str, List[float]]], target_pos: List[int], expected_input_dim: int) -> List[float]:
    # Current training data uses 18 features: curr_pos(6) + curr_vel(6) + (curr_pos-target_pos)(6)
    if expected_input_dim == 18:
        frame = frames[-1]
        curr_pos = [float(v) for v in frame["positions"][:6]]
        curr_vel = [float(v) for v in frame["velocities"][:6]]
        target_pos_6 = [float(v) for v in target_pos[:6]]
        sample = curr_pos + curr_vel + [curr_pos[i] - target_pos_6[i] for i in range(6)]
        return sample

    # Windowed format: 10 frames of [pos(6), vel(6), pos-target(6)] + target_pos(6) = 186
    if expected_input_dim == 186:
        sample: List[float] = []
        target_pos_6 = [float(v) for v in target_pos[:6]]
        frames_10 = frames[-10:]
        if len(frames_10) < 10:
            pad = [{"positions": [0.0] * 6, "velocities": [0.0] * 6}] * (10 - len(frames_10))
            frames_10 = pad + frames_10
        for frame in frames_10:
            curr_pos = [float(v) for v in frame["positions"][:6]]
            curr_vel = [float(v) for v in frame["velocities"][:6]]
            sample.extend(curr_pos)
            sample.extend(curr_vel)
            sample.extend([curr_pos[i] - target_pos_6[i] for i in range(6)])
        sample.extend(target_pos_6)
        return sample

    # Legacy format: 10-frame history with positions+velocities, then target_pos (126 features)
    if expected_input_dim == 126:
        sample: List[float] = []
        frames_10 = frames[-10:]
        if len(frames_10) < 10:
            pad = [{"positions": [0.0] * 6, "velocities": [0.0] * 6}] * (10 - len(frames_10))
            frames_10 = pad + frames_10
        for frame in frames_10:
            sample.extend([float(v) for v in frame["positions"][:6]])
            sample.extend([float(v) for v in frame["velocities"][:6]])
        sample.extend([float(v) for v in target_pos[:6]])
        return sample

    raise ValueError(
        f"Unsupported model input_dim={expected_input_dim}. Expected 18, 186, or 126 (legacy)."
    )


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
    frames: List[dict[str, List[float]]],
    target_pos: List[int],
    expected_input_dim: int,
    X_mean: List[float] | None = None,
    X_std: List[float] | None = None,
    Y_mean: List[float] | None = None,
    Y_std: List[float] | None = None,
) -> List[float]:
    sample = _build_sample(frames, target_pos, expected_input_dim)
    sample = _normalize_sample(sample, X_mean, X_std)
    tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor).squeeze(0)
    
    # Denormalize output if statistics are available
    if Y_mean is not None and Y_std is not None:
        Y_mean_tensor = torch.tensor(Y_mean, dtype=torch.float32)
        Y_std_tensor = torch.tensor(Y_std, dtype=torch.float32)
        output = output * Y_std_tensor + Y_mean_tensor
    
    return [float(v) for v in output.tolist()]


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


def _all_joints_within_tolerance(predicted: List[int], target: List[int], tolerance: int) -> bool:
    return all(abs(predicted[i] - target[i]) <= tolerance for i in range(len(JOINT_ORDER)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained model to drive follower arm")
    parser.add_argument("--model", type=str, default="my_model.pth", help="Path to model checkpoint")
    parser.add_argument("--leader-port", type=str, default="COM5")
    parser.add_argument("--follower-port", type=str, default="COM6")
    parser.add_argument("--calib", type=str, default=r"C:\Users\Tusha\.cache\huggingface\lerobot\calibration\robots\so_follower\my_follower_arm.json", help="Path to follower calibration JSON")
    parser.add_argument("--record-hz", type=float, default=20.0, help="Hz for initial leader recording")
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
        input("Press Enter to record the first 10 frames from the leader...")
        leader_frames = _capture_frames(leader, 10, args.record_hz)
        print("Captured 10 frames.")

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
    follower_frames: List[dict[str, List[float]]] = []
    prev_pos: List[int] | None = None
    prev_t = time.time()
    
    # Residual accumulator to resolve integer truncation for small deltas
    residual = [0.0] * len(JOINT_ORDER)

    try:
        print("Sending first model-driven command (slow mode)...")
        current_pos = _dict_to_list(read_all_joint_angles(follower))
        delta = _predict_delta(model, leader_frames, target_pos, expected_input_dim, X_mean, X_std, Y_mean, Y_std)
        next_pos = []
        for i in range(len(current_pos)):
            desired = current_pos[i] + delta[i] + residual[i]
            desired_int = int(round(desired))
            residual[i] = desired - desired_int
            next_pos.append(desired_int)

        if _all_joints_within_tolerance(next_pos, target_pos, TARGET_STOP_TOLERANCE):
            print(
                f"Predicted position is within +/- {TARGET_STOP_TOLERANCE} ticks of the target on all joints. Stopping."
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
            now = time.time()
            dt = max(1e-6, now - prev_t)
            prev_t = now

            pos = _dict_to_list(read_all_joint_angles(follower))
            # Use the training hz to compute velocity so the model sees familiar velocity scale
            fake_dt = 1.0 / args.record_hz 
            vel = _compute_velocities(prev_pos, pos, fake_dt)
            
            follower_frames.append({"positions": pos, "velocities": vel})
            if len(follower_frames) > 10:
                follower_frames.pop(0)
            prev_pos = pos

            prediction_frames = follower_frames if len(follower_frames) >= 10 else leader_frames
            if prediction_frames is leader_frames:
                print("Using recorded leader frames until follower history reaches 10 frames.")

            delta = _predict_delta(model, prediction_frames, target_pos, expected_input_dim, X_mean, X_std, Y_mean, Y_std)
            delta = [d * 5 for d in delta]  # Scale up prediction
            
            next_pos = []
            for i in range(len(pos)):
                desired = pos[i] + delta[i] + residual[i]
                desired_int = int(round(desired))
                residual[i] = desired - desired_int
                next_pos.append(desired_int)

            if _all_joints_within_tolerance(next_pos, target_pos, TARGET_STOP_TOLERANCE):
                print(
                    f"Predicted position is within +/- {TARGET_STOP_TOLERANCE} ticks of the target on all joints. Stopping."
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
