import json
import time
from pathlib import Path
from angle_calculator import calculate_angles
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

LINK_LENGTHS = [11.5, 13.5, 16]

MIN_DEGREE_OFFSETS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -10.0,
    "elbow_flex": -10.0,
    "wrist_flex": 80.0,
}

# User-provided conversion functions
def degrees_to_ticks(degrees: float, min_range: float, max_range: float, joint_name: str = "") -> int:
    offset = MIN_DEGREE_OFFSETS.get(joint_name, 0.0)
    return int(round((degrees - offset) * 10 + min_range))

def ticks_to_degrees(ticks: int, min_range: float, max_range: float, joint_name: str = "") -> float:
    offset = MIN_DEGREE_OFFSETS.get(joint_name, 0.0)
    return (ticks - min_range) / 10.0 + offset

def _get_calibration_data(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Calibration file not found at: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def enforce_limits(ticks: int, joint_name: str, calib_data: dict) -> int:
    """Strictly ensure the computed ticks are within the calibration limits."""
    range_min = calib_data[joint_name]["range_min"]
    range_max = calib_data[joint_name]["range_max"]
    clamped_ticks = max(range_min, min(range_max, ticks))
    if clamped_ticks != ticks:
        print(f"Warning: Calculated ticks {ticks} for {joint_name} was out of range [{range_min}, {range_max}]. Clamped to {clamped_ticks}.")
    return clamped_ticks

def _angles_to_target_positions(angles: dict, calib_data: dict) -> dict:
    target_positions = {}
    for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
        if joint in angles:
            deg = angles[joint]
            cal = calib_data[joint]
            ticks = degrees_to_ticks(deg, cal["range_min"], cal["range_max"], joint)

            # Clamp within safety limits from calibration.
            safe_ticks = enforce_limits(ticks, joint, calib_data)
            target_positions[joint] = safe_ticks

    # Keep remaining joints stable to avoid large posture changes.
    if "wrist_roll" in calib_data:
        target_positions["wrist_roll"] = enforce_limits(
            calib_data["wrist_roll"]["homing_offset"], "wrist_roll", calib_data
        )
    if "gripper" in calib_data:
        target_positions["gripper"] = enforce_limits(
            calib_data["gripper"]["homing_offset"], "gripper", calib_data
        )

    return target_positions

def _interpolate_positions(start_positions: dict, target_positions: dict, steps: int) -> list[dict]:
    all_joints = sorted(set(start_positions) | set(target_positions))
    path = []
    for step in range(1, steps + 1):
        fraction = step / steps
        intermediate = {}
        for joint in all_joints:
            start_value = start_positions.get(joint, target_positions.get(joint, 0))
            end_value = target_positions.get(joint, start_value)
            intermediate[joint] = int(round(start_value + (end_value - start_value) * fraction))
        path.append(intermediate)
    return path

def _send_positions(robot, target_positions: dict, angles: dict | None = None, start_positions: dict | None = None) -> None:
    if angles is not None:
        print(f"Calculated Joint Angles (degrees):\n{angles}")
    print(f"Sending clamped safe ticks to the motors:\n{target_positions}")
    if start_positions is None:
        start_positions = target_positions

    motion_path = _interpolate_positions(start_positions, target_positions, steps=120)
    if hasattr(robot, "bus") and hasattr(robot.bus, "sync_write"):
        for positions in motion_path:
            robot.bus.sync_write("Goal_Position", positions, normalize=False)
            time.sleep(0.08)
        for _ in range(10):
            robot.bus.sync_write("Goal_Position", target_positions, normalize=False)
            time.sleep(0.05)
    elif hasattr(robot, "set_joint_positions"):
        for positions in motion_path:
            robot.set_joint_positions(positions)
            time.sleep(0.08)
        for _ in range(10):
            robot.set_joint_positions(target_positions)
            time.sleep(0.05)
    else:
        print("Warning: Could not automatically detect internal set method. Make sure to define how you write to follower arm in this env.")

def main():
    # 1. Load calibration data
    calib_path = Path(r"C:\Users\Tusha\.cache\huggingface\lerobot\calibration\robots\so_follower\my_follower_arm.json")
    try:
        calib_data = _get_calibration_data(calib_path)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return

    # 2. Setup the LeRobot robot on COM6
    config = SO101FollowerConfig(
        port="COM6",
        id="my_follower_arm",
    )
    robot = SO101Follower(config)

    print("Connecting to the SO101 Follower Arm on COM6...")
    try:
        # Connecting without requiring a new calibration cycle
        robot.connect(calibrate=False)
    except Exception as e:
        print(f"Failed to connect to the robot on COM6: {e}")
        return

    try:
        print("\n--- Automatic IK Goal Positioning ---")
        last_target_positions = None
        
        while True:
            # 3. Prompt user for coordinates
            try:
                x = float(input("Enter target X coordinate: "))
                y = float(input("Enter target Y coordinate: "))
                z = float(input("Enter target Z coordinate: "))
            except ValueError:
                print("Invalid coordinates. Must be numeric floats based on link length scale.")
                continue

            print(f"\nCalculating IK mapped angles for coords ({x}, {y}, {z}) ...")
            angles = calculate_angles(x, y, z, LINK_LENGTHS, 10, 30)
            print(f"Calculated Joint Angles (degrees):\n{angles}")

            target_positions = _angles_to_target_positions(angles, calib_data)
            _send_positions(robot, target_positions, angles, last_target_positions)
            last_target_positions = target_positions

            print("Position held. Ready for new coordinates.\n")

    except:
        print("program stopped")

if __name__ == "__main__":
    main()
