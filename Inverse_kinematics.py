import json
import time
from pathlib import Path
from angle_calculator import calculate_angles
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

LINK_LENGTHS = [10, 10, 15]

# User-provided conversion functions
def degrees_to_ticks(degrees: float, min_range: float, max_range: float) -> int:
    return int(round(degrees * 10 + min_range))

def ticks_to_degrees(ticks: int, min_range: float, max_range: float) -> float:
    return (ticks - min_range) / 10.0

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
        # 3. Prompt user for coordinates
        print("\n--- Automatic IK Goal Positioning ---")
        try:
            x = float(input("Enter target X coordinate: "))
            y = float(input("Enter target Y coordinate: "))
            z = float(input("Enter target Z coordinate: "))
        except ValueError:
            print("Invalid coordinates. Must be numeric floats based on link length scale.")
            return

        print(f"\nCalculating IK mapped angles for coords ({x}, {y}, {z}) ...")
        angles = calculate_angles(x, y, z, LINK_LENGTHS)
        print(f"Calculated Joint Angles (degrees):\n{angles}")

        # 4. Map the newly calculated angles to motor ticks correctly applying the required formulas 
        target_positions = {}
        for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
            if joint in angles:
                deg = angles[joint]
                cal = calib_data[joint]
                if (joint == "elbow_flex"):
                    ticks = degrees_to_ticks(180-deg, cal["range_min"], cal["range_max"])
                else:
                    ticks = degrees_to_ticks(deg, cal["range_min"], cal["range_max"])
                # STRICTLY Ensure safe ranges are never breached
                safe_ticks = enforce_limits(ticks, joint, calib_data)
                target_positions[joint] = safe_ticks
        
        # Adding some fallbacks to remaining joints to not drastically throw off posture
        if "wrist_roll" in calib_data:
            target_positions["wrist_roll"] = enforce_limits(calib_data["wrist_roll"]["homing_offset"], "wrist_roll", calib_data)
        if "gripper" in calib_data:
            target_positions["gripper"] = enforce_limits(calib_data["gripper"]["homing_offset"], "gripper", calib_data)

        # 5. Send target parameters to motors using Lerobot Bus primitives directly
        print(f"Sending clamped safe ticks to the motors:\n{target_positions}")
        if hasattr(robot, "bus") and hasattr(robot.bus, "sync_write"):
            # Rely on sync_write or internal dictionary writing methods built into the LeRobot device handler framework.
            while True: 
                robot.bus.sync_write("Goal_Position", target_positions, normalize=False)
        elif hasattr(robot, "set_joint_positions"):
            while True:
                robot.set_joint_positions(target_positions)
        else:
             print("Warning: Could not automatically detect internal set method. Make sure to define how you write to follower arm in this env.")
             
        time.sleep(1.0)
        print("Done!")

    except:
        print("program stopped")

if __name__ == "__main__":
    main()
