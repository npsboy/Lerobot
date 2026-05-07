import sys
import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MIN_DEGREE_OFFSETS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -10.0,
    "elbow_flex": -20.0,
    "wrist_flex": -110,
}

def read_all_joint_angles(robot):
    bus = getattr(robot, "bus", robot)
    values = bus.sync_read("Present_Position", normalize=False)
    return {joint: int(pos) for joint, pos in values.items()}

def ticks_to_degrees(ticks: int, min_range: float, max_range: float, joint_name: str = "") -> float:
    offset = MIN_DEGREE_OFFSETS.get(joint_name, 0.0)
    return (ticks - min_range) / 10.0 + offset

def main():
    config = SO101FollowerConfig(
        port="COM6",
        id="my_follower_arm",
    )
    robot = SO101Follower(config)

    print("Connecting to the SO101 Follower Arm on COM6...")
    try:
        robot.connect(calibrate=False)
        
        # Disable torque so the robot can be moved freely by hand
        print("Disabling torque on all joints...")
        bus = getattr(robot, "bus", robot)
        if hasattr(bus, "disable_torque"):
            bus.disable_torque()
            
    except Exception as e:
        print(f"Failed to connect to the robot: {e}")
        return

    # Check if we have calibration data on the bus
    bus = getattr(robot, "bus", robot)
    if not hasattr(bus, "calibration") or bus.calibration is None or not len(bus.calibration):
        print("Warning: Calibration data not found on bus. Cannot convert ticks to degrees.")
        has_calib = False
    else:
        has_calib = True

    print("\nReading live joint angles... Press Ctrl+C to stop.")
    
    try:
        while True:
            current_ticks = read_all_joint_angles(robot)
            
            output = []
            for joint_name, ticks in current_ticks.items():
                if has_calib and joint_name in bus.calibration:
                    cal = bus.calibration[joint_name]
                    deg = ticks_to_degrees(ticks, cal.range_min, cal.range_max, joint_name)
                    output.append(f"{joint_name}: {ticks} ({deg:.1f}°)")
                else:
                    output.append(f"{joint_name}: {ticks}")
            
            # Print live on the same line using \r (carriage return)
            # padded with spaces to overwrite previous potentially longer text
            line = " | ".join(output)
            sys.stdout.write(f"\r{line:<120}")
            sys.stdout.flush()
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        print("Disconnecting robot...")
        robot.disconnect()

if __name__ == "__main__":
    main()
