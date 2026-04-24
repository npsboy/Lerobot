"""Joint angle helpers built on top of LeRobot's motor bus API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def _get_bus(robot_or_bus: Any):
	"""Return a LeRobot MotorsBus from either a robot object or a bus itself."""
	return getattr(robot_or_bus, "bus", robot_or_bus)


def _ensure_bus_calibration(robot_or_bus: Any) -> bool:
	"""Ensure bus calibration is available; fallback to reading it from motors."""
	bus = _get_bus(robot_or_bus)
	if getattr(bus, "calibration", None):
		return True

	try:
		calibration = bus.read_calibration()
	except Exception:
		return False

	if not calibration:
		return False

	bus.calibration = calibration
	if hasattr(robot_or_bus, "calibration"):
		robot_or_bus.calibration = calibration
	return True


def _clamp_to_calibration_limits(
	robot_or_bus: Any,
	goal_pos: Mapping[str, int],
) -> dict[str, int]:
	"""Clamp target ticks to each joint's calibration range_min/range_max."""
	bus = _get_bus(robot_or_bus)
	if not _ensure_bus_calibration(robot_or_bus):
		raise RuntimeError(
			"No calibration found in file or motors. Cannot enforce range limits."
		)

	clamped: dict[str, int] = {}
	for joint, target in goal_pos.items():
		if joint not in bus.calibration:
			raise KeyError(
				f"Joint '{joint}' has no calibration entry; cannot enforce limits."
			)
		cal = bus.calibration[joint]
		lo = int(cal.range_min)
		hi = int(cal.range_max)
		clamped[joint] = max(lo, min(hi, int(target)))

	return clamped


def read_all_joint_angles(robot_or_bus: Any) -> dict[str, int]:
	"""Read all current joint positions in raw motor ticks.

	Args:
		robot_or_bus: Either a LeRobot robot object that has ``.bus`` or a
			MotorsBus instance directly.

	Returns:
		Mapping ``joint_name -> position_ticks``.
	"""
	bus = _get_bus(robot_or_bus)
	# Use raw ticks (same scale as calibration values).
	values = bus.sync_read("Present_Position", normalize=False)
	return {joint: int(pos) for joint, pos in values.items()}


def set_joint_angles(
	robot_or_bus: Any,
	joint_angles: Mapping[str, int | float],
	*,
	max_relative_target: int | float | None = None,
) -> dict[str, int]:
	"""Set target joint positions in raw motor ticks.

	Args:
		robot_or_bus: Either a LeRobot robot object that has ``.bus`` or a
			MotorsBus instance directly.
		joint_angles: Desired targets as ``joint_name -> position_ticks``.
		max_relative_target: Optional per-step safety clamp in ticks. If set,
			each target is clipped to be at most this far from the current angle.

	Returns:
		The exact goal angles that were sent.
	"""
	bus = _get_bus(robot_or_bus)

	goal_pos = {joint: int(round(angle)) for joint, angle in joint_angles.items()}

	if max_relative_target is not None:
		if isinstance(max_relative_target, (int, float)):
			max_relative_target = float(max_relative_target)
		present_pos = bus.sync_read("Present_Position", normalize=False)
		goal_present_pos = {
			joint: (target, int(present_pos[joint]))
			for joint, target in goal_pos.items()
		}
		goal_pos = ensure_safe_goal_position(goal_present_pos, max_relative_target)
		goal_pos = {joint: int(round(pos)) for joint, pos in goal_pos.items()}

	# Hard safety bound: never exceed calibration limits.
	goal_pos = _clamp_to_calibration_limits(robot_or_bus, goal_pos)

	bus.sync_write("Goal_Position", goal_pos, normalize=False)
	return {joint: int(pos) for joint, pos in goal_pos.items()}


def example_usage_so101_com6() -> None:
	"""Example usage for an SO101 follower connected on COM6."""
	config = SO101FollowerConfig(
		port="COM6",
		id="my_follower_arm",
		max_relative_target=15.0,
	)
	robot = SO101Follower(config)

	# Reuse stored calibration files and do not run calibration.
	robot.connect(calibrate=False)
	try:
		if not _ensure_bus_calibration(robot):
			answer = input(
				"Calibration not found in file or motors. Run calibration now? [y/N]: "
			).strip().lower()
			if answer in {"y", "yes"}:
				robot.calibrate()
				_ensure_bus_calibration(robot)
			else:
				raise RuntimeError("Calibration required to read/set normalized joint angles.")

		current = read_all_joint_angles(robot)
		print("Current joint positions (ticks):")
		for name, angle in current.items():
			print(f"  {name}: {angle}")

		target = {
			"shoulder_pan": current["shoulder_pan"] + 40,
			"shoulder_lift": current["shoulder_lift"] + 30,
			"elbow_flex": current["elbow_flex"] - 40,
			"wrist_flex": current["wrist_flex"],
			"wrist_roll": current["wrist_roll"],
			"gripper": current["gripper"],
		}

		sent = set_joint_angles(robot, target, max_relative_target=80)
		print("Sent target joint positions (ticks):", sent)
	finally:
		robot.disconnect()


def keyboard_control_shoulder_pan_so101_com6(step_ticks: int = 20) -> None:
	"""Control shoulder_pan with keyboard arrows (Left/Right) on SO101 COM6."""
	import msvcrt

	config = SO101FollowerConfig(
		port="COM6",
		id="my_follower_arm",
		max_relative_target=15.0,
	)
	robot = SO101Follower(config)

	robot.connect(calibrate=False)
	try:
		if not _ensure_bus_calibration(robot):
			answer = input(
				"Calibration not found in file or motors. Run calibration now? [y/N]: "
			).strip().lower()
			if answer in {"y", "yes"}:
				robot.calibrate()
				_ensure_bus_calibration(robot)
			else:
				raise RuntimeError("Calibration required to read/set normalized joint angles.")

		print("Keyboard control ready.")
		print("Left arrow: shoulder_pan - step_ticks")
		print("Right arrow: shoulder_pan + step_ticks")
		print("Press SPACE or 'q' to quit.")

		while True:
			key = msvcrt.getch()

			if key in (b" ", b"q", b"Q"):
				print("Exiting keyboard control.")
				break

			if key in (b"\x00", b"\xe0"):
				arrow = msvcrt.getch()
				if arrow not in (b"K", b"M"):
					continue

				current = read_all_joint_angles(robot)
				shoulder_pan = current["shoulder_pan"]
				if arrow == b"K":
					target = shoulder_pan - step_ticks
					label = "LEFT"
				else:
					target = shoulder_pan + step_ticks
					label = "RIGHT"

				sent = set_joint_angles(
					robot,
					{"shoulder_pan": target},
						max_relative_target=step_ticks,
				)
				print(
					f"{label}: shoulder_pan {shoulder_pan} -> {sent['shoulder_pan']} ticks"
				)
	finally:
		robot.disconnect()


if __name__ == "__main__":
	keyboard_control_shoulder_pan_so101_com6(step_ticks=20)

