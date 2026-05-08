"""Simple record / replay utility.

Record: reads raw motor ticks from a leader arm on COM5 and saves a JSON file.
Replay: loads JSON and sends recorded ticks to follower on COM6, preserving timing.

Usage examples:
  python record_replay.py record --out recording.json
  python record_replay.py replay --in recording.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# reuse helpers from project
from Keyboard_controll import read_all_joint_angles, set_joint_angles


def record(leader_port: str, out_path: Path, hz: float = 20.0) -> None:
	config = SO101FollowerConfig(port=leader_port, id="leader_arm")
	leader = SO101Follower(config)

	print(f"Connecting to leader on {leader_port}...")
	try:
		leader.connect(calibrate=False)
		# Ensure we don't send any torque/goal commands to the motors while recording
		bus = getattr(leader, "bus", leader)
		if hasattr(bus, "disable_torque"):
			try:
				bus.disable_torque()
				print("Disabled motor torque on leader so it can be moved by hand.")
			except Exception:
				# non-fatal: continue reading if disable_torque isn't available
				pass
	except Exception as e:
		print("Failed to connect to leader:", e)
		return

	frames: list[dict[str, Any]] = []
	period = 1.0 / float(hz)
	print("Recording... move the leader arm. Press Ctrl+C to stop.")
	start_t = time.time()
	try:
		while True:
			ts = time.time() - start_t
			pos = read_all_joint_angles(leader)
			frames.append({"t": ts, "positions": pos})
			time.sleep(period)
	except KeyboardInterrupt:
		print("\nStopped recording by user.")
	finally:
		# restore torque if possible before disconnecting
		try:
			bus = getattr(leader, "bus", leader)
			if hasattr(bus, "enable_torque"):
				try:
					bus.enable_torque()
					print("Re-enabled motor torque on leader.")
				except Exception:
					pass
		except Exception:
			pass
		leader.disconnect()

	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8") as f:
		json.dump({"hz": hz, "frames": frames}, f, indent=2)

	print(f"Saved recording to {out_path}")


def replay(follower_port: str, in_path: Path, speed: float = 1.0) -> None:
	if not in_path.exists():
		print("Recording file not found:", in_path)
		return

	with in_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	frames = data.get("frames", [])
	if not frames:
		print("No frames found in recording.")
		return

	config = SO101FollowerConfig(port=follower_port, id="my_follower_arm", max_relative_target=80)
	follower = SO101Follower(config)
	print(f"Connecting to follower on {follower_port}...")
	try:
		follower.connect(calibrate=False)
	except Exception as e:
		print("Failed to connect to follower:", e)
		return

	print("Replaying to follower. Press Ctrl+C to stop.")
	start_t = time.time()
	try:
		base_t = frames[0]["t"] if "t" in frames[0] else 0.0
		for frame in frames:
			desired_t = start_t + (frame["t"] - base_t) / float(speed)
			now = time.time()
			to_wait = desired_t - now
			if to_wait > 0:
				time.sleep(to_wait)

			positions = frame["positions"]
			try:
				set_joint_angles(follower, positions, max_relative_target=80)
			except RuntimeError as e:
				print("Failed to send positions (calibration/motor limits?):", e)
				break
	except KeyboardInterrupt:
		print("\nReplay stopped by user.")
	finally:
		follower.disconnect()


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Record or replay joint recordings")
	sub = p.add_subparsers(dest="cmd", required=True)

	r = sub.add_parser("record")
	r.add_argument("--out", type=Path, default=Path("recording.json"))
	r.add_argument("--leader-port", default="COM5")
	r.add_argument("--hz", type=float, default=20.0)

	p2 = sub.add_parser("replay")
	p2.add_argument("--in", dest="in_path", type=Path, required=True)
	p2.add_argument("--follower-port", default="COM6")
	p2.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")

	return p


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if args.cmd == "record":
		record(args.leader_port, args.out, hz=args.hz)
	elif args.cmd == "replay":
		replay(args.follower_port, args.in_path, speed=args.speed)


if __name__ == "__main__":
	main()
