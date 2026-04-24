"""Simple keyboard-controlled SO101 simulator using real joint limits.

Controls:
- 1..6: Select active joint
- Left/Down: Decrease selected joint raw value
- Right/Up: Increase selected joint raw value
- Shift + Arrow: Larger step
- r: Reset all joints to mid-range
- h: Print controls in terminal

Notes:
- shoulder_pan and wrist_roll are not true planar joints, so they are shown in text.
- gripper is simulated as a two-finger opening at the wrist tip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class JointConfig:
	name: str
	servo_id: int
	drive_mode: int
	homing_offset: int
	range_min: int
	range_max: int


JOINT_CONFIGS: list[JointConfig] = [
	JointConfig("shoulder_pan", 1, 0, -512, 1099, 2910),
	JointConfig("shoulder_lift", 2, 0, 1022, 1457, 3815),
	JointConfig("elbow_flex", 3, 0, 1288, 536, 2749),
	JointConfig("wrist_flex", 4, 0, -1112, 313, 2632),
	JointConfig("wrist_roll", 5, 0, -1763, 0, 4095),
	JointConfig("gripper", 6, 0, 1696, 2037, 3296),
]


@dataclass
class SO101Simulator:
	"""A basic 2D line simulator that uses the actual SO101 joint limits."""

	link_lengths: np.ndarray = field(
		default_factory=lambda: np.array([1.0, 0.9, 0.65, 0.28], dtype=float)
	)
	selected_joint: int = 0
	step_counts: int = 20
	coarse_step_counts: int = 80

	def __post_init__(self) -> None:
		self.joint_names = np.array([cfg.name for cfg in JOINT_CONFIGS], dtype=object)
		self.range_min = np.array([cfg.range_min for cfg in JOINT_CONFIGS], dtype=float)
		self.range_max = np.array([cfg.range_max for cfg in JOINT_CONFIGS], dtype=float)
		self.homing_offset = np.array([cfg.homing_offset for cfg in JOINT_CONFIGS], dtype=float)

		self.neutral_raw = (self.range_min + self.range_max) / 2.0
		self.raw_positions = self.neutral_raw.copy()
		self.count_to_deg = 360.0 / 4096.0

		self.fig, self.ax = plt.subplots(figsize=(9, 8))
		self.fig.canvas.manager.set_window_title("SO101 Matplotlib Simulator")

		(self.arm_line,) = self.ax.plot([], [], "-o", lw=3, ms=7, color="#1f77b4")
		(self.grip_left,) = self.ax.plot([], [], "-", lw=3, color="#ff7f0e")
		(self.grip_right,) = self.ax.plot([], [], "-", lw=3, color="#ff7f0e")
		(self.pan_indicator,) = self.ax.plot([], [], "--", lw=2, color="#2ca02c", alpha=0.9)
		self.base_marker = self.ax.scatter([0.0], [0.0], s=80, zorder=4, color="black")

		self.info_text = self.ax.text(
			0.02,
			0.98,
			"",
			transform=self.ax.transAxes,
			va="top",
			family="monospace",
			fontsize=9,
			bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
		)
		self.warning_text = self.ax.text(
			0.02,
			0.03,
			"",
			transform=self.ax.transAxes,
			va="bottom",
			family="monospace",
			fontsize=10,
			color="red",
			fontweight="bold",
		)

		self.ax.set_title("SO101 2D Simple Simulator (real joint limits)")
		self.ax.set_xlabel("X (projected)")
		self.ax.set_ylabel("Y")
		self.ax.set_aspect("equal", adjustable="box")
		self.ax.grid(True, alpha=0.3)

		reach = float(np.sum(self.link_lengths))
		pad = 0.35
		self.ax.set_xlim(-reach - pad, reach + pad)
		self.ax.set_ylim(-reach - pad, reach + pad)

		self.fig.canvas.mpl_connect("key_press_event", self.on_key)
		self.draw()

	def _joint_angle_deg(self, index: int) -> float:
		return float((self.raw_positions[index] - self.neutral_raw[index]) * self.count_to_deg)

	def _limit_violations(self) -> list[str]:
		violations: list[str] = []
		for idx, name in enumerate(self.joint_names):
			pos = self.raw_positions[idx]
			if pos < self.range_min[idx]:
				violations.append(f"{name}: {pos:.0f} < min {self.range_min[idx]:.0f}")
			elif pos > self.range_max[idx]:
				violations.append(f"{name}: {pos:.0f} > max {self.range_max[idx]:.0f}")
		return violations

	def forward_kinematics(self) -> tuple[np.ndarray, np.ndarray, float, float]:
		"""Return projected arm points plus wrist direction and pan angle."""
		pitch_indices = [1, 2, 3]  # shoulder_lift, elbow_flex, wrist_flex
		pitch_angles_deg = [self._joint_angle_deg(idx) for idx in pitch_indices]

		segment_angle_deg = [pitch_angles_deg[0], pitch_angles_deg[1], pitch_angles_deg[2], 0.0]
		cumulative = 0.0
		x_points = [0.0]
		y_points = [0.0]
		x, y = 0.0, 0.0

		for length, angle_deg in zip(self.link_lengths, segment_angle_deg):
			cumulative += math.radians(angle_deg)
			x += float(length) * math.cos(cumulative)
			y += float(length) * math.sin(cumulative)
			x_points.append(x)
			y_points.append(y)

		# shoulder_pan is yaw in 3D; approximate by X foreshortening in this 2D view.
		pan_rad = math.radians(self._joint_angle_deg(0))
		x_points = [px * math.cos(pan_rad) for px in x_points]

		return np.array(x_points), np.array(y_points), cumulative, pan_rad

	def _gripper_width(self) -> float:
		gripper_idx = 5
		span = self.range_max[gripper_idx] - self.range_min[gripper_idx]
		ratio = (self.raw_positions[gripper_idx] - self.range_min[gripper_idx]) / span
		ratio = float(np.clip(ratio, 0.0, 1.0))
		return 0.06 + ratio * 0.22

	def draw(self) -> None:
		x_points, y_points, wrist_dir_rad, pan_rad = self.forward_kinematics()
		violations = self._limit_violations()

		arm_color = "#d62728" if violations else "#1f77b4"
		self.arm_line.set_color(arm_color)
		self.arm_line.set_data(x_points, y_points)

		end = np.array([x_points[-1], y_points[-1]], dtype=float)
		direction = np.array([math.cos(wrist_dir_rad), math.sin(wrist_dir_rad)], dtype=float)
		normal = np.array([-direction[1], direction[0]], dtype=float)
		grip_len = 0.22
		grip_width = self._gripper_width()

		left_base = end + normal * (grip_width / 2.0)
		right_base = end - normal * (grip_width / 2.0)
		left_tip = left_base + direction * grip_len
		right_tip = right_base + direction * grip_len

		self.grip_left.set_data([left_base[0], left_tip[0]], [left_base[1], left_tip[1]])
		self.grip_right.set_data([right_base[0], right_tip[0]], [right_base[1], right_tip[1]])

		pan_len = 0.4
		self.pan_indicator.set_data(
			[0.0, pan_len * math.cos(pan_rad)],
			[0.0, pan_len * math.sin(pan_rad)],
		)

		lines: list[str] = [f"Selected: J{self.selected_joint + 1} ({self.joint_names[self.selected_joint]})"]
		for idx, name in enumerate(self.joint_names):
			pos = self.raw_positions[idx]
			marker = " <" if idx == self.selected_joint else ""
			flag = " !" if (pos < self.range_min[idx] or pos > self.range_max[idx]) else ""
			lines.append(
				f"J{idx + 1} {name:<13} {pos:>6.0f} [{self.range_min[idx]:.0f}, {self.range_max[idx]:.0f}]"
				f"{marker}{flag}"
			)

		lines.append(
			"Keys: 1-6 select | arrows adjust raw | shift+arrow coarse | r reset"
		)
		self.info_text.set_text("\n".join(lines))

		if violations:
			self.warning_text.set_text("WARNING limit exceeded: " + " | ".join(violations))
		else:
			self.warning_text.set_text("")

		self.fig.canvas.draw_idle()

	def adjust_selected_joint(self, delta_counts: float) -> None:
		self.raw_positions[self.selected_joint] += delta_counts

	def on_key(self, event) -> None:  # Matplotlib callback signature
		key = (event.key or "").lower()
		if not key:
			return

		if key in {"1", "2", "3", "4", "5", "6"}:
			self.selected_joint = int(key) - 1
			self.draw()
			return

		if key == "r":
			self.raw_positions = self.neutral_raw.copy()
			self.draw()
			return

		if key == "h":
			print(__doc__)
			return

		step = self.coarse_step_counts if "shift" in key else self.step_counts
		if "left" in key or "down" in key:
			self.adjust_selected_joint(-step)
			self.draw()
			return

		if "right" in key or "up" in key:
			self.adjust_selected_joint(step)
			self.draw()

	def run(self) -> None:
		print(__doc__)
		plt.show()


def main() -> None:
	simulator = SO101Simulator()
	simulator.run()


if __name__ == "__main__":
	main()
