"""Simple keyboard-controlled SO101 simulator using real joint limits.

Controls:
- 1..6: Select active joint
- Left/Down: Decrease selected joint raw value
- Right/Up: Increase selected joint raw value
- Shift + Arrow: Larger step
- r: Reset all joints to startup pose
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
from matplotlib.patches import Arc


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


def degrees_to_ticks(degrees: float, min_range: float, max_range: float) -> int:
	return int(round(degrees * 10 + min_range))


def ticks_to_degrees(ticks: int, min_range: float, max_range: float) -> float:
	return (ticks - min_range) / 10.0


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
		self.start_raw = self.neutral_raw.copy()
		self.start_raw[[0, 1, 2, 3]] = self.range_min[[0, 1, 2, 3]]
		self.raw_positions = self.start_raw.copy()

		self.fig, self.ax = plt.subplots(figsize=(13, 11))
		self.fig.canvas.manager.set_window_title("SO101 Matplotlib Simulator")

		(self.arm_line,) = self.ax.plot([], [], "-o", lw=4, ms=9, color="#1f77b4")
		(self.grip_left,) = self.ax.plot([], [], "-", lw=4, color="#ff7f0e")
		(self.grip_right,) = self.ax.plot([], [], "-", lw=4, color="#ff7f0e")
		(self.pan_indicator,) = self.ax.plot([], [], "--", lw=2, color="#2ca02c", alpha=0.9)
		(self.base_ref_line,) = self.ax.plot([], [], ":", lw=2, color="#7f7f7f", alpha=0.9)
		self.base_marker = self.ax.scatter([0.0], [0.0], s=120, zorder=4, color="black")

		self.included_arcs: list[Arc] = []
		self.included_arc_labels = []
		for _ in range(3):
			arc = Arc((0.0, 0.0), width=0.1, height=0.1, theta1=0.0, theta2=1.0, lw=2.5, color="#9467bd")
			self.ax.add_patch(arc)
			self.included_arcs.append(arc)
			label = self.ax.text(
				0.0, 0.0, "", fontsize=10, family="monospace", color="#9467bd",
				bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
			)
			self.included_arc_labels.append(label)

		self.info_text = self.ax.text(
			0.02, 0.98, "",
			transform=self.ax.transAxes,
			va="top",
			family="monospace",
			fontsize=11,
			bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
		)
		self.warning_text = self.ax.text(
			0.02, 0.03, "",
			transform=self.ax.transAxes,
			va="bottom",
			family="monospace",
			fontsize=12,
			color="red",
			fontweight="bold",
		)

		self.ax.set_title("SO101 2D Simple Simulator (real joint limits)", fontsize=14)
		self.ax.set_xlabel("X (projected)", fontsize=12)
		self.ax.set_ylabel("Y", fontsize=12)
		self.ax.set_aspect("equal", adjustable="box")
		self.ax.grid(True, alpha=0.3)

		reach = float(np.sum(self.link_lengths))
		pad = 0.6
		self.ax.set_xlim(-reach - pad, reach + pad)
		self.ax.set_ylim(-reach - pad, reach + pad)

		self.fig.canvas.mpl_connect("key_press_event", self.on_key)
		self.draw()

	def _joint_angle_deg(self, index: int) -> float:
		ticks = int(round(float(self.raw_positions[index])))
		return float(ticks_to_degrees(ticks, self.range_min[index], self.range_max[index]))

	@staticmethod
	def _format_signed_deg(angle_deg: float) -> str:
		if math.isclose(angle_deg, 0.0, abs_tol=1e-9):
			return "0.0deg"
		if math.isclose(abs(angle_deg), 180.0, abs_tol=1e-9):
			return "180.0deg"
		return f"{angle_deg:+.1f}deg"

	@staticmethod
	def _vector_angle_deg(vec: np.ndarray) -> float:
		return math.degrees(math.atan2(float(vec[1]), float(vec[0])))

	@staticmethod
	def _signed_angle_between_vectors_deg(prev_vec: np.ndarray, next_vec: np.ndarray) -> float:
		cross_z = float(prev_vec[0] * next_vec[1] - prev_vec[1] * next_vec[0])
		dot = float(prev_vec[0] * next_vec[0] + prev_vec[1] * next_vec[1])
		signed_delta = math.degrees(math.atan2(cross_z, dot))
		if math.isclose(abs(signed_delta), 180.0, abs_tol=1e-9):
			return 180.0
		return signed_delta

	@staticmethod
	def _between_lines_signed_deg(signed_angle_deg: float) -> float:
		if math.isclose(signed_angle_deg, 0.0, abs_tol=1e-9):
			return 0.0
		if math.isclose(abs(signed_angle_deg), 180.0, abs_tol=1e-9):
			return 180.0
		return signed_angle_deg

	def _set_included_arc(
		self,
		arc: Arc,
		center: np.ndarray,
		theta_from_deg: float,
		delta_deg: float,
		radius: float,
	) -> np.ndarray:
		"""Draw arc from theta_from_deg sweeping by delta_deg (CCW positive, matching matplotlib).
		Returns label position."""
		start_deg = theta_from_deg
		end_deg = theta_from_deg + delta_deg
		arc.center = (float(center[0]), float(center[1]))
		arc.width = float(2.0 * radius)
		arc.height = float(2.0 * radius)
		arc.theta1 = float(min(start_deg, end_deg))
		arc.theta2 = float(max(start_deg, end_deg))

		mid_deg = start_deg + 0.5 * delta_deg
		mid_rad = math.radians(mid_deg)
		label_pos = np.array(
			[
				center[0] + (radius + 0.18) * math.cos(mid_rad),
				center[1] + (radius + 0.18) * math.sin(mid_rad),
			],
			dtype=float,
		)
		return label_pos

	def _limit_violations(self) -> list[str]:
		violations: list[str] = []
		for idx, name in enumerate(self.joint_names):
			pos = self.raw_positions[idx]
			if pos < self.range_min[idx]:
				violations.append(f"{name}: {pos:.0f} < min {self.range_min[idx]:.0f}")
			elif pos > self.range_max[idx]:
				violations.append(f"{name}: {pos:.0f} > max {self.range_max[idx]:.0f}")
		return violations

	def forward_kinematics(self) -> tuple[np.ndarray, np.ndarray, float]:
		pitch_indices = [1, 2, 3]
		pitch_angles_deg = [self._joint_angle_deg(idx) for idx in pitch_indices]
		segment_angle_deg = [pitch_angles_deg[0], pitch_angles_deg[1], pitch_angles_deg[2], 0.0]
		cumulative = math.radians(180.0)
		x_points = [0.0]
		y_points = [0.0]
		x, y = 0.0, 0.0

		for length, angle_deg in zip(self.link_lengths, segment_angle_deg):
			cumulative -= math.radians(angle_deg)
			x += float(length) * math.cos(cumulative)
			y += float(length) * math.sin(cumulative)
			x_points.append(x)
			y_points.append(y)

		pan_rad = math.radians(-self._joint_angle_deg(0))
		x_points = [px * math.cos(pan_rad) for px in x_points]
		return np.array(x_points), np.array(y_points), pan_rad

	def _gripper_width(self) -> float:
		gripper_idx = 5
		span = self.range_max[gripper_idx] - self.range_min[gripper_idx]
		ratio = (self.raw_positions[gripper_idx] - self.range_min[gripper_idx]) / span
		ratio = float(np.clip(ratio, 0.0, 1.0))
		return 0.06 + ratio * 0.22

	def draw(self) -> None:
		x_points, y_points, pan_rad = self.forward_kinematics()
		violations = self._limit_violations()

		arm_color = "#d62728" if violations else "#1f77b4"
		self.arm_line.set_color(arm_color)
		self.arm_line.set_data(x_points, y_points)

		segment_vecs = [
			np.array([x_points[i + 1] - x_points[i], y_points[i + 1] - y_points[i]], dtype=float)
			for i in range(len(x_points) - 1)
		]
		segment_dirs_deg = [self._vector_angle_deg(vec) for vec in segment_vecs]
		wrist_dir_rad = math.radians(segment_dirs_deg[-1])

		end = np.array([x_points[-1], y_points[-1]], dtype=float)
		direction = np.array([math.cos(wrist_dir_rad), math.sin(wrist_dir_rad)], dtype=float)
		normal = np.array([-direction[1], direction[0]], dtype=float)
		grip_len = 0.22
		grip_width = self._gripper_width()

		left_base = end + normal * (grip_width / 2.0)
		right_base = end - normal * (grip_width / 2.0)
		self.grip_left.set_data([left_base[0], (left_base + direction * grip_len)[0]],
								[left_base[1], (left_base + direction * grip_len)[1]])
		self.grip_right.set_data([right_base[0], (right_base + direction * grip_len)[0]],
								 [right_base[1], (right_base + direction * grip_len)[1]])
		self.base_ref_line.set_data([0.0, 0.45], [0.0, 0.0])
		pan_len = 0.4
		self.pan_indicator.set_data(
			[0.0, pan_len * math.cos(pan_rad)],
			[0.0, pan_len * math.sin(pan_rad)],
		)

		base_ref_vec = np.array([1.0, 0.0], dtype=float)
		joint_vec_specs = [
			("shoulder_lift", base_ref_vec, segment_vecs[0]),
			("elbow_flex",    -segment_vecs[0], segment_vecs[1]),
			("wrist_flex",    -segment_vecs[1], segment_vecs[2]),
		]
		raw_joint_angles_deg = [
			self._signed_angle_between_vectors_deg(prev_vec, next_vec)
			for _, prev_vec, next_vec in joint_vec_specs
		]
		joint_angles_deg = [self._between_lines_signed_deg(a) for a in raw_joint_angles_deg]
		# Wrist flex: clockwise positive convention — flip sign for display.
		joint_angles_deg[2] *= -1.0

		joint_display_deg = [self._joint_angle_deg(i) for i in range(6)]
		joint_display_deg[1] = joint_angles_deg[0]
		joint_display_deg[2] = joint_angles_deg[1]
		joint_display_deg[3] = joint_angles_deg[2]

		# Arc sweep deltas passed directly (matplotlib CCW positive).
		# shoulder_lift and elbow_flex: CCW positive already matches.
		# wrist_flex: joint_angles_deg[2] is clockwise-positive, so negate for the arc sweep.
		joint_arc_specs = [
			(0, self._vector_angle_deg(base_ref_vec),      joint_angles_deg[0],   joint_vec_specs[0][0]),
			(1, self._vector_angle_deg(-segment_vecs[0]),  joint_angles_deg[1],   joint_vec_specs[1][0]),
			(2, self._vector_angle_deg(-segment_vecs[1]),  -joint_angles_deg[2],  joint_vec_specs[2][0]),
		]
		for arc_idx, (center_idx, theta_from, delta, label_prefix) in enumerate(joint_arc_specs):
			center = np.array([x_points[center_idx], y_points[center_idx]], dtype=float)
			label_pos = self._set_included_arc(
				self.included_arcs[arc_idx],
				center,
				theta_from,
				delta,
				radius=0.25,
			)
			self.included_arc_labels[arc_idx].set_position((float(label_pos[0]), float(label_pos[1])))
			self.included_arc_labels[arc_idx].set_text(
				f"{label_prefix}\n{self._format_signed_deg(joint_display_deg[arc_idx + 1])}"
			)

		lines: list[str] = [f"Selected: J{self.selected_joint + 1} ({self.joint_names[self.selected_joint]})"]
		for idx, name in enumerate(self.joint_names):
			pos = self.raw_positions[idx]
			display_ticks = int(round(float(pos)))
			if idx == 3:
				display_ticks = degrees_to_ticks(joint_display_deg[3], self.range_min[idx], self.range_max[idx])
			marker = " <" if idx == self.selected_joint else ""
			flag = " !" if (pos < self.range_min[idx] or pos > self.range_max[idx]) else ""
			deg_text = self._format_signed_deg(joint_display_deg[idx])
			lines.append(
				f"J{idx + 1} {name:<13} {display_ticks:>6}t {deg_text:>9} [{self.range_min[idx]:.0f}, {self.range_max[idx]:.0f}]"
				f"{marker}{flag}"
			)
		lines.append("Keys: 1-6 select | arrows adjust raw | shift+arrow coarse | r startup pose")
		self.info_text.set_text("\n".join(lines))

		if violations:
			self.warning_text.set_text("WARNING limit exceeded: " + " | ".join(violations))
		else:
			self.warning_text.set_text("")

		self.fig.canvas.draw_idle()

	def adjust_selected_joint(self, delta_counts: float) -> None:
		self.raw_positions[self.selected_joint] += delta_counts

	def on_key(self, event) -> None:
		key = (event.key or "").lower()
		if not key:
			return
		if key in {"1", "2", "3", "4", "5", "6"}:
			self.selected_joint = int(key) - 1
			self.draw()
			return
		if key == "r":
			self.raw_positions = self.start_raw.copy()
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