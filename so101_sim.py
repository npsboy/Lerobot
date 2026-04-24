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
- Link lengths are adjustable via sliders in the left sidebar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyBboxPatch
from matplotlib.widgets import Slider


@dataclass(frozen=True)
class JointConfig:
    name: str
    servo_id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


JOINT_CONFIGS: list[JointConfig] = [
    JointConfig("shoulder_pan",  1, 0,  -512, 1099, 2910),
    JointConfig("shoulder_lift", 2, 0,  1022, 1457, 3815),
    JointConfig("elbow_flex",    3, 0,  1288,  536, 2749),
    JointConfig("wrist_flex",    4, 0, -1112,  313, 2632),
    JointConfig("wrist_roll",    5, 0, -1763,    0, 4095),
    JointConfig("gripper",       6, 0,  1696, 2037, 3296),
]

DEFAULT_LINK_LENGTHS = [1.00, 0.90, 0.65, 0.28]
LINK_NAMES           = ["Base", "Upper arm", "Forearm", "Wrist"]
LINK_MIN, LINK_MAX   = 0.05, 3.0


def degrees_to_ticks(degrees: float, lo: float, hi: float) -> int:
    return int(round(degrees * 10 + lo))


def ticks_to_degrees(ticks: float, lo: float, hi: float) -> float:
    return (ticks - lo) / 10.0


class SO101Simulator:
    """A basic 2D line simulator that uses the actual SO101 joint limits."""

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.link_lengths  = list(DEFAULT_LINK_LENGTHS)
        self.selected_joint = 0
        self.step_counts   = 20
        self.coarse_step   = 80

        self.joint_names = [cfg.name for cfg in JOINT_CONFIGS]
        self.range_min   = np.array([cfg.range_min for cfg in JOINT_CONFIGS], dtype=float)
        self.range_max   = np.array([cfg.range_max for cfg in JOINT_CONFIGS], dtype=float)

        neutral          = (self.range_min + self.range_max) / 2.0
        self.start_raw   = neutral.copy()
        self.start_raw[[0, 1, 2, 3]] = self.range_min[[0, 1, 2, 3]]
        self.raw_positions = self.start_raw.copy()

        self._build_figure()
        self.draw()

    def _build_figure(self) -> None:
        # ── figure with explicit left margin for sidebar ───────────────────────
        # sidebar occupies the leftmost ~28 % of the figure in figure coords.
        # The arm axes occupies the rest.  We use add_axes with figure fractions
        # so slider axes can sit cleanly inside the sidebar without GridSpec fights.

        self.fig = plt.figure(figsize=(19, 12))
        self.fig.canvas.manager.set_window_title("SO101 Matplotlib Simulator")

        SB_LEFT   = 0.01   # sidebar left edge (fig fraction)
        SB_WIDTH  = 0.26   # sidebar width
        SB_RIGHT  = SB_LEFT + SB_WIDTH
        ARM_LEFT  = SB_RIGHT + 0.03
        ARM_RIGHT = 0.985

        # Shaded sidebar background rectangle
        self.fig.add_artist(
            FancyBboxPatch(
                (SB_LEFT, 0.01), SB_WIDTH, 0.97,
                boxstyle="round,pad=0.005",
                facecolor="#f0f2f5", edgecolor="#c0c4cc", linewidth=1.2,
                transform=self.fig.transFigure, zorder=0,
            )
        )

        # ── sidebar text axis (top portion) ───────────────────────────────────
        INFO_BOTTOM = 0.42   # info text occupies [INFO_BOTTOM, 0.98]
        self.info_ax = self.fig.add_axes(
            [SB_LEFT + 0.005, INFO_BOTTOM, SB_WIDTH - 0.01, 0.98 - INFO_BOTTOM]
        )
        self.info_ax.set_axis_off()

        self.info_text = self.info_ax.text(
            0.0, 1.0, "",
            va="top", ha="left",
            family="monospace", fontsize=10,
            transform=self.info_ax.transAxes,
        )

        # ── warning text axis (very bottom of sidebar) ────────────────────────
        self.warn_ax = self.fig.add_axes(
            [SB_LEFT + 0.005, 0.02, SB_WIDTH - 0.01, 0.06]
        )
        self.warn_ax.set_axis_off()
        self.warning_text = self.warn_ax.text(
            0.0, 0.0, "",
            va="bottom", ha="left",
            family="monospace", fontsize=9,
            color="#cc0000", fontweight="bold",
            transform=self.warn_ax.transAxes,
            wrap=True,
        )

        # ── link-length sliders ───────────────────────────────────────────────
        # Section label
        self.fig.text(
            SB_LEFT + 0.012, 0.395,
            "── Link lengths ──────────────",
            fontsize=9, family="monospace", color="#333",
            transform=self.fig.transFigure,
        )

        SL_LEFT    = SB_LEFT + 0.018
        SL_WIDTH   = SB_WIDTH - 0.04
        SL_H       = 0.028
        SL_GAP     = 0.053
        SL_BOTTOM0 = 0.10   # bottom of first slider

        self.link_sliders: list[Slider] = []
        for i, (name, default) in enumerate(zip(LINK_NAMES, DEFAULT_LINK_LENGTHS)):
            bottom = SL_BOTTOM0 + i * SL_GAP
            ax_s = self.fig.add_axes(
                [SL_LEFT, bottom, SL_WIDTH, SL_H],
                facecolor="#dde3ec",
            )
            s = Slider(
                ax_s, name, LINK_MIN, LINK_MAX,
                valinit=default, valstep=0.01, color="#4a7fbf",
            )
            s.label.set_fontsize(9)
            s.label.set_family("monospace")
            s.valtext.set_fontsize(9)
            s.valtext.set_family("monospace")
            s.on_changed(self._on_slider)
            self.link_sliders.append(s)

        # Section label above sliders
        self.fig.text(
            SB_LEFT + 0.012, SL_BOTTOM0 + len(LINK_NAMES) * SL_GAP + 0.005,
            "──────────────────────────────",
            fontsize=9, family="monospace", color="#aaa",
            transform=self.fig.transFigure,
        )

        # ── arm plot ──────────────────────────────────────────────────────────
        self.ax = self.fig.add_axes(
            [ARM_LEFT, 0.06, ARM_RIGHT - ARM_LEFT, 0.90]
        )

        (self.arm_line,)      = self.ax.plot([], [], "-o",  lw=5,  ms=11, color="#1f77b4")
        (self.grip_left,)     = self.ax.plot([], [], "-",   lw=5,  color="#ff7f0e")
        (self.grip_right,)    = self.ax.plot([], [], "-",   lw=5,  color="#ff7f0e")
        (self.pan_indicator,) = self.ax.plot([], [], "--",  lw=2,  color="#2ca02c", alpha=0.8)
        (self.base_ref_line,) = self.ax.plot([], [], ":",   lw=2,  color="#888",    alpha=0.8)
        self.base_marker      = self.ax.scatter([0], [0], s=180, zorder=5, color="black")

        self.included_arcs: list[Arc] = []
        for _ in range(3):
            arc = Arc((0, 0), 0.1, 0.1, theta1=0, theta2=1, lw=2.5, color="#9467bd")
            self.ax.add_patch(arc)
            self.included_arcs.append(arc)

        self.ax.set_title("SO101 2D Simple Simulator  (real joint limits)", fontsize=14)
        self.ax.set_xlabel("X (projected)", fontsize=12)
        self.ax.set_ylabel("Y", fontsize=12)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.3)
        self._update_limits()

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _update_limits(self) -> None:
        reach = sum(self.link_lengths)
        pad = 0.7
        self.ax.set_xlim(-reach - pad, reach + pad)
        self.ax.set_ylim(-reach - pad, reach + pad)

    def _on_slider(self, _val: float) -> None:
        for i, s in enumerate(self.link_sliders):
            self.link_lengths[i] = float(s.val)
        self._update_limits()
        self.draw()

    def _jdeg(self, idx: int) -> float:
        return ticks_to_degrees(self.raw_positions[idx], self.range_min[idx], self.range_max[idx])

    @staticmethod
    def _vang(v: np.ndarray) -> float:
        return math.degrees(math.atan2(float(v[1]), float(v[0])))

    @staticmethod
    def _signed_angle(a: np.ndarray, b: np.ndarray) -> float:
        cross = float(a[0] * b[1] - a[1] * b[0])
        dot   = float(a[0] * b[0] + a[1] * b[1])
        ang   = math.degrees(math.atan2(cross, dot))
        return 180.0 if math.isclose(abs(ang), 180.0, abs_tol=1e-9) else ang

    @staticmethod
    def _fmt(deg: float) -> str:
        if math.isclose(deg, 0.0, abs_tol=1e-9):   return "  0.0°"
        if math.isclose(abs(deg), 180.0, abs_tol=1e-9): return "180.0°"
        return f"{deg:+.1f}°"

    def _set_arc(self, arc: Arc, center: np.ndarray,
                 theta_from: float, delta: float, radius: float) -> np.ndarray:
        arc.center = (float(center[0]), float(center[1]))
        arc.width  = arc.height = float(2.0 * radius)
        arc.theta1 = float(min(theta_from, theta_from + delta))
        arc.theta2 = float(max(theta_from, theta_from + delta))
        mid = math.radians(theta_from + 0.5 * delta)
        return np.array([center[0] + (radius + 0.22) * math.cos(mid),
                         center[1] + (radius + 0.22) * math.sin(mid)])

    def _violations(self) -> list[str]:
        out = []
        for i, name in enumerate(self.joint_names):
            v = self.raw_positions[i]
            if v < self.range_min[i]:
                out.append(f"{name}: {v:.0f} < {self.range_min[i]:.0f}")
            elif v > self.range_max[i]:
                out.append(f"{name}: {v:.0f} > {self.range_max[i]:.0f}")
        return out

    # ── forward kinematics ────────────────────────────────────────────────────

    def fk(self) -> tuple[np.ndarray, np.ndarray, float]:
        angles = [self._jdeg(i) for i in [1, 2, 3]]
        
        # Start with a fixed vertical base
        L0 = self.link_lengths[0]
        xs, ys = [0.0, 0.0], [0.0, L0]
        x, y = 0.0, L0
        
        segs = [angles[0], angles[1], angles[2]]
        cum  = math.radians(180.0)
        for L, a in zip(self.link_lengths[1:], segs):
            cum -= math.radians(a)
            x   += L * math.cos(cum)
            y   += L * math.sin(cum)
            xs.append(x); ys.append(y)
            
        pan = math.radians(-self._jdeg(0))
        xs  = [px * math.cos(pan) for px in xs]
        return np.array(xs), np.array(ys), pan

    def _gripper_w(self) -> float:
        i    = 5
        r    = (self.raw_positions[i] - self.range_min[i]) / (self.range_max[i] - self.range_min[i])
        return 0.06 + float(np.clip(r, 0, 1)) * 0.30

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self) -> None:
        xs, ys, pan = self.fk()
        viols = self._violations()

        self.arm_line.set_color("#d62728" if viols else "#1f77b4")
        self.arm_line.set_data(xs, ys)

        svecs = [np.array([xs[i+1]-xs[i], ys[i+1]-ys[i]]) for i in range(len(xs)-1)]
        wdir  = math.radians(self._vang(svecs[-1]))
        end   = np.array([xs[-1], ys[-1]])
        d     = np.array([math.cos(wdir), math.sin(wdir)])
        n     = np.array([-d[1], d[0]])
        gl    = self.link_lengths[-1] * 0.85
        gw    = self._gripper_w()
        for line, side in ((self.grip_left, 1), (self.grip_right, -1)):
            b = end + n * (side * gw / 2)
            line.set_data([b[0], b[0]+d[0]*gl], [b[1], b[1]+d[1]*gl])

        self.base_ref_line.set_data([0, 0.5], [0, 0])
        self.pan_indicator.set_data([0, 0.45*math.cos(pan)], [0, 0.45*math.sin(pan)])

        # Angle computation
        bref  = np.array([1.0, 0.0])
        specs = [
            (bref,       svecs[1]),
            (-svecs[1],  svecs[2]),
            (-svecs[2],  svecs[3]),
        ]
        jang = [self._signed_angle(a, b) for a, b in specs]
        # clamp 0 / 180
        for k in range(3):
            if math.isclose(jang[k], 0.0, abs_tol=1e-9): jang[k] = 0.0
            elif math.isclose(abs(jang[k]), 180.0, abs_tol=1e-9): jang[k] = 180.0
        jang[2] *= -1.0   # wrist: CW positive

        jdisp = [self._jdeg(i) for i in range(6)]
        jdisp[1], jdisp[2], jdisp[3] = jang[0], jang[1], jang[2]

        arc_r = max(0.18, min(self.link_lengths) * 0.28)
        arc_specs = [
            (1, self._vang(bref),       jang[0],  "shoulder_lift"),
            (2, self._vang(-svecs[1]),  jang[1],  "elbow_flex"),
            (3, self._vang(-svecs[2]), -jang[2],  "wrist_flex"),
        ]
        for ai, (ci, tf, delta, name) in enumerate(arc_specs):
            center = np.array([xs[ci], ys[ci]])
            self._set_arc(self.included_arcs[ai], center, tf, delta, arc_r)

        # ── sidebar info text ──────────────────────────────────────────────────
        sel_name = self.joint_names[self.selected_joint]
        lines = [
            f"Selected: J{self.selected_joint+1}",
            f"  {sel_name}",
            "",
            f"{'Joint':<14} {'ticks':>6}  {'angle':>8}",
            "─" * 35,
        ]
        for i, name in enumerate(self.joint_names):
            pos  = self.raw_positions[i]
            dtix = int(round(float(pos)))
            if i == 3:
                dtix = degrees_to_ticks(jdisp[3], self.range_min[i], self.range_max[i])
            flag   = " !" if (pos < self.range_min[i] or pos > self.range_max[i]) else "  "
            marker = " ◀" if i == self.selected_joint else ""
            lines.append(
                f"J{i+1} {name:<13} {dtix:>6}t  {self._fmt(jdisp[i])}{flag}{marker}"
            )
        lines += [
            "─" * 35,
            f"  range  [{self.range_min[self.selected_joint]:.0f}"
            f" .. {self.range_max[self.selected_joint]:.0f}] ticks",
            "",
            "Keys:",
            "  1-6          select joint",
            "  ←↓ / →↑      adjust (fine)",
            "  shift+arrow  coarse step",
            "  r            startup pose",
            "  h            help in terminal",
        ]
        self.info_text.set_text("\n".join(lines))
        self.warning_text.set_text(
            ("⚠ LIMIT EXCEEDED:\n" + "  ".join(viols)) if viols else ""
        )

        self.fig.canvas.draw_idle()

    # ── input ─────────────────────────────────────────────────────────────────

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in {"1","2","3","4","5","6"}:
            self.selected_joint = int(key) - 1
            self.draw(); return
        if key == "r":
            self.raw_positions = self.start_raw.copy()
            self.draw(); return
        if key == "h":
            print(__doc__); return
        step = self.coarse_step if "shift" in key else self.step_counts
        if "left" in key or "down" in key:
            self.raw_positions[self.selected_joint] -= step; self.draw()
        elif "right" in key or "up" in key:
            self.raw_positions[self.selected_joint] += step; self.draw()

    def run(self) -> None:
        print(__doc__)
        plt.show()


def main() -> None:
    SO101Simulator().run()


if __name__ == "__main__":
    main()