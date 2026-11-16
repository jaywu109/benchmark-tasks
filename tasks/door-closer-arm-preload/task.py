"""On the station bench: a surface door closer of the LCN 4040XP / Norton 1601 kind and hinged upper door leaf tabletop service
rig (`header_frame`, 450 x 145 x 72 mm fixed anodized aluminum transom header and dark-bronze cast-iron closer body at
x = 0.3275 m, y = -0.0250 m). On the right (-y) side of the header frame, a hinged upper door leaf section (`door_leaf`,
260 x 45 x 64 mm blade with 140 mm rear hinge jamb post, 0.380 kg) stands 18.0 deg ajar on a low-friction vertical hinge
at P_D = (0.2900, -0.2400) m with zero spring drive of its own (`stiffness = 0.0`). Mounted on the front face of
`door_leaf` at B(18 deg) = (0.1868, -0.2107) m is the swiveling door-shoe bracket (`door_shoe`) carrying the 215 mm
steel forearm link rod (`forearm_rod`, 215 x 28 x 16 mm, 0.180 kg, 185.0 mm center-to-center link length, 24.0 x 16.0 mm
central grip flat and a distal Ø18.0 mm clevis eye with a 4.0 mm bottom entry chamfer,
spawned swung +25.0 deg forward to world yaw +90.0 deg).

On the left (+y) side of the header frame at pinion shaft P_M = (0.2650, +0.1550) m sits the 228 mm forged steel main
arm (`main_arm`, 228 x 64 x 36 mm overall, 0.240 kg, 198.0 mm pinion-to-stud center distance, 26.0 x 16.0 mm grip bar),
biased back toward its 0.0 deg slack stop (`world yaw = -48.0 deg`) by the closer's physical torsion spring drive
(`MAIN_ARM_STIFFNESS = 0.022 N*m/deg`, 0.93 N*m at the 42.2 deg stop). The main arm's distal knuckle carries an upward-projecting
Ø14.0 mm chamfered steel elbow stud (5.0 mm conical lead-in, 4.0 mm diametral / 2.0 mm radial clearance inside the
Ø18.0 mm forearm clevis bore) and a captive top quarter-turn cam lock lever (`elbow_lock`, 34 x 12 x 14 mm top paddle
above a 52 x 12 x 5 mm cam keeper, 0.040 kg).

When `main_arm` sits unheld at its 0.0 deg slack stop (`stud at (0.3975, +0.0079) m`), the elbow stud is 303.5 mm away
from the door shoe (`118.5 mm` beyond the 185.0 mm forearm link reach), and releasing `main_arm` before coupling snaps
it back to 0.0 deg under spring tension. Left hand must pull and hold `main_arm` rotated +42.0 deg forward against the
closer spring (bringing the elbow stud to S(42 deg) = (0.2650, -0.0430) m, 185.0 mm from the door shoe), while Right hand lifts `forearm_rod`, swings its Ø18.0 mm clevis eye -25.0 deg clockwise
over the held Ø14.0 mm stud (`<= 4.0 mm` radial and vertical seat error), and turns `elbow_lock` >= 75.0 deg clockwise
(to 90.0 deg) over the seated clevis to capture it vertically. When both hands open and retract >= 40.0 mm clear, the
preloaded main arm's spring drive pulls the coupled forearm link and draws `door_leaf` from 18.0 deg ajar to <= 2.5 deg
against the header frame stop. Turning `elbow_lock` past 45.0 deg before the clevis eye is seated (which physically
blocks the clevis from seating onto the stud) or forcing `door_leaf` past its stop with > 25.0 N from any robot link
latches a fault and scores 0.0 for the rest of the trial.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402

import yam_arena.demo as _demo  # noqa: E402
from yam_arena import YAM_URDF, analysis as _an, ik, station, urdf  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, ObjectState, Sphere  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix, rpy_matrix  # noqa: E402

SCENE = "station"
TOP = float(TABLE_TOP[SCENE])
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
CONTROL_HZ = 20.0

INSTRUCTION = "Reconnect this door closer's arm so that it pulls the door shut against its stop."


def f3(v) -> tuple[float, float, float]:
    return tuple(float(x) for x in v)


class Grip(float):
    """A 3x3 TCP rotation matrix carried in Waypoint.yaw (a float field)."""

    def __new__(cls, m):
        obj = super().__new__(cls, 1.0)
        obj.m = np.asarray(m, float)
        return obj


_orig_grip_rotation = getattr(_demo, "_orig_grip_rotation", _demo.grip_rotation)
_demo._orig_grip_rotation = _orig_grip_rotation


def _closer_grip_rotation(y=0.0) -> np.ndarray:
    if hasattr(y, "m"):
        return y.m
    if isinstance(y, np.ndarray):
        return y
    return _orig_grip_rotation(float(y))


_demo.grip_rotation = _closer_grip_rotation
DOWN = _orig_grip_rotation(0.0)


def rx_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(deg))


def ry_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(deg))


def rz_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(deg))


def grip_rot(pitch_deg: float, yaw_deg: float = 0.0) -> Grip:
    """TCP rotation pitched by ``pitch_deg`` about world Y and yawed by ``yaw_deg`` about world Z."""
    return Grip(rz_mat(yaw_deg) @ ry_mat(pitch_deg) @ DOWN)


# -- Fast URDF forward kinematics for the arm chain --------------------------------------------------------------------
_JOINTS_META = urdf.parse_joints(YAM_URDF)
_CHAIN = []
for _jname in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"):
    _j = _JOINTS_META[_jname]
    _t0 = np.eye(4)
    _t0[:3, :3] = rpy_matrix(_j.rpy)
    _t0[:3, 3] = _j.xyz
    _ax = _j.axis / np.linalg.norm(_j.axis)
    _k = np.array([[0.0, -_ax[2], _ax[1]], [_ax[2], 0.0, -_ax[0]], [-_ax[1], _ax[0], 0.0]])
    _CHAIN.append((_j.name, _j.parent, _j.child, _j.type, _t0, _ax, _k, _k @ _k))


def fast_fk(urdf_path, q: dict[str, float], root: str = "base") -> dict[str, np.ndarray]:
    poses = {root: np.eye(4)}
    for name, parent, child, jtype, t0, ax, k, k2 in _CHAIN:
        if name not in q and name in ("joint7", "joint8"):
            continue
        v = float(q.get(name, 0.0))
        tj = np.eye(4)
        if jtype in ("revolute", "continuous"):
            s, c = math.sin(v), math.cos(v)
            tj[:3, :3] = np.eye(3) + s * k + (1.0 - c) * k2
        else:
            tj[:3, 3] = ax * v
        poses[child] = poses[parent] @ (t0 @ tj)
    return poses


urdf.forward_kinematics = fast_fk
ik.forward_kinematics = fast_fk
_an.forward_kinematics = fast_fk


# -- Colours -----------------------------------------------------------------------------------------------------------
FRAME_ALUMINUM = (0.64, 0.67, 0.70)    # anodized aluminum transom header frame & stop rail
CLOSER_BRONZE = (0.26, 0.21, 0.17)     # dark-bronze cast-iron LCN 4040XP / Norton 1601 closer body
DOOR_OAK = (0.72, 0.56, 0.36)          # commercial wood/metal upper door leaf section
SHOE_STEEL = (0.45, 0.48, 0.52)        # steel door-shoe bracket on door face
ARM_STEEL = (0.34, 0.36, 0.40)         # forged steel main arm & forearm link rod
STUD_BRASS = (0.86, 0.72, 0.26)        # chamfered Ø14.0 mm elbow stud & alignment index
LOCK_RED = (0.84, 0.18, 0.16)          # high-contrast captive quarter-turn elbow lock lever


# -- Closer dimensions (arm lengths assumed; LCN sheets give none) and physical constants -----------------------------
MAIN_ARM_LENGTH_M = 0.228              # 228.0 mm forged steel main arm overall length
MAIN_ARM_RADIUS_M = 0.198              # 198.0 mm center-to-center distance from pinion shaft P_M to elbow stud S
MAIN_ARM_GRIP_WIDTH_M = 0.026          # 26.0 mm wide x 16.0 mm thick forged steel grip bar section
MAIN_ARM_GRIP_THICK_M = 0.016          # 16.0 mm vertical thickness
MAIN_ARM_MASS = 0.240                  # 0.240 kg

FOREARM_LENGTH_M = 0.215               # 215.0 mm steel forearm link rod overall length
FOREARM_LINK_M = 0.185                 # 185.0 mm center-to-center distance from door-shoe pin B to clevis eye E
FOREARM_GRIP_WIDTH_M = 0.024           # 24.0 mm wide x 16.0 mm thick central grip flat
FOREARM_GRIP_THICK_M = 0.016           # 16.0 mm vertical thickness
FOREARM_MASS = 0.180                   # 0.180 kg

STUD_DIAMETER_M = 0.014                # Ø14.0 mm upward elbow stud on main_arm distal knuckle
STUD_CHAMFER_M = 0.005                 # 5.0 mm conical lead-in on top of the Ø14.0 mm stud
CLEVIS_BORE_M = 0.018                  # Ø18.0 mm bore in forearm_rod distal clevis eye (+4.0 mm diametral / 2.0 mm radial clearance)
CLEVIS_RADIAL_CLEARANCE_M = 0.5 * (CLEVIS_BORE_M - STUD_DIAMETER_M)  # 0.0020 m (2.0 mm radial / 4.0 mm diametral)
CLEVIS_CHAMFER_M = 0.004               # 4.0 mm bottom entry chamfer on clevis eye (flared to 22.0 x 22.0 mm entry)

ELBOW_LOCK_PADDLE_LEN_M = 0.034        # 34.0 mm long x 12.0 mm wide x 14.0 mm tall top quarter-turn paddle
ELBOW_LOCK_CAM_LEN_M = 0.052           # 52.0 mm symmetric lower cam keeper overhanging the seated clevis
ELBOW_LOCK_WIDTH_M = 0.012             # 12.0 mm width across jaws
ELBOW_LOCK_MASS = 0.040                # 0.040 kg

DOOR_LEAF_SIZE = (0.045, 0.260, 0.064) # 260 x 45 x 64 mm hinged upper door leaf blade
DOOR_LEAF_MASS = 0.380                 # 0.380 kg
DOOR_INITIAL_DEG = 18.0                # 18.0 deg ajar at spawn

# Pivots and 4-bar linkage geometry in station world coordinates:
PINION_XY = (0.2650, +0.1550)          # Closer pinion shaft P_M on left (+y) side of header_frame
DOOR_HINGE_XY = (0.2900, -0.2400)      # Door leaf vertical hinge P_D on right (-y) side of header_frame

MAIN_ARM_SLACK_YAW_DEG = -48.0         # World yaw of main_arm at 0.0 deg slack stop (-90.0 deg at +42.0 deg preload)
MAIN_ARM_PRELOAD_TARGET_DEG = 42.0     # Nominal preload pull angle (+42.0 deg from slack stop -> world yaw -90.0 deg)
MAIN_ARM_MAX_PULL_DEG = 42.2           # Mechanical preload stop on closer pinion housing (-90.2 deg world yaw)
MAIN_ARM_GRIP_R_M = 0.0880             # 88.0 mm radial distance from P_M to main_arm grip center

# At main_arm = +42.0 deg preload (world yaw = -90.0 deg), stud S(42) is at (0.2650, -0.0430) m:
STUD_PRELOAD_XY = (PINION_XY[0], PINION_XY[1] - MAIN_ARM_RADIUS_M)           # (0.2650, -0.0430)
MAIN_GRIP_PRELOAD_XY = (PINION_XY[0], PINION_XY[1] - MAIN_ARM_GRIP_R_M)      # (0.2650, +0.0670)
LOCK_PIVOT_OFFSET_Y_LOCAL = +0.0270    # +27.0 mm in +y_local from stud S -> world (0.2920, -0.0430) at 42 deg preload
LOCK_PIVOT_PRELOAD_XY = (STUD_PRELOAD_XY[0] + LOCK_PIVOT_OFFSET_Y_LOCAL, STUD_PRELOAD_XY[1])  # (0.2920, -0.0430)

# At door_angle = 18.0 deg ajar and main_arm = 42.0 deg preload, forearm_rod couples at world yaw +65.0 deg:
FOREARM_COUPLED_WORLD_YAW_DEG = 65.0
FOREARM_ZERO_YAW_IN_DOOR_DEG = FOREARM_COUPLED_WORLD_YAW_DEG - DOOR_INITIAL_DEG  # +47.0 deg when door_leaf is at 0.0 deg
FOREARM_SPAWN_SWIVEL_DEG = 25.0        # Spawned +25.0 deg CCW from coupled -> world yaw = 65.0 + 25.0 = +90.0 deg!
FOREARM_SPAWN_IN_DOOR_DEG = FOREARM_ZERO_YAW_IN_DOOR_DEG + FOREARM_SPAWN_SWIVEL_DEG  # +72.0 deg in door_leaf frame
FOREARM_SPAWN_WORLD_YAW_DEG = FOREARM_COUPLED_WORLD_YAW_DEG + FOREARM_SPAWN_SWIVEL_DEG  # +90.0 deg

_U_COUPLED = np.array(
    [math.cos(math.radians(FOREARM_COUPLED_WORLD_YAW_DEG)), math.sin(math.radians(FOREARM_COUPLED_WORLD_YAW_DEG))],
    dtype=float,
)
SHOE_AJAR_XY = (
    float(STUD_PRELOAD_XY[0] - FOREARM_LINK_M * _U_COUPLED[0]),
    float(STUD_PRELOAD_XY[1] - FOREARM_LINK_M * _U_COUPLED[1]),
)  # (0.186816, -0.210667)

# Shoe pivot B in door_leaf's 0.0 deg closed frame relative to DOOR_HINGE_XY:
_A18 = math.radians(DOOR_INITIAL_DEG)
_R18_2D = np.array([[math.cos(_A18), -math.sin(_A18)], [math.sin(_A18), math.cos(_A18)]], dtype=float)
SHOE_REL_HINGE_CLOSED_XY = _R18_2D.T @ (np.array(SHOE_AJAR_XY) - np.array(DOOR_HINGE_XY))  # (-0.089056, +0.059819)
SHOE_CLOSED_XY = (
    float(DOOR_HINGE_XY[0] + SHOE_REL_HINGE_CLOSED_XY[0]),
    float(DOOR_HINGE_XY[1] + SHOE_REL_HINGE_CLOSED_XY[1]),
)  # (0.200944, -0.180181)

# Vertical stack heights above table TOP (m):
Z_SHOULDER_REL = 0.0960                # 96.0 mm above table: main_arm knuckle shoulder & forearm_rod underside when seated
Z_MAIN_BAR_LO_REL = 0.0800             # 80.0 mm..96.0 mm: 16.0 mm thick main_arm grip bar (center at 88.0 mm)
Z_STUD_TOP_REL = 0.1140                # 96.0 mm..114.0 mm: 18.0 mm tall Ø14.0 mm elbow stud (13 mm shaft + 5 mm chamfer)
Z_FOREARM_SEATED_CTR_REL = Z_SHOULDER_REL + 0.5 * FOREARM_GRIP_THICK_M  # 0.1040 m (spans 96.0..112.0 mm when seated)
Z_FOREARM_TOP_SEATED_REL = Z_SHOULDER_REL + FOREARM_GRIP_THICK_M        # 0.1120 m
Z_LOCK_CAM_LO_REL = 0.1145             # 114.5 mm (2.5 mm above seated clevis top at 112.0 mm)
Z_LOCK_TOP_REL = 0.1335                # 114.5..133.5 mm: 19.0 mm tall elbow_lock (5 mm cam keeper + 14 mm top paddle)
Z_LOCK_CTR_REL = 0.5 * (Z_LOCK_CAM_LO_REL + Z_LOCK_TOP_REL)             # 0.1240 m

# Physical joint drive stiffnesses and dampings:
# Spring: the left hand gave way at ~1.40 N*m (39.0 deg with 0.036 N*m/deg); 0.022 N*m/deg (~0.93 N*m at 42.2 deg)
# is ~65% of that. A real closer is far stronger (EN 1154 grade 3: >= 18 N*m closing moment at 0-4 deg).
MAIN_ARM_STIFFNESS = 0.022             # N*m/deg (0.93 N*m at the 42.2 deg stop, 10.5 N at the grip bar)
MAIN_ARM_DAMPING = 0.0024              # N*m*s/deg hydraulic sweep damping, scaled with the spring (same closing speed)
DOOR_LEAF_DAMPING = 0.0003             # N*m*s/deg low-friction ball-bearing door hinge damping (stiffness = 0.0!)
DOOR_SHOE_DAMPING = 0.00005            # N*m*s/deg shoe swivel pin damping
FOREARM_SLIDE_DAMPING = 0.35           # N*s/m vertical clevis lift damping
ELBOW_LOCK_DAMPING = 0.0004            # N*m*s/deg quarter-turn cam bearing damping

# Captive spring-ball detent: the ball rises into an underside groove at 90 deg CW.
# The symmetric second groove preserves the cam's original bounds and body origin.
DETENT_RADIUS_M = 0.012
DETENT_BALL_RADIUS_M = 0.003
DETENT_NOTCH_HALF_WIDTH_M = 0.0025
DETENT_NOTCH_DEPTH_M = 0.002
DETENT_PROTRUSION_M = 0.0018
DETENT_SPRING_N_M = 1500.0
DETENT_SPRING_PRELOAD_M = 0.002

YAM_MAX_OPENING_M = 2.0 * abs(float(FINGER_OPEN))  # 0.0939 m (93.9 mm)


# -- 1. Fixed Transom Header Frame & LCN 4040XP Closer Body (`header_frame`, 450 x 145 x 72 mm) -----------------------
# Placed strictly at x >= 0.3125 m (behind the door stop plane x = 0.3125 m) and z_rel <= 0.0580 m across the main
# sweep zone, with only the pinion shaft pedestal at P_M = (0.2650, +0.1550) rising to z_rel = 0.0720 m (6.0 mm below
# main_arm's hub at z_rel = 0.0780 m) so neither `door_leaf` nor the robot finger tips ever collide with `header_frame`.
HEADER_POS = f3((0.3275, -0.0250, TOP + 0.0360))
HEADER_SIZE = f3((0.1450, 0.4500, 0.0720))


def _header_piece(lo_w_rel, hi_w_rel, color) -> rx.Piece:
    lo_w = (lo_w_rel[0], lo_w_rel[1], TOP + lo_w_rel[2])
    hi_w = (hi_w_rel[0], hi_w_rel[1], TOP + hi_w_rel[2])
    return rx.Piece(
        size=f3(h - l for l, h in zip(lo_w, hi_w)),
        offset=f3(0.5 * (l + h) - c for l, h, c in zip(lo_w, hi_w, HEADER_POS)),
        color=color,
    )


HEADER_PIECES = (
    # Rear anodized aluminum transom header beam across full width (x = 0.3220..0.4000 m, y = -0.2500..+0.2000 m, z_rel = 0..56 mm):
    _header_piece((0.3220, -0.2500, 0.0000), (0.4000, +0.2000, 0.0560), FRAME_ALUMINUM),
    # Front door stop lip at x = 0.3125..0.3220 m across the latching span y = -0.1800..+0.2000 m (z_rel = 0..56 mm):
    _header_piece((0.3125, -0.1800, 0.0000), (0.3220, +0.2000, 0.0560), FRAME_ALUMINUM),
    # LCN 4040XP cast-iron closer body behind stop plane (310 mm long across y = -0.1100..+0.2000 m, z_rel = 0..58 mm):
    _header_piece((0.3125, -0.1100, 0.0000), (0.4000, +0.2000, 0.0580), CLOSER_BRONZE),
    # Closer pinion shaft mounting arm & boss at P_M = (0.2650, +0.1550), y = +0.1400..+0.1900 m, z_rel = 0..72 mm:
    _header_piece((0.2550, +0.1400, 0.0000), (0.3125, +0.1900, 0.0720), CLOSER_BRONZE),
)
HEADER_COMPOUND = rx.Compound(pieces=HEADER_PIECES)
HEADER_FRAME = Box("header_frame", pos=HEADER_POS, size=HEADER_SIZE, color=FRAME_ALUMINUM)


# -- 2. Hinged Upper Door Leaf Section (`door_leaf`, 260 mm wide x 45 mm thick x 64 mm blade, 0.380 kg) ---------------
# Zero pose (0.0 deg closed against header_frame stop at x = 0.3125 m, spec.yaw = 0.0):
# Hinged at P_D = (0.2900, -0.2400) about vertical Z; spans y in [-0.2400, +0.0200] m (260.0 mm), x in [0.2675, 0.3125] m
# (45.0 mm), z_rel in [0.0040, 0.0680] m (64.0 mm, leaving >= 28.0 mm vertical clearance below forearm_rod at 0.0960 m!).
DOOR_LEAF_ZERO_POS = f3((0.2900, -0.1100, TOP + 0.0360))
DOOR_LEAF_BOX_SIZE = f3((0.0450, 0.2600, 0.0640))
DOOR_HINGE_ANCHOR = f3((DOOR_HINGE_XY[0] - DOOR_LEAF_ZERO_POS[0], DOOR_HINGE_XY[1] - DOOR_LEAF_ZERO_POS[1], 0.0))  # (0.0, -0.1300, 0.0)


def _door_piece(lo_w_rel, hi_w_rel, color) -> rx.Piece:
    lo_w = (lo_w_rel[0], lo_w_rel[1], TOP + lo_w_rel[2])
    hi_w = (hi_w_rel[0], hi_w_rel[1], TOP + hi_w_rel[2])
    return rx.Piece(
        size=f3(h - l for l, h in zip(lo_w, hi_w)),
        offset=f3(0.5 * (l + h) - c for l, h, c in zip(lo_w, hi_w, DOOR_LEAF_ZERO_POS)),
        color=color,
    )


DOOR_LEAF_PIECES = (
    _door_piece((0.2675, -0.2400, 0.0040), (0.3125, +0.0200, 0.0680), DOOR_OAK),
)
DOOR_LEAF_COMPOUND = rx.Compound(pieces=DOOR_LEAF_PIECES)
DOOR_LEAF = Box("door_leaf", pos=DOOR_LEAF_ZERO_POS, size=DOOR_LEAF_BOX_SIZE, mass=DOOR_LEAF_MASS, color=DOOR_OAK)
DOOR_LEAF_JOINT = rx.Revolute(
    anchor=DOOR_HINGE_ANCHOR,
    axis="Z",
    lower=0.0,
    upper=18.2,
    target=0.0,
    stiffness=0.0,
    damping=DOOR_LEAF_DAMPING,
    initial=DOOR_INITIAL_DEG,
)


# -- 3. Door-Shoe Swivel Clevis Bracket (`door_shoe`, 28 x 28 x 16 mm, 0.100 kg) --------------------------------------
# Zero pose has spec.yaw = 0.0 (matching door_leaf so LocalRot0 is identity!).
# Pivots about vertical Z on `door_leaf` at SHOE_CLOSED_XY = (0.200944, -0.180181) at z_rel = 0.0730..0.0890 m
# (5.0 mm above door_leaf and 7.0 mm below forearm_rod).
# In door_leaf's frame:
#   - Coupled angle at door = 18.0 deg ajar is FOREARM_ZERO_YAW_IN_DOOR_DEG = +47.0 deg (world yaw 47 + 18 = +65.0 deg).
#   - Spawn angle is FOREARM_SPAWN_IN_DOOR_DEG = +72.0 deg (world yaw 72 + 18 = +90.0 deg, pointing straight along +Y!).
DOOR_SHOE_ZERO_POS = f3((SHOE_CLOSED_XY[0], SHOE_CLOSED_XY[1], TOP + 0.0810))
DOOR_SHOE_SIZE = f3((0.0280, 0.0280, 0.0160))
DOOR_SHOE_MASS = 0.100
DOOR_SHOE = Box(
    "door_shoe",
    pos=DOOR_SHOE_ZERO_POS,
    size=DOOR_SHOE_SIZE,
    mass=DOOR_SHOE_MASS,
    yaw=0.0,
    color=SHOE_STEEL,
)
DOOR_SHOE_JOINT = rx.Revolute(
    anchor=(0.0, 0.0, 0.0),
    axis="Z",
    lower=FOREARM_ZERO_YAW_IN_DOOR_DEG - 0.3,
    upper=FOREARM_SPAWN_IN_DOOR_DEG + 1.0,
    target=FOREARM_ZERO_YAW_IN_DOOR_DEG,
    stiffness=0.0,
    damping=DOOR_SHOE_DAMPING,
    parent="door_leaf",
    initial=FOREARM_SPAWN_IN_DOOR_DEG,
)


# -- 4. Door-Shoe Forearm Link Rod (`forearm_rod`, 215 x 28 x 16 mm, 0.180 kg) ----------------------------------
# Zero pose has spec.yaw = 0.0 (pointing along +X from SHOE_CLOSED_XY, matching door_shoe so LocalRot0 is identity!).
# Slides vertically along Z (`rx.Prismatic` axis="Z", lower=0.0, upper=0.028 m, parent="door_shoe") so the right hand
# can lift the forearm +20 mm over the Ø14.0 mm stud top, swing `door_shoe` + `forearm_rod` -25.0 deg clockwise from
# +90.0 deg to +65.0 deg, and lower the Ø18.0 mm clevis eye straight down onto the Ø14.0 mm stud shoulder (z_rel = 0.0960 m).
FOREARM_ZERO_POS = f3((SHOE_CLOSED_XY[0] + 0.5 * FOREARM_LINK_M, SHOE_CLOSED_XY[1], TOP + Z_FOREARM_SEATED_CTR_REL))
FOREARM_SIZE = f3((FOREARM_LENGTH_M, 0.0280, FOREARM_GRIP_THICK_M))
FOREARM_ANCHOR = f3((-0.5 * FOREARM_LINK_M, 0.0, 0.0))  # (-0.0925, 0.0, 0.0) at shoe pin B

FOREARM_PIECES = (
    # Proximal shoe end lug (x_local in [-0.1075, -0.0775], y_local in [-0.0140, +0.0140], z_local in [-0.0080, +0.0080]):
    rx.Piece(size=(0.0300, 0.0280, 0.0160), offset=(-0.0925, 0.0, 0.0), color=ARM_STEEL),
    # Central 24.0 mm wide x 16.0 mm thick grip flat (x_local in [-0.0775, +0.0775], y_local in [-0.0120, +0.0120]):
    rx.Piece(size=(0.1550, FOREARM_GRIP_WIDTH_M, FOREARM_GRIP_THICK_M), offset=(0.0, 0.0, 0.0), color=ARM_STEEL),
    # Distal clevis eye upper bore walls (z_local in [-0.0040, +0.0080], 18.0 x 18.0 mm bore centered at x_local = +0.0925,
    # outer clevis width 28.0 mm in y_local):
    rx.Piece(size=(0.0060, 0.0280, 0.0120), offset=(+0.0805, 0.0, +0.0020), color=ARM_STEEL),
    rx.Piece(size=(0.0060, 0.0280, 0.0120), offset=(+0.1045, 0.0, +0.0020), color=ARM_STEEL),
    rx.Piece(size=(0.0180, 0.0050, 0.0120), offset=(+0.0925, +0.0115, +0.0020), color=ARM_STEEL),
    rx.Piece(size=(0.0180, 0.0050, 0.0120), offset=(+0.0925, -0.0115, +0.0020), color=ARM_STEEL),
    # Distal clevis eye lower 4.0 mm entry chamfer walls (z_local in [-0.0080, -0.0040], 22.0 x 22.0 mm flared mouth):
    rx.Piece(size=(0.0040, 0.0280, 0.0040), offset=(+0.0795, 0.0, -0.0060), color=ARM_STEEL),
    rx.Piece(size=(0.0040, 0.0280, 0.0040), offset=(+0.1055, 0.0, -0.0060), color=ARM_STEEL),
    rx.Piece(size=(0.0220, 0.0030, 0.0040), offset=(+0.0925, +0.0125, -0.0060), color=ARM_STEEL),
    rx.Piece(size=(0.0220, 0.0030, 0.0040), offset=(+0.0925, -0.0125, -0.0060), color=ARM_STEEL),
)
FOREARM_COMPOUND = rx.Compound(pieces=FOREARM_PIECES)
FOREARM_ROD = Box(
    "forearm_rod",
    pos=FOREARM_ZERO_POS,
    size=FOREARM_SIZE,
    mass=FOREARM_MASS,
    yaw=0.0,
    color=ARM_STEEL,
)
FOREARM_JOINT = rx.Prismatic(
    anchor=FOREARM_ANCHOR,
    axis="Z",
    lower=0.0,
    upper=0.028,
    target=0.0,
    stiffness=0.0,
    damping=FOREARM_SLIDE_DAMPING,
    parent="door_shoe",
    initial=0.0,
)


# -- 5. Spring-Loaded Main Arm (`main_arm`, 228 x 64 x 36 mm overall, 0.240 kg) ---------------------------------------
# Zero pose has spec.yaw = 0.0 (pointing along +X from PINION_XY = (0.2650, +0.1550)).
# Spawns at initial = MAIN_ARM_SLACK_YAW_DEG (-48.0 deg, the 0.0 deg slack stop), with joint limits
# lower = -90.2 deg (+42.2 deg preload stop) and upper = -48.0 deg (0.0 deg slack stop), and a physical torsion spring
# drive (`stiffness = MAIN_ARM_STIFFNESS`, `target = -48.0 deg`) pulling `main_arm` back toward -48.0 deg.
MAIN_ARM_ZERO_POS = f3((PINION_XY[0] + 0.5 * MAIN_ARM_RADIUS_M, PINION_XY[1], TOP + Z_SHOULDER_REL))
MAIN_ARM_SIZE = f3((MAIN_ARM_LENGTH_M, 0.0640, 0.0360))
MAIN_ARM_ANCHOR = f3((-0.5 * MAIN_ARM_RADIUS_M, 0.0, 0.0))  # (-0.0990, 0.0, 0.0) at pinion shaft P_M

MAIN_ARM_PIECES = (
    # Proximal pinion hub (reaches x_local = -0.1140, y_local = +-0.0320, z_local = +-0.0180):
    rx.Piece(size=(0.0300, 0.0640, 0.0360), offset=(-0.0990, 0.0000, 0.0000), color=ARM_STEEL),
    # Central 26.0 mm wide x 16.0 mm thick forged steel grip bar (x_local in [-0.0840, +0.0680], z_local in [-0.0160, 0.0000]):
    rx.Piece(size=(0.1520, MAIN_ARM_GRIP_WIDTH_M, MAIN_ARM_GRIP_THICK_M), offset=(-0.0080, 0.0000, -0.0080), color=ARM_STEEL),
    # Distal knuckle shoulder under stud S (x_local in [+0.0850, +0.1140], y_local in [-0.0140, +0.0140], z_local in [-0.0180, 0.0000]):
    rx.Piece(size=(0.0290, 0.0280, 0.0180), offset=(+0.0995, 0.0000, -0.0090), color=ARM_STEEL),
    # Lower knuckle base to lock pivot post L_piv (x_local in [+0.0680, +0.1140], y_local in [-0.0320, +0.0320], z_local in [-0.0180, -0.0060],
    # 6.0 mm below the seated forearm_rod underside at z_local = 0.0000!):
    rx.Piece(size=(0.0460, 0.0640, 0.0120), offset=(+0.0910, 0.0000, -0.0120), color=ARM_STEEL),
    # Lock-pivot riser post at L_piv (x_local = +0.0990, y_local = +0.0270, z_local in [-0.0060, +0.0180]):
    rx.Piece(size=(0.0100, 0.0100, 0.0240), offset=(+0.0990, LOCK_PIVOT_OFFSET_Y_LOCAL, +0.0060), color=ARM_STEEL),
    # Upward Ø14.0 mm octagonal elbow stud shaft at S (x_local = +0.0990, y_local = 0.0, z_local in [0.0000, +0.0130]):
    rx.Piece(size=(0.0140, 0.0076, 0.0130), offset=(+0.0990, 0.0000, +0.0065), color=STUD_BRASS),
    rx.Piece(size=(0.0076, 0.0140, 0.0130), offset=(+0.0990, 0.0000, +0.0065), color=STUD_BRASS),
    rx.Piece(size=(0.0104, 0.0104, 0.0130), offset=(+0.0990, 0.0000, +0.0065), color=STUD_BRASS),
    # 5.0 mm conical lead-in chamfer on top of the Ø14.0 mm stud (z_local in [+0.0130, +0.0180]):
    rx.Piece(size=(0.0100, 0.0100, 0.0030), offset=(+0.0990, 0.0000, +0.0145), color=STUD_BRASS),
    rx.Piece(size=(0.0060, 0.0060, 0.0020), offset=(+0.0990, 0.0000, +0.0170), color=STUD_BRASS),
)
MAIN_ARM_COMPOUND = rx.Compound(pieces=MAIN_ARM_PIECES)
MAIN_ARM = Box(
    "main_arm",
    pos=MAIN_ARM_ZERO_POS,
    size=MAIN_ARM_SIZE,
    mass=MAIN_ARM_MASS,
    yaw=0.0,
    color=ARM_STEEL,
)
MAIN_ARM_JOINT = rx.Revolute(
    anchor=MAIN_ARM_ANCHOR,
    axis="Z",
    lower=MAIN_ARM_SLACK_YAW_DEG - MAIN_ARM_MAX_PULL_DEG,
    upper=MAIN_ARM_SLACK_YAW_DEG,
    target=MAIN_ARM_SLACK_YAW_DEG,
    stiffness=MAIN_ARM_STIFFNESS,
    damping=MAIN_ARM_DAMPING,
    initial=MAIN_ARM_SLACK_YAW_DEG,
)


# -- 6. Captive Quarter-Turn Elbow Lock Lever (`elbow_lock`, 52 x 12 x 19 mm overall, 0.040 kg) -----------------------
# Zero pose has spec.yaw = 0.0 (matching main_arm so LocalRot0 is identity!).
# Pivots about vertical Z on `main_arm` at L_piv (x_local = +0.0990, y_local = +0.0270 in main_arm).
# At lock_angle = 0.0 deg (unlocked at spawn):
#   Its 52.0 mm / 34.0 mm long axis is aligned with main_arm's axis (-Y_world at 42 deg preload), while its narrow
#   12.0 mm width spans +-6.0 mm in X_world around x = 0.2920 m (`x = 0.2860..0.2980 m`), sitting completely outside
#   the rotated 28.0 x 28.0 mm clevis eye (`x <= 0.2848 m`) so the clevis eye has 100% open vertical access to the stud.
# When rotated clockwise (`-45.0..-95.0 deg` about +Z):
#   Its 52.0 mm lower cam keeper (z_rel = 0.1145..0.1195 m) and 34.0 mm top paddle (z_rel = 0.1195..0.1335 m) swing
#   across `x = 0.2660..0.3180 m` directly over the clevis eye wall and stud (`x = 0.2650 m`), blocking the clevis from
#   seating if turned > 45.0 deg early and locking the clevis down on the stud when turned >= 75.0 deg!
ELBOW_LOCK_ZERO_POS = f3((PINION_XY[0] + MAIN_ARM_RADIUS_M, PINION_XY[1] + LOCK_PIVOT_OFFSET_Y_LOCAL, TOP + Z_LOCK_CTR_REL))
ELBOW_LOCK_SIZE = f3((ELBOW_LOCK_CAM_LEN_M, ELBOW_LOCK_WIDTH_M, Z_LOCK_TOP_REL - Z_LOCK_CAM_LO_REL))  # (0.052, 0.012, 0.019)

ELBOW_LOCK_PIECES = (
    # Continuous upper web and three lower lands leave two 5 x 12 x 2 mm
    # underside grooves at x = +/-12 mm. The outward groove meets the ball
    # at 90 deg CW; rounded ball/land-edge contact supplies the release ramp.
    rx.Piece(
        size=(ELBOW_LOCK_CAM_LEN_M, ELBOW_LOCK_WIDTH_M, 0.0050 - DETENT_NOTCH_DEPTH_M),
        offset=(0.0, 0.0, -0.0070 + 0.5 * DETENT_NOTCH_DEPTH_M), color=LOCK_RED,
    ),
    *tuple(
        rx.Piece(
            size=f3((hi - lo, ELBOW_LOCK_WIDTH_M, DETENT_NOTCH_DEPTH_M)),
            offset=f3((0.5 * (lo + hi), 0.0, -0.0095 + 0.5 * DETENT_NOTCH_DEPTH_M)),
            color=LOCK_RED,
        )
        for lo, hi in (
            (-0.5 * ELBOW_LOCK_CAM_LEN_M, -DETENT_RADIUS_M - DETENT_NOTCH_HALF_WIDTH_M),
            (-DETENT_RADIUS_M + DETENT_NOTCH_HALF_WIDTH_M, DETENT_RADIUS_M - DETENT_NOTCH_HALF_WIDTH_M),
            (DETENT_RADIUS_M + DETENT_NOTCH_HALF_WIDTH_M, 0.5 * ELBOW_LOCK_CAM_LEN_M),
        )
    ),
    # Upper 34.0 x 12.0 x 14.0 mm quarter-turn grip paddle (z_local in [-0.0045, +0.0095] -> z_rel = 0.1195..0.1335 m):
    rx.Piece(size=(ELBOW_LOCK_PADDLE_LEN_M, ELBOW_LOCK_WIDTH_M, 0.0140), offset=(0.0, 0.0, +0.0025), color=LOCK_RED),
)
ELBOW_LOCK_COMPOUND = rx.Compound(pieces=ELBOW_LOCK_PIECES)
ELBOW_LOCK = Box(
    "elbow_lock",
    pos=ELBOW_LOCK_ZERO_POS,
    size=ELBOW_LOCK_SIZE,
    mass=ELBOW_LOCK_MASS,
    yaw=0.0,
    color=LOCK_RED,
)
ELBOW_LOCK_JOINT = rx.Revolute(
    anchor=(0.0, 0.0, 0.0),
    axis="Z",
    lower=-100.0,
    upper=5.0,
    target=0.0,
    stiffness=0.0,
    damping=ELBOW_LOCK_DAMPING,
    parent="main_arm",
    initial=0.0,
)

# The spring and guide are represented by a constant-drive prismatic joint
# anchored to main_arm. Its ball is outside the clevis insertion path, on the
# outward side of the lock pivot. Only contact with elbow_lock makes a detent;
# the lock's zero-stiffness revolute and all of its settings stay unchanged.
DETENT_ZERO_POS = f3((
    ELBOW_LOCK_ZERO_POS[0],
    ELBOW_LOCK_ZERO_POS[1] + DETENT_RADIUS_M,
    TOP + Z_LOCK_CAM_LO_REL - DETENT_BALL_RADIUS_M + DETENT_PROTRUSION_M,
))
ELBOW_DETENT = Sphere(
    "elbow_detent", pos=DETENT_ZERO_POS, radius=DETENT_BALL_RADIUS_M,
    mass=0.001, color=STUD_BRASS,
)
ELBOW_DETENT_JOINT = rx.Prismatic(
    anchor=(0.0, 0.0, 0.0), axis="Z", lower=-0.0025, upper=0.0,
    target=DETENT_SPRING_PRELOAD_M, stiffness=DETENT_SPRING_N_M,
    damping=1.0, parent="main_arm", initial=0.0,
)

OBJECTS = [HEADER_FRAME, DOOR_LEAF, DOOR_SHOE, FOREARM_ROD, MAIN_ARM, ELBOW_LOCK, ELBOW_DETENT]

rx.install(
    {
        "header_frame": HEADER_COMPOUND,
        "door_leaf": (DOOR_LEAF_COMPOUND, DOOR_LEAF_JOINT),
        "door_shoe": DOOR_SHOE_JOINT,
        "forearm_rod": (FOREARM_COMPOUND, FOREARM_JOINT),
        "main_arm": (MAIN_ARM_COMPOUND, MAIN_ARM_JOINT),
        "elbow_lock": (ELBOW_LOCK_COMPOUND, ELBOW_LOCK_JOINT),
        "elbow_detent": ELBOW_DETENT_JOINT,
    }
)


# -- Scene physics patch: 20 Hz control, high grip friction, solver iterations, and contact sensor on door_leaf --------
_SCENE_MOD = "yam_arena.scene"
_RUNTIME_HANDLES: dict[str, Any] = {}
_CONTACT_VIEWS: dict[str, Any] = {}


def _patch_scene_physics(scene_mod) -> None:
    if getattr(scene_mod, "_door_closer_patched", False):
        return
    import isaaclab.sim as sim_utils

    orig_task_object = scene_mod.task_object
    orig_build_task_env = scene_mod.build_task_environment

    grip_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=2.0,
        dynamic_friction=1.8,
        friction_combine_mode="max",
        restitution=0.0,
    )
    linkage_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.10,
        dynamic_friction=0.08,
        friction_combine_mode="min",
        restitution=0.0,
    )

    def task_object(spec):
        obj = orig_task_object(spec)
        if spec.name in ("main_arm", "forearm_rod", "elbow_lock"):
            obj.object_cfg.spawn.physics_material = grip_mat
            obj.spawner_cfg.physics_material = grip_mat
        elif spec.name in ("door_leaf", "door_shoe", "header_frame", "elbow_detent"):
            obj.object_cfg.spawn.physics_material = linkage_mat
            obj.spawner_cfg.physics_material = linkage_mat

        if getattr(obj.object_cfg.spawn, "collision_props", None) is not None:
            cp = obj.object_cfg.spawn.collision_props
            cp.contact_offset = 0.001
            cp.rest_offset = 0.0

        if getattr(obj.object_cfg.spawn, "rigid_props", None) is not None:
            rp = obj.object_cfg.spawn.rigid_props
            rp.solver_position_iteration_count = 24
            rp.solver_velocity_iteration_count = 8
            rp.max_depenetration_velocity = 1.0

        if spec.name == "door_leaf":
            obj.object_cfg.spawn.activate_contact_sensors = True
            obj.spawner_cfg.activate_contact_sensors = True
            prev_door_spawn = obj.object_cfg.spawn.func

            def _spawn_door(prim_path, cfg, *args, _prev=prev_door_spawn, **kwargs):
                prim = _prev(prim_path, cfg, *args, **kwargs)
                from pxr import PhysxSchema

                PhysxSchema.PhysxContactReportAPI.Apply(prim)
                _CONTACT_VIEWS.clear()
                return prim

            obj.object_cfg.spawn.func = _spawn_door
            obj.spawner_cfg.func = _spawn_door

        if spec.name == "forearm_rod":
            prev_fore_spawn = obj.object_cfg.spawn.func

            def _spawn_forearm(prim_path, cfg, *args, _prev=prev_fore_spawn, **kwargs):
                prim = _prev(prim_path, cfg, *args, **kwargs)
                from pxr import UsdPhysics

                stage = prim.GetStage()
                env_dir = prim_path.rsplit("/", 1)[0]
                j_prim = stage.GetPrimAtPath(f"{env_dir}/forearm_rod_joint")
                _RUNTIME_HANDLES.clear()
                _RUNTIME_HANDLES["stage"] = stage
                _RUNTIME_HANDLES["env_dir"] = env_dir
                if j_prim.IsValid():
                    _RUNTIME_HANDLES["forearm_joint"] = UsdPhysics.PrismaticJoint(j_prim)
                return prim

            obj.object_cfg.spawn.func = _spawn_forearm
            obj.spawner_cfg.func = _spawn_forearm
        return obj

    def build_task_environment(embodiment, task, control_hz=None):
        return orig_build_task_env(embodiment, task, control_hz=control_hz or CONTROL_HZ)

    scene_mod.task_object = task_object
    scene_mod.build_task_environment = build_task_environment
    scene_mod._door_closer_patched = True


_orig_rx_patch = getattr(rx, "_orig_rx_patch", rx._patch)
rx._orig_rx_patch = _orig_rx_patch


def _door_closer_rx_patch(scene_mod) -> None:
    _orig_rx_patch(scene_mod)
    _patch_scene_physics(scene_mod)


rx._patch = _door_closer_rx_patch
if _SCENE_MOD in sys.modules:
    _door_closer_rx_patch(sys.modules[_SCENE_MOD])


# -- Scoring and fault thresholds --------------------------------------------------------------------------------------
MAIN_ARM_COUPLED_MIN_DEG = 20.0        # >= 20.0 deg main_arm tension for every scored tier (26.4 deg once the door is shut)
CLEVIS_MAX_XY_ERR_M = 0.004            # <= 4.0 mm radial offset between clevis eye center E and elbow stud center S
CLEVIS_MAX_Z_ERR_M = 0.004             # <= 4.0 mm vertical gap between forearm_rod underside and knuckle shoulder
LOCK_EARLY_FAULT_DEG = 45.0            # > 45.0 deg clockwise lock rotation before clevis is seated latches fault
LOCK_MIN_DEG = 75.0                    # >= 75.0 deg clockwise lock rotation over seated clevis (demo turns 95.0 deg)
DOOR_CLOSED_MAX_DEG = 2.5              # <= 2.5 deg door leaf angle from closed frame stop for 1.0 success
MAX_DOOR_STOP_FORCE_N = 25.0           # > 25.0 N robot contact force forcing door_leaf past its stop latches fault
GRIPPER_CLEAR = 0.040                  # >= 40.0 mm TCP clearance of both grippers from all task objects
STILL_MAX_SPEED = 0.05                 # < 0.05 m/s linear speed for 1.0 s rest hold


# -- State readout helpers ---------------------------------------------------------------------------------------------
def _yaw_deg_from_rot(rot: np.ndarray) -> float:
    return math.degrees(math.atan2(float(rot[1, 0]), float(rot[0, 0])))


def _wrap_deg(angle_deg: float) -> float:
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def door_angle_deg(objects: dict[str, ObjectState]) -> float:
    """Angle (deg) of ``door_leaf`` from its 0.0 deg closed stop (18.0 deg ajar at spawn)."""
    r_door = quat_xyzw_to_matrix(np.asarray(objects["door_leaf"].quat_xyzw, dtype=float))
    return float(max(0.0, _wrap_deg(_yaw_deg_from_rot(r_door))))


def main_arm_deg(objects: dict[str, ObjectState]) -> float:
    """Preload pull angle (deg) of ``main_arm`` away from its 0.0 deg slack stop (+42.0 deg at preload)."""
    r_main = quat_xyzw_to_matrix(np.asarray(objects["main_arm"].quat_xyzw, dtype=float))
    yaw = _yaw_deg_from_rot(r_main)
    return float(max(0.0, _wrap_deg(MAIN_ARM_SLACK_YAW_DEG - yaw)))


def elbow_lock_deg(objects: dict[str, ObjectState]) -> float:
    """Clockwise rotation (deg) of ``elbow_lock`` relative to ``main_arm`` (0.0 deg unlocked, ~90.0 deg locked)."""
    r_main = quat_xyzw_to_matrix(np.asarray(objects["main_arm"].quat_xyzw, dtype=float))
    r_lock = quat_xyzw_to_matrix(np.asarray(objects["elbow_lock"].quat_xyzw, dtype=float))
    yaw_main = _yaw_deg_from_rot(r_main)
    yaw_lock = _yaw_deg_from_rot(r_lock)
    return float(max(0.0, _wrap_deg(yaw_main - yaw_lock)))


def stud_world_pos(objects: dict[str, ObjectState]) -> np.ndarray:
    """World 3D position `(x_S, y_S, z_shoulder)` of ``main_arm``'s Ø14.0 mm elbow stud shoulder."""
    p_main = np.asarray(objects["main_arm"].pos, dtype=float)
    r_main = quat_xyzw_to_matrix(np.asarray(objects["main_arm"].quat_xyzw, dtype=float))
    return p_main + r_main @ np.array([+0.5 * MAIN_ARM_RADIUS_M, 0.0, 0.0], dtype=float)


def clevis_world_pos(objects: dict[str, ObjectState]) -> np.ndarray:
    """World 3D position `(x_E, y_E, z_underside)` of ``forearm_rod``'s Ø18.0 mm clevis eye bottom face."""
    p_fore = np.asarray(objects["forearm_rod"].pos, dtype=float)
    r_fore = quat_xyzw_to_matrix(np.asarray(objects["forearm_rod"].quat_xyzw, dtype=float))
    return p_fore + r_fore @ np.array([+0.5 * FOREARM_LINK_M, 0.0, -0.5 * FOREARM_GRIP_THICK_M], dtype=float)


def clevis_seat_errors(objects: dict[str, ObjectState]) -> tuple[float, float]:
    """Return ``(xy_err_m, z_seat_err_m)`` between ``forearm_rod``'s Ø18.0 mm clevis eye and ``main_arm``'s Ø14.0 mm stud."""
    s_pos = stud_world_pos(objects)
    e_pos = clevis_world_pos(objects)
    xy_err_m = float(math.hypot(float(e_pos[0] - s_pos[0]), float(e_pos[1] - s_pos[1])))
    z_seat_err_m = float(abs(float(e_pos[2] - s_pos[2])))
    return xy_err_m, z_seat_err_m


def clevis_seated(objects: dict[str, ObjectState]) -> bool:
    """True when ``forearm_rod``'s Ø18.0 mm clevis eye is seated on ``main_arm``'s Ø14.0 mm elbow stud
    (``xy_err <= 4.0 mm`` and ``z_seat_err <= 4.0 mm``)."""
    xy_err, z_err = clevis_seat_errors(objects)
    return bool(xy_err <= CLEVIS_MAX_XY_ERR_M and z_err <= CLEVIS_MAX_Z_ERR_M)


def linkage_locked(objects: dict[str, ObjectState]) -> bool:
    """True when the clevis eye is seated on the elbow stud and ``elbow_lock`` is rotated >= 75.0 deg clockwise."""
    return bool(clevis_seated(objects) and elbow_lock_deg(objects) >= LOCK_MIN_DEG)


def door_closed(objects: dict[str, ObjectState]) -> bool:
    """True when ``door_leaf`` has been drawn shut against the header stop (``door_angle <= 2.5 deg``)."""
    return bool(door_angle_deg(objects) <= DOOR_CLOSED_MAX_DEG)


# -- Embodiment, PhysX RigidContactView across ALL robot links, and TCP clearance --------------------------------------
def _embodiment():
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and getattr(s, "_base", None) is not None and hasattr(s, "_ids") and hasattr(s, "_t"):
            return s
        f = f.f_back
    return None


def _to_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "numpy"):
        return np.asarray(arr.numpy(), dtype=np.float64)
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy().astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def _ensure_contact_views(base: Any) -> dict[str, Any]:
    """Build a PhysX RigidContactView on ``door_leaf`` against ALL robot rigid-body links (palm, wrist, fingers, arm links)."""
    if _CONTACT_VIEWS.get("ready") or _CONTACT_VIEWS.get("error"):
        return _CONTACT_VIEWS
    try:
        try:
            import carb.settings

            carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        except Exception:
            pass

        from pxr import Usd, UsdPhysics

        stage = base.scene.stage
        robot_link_paths: list[str] = []
        for arm_prim in ("LeftArm", "RightArm"):
            root = stage.GetPrimAtPath(f"/World/envs/env_0/{arm_prim}")
            if root.IsValid():
                for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        robot_link_paths.append(prim.GetPath().pathString)

        sim_view = base.sim.physics_sim_view
        door_view = sim_view.create_rigid_contact_view(
            ["/World/envs/env_0/door_leaf"],
            filter_patterns=[robot_link_paths],
            max_contact_data_count=32,
        )
        _CONTACT_VIEWS["door_view"] = door_view
        _CONTACT_VIEWS["robot_link_count"] = len(robot_link_paths)
        _CONTACT_VIEWS["ready"] = True
    except Exception as exc:
        _CONTACT_VIEWS["error"] = f"{type(exc).__name__}: {exc}"
        print(f"CLOSER_CONTACT_VIEW_ERROR: {_CONTACT_VIEWS['error']}", flush=True)
    return _CONTACT_VIEWS


def read_door_stop_force_n(base: Any, objects: dict[str, ObjectState]) -> float:
    """Read the contact force (N) from all robot links (palm, wrist, fingers, links) forcing ``door_leaf`` past its stop."""
    if base is None:
        return 0.0
    if door_angle_deg(objects) > DOOR_CLOSED_MAX_DEG + 1.0:
        return 0.0
    views = _ensure_contact_views(base)
    if not views.get("ready"):
        return 0.0
    dv = views.get("door_view")
    if dv is None or getattr(dv, "filter_count", 0) <= 0:
        return 0.0
    dt = float(base.sim.get_physics_dt())
    fmat = _to_numpy(dv.get_contact_force_matrix(dt)).reshape(dv.sensor_count, dv.filter_count, 3)
    f_vec = fmat.sum(axis=(0, 1))
    return float(np.linalg.norm(f_vec))


def _robot_state(emb) -> dict[str, tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    out = {}
    for side in ("left", "right"):
        arm_ids, finger_ids = emb._ids[f"{side}_robot"]
        q_all = emb._base.scene[f"{side}_robot"].data.joint_pos.torch[0].cpu().numpy()
        q6 = q_all[arm_ids].astype(float)
        fingers = q_all[finger_ids].astype(float)
        tcp, r_tcp = ik.tcp_pose(q6, station.BASES[side])
        out[side] = (q6, float(fingers.mean()), tcp, r_tcp)
    return out


def tcp_points() -> list[np.ndarray]:
    """World TCP positions of both arms this step; ``[]`` offline."""
    emb = _embodiment()
    if emb is None:
        return []
    rs = _robot_state(emb)
    return [rs["left"][2], rs["right"][2]]


def box_distance(state: ObjectState, size: tuple[float, float, float], point: np.ndarray) -> float:
    """Distance (m) from ``point`` to the nearest point on an oriented box of ``size`` at ``state``."""
    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    local = rot.T @ (np.asarray(point, dtype=float) - np.asarray(state.pos, dtype=float))
    half = np.asarray(size, dtype=float) / 2.0
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def min_tcp_clearance_m(objects: dict[str, ObjectState], tcps: list[np.ndarray] | None = None) -> float:
    """Smallest distance (m) from any TCP to any task object. Returns ``+inf`` offline when ``tcps`` is empty."""
    if tcps is None:
        tcps = tcp_points()
    if not tcps:
        return float("inf")
    part_sizes = (
        ("header_frame", HEADER_SIZE),
        ("door_leaf", DOOR_LEAF_BOX_SIZE),
        ("door_shoe", DOOR_SHOE_SIZE),
        ("forearm_rod", FOREARM_SIZE),
        ("main_arm", MAIN_ARM_SIZE),
        ("elbow_lock", ELBOW_LOCK_SIZE),
    )
    return float(
        min(box_distance(objects[name], size, np.asarray(p, dtype=float)) for name, size in part_sizes for p in tcps)
    )


def hands_clear(objects: dict[str, ObjectState], tcps: list[np.ndarray] | None = None) -> bool:
    """True when both grippers are at least GRIPPER_CLEAR (40 mm) from all task objects."""
    return bool(min_tcp_clearance_m(objects, tcps) >= GRIPPER_CLEAR)


def all_resting(objects: dict[str, ObjectState]) -> bool:
    """True when all moving parts are at rest (< 0.05 m/s linear speed, < 0.2 rad/s angular speed)."""
    for k in ("door_leaf", "door_shoe", "forearm_rod", "main_arm", "elbow_lock"):
        st = objects[k]
        if float(np.linalg.norm(st.lin_vel)) >= STILL_MAX_SPEED or not st.resting():
            return False
    return True


# -- Episode state and latched faults ----------------------------------------------------------------------------------
_S: dict[str, Any] = {}


def new_episode() -> None:
    _S.clear()
    _S.update(
        t=None,
        k=0,
        max_main_arm_deg=0.0,
        max_lock_deg=0.0,
        peak_stop_force_n=0.0,
        fault=None,
    )
    fj = _RUNTIME_HANDLES.get("forearm_joint")
    if fj is not None:
        fj.GetLowerLimitAttr().Set(0.0)
        fj.GetUpperLimitAttr().Set(0.028)


new_episode()


def advance(objects: dict[str, ObjectState], door_stop_force_n: float | None = None) -> str | None:
    """Update linkage state and latched faults once per control step."""
    emb = _embodiment()
    base = emb._base if emb is not None else None
    if emb is not None and door_stop_force_n is None:
        t = float(emb._t)
        if _S["t"] is not None and t < _S["t"]:
            new_episode()
        if t == _S["t"]:
            return _S["fault"]
        _S["t"] = t

    _S["k"] += 1
    st = _S["k"]

    m_deg = main_arm_deg(objects)
    d_deg = door_angle_deg(objects)
    l_deg = elbow_lock_deg(objects)
    xy_err, z_err = clevis_seat_errors(objects)
    seated = clevis_seated(objects)

    _S["max_main_arm_deg"] = max(float(_S["max_main_arm_deg"]), m_deg)
    _S["max_lock_deg"] = max(float(_S["max_lock_deg"]), l_deg)

    # 1. Latched fault: turning captive elbow lock lever past 45 deg before forearm clevis eye is seated on the stud:
    if l_deg > LOCK_EARLY_FAULT_DEG and not seated and _S["fault"] is None:
        _S["fault"] = "lock_turned_before_clevis_seated"
        print(
            f"CLOSER_FAULT lock_turned_before_clevis_seated step={st} "
            f"lock_deg={l_deg:.1f} xy_err_mm={xy_err * 1000:.1f} z_err_mm={z_err * 1000:.1f}",
            flush=True,
        )

    # 2. Latched fault: forcing the door leaf past its frame stop (> 25.0 N from any robot link):
    f_stop_n = float(door_stop_force_n) if door_stop_force_n is not None else read_door_stop_force_n(base, objects)
    _S["peak_stop_force_n"] = max(float(_S["peak_stop_force_n"]), f_stop_n)
    if f_stop_n > MAX_DOOR_STOP_FORCE_N and _S["fault"] is None:
        _S["fault"] = "door_forced_past_stop"
        print(
            f"CLOSER_FAULT door_forced_past_stop step={st} "
            f"force_n={f_stop_n:.2f} door_deg={d_deg:.2f}",
            flush=True,
        )

    if emb is not None and (st == 1 or st % 10 == 0):
        clr_mm = min_tcp_clearance_m(objects) * 1000.0
        print(
            f"CLOSER_STEP step={st:3d} t={st / CONTROL_HZ:5.2f}s "
            f"main_deg={m_deg:5.1f} door_deg={d_deg:5.2f} "
            f"clevis_err=({xy_err * 1000:4.1f}mm,{z_err * 1000:4.1f}mm,seat={int(seated)}) "
            f"lock_deg={l_deg:5.1f}(lock={int(linkage_locked(objects))}) "
            f"stop_N={f_stop_n:4.1f}(peak={_S['peak_stop_force_n']:4.1f}) "
            f"tcp_clr={clr_mm:5.1f}mm fault={_S['fault']}",
            flush=True,
        )
    return _S["fault"]


def score(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
    door_stop_force_n: float | None = None,
) -> float:
    """Tiered scoring:
    - 1.00: linkage coupled (clevis seated on stud: xy_err <= 4 mm, z_err <= 4 mm) and locked (elbow_lock >= 75 deg),
      preloaded closer arm (>= 20 deg spring tension) has drawn door_leaf closed against the frame stop
      (door_angle <= 2.5 deg), both grippers >= 40 mm clear, and all bodies at rest (< 0.05 m/s for 1.0 s).
    - 0.70: eye seated on the stud, main_arm >= 20 deg, elbow_lock >= 75 deg clockwise, but the door is not within 2.5 deg of its stop,
      or something still moves, or a gripper is within 40 mm of a part.
    - 0.35: forearm clevis eye seated on the elbow stud (xy_err <= 4 mm, z_err <= 4 mm) with main_arm >= 20 deg,
      lock not yet turned.
    - 0.00: otherwise, or if any latched fault has occurred."""
    if advance(objects, door_stop_force_n=door_stop_force_n) is not None:
        return 0.0
    if not clevis_seated(objects):
        return 0.0
    m_deg = main_arm_deg(objects)
    if m_deg < MAIN_ARM_COUPLED_MIN_DEG:
        return 0.0
    if linkage_locked(objects):
        if door_closed(objects) and all_resting(objects) and hands_clear(objects, tcps):
            return 1.0
        return 0.70
    return 0.35


def success(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
    door_stop_force_n: float | None = None,
) -> bool:
    """The 1.0 condition (held for SUCCESS_HOLD_S = 1.0 s by the runner)."""
    return bool(score(objects, tcps, door_stop_force_n=door_stop_force_n) == 1.0)


def distance(objects: dict[str, ObjectState]) -> float:
    """Normalized remaining distance across clevis seating (1/3), elbow lock rotation (1/3), and door closure (1/3)."""
    xy_err, z_err = clevis_seat_errors(objects)
    d_clevis = 0.0 if clevis_seated(objects) else min(1.0, math.hypot(xy_err, z_err) / 0.18)
    d_lock = max(0.0, LOCK_MIN_DEG - elbow_lock_deg(objects)) / LOCK_MIN_DEG if clevis_seated(objects) else 1.0
    d_door = max(0.0, door_angle_deg(objects) - DOOR_CLOSED_MAX_DEG) / max(1.0, DOOR_INITIAL_DEG - DOOR_CLOSED_MAX_DEG)
    return float((d_clevis + d_lock + d_door) / 3.0)


# -- Scripted bimanual demonstration (23.2 s = 464 steps at 20 Hz, run_limit = 580 steps) ------------------------------
def _main_grip_xy(theta_m_deg: float) -> tuple[float, float]:
    phi = math.radians(MAIN_ARM_SLACK_YAW_DEG - theta_m_deg)
    return (
        float(PINION_XY[0] + MAIN_ARM_GRIP_R_M * math.cos(phi)),
        float(PINION_XY[1] + MAIN_ARM_GRIP_R_M * math.sin(phi)),
    )


def _forearm_grip_xy(world_yaw_deg: float) -> tuple[float, float]:
    r_grip_f = 0.5 * FOREARM_LINK_M
    rad = math.radians(world_yaw_deg)
    return (
        float(SHOE_AJAR_XY[0] + r_grip_f * math.cos(rad)),
        float(SHOE_AJAR_XY[1] + r_grip_f * math.sin(rad)),
    )


G_MAIN_0 = _main_grip_xy(0.0)          # (0.3239, +0.0896) at 0.0 deg slack stop
G_MAIN_21 = _main_grip_xy(24.0)        # (0.2923, +0.0706) midway along preload arc
# Over-target along the arc so wrist PD compliance seats main_arm cleanly against its -42.2 deg preload stop:
G_MAIN_42 = _main_grip_xy(55.0)        # 12.8 deg past the stop (~4 N*m PD push, offline) holds main_arm against the 42.0..42.2 deg preload stop

GF_SPAWN = _forearm_grip_xy(90.0)      # (0.1868, -0.1182) at +90.0 deg spawn yaw
GF_MID = _forearm_grip_xy(75.0)        # (0.2108, -0.1213) midway along -25 deg CW swing
# Over-target by 4.5 deg CW so wrist PD compliance seats door_shoe against its 46.7 deg alignment stop (64.7 deg world):
GF_COUPLED = _forearm_grip_xy(60.5)    # (0.2324, -0.1302) seats Ø18 mm clevis eye coaxially over S(42)

GL_0_HOV = grip_rot(-12.0, -48.0)
GL_0 = grip_rot(0.0, -48.0)
GL_21 = grip_rot(0.0, -72.0)
GL_42 = grip_rot(0.0, -94.0)
GL_42_HOV = grip_rot(-12.0, -90.0)
GP = grip_rot(-15.0, 0.0)

GR_F90_HOV = grip_rot(-12.0, +90.0)
GR_F90 = grip_rot(0.0, +90.0)
GR_F77 = grip_rot(0.0, +75.0)
GR_F65 = grip_rot(0.0, +61.0)
GR_F65_HOV = grip_rot(-12.0, +65.0)

GR_L0_HOV = grip_rot(-12.0, +90.0)
GR_L0 = grip_rot(0.0, +90.0)
GR_L45 = grip_rot(0.0, +45.0)
GR_L95 = grip_rot(0.0, -5.0)
GR_L_EXIT = grip_rot(-12.0, -5.0)

Z_MAIN_GRIP = TOP + 0.092
Z_FORE_LOW = TOP + 0.107
Z_FORE_SEAT = TOP + 0.103
Z_FORE_LIFT = TOP + 0.129
Z_LOCK_GRIP = TOP + 0.127
Z_HOVER = TOP + 0.154

PARK_LEFT = f3((0.180, +0.280, TOP + 0.155))
PARK_RIGHT = f3((0.180, -0.280, TOP + 0.155))


def demo() -> dict[str, list[Waypoint]]:
    """Scripted bimanual demonstration (23.2 s = 464 steps at 20 Hz)."""
    left = [
        Waypoint(1.2, f3((G_MAIN_0[0], G_MAIN_0[1], Z_HOVER)), FINGER_OPEN, "l_above_main_arm", GL_0_HOV),
        Waypoint(1.0, f3((G_MAIN_0[0], G_MAIN_0[1], Z_MAIN_GRIP)), FINGER_OPEN, "l_down_to_main_arm", GL_0),
        Waypoint(1.0, f3((G_MAIN_0[0], G_MAIN_0[1], Z_MAIN_GRIP)), FINGER_CLOSED, "l_grasp_main_arm", GL_0),
        Waypoint(1.2, f3((G_MAIN_21[0], G_MAIN_21[1], Z_MAIN_GRIP)), FINGER_CLOSED, "l_pull_preload_mid", GL_21),
        Waypoint(1.2, f3((G_MAIN_42[0], G_MAIN_42[1], Z_MAIN_GRIP)), FINGER_CLOSED, "l_pull_preload_42", GL_42),
        Waypoint(11.8, f3((G_MAIN_42[0], G_MAIN_42[1], Z_MAIN_GRIP)), FINGER_CLOSED, "l_hold_preload_42", GL_42),
        Waypoint(0.6, f3((G_MAIN_42[0], G_MAIN_42[1], Z_MAIN_GRIP)), FINGER_OPEN, "l_release_main_arm", GL_42),
        Waypoint(1.0, f3((G_MAIN_42[0], G_MAIN_42[1], Z_HOVER)), FINGER_OPEN, "l_lift_clear", GL_42_HOV),
        Waypoint(1.2, PARK_LEFT, FINGER_OPEN, "l_park", GP),
        Waypoint(3.0, PARK_LEFT, FINGER_OPEN, "l_done", GP),
    ]
    right = [
        Waypoint(3.8, PARK_RIGHT, FINGER_OPEN, "r_wait_for_preload", GP),
        Waypoint(1.2, f3((GF_SPAWN[0], GF_SPAWN[1], Z_HOVER)), FINGER_OPEN, "r_above_forearm", GR_F90_HOV),
        Waypoint(1.0, f3((GF_SPAWN[0], GF_SPAWN[1], Z_FORE_LOW)), FINGER_OPEN, "r_down_to_forearm", GR_F90),
        Waypoint(0.6, f3((GF_SPAWN[0], GF_SPAWN[1], Z_FORE_LOW)), FINGER_CLOSED, "r_grasp_forearm", GR_F90),
        Waypoint(0.9, f3((GF_SPAWN[0], GF_SPAWN[1], Z_FORE_LIFT)), FINGER_CLOSED, "r_lift_forearm", GR_F90),
        Waypoint(1.0, f3((GF_MID[0], GF_MID[1], Z_FORE_LIFT)), FINGER_CLOSED, "r_swing_forearm_mid", GR_F77),
        Waypoint(1.0, f3((GF_COUPLED[0], GF_COUPLED[1], Z_FORE_LIFT)), FINGER_CLOSED, "r_align_clevis_over_stud", GR_F65),
        Waypoint(1.0, f3((GF_COUPLED[0], GF_COUPLED[1], Z_FORE_SEAT)), FINGER_CLOSED, "r_seat_clevis_on_stud", GR_F65),
        Waypoint(0.4, f3((GF_COUPLED[0], GF_COUPLED[1], Z_FORE_SEAT)), FINGER_OPEN, "r_release_forearm", GR_F65),
        Waypoint(0.9, f3((GF_COUPLED[0], GF_COUPLED[1], Z_HOVER)), FINGER_OPEN, "r_lift_from_forearm", GR_F65_HOV),
        Waypoint(1.0, f3((LOCK_PIVOT_PRELOAD_XY[0], LOCK_PIVOT_PRELOAD_XY[1], Z_HOVER)), FINGER_OPEN, "r_above_elbow_lock", GR_L0_HOV),
        Waypoint(0.9, f3((LOCK_PIVOT_PRELOAD_XY[0], LOCK_PIVOT_PRELOAD_XY[1], Z_LOCK_GRIP)), FINGER_OPEN, "r_down_to_elbow_lock", GR_L0),
        Waypoint(0.5, f3((LOCK_PIVOT_PRELOAD_XY[0], LOCK_PIVOT_PRELOAD_XY[1], Z_LOCK_GRIP)), FINGER_CLOSED, "r_grasp_elbow_lock", GR_L0),
        Waypoint(1.0, f3((LOCK_PIVOT_PRELOAD_XY[0], LOCK_PIVOT_PRELOAD_XY[1], Z_LOCK_GRIP)), FINGER_CLOSED, "r_turn_lock_cw_45", GR_L45),
        Waypoint(1.0, f3((LOCK_PIVOT_PRELOAD_XY[0], LOCK_PIVOT_PRELOAD_XY[1], Z_LOCK_GRIP)), FINGER_CLOSED, "r_turn_lock_cw_95", GR_L95),
        Waypoint(0.4, f3((LOCK_PIVOT_PRELOAD_XY[0], LOCK_PIVOT_PRELOAD_XY[1], Z_LOCK_GRIP)), FINGER_OPEN, "r_release_elbow_lock", GR_L95),
        Waypoint(0.9, f3((LOCK_PIVOT_PRELOAD_XY[0], LOCK_PIVOT_PRELOAD_XY[1], Z_HOVER)), FINGER_OPEN, "r_lift_from_lock", GR_L_EXIT),
        Waypoint(1.2, PARK_RIGHT, FINGER_OPEN, "r_park", GP),
        Waypoint(4.5, PARK_RIGHT, FINGER_OPEN, "r_done", GP),
    ]
    return {"left": left, "right": right}


if __name__ == "__main__":
    for side, plan in demo().items():
        q = None
        for w in plan:
            q, pe, re = ik.solve(
                np.array(w.pos),
                ik.matrix_to_quat_xyzw(_closer_grip_rotation(w.yaw)),
                q_init=q,
                base_pos=station.BASES[side],
                iters=400,
            )
            m = np.degrees(np.minimum(q - ik.LOWER, ik.UPPER - q)).min()
            flag = "BAD" if pe > 0.002 or re > 0.02 or m < 5.0 else ""
            print(
                f"{side:5s} {w.label:24s} pos=({w.pos[0]:.3f},{w.pos[1]:+.3f},{w.pos[2] - TOP:.4f}) "
                f"pos_err={pe * 1000:.2f}mm rot={re:.4f} margin={m:.1f}deg {flag}"
            )
        print(side, "plan s", round(sum(w.duration for w in plan), 2))
