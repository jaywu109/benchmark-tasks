"""On the station table: a desktop friction-feed paper folder feed-table, retarder-frame, and overhead rubber
feed-wheel assembly (modelled after the Martin Yale 1501X AutoFolder feed table and retarder assembly,
https://www.martinyale.com/product/1501x-paper-folder/).

Centred near the midline at (0.255, +0.010) m sits the folder housing (`autofolder_housing`, 252 mm wide along x by
380 mm long along y by 176 mm tall, 14.1 kg / 31 lb powder-coated steel frame resting on the station table on non-skid
rubber feet). Its pre-set front and rear side paper guides are spaced 223 mm apart along x (x = 0.1435..0.3665 m,
giving 3.5 mm clearance per side for 216 mm letter-width paper). Mounted on the overhead feed shaft at y = +0.092 m is
the central 58 mm diameter rubber feed wheel (46 mm wide along x = 0.232..0.278 m) directly ahead of the retarder
frame stop at y = +0.130 m.

Hinged on `autofolder_housing` about the horizontal X axis at (0.255, -0.036, TOP + 0.112) m is the spring-elevated
sheet-metal feed table (`feed_table`, 210 mm wide along x by 226 mm long along y by 28 mm tall, 0.240 kg, with a front
retarder stop lip at y = +0.130 m and a left-hand press-down paddle at y = +0.136..+0.190 m). A constant passive upward
torsion spring (`stiffness = 0.055 N*m/deg`, preloaded toward `target = +13.0 deg` against `upper = 0.0 deg`) holds the
unloaded feed table up at `0.0 deg`, where the vertical opening under the overhead rubber feed wheel is only 7.4..8.9 mm
(blocking the 20 mm-thick newsletter stack by contact). Pressing the feed table down to about `-14.0 deg` (above its
`-16.5 deg` mechanical bottom stop) opens a ~39 mm vertical gap under the feed wheel, which exists only while a hand
holds the feed table down.

Resting on the folder's right-hand entry staging apron at (0.255, -0.1060, TOP + 0.124) m is a squared 150-sheet
half-letter (statement-size) newsletter stack (`newsletter_packet`, 216 mm width along x by 140 mm length along y by
20 mm height along z, 0.540 kg). There is no scripted interlock, no joint-limit switching, and no drive modification at
runtime: order comes purely from physical contact between the spring-elevated feed table, the overhead rubber feed
wheel, and the 20 mm newsletter stack.

The goal: hold the spring-loaded feed table depressed with one hand to open the gap under the overhead rubber feed
wheel, slide the 20 mm newsletter stack between the side paper guides with the other hand until its leading edge is
within 5 mm of the retarder stop (yaw <= 5 deg), release the feed table so its spring lifts and clamps the stack
against the underside of the feed wheel (`-7.5` to `-3.5 deg`), and move both grippers at least 30 mm clear with all
parts at rest for 1 s. Moving the folder > 15 mm, dropping the unheld packet by > 50 mm, or pushing the feed table past
its `-16.5 deg` stop latches a fault and scores 0 for the rest of the trial.
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
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, ObjectState  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix, rpy_matrix  # noqa: E402

SCENE = "station"
TOP = float(TABLE_TOP[SCENE])
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
CONTROL_HZ = 20.0

INSTRUCTION = "Load this newsletter stack into the folder so it is ready to feed."

YAM_MAX_OPENING_M = float(2.0 * abs(FINGER_OPEN))  # 0.0939 m (93.9 mm)


def f3(v) -> tuple[float, float, float]:
    return tuple(float(x) for x in v)


class Grip(float):
    """A 3x3 TCP rotation matrix carried in Waypoint.yaw (a float field)."""

    def __new__(cls, m):
        obj = super().__new__(cls, 1.0)
        obj.m = np.asarray(m, dtype=float)
        return obj


_orig_grip_rotation = getattr(_demo, "_orig_grip_rotation", _demo.grip_rotation)
_demo._orig_grip_rotation = _orig_grip_rotation


def _autofolder_grip_rotation(y=0.0) -> np.ndarray:
    if hasattr(y, "m"):
        return y.m
    if isinstance(y, np.ndarray):
        return y
    return _orig_grip_rotation(float(y))


_demo.grip_rotation = _autofolder_grip_rotation
DOWN = _orig_grip_rotation(0.0)


def rx_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(deg))


def ry_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(deg))


def rz_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(deg))


def grip_rot(pitch_deg: float = 0.0, roll_deg: float = 0.0, jaw_rot_deg: float = 0.0) -> Grip:
    """TCP rotation pitched about world y, rolled about world x, and turned jaw_rot_deg about gripper z."""
    return Grip(ry_mat(pitch_deg) @ rx_mat(roll_deg) @ DOWN @ rz_mat(jaw_rot_deg))


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
FOLDER_BEIGE = (0.76, 0.74, 0.70)       # Martin Yale 1501X light putty/beige powder-coated steel side frames
FOLDER_CHARCOAL = (0.20, 0.22, 0.25)    # dark charcoal base rails, overhead shaft arch & retarder frame
GUIDE_STEEL = (0.62, 0.65, 0.69)        # brushed steel side paper guides
WHEEL_RUBBER = (0.14, 0.15, 0.17)       # dark grey EPDM rubber overhead feed wheel (58 mm dia)
TABLE_SILVER = (0.70, 0.73, 0.77)       # sheet-metal spring-loaded feed table tray
CORK_PAD_TAN = (0.68, 0.50, 0.30)       # cork/rubber friction pad insert on feed table under feed wheel
PADDLE_AMBER = (0.86, 0.50, 0.12)       # high-contrast amber press-down paddle on feed table
PAPER_WHITE = (0.94, 0.94, 0.92)        # 20 lb white bond newsletter stack (150 sheets, 20 mm thick)
PAPER_BAND_BLUE = (0.24, 0.46, 0.72)    # blue printed header / belly band on the squared newsletter stack


# -- 1. AutoFolder Housing (autofolder_housing, free rigid body on rubber feet, 252 x 380 x 176 mm, 14.1 kg) -----------
X_CENTER = 0.2550
Y_FOLDER_CENTER = +0.0100
FOLDER_LX = 0.2520
FOLDER_LY = 0.3800
FOLDER_H = 0.1760
FOLDER_MASS = 14.100

FOLDER_X0 = X_CENTER - 0.5 * FOLDER_LX          # 0.1290 m (front outer frame face)
FOLDER_X1 = X_CENTER + 0.5 * FOLDER_LX          # 0.3810 m (rear outer frame face)
FOLDER_Y0 = Y_FOLDER_CENTER - 0.5 * FOLDER_LY   # -0.1800 m (right staging apron outer face)
FOLDER_Y1 = Y_FOLDER_CENTER + 0.5 * FOLDER_LY   # +0.2000 m (left exit housing outer face)

# Pre-set side paper guides: 223.0 mm inner spacing for 216.0 mm letter width (3.5 mm clearance per side):
GUIDE_SPACING_M = 0.2230
GUIDE_INNER_X0 = X_CENTER - 0.5 * GUIDE_SPACING_M  # 0.1435 m
GUIDE_INNER_X1 = X_CENTER + 0.5 * GUIDE_SPACING_M  # 0.3665 m

# Central 104 mm finger-clearance channel (x = 0.203..0.307 m) between the front and rear support decks:
SLOT_X0 = 0.2030
SLOT_X1 = 0.3070

# Feed table hinge pivot & retarder stop coordinates:
Y_PIVOT = -0.0360
Z_PIVOT_REL = 0.1120
Z_PIVOT = TOP + Z_PIVOT_REL
Y_WHEEL = +0.0920
Y_STOP = +0.1300
WHEEL_DIAMETER_M = 0.0580

# Stepped bottom contact shoes of the 58 mm overhead rubber feed wheel (x = 0.232..0.278 m, y = +0.080..+0.104 m):
# At theta_table = 0.0 deg (unloaded rest), the vertical gap above the feed table deck (z_rel = 0.112 m) is
# 7.38..8.92 mm (<< 20.0 mm newsletter stack height, blocking loading unless the feed table is depressed).
# When the 20.0 mm newsletter stack sits on the feed table and the table is released, all 3 shoes contact the top face
# of the 20.0 mm stack simultaneously at theta_table = -5.50 deg (centre of the scored [-7.5, -3.5] deg window).
WHEEL_SHOE_1_Z0 = 0.12092  # empty gap 8.92 mm; contacts 20 mm packet top at theta_table = -5.50 deg
WHEEL_SHOE_2_Z0 = 0.12015  # empty gap 8.15 mm; contacts 20 mm packet top at theta_table = -5.50 deg
WHEEL_SHOE_3_Z0 = 0.11938  # empty gap 7.38 mm; contacts 20 mm packet top at theta_table = -5.50 deg
EMPTY_NIP_MIN_GAP_M = WHEEL_SHOE_3_Z0 - Z_PIVOT_REL  # 0.00738 m (7.38 mm)

FOLDER_SIZE = f3((FOLDER_LX, FOLDER_LY, FOLDER_H))
FOLDER_SPAWN_POS = f3((X_CENTER, Y_FOLDER_CENTER, TOP + 0.5 * FOLDER_H))  # (0.255, +0.010, TOP + 0.088)


def _folder_piece(lo, hi, color=FOLDER_BEIGE) -> rx.Piece:
    lo_w = (lo[0], lo[1], TOP + lo[2])
    hi_w = (hi[0], hi[1], TOP + hi[2])
    return rx.Piece(
        size=f3(h - l for l, h in zip(lo_w, hi_w)),
        offset=f3(0.5 * (l + h) - c for l, h, c in zip(lo_w, hi_w, FOLDER_SPAWN_POS)),
        color=color,
    )


FOLDER_PIECE_BOUNDS = (
    # Heavy rubber-footed lower side rails & cross-ties:
    ("base_floor_front", (FOLDER_X0, FOLDER_Y0, 0.000), (SLOT_X0, FOLDER_Y1, 0.016), FOLDER_CHARCOAL),
    ("base_floor_rear", (SLOT_X1, FOLDER_Y0, 0.000), (FOLDER_X1, FOLDER_Y1, 0.016), FOLDER_CHARCOAL),
    ("base_cross_left", (SLOT_X0, +0.048, 0.000), (SLOT_X1, +0.115, 0.012), FOLDER_CHARCOAL),
    ("base_cross_right", (SLOT_X0, FOLDER_Y0, 0.000), (SLOT_X1, -0.140, 0.012), FOLDER_CHARCOAL),
    # Right-hand entry staging apron (y = -0.180..-0.046 m, z_rel = 0.114 m, 2 mm above Z_PIVOT_REL for smooth entry):
    ("staging_deck_front", (0.147, FOLDER_Y0, 0.016), (SLOT_X0, -0.046, 0.114), FOLDER_BEIGE),
    ("staging_deck_rear", (SLOT_X1, FOLDER_Y0, 0.016), (0.363, -0.046, 0.114), FOLDER_BEIGE),
    # Pre-set 223 mm-spaced side paper guides (x = 0.129..0.1435 m and 0.3665..0.381 m, y = -0.180..+0.136 m):
    ("side_guide_front", (FOLDER_X0, FOLDER_Y0, 0.016), (GUIDE_INNER_X0, +0.136, 0.150), GUIDE_STEEL),
    ("side_guide_rear", (GUIDE_INNER_X1, FOLDER_Y0, 0.016), (FOLDER_X1, +0.136, 0.150), GUIDE_STEEL),
    # Overhead feed-roller shaft arch and upper retarder frame (y <= +0.136 m so y >= +0.140 m is clear above paddle):
    ("feed_arch_pillar_front", (FOLDER_X0, +0.076, 0.150), (GUIDE_INNER_X0, +0.136, FOLDER_H), FOLDER_CHARCOAL),
    ("feed_arch_pillar_rear", (GUIDE_INNER_X1, +0.076, 0.150), (FOLDER_X1, +0.136, FOLDER_H), FOLDER_CHARCOAL),
    ("feed_arch_bridge", (FOLDER_X0, +0.080, 0.162), (FOLDER_X1, +0.128, FOLDER_H), FOLDER_CHARCOAL),
    ("retarder_frame_upper", (0.165, +0.128, 0.138), (0.345, +0.136, 0.162), FOLDER_CHARCOAL),
    # Central 58 mm diameter rubber feed wheel (x = 0.232..0.278 m) with 3-step -5.50 deg bottom contact contour:
    ("wheel_shoe_1", (0.232, +0.080, WHEEL_SHOE_1_Z0), (0.278, +0.088, 0.162), WHEEL_RUBBER),
    ("wheel_shoe_2", (0.232, +0.088, WHEEL_SHOE_2_Z0), (0.278, +0.096, 0.162), WHEEL_RUBBER),
    ("wheel_shoe_3", (0.232, +0.096, WHEEL_SHOE_3_Z0), (0.278, +0.104, 0.162), WHEEL_RUBBER),
    ("wheel_drum_upper", (0.232, +0.066, 0.12600), (0.278, +0.118, 0.162), WHEEL_RUBBER),
)

AUTOFOLDER_COMPOUND = rx.Compound(pieces=tuple(_folder_piece(lo, hi, col) for _, lo, hi, col in FOLDER_PIECE_BOUNDS))
AUTOFOLDER = Box(
    "autofolder_housing",
    pos=FOLDER_SPAWN_POS,
    size=FOLDER_SIZE,
    mass=FOLDER_MASS,
    color=FOLDER_BEIGE,
)


# -- 2. Spring-Elevated Feed Table (feed_table, loose Box + rx.Compound + rx.Revolute on autofolder_housing, 0.240 kg) --
# Hinged about autofolder_housing's X axis at world (X_CENTER, Y_PIVOT, Z_PIVOT) = (0.255, -0.036, TOP + 0.112).
# Zero pose (0.0 deg in OBJECTS) is the unloaded upper rest stop (upper = 0.0 deg), held up by the constant passive
# spring (target = +13.0 deg, stiffness = 0.055 N*m/deg, damping = 0.0045 N*m*s/deg).
# Depressing the left-side paddle (y = +0.136..+0.190 m) rotates feed_table to negative angles:
#   * -14.0 deg during loading (opens a ~39 mm vertical gap under the overhead rubber feed wheel),
#   * -16.5 deg rated mechanical bottom stop (pushing below TABLE_STOP_DEG = -16.5 deg latches table_pushed_past_stop),
#   * -5.50 deg when released with the 20 mm newsletter packet under the feed wheel (clamped inside [-7.5, -3.5] deg).
FEED_TABLE_Y0 = Y_PIVOT                         # -0.0360 m
FEED_TABLE_Y1 = +0.1900                         # +0.1900 m
FEED_TABLE_Z0_REL = 0.1020                      # 102.0 mm above table
FEED_TABLE_Z1_REL = 0.1300                      # 130.0 mm above table (top of retarder stop lip)

FEED_TABLE_LX = 0.2100
FEED_TABLE_LY = float(FEED_TABLE_Y1 - FEED_TABLE_Y0)  # 0.2260 m
FEED_TABLE_H = 0.0280
FEED_TABLE_MASS = 0.240

FEED_TABLE_ZERO_POS = f3((
    X_CENTER,
    0.5 * (FEED_TABLE_Y0 + FEED_TABLE_Y1),      # +0.0770 m
    TOP + 0.5 * (FEED_TABLE_Z0_REL + FEED_TABLE_Z1_REL),  # TOP + 0.1160 m
))
FEED_TABLE_SIZE = f3((FEED_TABLE_LX, FEED_TABLE_LY, FEED_TABLE_H))
FEED_TABLE_HINGE_WORLD = f3((X_CENTER, Y_PIVOT, Z_PIVOT))
FEED_TABLE_ANCHOR = f3(h - c for h, c in zip(FEED_TABLE_HINGE_WORLD, FEED_TABLE_ZERO_POS))  # (0.0, -0.107, -0.004)

FEED_TABLE_LOWER_DEG = -22.0
FEED_TABLE_UPPER_DEG = 0.0
FEED_TABLE_TARGET_DEG = +13.0
FEED_TABLE_STIFFNESS = 0.055
FEED_TABLE_DAMPING = 0.0045

Y_PADDLE = +0.1750
PADDLE_WIDTH_X = 0.0600
PADDLE_THICK_Z = 0.0180


def _table_piece(lo, hi, color=TABLE_SILVER) -> rx.Piece:
    lo_w = (lo[0], lo[1], TOP + lo[2])
    hi_w = (hi[0], hi[1], TOP + hi[2])
    return rx.Piece(
        size=f3(h - l for l, h in zip(lo_w, hi_w)),
        offset=f3(0.5 * (l + h) - c for l, h, c in zip(lo_w, hi_w, FEED_TABLE_ZERO_POS)),
        color=color,
    )


FEED_TABLE_PIECE_BOUNDS = (
    # Entry side decks (y = -0.036..+0.042 m) flanking the 104 mm central finger-clearance slot (x = 0.203..0.307 m):
    ("deck_entry_front", (0.150, -0.036, 0.102), (SLOT_X0, +0.042, Z_PIVOT_REL), TABLE_SILVER),
    ("deck_entry_rear", (SLOT_X1, -0.036, 0.102), (0.360, +0.042, Z_PIVOT_REL), TABLE_SILVER),
    # Full-width clamping platen (y = +0.042..+0.130 m) and flush cork friction pad under the overhead feed wheel:
    ("deck_platen_front", (0.150, +0.042, 0.102), (0.228, Y_STOP, Z_PIVOT_REL), TABLE_SILVER),
    ("deck_platen_rear", (0.282, +0.042, 0.102), (0.360, Y_STOP, Z_PIVOT_REL), TABLE_SILVER),
    ("deck_platen_mid_entry", (0.228, +0.042, 0.102), (0.282, +0.072, Z_PIVOT_REL), TABLE_SILVER),
    ("cork_feed_pad", (0.228, +0.072, 0.102), (0.282, Y_STOP, Z_PIVOT_REL), CORK_PAD_TAN),
    # Retarder stop lip at y = Y_STOP = +0.130 m (stops the leading +y edge of newsletter_packet):
    ("retarder_stop_lip", (0.158, Y_STOP, 0.102), (0.352, +0.136, 0.130), FOLDER_CHARCOAL),
    # Left-side (+y) press-down paddle (60 x 54 x 18 mm at y = +0.136..+0.190 m) for the left hand:
    ("depress_paddle", (0.225, +0.136, 0.102), (0.285, FEED_TABLE_Y1, 0.120), PADDLE_AMBER),
)

FEED_TABLE_COMPOUND = rx.Compound(
    pieces=tuple(_table_piece(lo, hi, col) for _, lo, hi, col in FEED_TABLE_PIECE_BOUNDS)
)
FEED_TABLE = Box(
    "feed_table",
    pos=FEED_TABLE_ZERO_POS,
    size=FEED_TABLE_SIZE,
    mass=FEED_TABLE_MASS,
    color=TABLE_SILVER,
)
FEED_TABLE_JOINT = rx.Revolute(
    anchor=FEED_TABLE_ANCHOR,
    axis="X",
    lower=FEED_TABLE_LOWER_DEG,
    upper=FEED_TABLE_UPPER_DEG,
    target=FEED_TABLE_TARGET_DEG,
    stiffness=FEED_TABLE_STIFFNESS,
    damping=FEED_TABLE_DAMPING,
    parent="autofolder_housing",
    initial=0.0,
)


# -- 3. Squared 150-Sheet Half-Letter Newsletter Stack (newsletter_packet, 216 x 140 x 20 mm, 0.540 kg) ----------------
# Half-letter (statement-size) dimensions: 216 mm width along x, 140 mm length along y, 20.0 mm stack height along z.
# No invented tabs, posts, handles, or rims: the right hand rolls +73 deg so its 34 x 40 mm flat finger pads pinch
# the real 20.0 mm stack thickness vertically (top and bottom faces) inside the 104 mm central tray cutout.
PACKET_LX = 0.2160
PACKET_LY = 0.1400
PACKET_LZ = 0.0200
PACKET_SIZE = f3((PACKET_LX, PACKET_LY, PACKET_LZ))
PACKET_MASS = 0.540

# At spawn, newsletter_packet rests flat on the right-hand staging apron (z_bottom = TOP + 0.114 m, y = -0.176..-0.036 m):
PACKET_SPAWN_X = X_CENTER
PACKET_SPAWN_Y = -0.1060
PACKET_SPAWN_Z = TOP + 0.1140 + 0.5 * PACKET_LZ  # TOP + 0.1240 m
PACKET_SPAWN_POS = f3((PACKET_SPAWN_X, PACKET_SPAWN_Y, PACKET_SPAWN_Z))

# Reference points in feed_table frame (relative to FEED_TABLE_ZERO_POS = (0.255, +0.077, TOP + 0.116)):
# Stop face at y = Y_STOP = +0.130 m, deck surface at z = TOP + Z_PIVOT_REL = TOP + 0.112 m.
STOP_Y_IN_TABLE = float(Y_STOP - FEED_TABLE_ZERO_POS[1])       # +0.0530 m
DECK_Z_IN_TABLE = float(Z_PIVOT - FEED_TABLE_ZERO_POS[2])      # -0.0040 m
_WHEEL_CONTACT_Z_AT_Y_WHEEL = 0.5 * (WHEEL_SHOE_2_Z0 + WHEEL_SHOE_3_Z0)
WHEEL_BOT_IN_FOLDER = np.array(
    [0.0, Y_WHEEL - Y_FOLDER_CENTER, _WHEEL_CONTACT_Z_AT_Y_WHEEL - 0.5 * FOLDER_H],
    dtype=float,
)


def _packet_piece(lo_local, hi_local, color=PAPER_WHITE) -> rx.Piece:
    return rx.Piece(
        size=f3(h - l for l, h in zip(lo_local, hi_local)),
        offset=f3(0.5 * (l + h) for l, h in zip(lo_local, hi_local)),
        color=color,
    )


PACKET_PIECE_BOUNDS = (
    # Left and right white bond paper stack halves with central blue printed newsletter header band (flush 216x140x20 mm):
    ("stack_left", (-0.108, -0.0700, -0.010), (-0.028, +0.0700, +0.010), PAPER_WHITE),
    ("stack_band", (-0.028, -0.0700, -0.010), (+0.028, +0.0700, +0.010), PAPER_BAND_BLUE),
    ("stack_right", (+0.028, -0.0700, -0.010), (+0.108, +0.0700, +0.010), PAPER_WHITE),
)

PACKET_COMPOUND = rx.Compound(pieces=tuple(_packet_piece(lo, hi, col) for _, lo, hi, col in PACKET_PIECE_BOUNDS))
PACKET = Box(
    "newsletter_packet",
    pos=PACKET_SPAWN_POS,
    size=PACKET_SIZE,
    mass=PACKET_MASS,
    color=PAPER_WHITE,
)


def packet_pose_on_table(
    table_deg: float = -5.50,
    stop_gap_m: float = 0.0,
    dx_m: float = 0.0,
    dz_deck_m: float = 0.0,
    yaw_deg: float = 0.0,
    folder_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """World (pos, quat_xyzw) of newsletter_packet resting on feed_table at joint angle ``table_deg`` with its +y
    leading edge ``stop_gap_m`` back from the retarder stop."""
    th = math.radians(float(table_deg))
    r_tbl = axis_angle_matrix(np.array([1.0, 0.0, 0.0]), th)
    r_yaw = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(float(yaw_deg)))
    r_pkt = r_tbl @ r_yaw
    dy_center = (Y_STOP - Y_PIVOT) - float(stop_gap_m) - 0.5 * PACKET_LY
    dz_center = 0.5 * PACKET_LZ + float(dz_deck_m)
    p_hinge = np.array([
        X_CENTER + float(folder_shift[0]),
        Y_PIVOT + float(folder_shift[1]),
        Z_PIVOT + float(folder_shift[2]),
    ], dtype=float)
    p_world = p_hinge + r_tbl @ np.array([float(dx_m), dy_center, dz_center], dtype=float)
    q_pkt = ik.matrix_to_quat_xyzw(r_pkt)
    return f3(p_world), tuple(float(v) for v in q_pkt)


PACKET_CLAMPED_DEG = -5.50
PACKET_LOADED_POS, PACKET_LOADED_QUAT = packet_pose_on_table(PACKET_CLAMPED_DEG, stop_gap_m=0.0)

OBJECTS = [AUTOFOLDER, FEED_TABLE, PACKET]

rx.install(
    {
        "autofolder_housing": AUTOFOLDER_COMPOUND,
        "feed_table": (FEED_TABLE_COMPOUND, FEED_TABLE_JOINT),
        "newsletter_packet": PACKET_COMPOUND,
    }
)


# -- Scene physics patch: 20 Hz control & friction materials (no runtime joint/drive manipulation) ---------------------
_SCENE_MOD = "yam_arena.scene"
_CONTACT_VIEWS: dict[str, Any] = {}


def _patch_scene_physics(scene_mod) -> None:
    if getattr(scene_mod, "_autofolder_patched", False):
        return
    import isaaclab.sim as sim_utils

    orig_task_object = scene_mod.task_object
    orig_build_task_env = scene_mod.build_task_environment

    folder_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.80,
        dynamic_friction=1.50,
        friction_combine_mode="average",
        restitution=0.0,
    )
    table_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.42,
        dynamic_friction=0.32,
        friction_combine_mode="min",
        restitution=0.0,
    )
    packet_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.85,
        dynamic_friction=0.70,
        friction_combine_mode="average",
        restitution=0.0,
    )

    def task_object(spec):
        obj = orig_task_object(spec)
        if spec.name == "autofolder_housing":
            obj.object_cfg.spawn.physics_material = folder_mat
            obj.spawner_cfg.physics_material = folder_mat
        elif spec.name == "feed_table":
            obj.object_cfg.spawn.physics_material = table_mat
            obj.spawner_cfg.physics_material = table_mat
        elif spec.name == "newsletter_packet":
            obj.object_cfg.spawn.physics_material = packet_mat
            obj.spawner_cfg.physics_material = packet_mat
            obj.object_cfg.spawn.activate_contact_sensors = True
            obj.spawner_cfg.activate_contact_sensors = True
            prev_pkt_spawn = obj.object_cfg.spawn.func

            def _spawn_packet(prim_path, cfg, *args, _prev=prev_pkt_spawn, **kwargs):
                prim = _prev(prim_path, cfg, *args, **kwargs)
                from pxr import PhysxSchema

                PhysxSchema.PhysxContactReportAPI.Apply(prim)
                _CONTACT_VIEWS.clear()
                return prim

            obj.object_cfg.spawn.func = _spawn_packet
            obj.spawner_cfg.func = _spawn_packet
        return obj

    def build_task_environment(embodiment, task, control_hz=None):
        return orig_build_task_env(embodiment, task, control_hz=control_hz or CONTROL_HZ)

    scene_mod.task_object = task_object
    scene_mod.build_task_environment = build_task_environment
    scene_mod._autofolder_patched = True


_orig_rx_patch = getattr(rx, "_orig_rx_patch", rx._patch)
rx._orig_rx_patch = _orig_rx_patch


def _autofolder_rx_patch(scene_mod) -> None:
    _orig_rx_patch(scene_mod)
    _patch_scene_physics(scene_mod)


rx._patch = _autofolder_rx_patch
if _SCENE_MOD in sys.modules:
    _autofolder_rx_patch(sys.modules[_SCENE_MOD])


# -- Scoring & latched fault thresholds (from job notes) ---------------------------------------------------------------
STOP_GAP_MAX_M = 0.005                  # 5 mm: packet leading edge within 5 mm of the retarder stop
PACKET_YAW_MAX_DEG = 5.0                # 5 deg: packet yaw <= 5 deg relative to folder guides
PACKET_TABLE_TILT_MAX_DEG = 4.0         # 4 deg: packet flat on the feed table plane within 4 deg
PACKET_LATERAL_MAX_M = 0.006            # 6 mm: packet centred laterally between the 223 mm side guides
PACKET_DECK_GAP_MAX_M = 0.004           # 4 mm: packet bottom resting on the feed table deck
CLAMP_ANGLE_MIN_DEG = -7.5              # -7.5 deg: lower bound of clamped feed-table angle
CLAMP_ANGLE_MAX_DEG = -3.5              # -3.5 deg: upper bound of clamped feed-table angle
WHEEL_NIP_GAP_MAX_M = 0.0025            # 2.5 mm: packet top face pressed against underside of rubber feed wheel
DEPRESS_CLEAR_DEG = -12.0               # -12.0 deg: feed-table depression needed to clear the 20 mm stack under wheel
GRIPPER_CLEAR = 0.030                   # 30 mm: both grippers >= 30 mm clear of all task objects
GRIPPER_TOUCH_MAX_M = 0.025             # 25 mm: gripper TCP box-distance threshold for holding/touching packet

FOLDER_SHIFT_MAX_M = 0.015              # 15 mm: folder moved > 15 mm over the trial latches folder_moved
FOLDER_TILT_MAX_DEG = 8.0               # 8 deg: folder tipped > 8 deg over the trial latches folder_moved
PACKET_DROP_MAX_M = 0.050               # 50 mm: unheld packet descent > 50 mm over the trial latches packet_dropped
TABLE_STOP_DEG = -16.5                  # -16.5 deg: table pushed past its stop (< -16.5 deg) latches fault


# -- State readout helpers ---------------------------------------------------------------------------------------------
def tilt_deg(state: ObjectState) -> float:
    """Angle (deg) between the object's local +Z axis and world +Z."""
    r = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    cos_t = max(-1.0, min(1.0, float(r[2, 2])))
    return float(math.degrees(math.acos(cos_t)))


def folder_shift_m(objects: dict[str, ObjectState]) -> float:
    """3D displacement (m) of autofolder_housing from its spawn spot FOLDER_SPAWN_POS."""
    p = np.asarray(objects["autofolder_housing"].pos, dtype=float)
    p0 = np.asarray(FOLDER_SPAWN_POS, dtype=float)
    return float(np.linalg.norm(p - p0))


def feed_table_angle_deg(objects: dict[str, ObjectState]) -> float:
    """Signed angle (deg) of feed_table about autofolder_housing's +X axis:
    - 0.0 deg when empty and held against the upper stop by its spring,
    - -5.5 deg (in [-7.5, -3.5] deg) when released and clamping the 20 mm packet against the feed wheel,
    - ~-14.0 deg when held depressed to open the feed-wheel nip."""
    r_fld = quat_xyzw_to_matrix(np.asarray(objects["autofolder_housing"].quat_xyzw, dtype=float))
    r_tbl = quat_xyzw_to_matrix(np.asarray(objects["feed_table"].quat_xyzw, dtype=float))
    r_rel = r_fld.T @ r_tbl
    return float(math.degrees(math.atan2(float(r_rel[2, 1]), float(r_rel[1, 1]))))


def packet_yaw_deg(objects: dict[str, ObjectState]) -> float:
    """Absolute yaw angle (deg) of newsletter_packet in feed_table's guide channel plane."""
    r_tbl = quat_xyzw_to_matrix(np.asarray(objects["feed_table"].quat_xyzw, dtype=float))
    r_pkt = quat_xyzw_to_matrix(np.asarray(objects["newsletter_packet"].quat_xyzw, dtype=float))
    r_rel = r_tbl.T @ r_pkt
    return float(abs(math.degrees(math.atan2(float(r_rel[1, 0]), float(r_rel[0, 0])))))


def packet_table_tilt_deg(objects: dict[str, ObjectState]) -> float:
    """Angle (deg) between newsletter_packet's +Z normal and feed_table's +Z normal."""
    r_tbl = quat_xyzw_to_matrix(np.asarray(objects["feed_table"].quat_xyzw, dtype=float))
    r_pkt = quat_xyzw_to_matrix(np.asarray(objects["newsletter_packet"].quat_xyzw, dtype=float))
    cos_t = max(-1.0, min(1.0, float(np.dot(r_tbl[:, 2], r_pkt[:, 2]))))
    return float(math.degrees(math.acos(cos_t)))


def packet_offsets_in_table_m(objects: dict[str, ObjectState]) -> tuple[float, float, float]:
    """(stop_gap_m, lateral_dx_m, deck_dz_m) of newsletter_packet in feed_table's frame:
    - stop_gap_m: distance (m) from newsletter_packet's +y leading edge centre to the retarder stop face (Y_STOP),
    - lateral_dx_m: absolute lateral offset |dx| (m) from the guide channel centreline X_CENTER,
    - deck_dz_m: distance (m) between newsletter_packet's bottom face and the feed_table deck surface."""
    tbl = objects["feed_table"]
    pkt = objects["newsletter_packet"]
    p_tbl = np.asarray(tbl.pos, dtype=float)
    r_tbl = quat_xyzw_to_matrix(np.asarray(tbl.quat_xyzw, dtype=float))
    p_pkt = np.asarray(pkt.pos, dtype=float)
    r_pkt = quat_xyzw_to_matrix(np.asarray(pkt.quat_xyzw, dtype=float))

    lead_world = p_pkt + r_pkt @ np.array([0.0, +0.5 * PACKET_LY, 0.0], dtype=float)
    lead_in_tbl = r_tbl.T @ (lead_world - p_tbl)
    pkt_in_tbl = r_tbl.T @ (p_pkt - p_tbl)

    stop_gap = abs(float(STOP_Y_IN_TABLE - lead_in_tbl[1]))
    lateral_dx = abs(float(pkt_in_tbl[0]))
    deck_dz = abs(float((pkt_in_tbl[2] - 0.5 * PACKET_LZ) - DECK_Z_IN_TABLE))
    return stop_gap, lateral_dx, deck_dz


def packet_leading_edge_gap_m(objects: dict[str, ObjectState]) -> float:
    """Distance (m) from newsletter_packet's +y leading edge to the retarder stop."""
    return packet_offsets_in_table_m(objects)[0]


def packet_wheel_gap_m(objects: dict[str, ObjectState]) -> float:
    """Vertical/normal gap (m) between the top face of newsletter_packet under the feed wheel and the underside of
    autofolder_housing's overhead rubber feed wheel."""
    fld = objects["autofolder_housing"]
    pkt = objects["newsletter_packet"]
    p_fld = np.asarray(fld.pos, dtype=float)
    r_fld = quat_xyzw_to_matrix(np.asarray(fld.quat_xyzw, dtype=float))
    p_pkt = np.asarray(pkt.pos, dtype=float)
    r_pkt = quat_xyzw_to_matrix(np.asarray(pkt.quat_xyzw, dtype=float))

    wheel_bot_w = p_fld + r_fld @ WHEEL_BOT_IN_FOLDER
    wheel_in_pkt = r_pkt.T @ (wheel_bot_w - p_pkt)
    return abs(float(wheel_in_pkt[2] - 0.5 * PACKET_LZ))


def any_part_off_table(objects: dict[str, ObjectState]) -> bool:
    """True if any task object has fallen below or outside the station table surface."""
    for name in ("autofolder_housing", "feed_table", "newsletter_packet"):
        s = objects[name]
        px, py, pz = (float(v) for v in s.pos)
        low_z = float(s.low[2])
        if pz < TOP or low_z < TOP - 0.008 or not (0.02 <= px <= 0.78 and abs(py) <= 0.52):
            return True
    return False


def folder_stable_on_table(objects: dict[str, ObjectState]) -> bool:
    """True when autofolder_housing has moved <= 15 mm and tilted <= 8 deg on the table."""
    if any_part_off_table(objects):
        return False
    fld = objects["autofolder_housing"]
    return bool(
        folder_shift_m(objects) <= FOLDER_SHIFT_MAX_M
        and tilt_deg(fld) <= FOLDER_TILT_MAX_DEG
        and abs(float(fld.low[2]) - TOP) <= 0.006
    )


def packet_at_retarder_stop(objects: dict[str, ObjectState]) -> bool:
    """True when newsletter_packet's leading edge is within 5 mm of the retarder stop, centred between the side paper
    guides (<= 6 mm), and resting on the feed table deck (<= 4 mm)."""
    stop_gap, lateral_dx, deck_dz = packet_offsets_in_table_m(objects)
    return bool(
        stop_gap <= STOP_GAP_MAX_M
        and lateral_dx <= PACKET_LATERAL_MAX_M
        and deck_dz <= PACKET_DECK_GAP_MAX_M
    )


def packet_square_in_guides(objects: dict[str, ObjectState]) -> bool:
    """True when newsletter_packet yaw <= 5 deg and tilt relative to feed_table <= 4 deg."""
    return bool(
        packet_yaw_deg(objects) <= PACKET_YAW_MAX_DEG
        and packet_table_tilt_deg(objects) <= PACKET_TABLE_TILT_MAX_DEG
    )


def feed_table_clamped(objects: dict[str, ObjectState]) -> bool:
    """True when feed_table is released and clamping newsletter_packet against the overhead feed wheel
    (theta_table in [-7.5, -3.5] deg and packet top within 2.5 mm of the wheel underside)."""
    ang = feed_table_angle_deg(objects)
    return bool(
        CLAMP_ANGLE_MIN_DEG <= ang <= CLAMP_ANGLE_MAX_DEG
        and packet_wheel_gap_m(objects) <= WHEEL_NIP_GAP_MAX_M
    )


def packet_loaded_and_clamped(objects: dict[str, ObjectState]) -> bool:
    """True when the folder is stable on the table, the packet is at the retarder stop and square in the guides, and
    the feed table is released and clamping the packet against the feed wheel."""
    return bool(
        folder_stable_on_table(objects)
        and packet_at_retarder_stop(objects)
        and packet_square_in_guides(objects)
        and feed_table_clamped(objects)
    )


# -- Embodiment & TCP clearance helpers --------------------------------------------------------------------------------
def _embodiment():
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and getattr(s, "_base", None) is not None and hasattr(s, "_ids") and hasattr(s, "_t"):
            return s
        f = f.f_back
    return None


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


def _to_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        return np.asarray(arr.numpy(), dtype=np.float64)
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy().astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def _ensure_contact_views(base: Any) -> dict[str, Any]:
    """Build a PhysX RigidContactView on ``newsletter_packet`` against all robot rigid-body links."""
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
        pkt_view = sim_view.create_rigid_contact_view(
            ["/World/envs/env_0/newsletter_packet"],
            filter_patterns=[robot_link_paths],
            max_contact_data_count=32,
        )
        _CONTACT_VIEWS["packet_view"] = pkt_view
        _CONTACT_VIEWS["ready"] = True
    except Exception as exc:
        _CONTACT_VIEWS["error"] = f"{type(exc).__name__}: {exc}"
    return _CONTACT_VIEWS


def read_packet_gripper_contact_force_n(base: Any) -> float:
    """Read total contact force magnitude (N) between either robot arm/gripper and ``newsletter_packet``."""
    if base is None:
        return 0.0
    views = _ensure_contact_views(base)
    if not views.get("ready"):
        return 0.0
    pv = views.get("packet_view")
    if pv is None or getattr(pv, "filter_count", 0) <= 0:
        return 0.0
    dt = float(base.sim.get_physics_dt())
    fmat = _to_numpy(pv.get_contact_force_matrix(dt)).reshape(pv.sensor_count, pv.filter_count, 3)
    return float(np.linalg.norm(fmat, axis=-1).sum())


def gripper_touches_packet(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
) -> bool:
    """True when either gripper touches or holds ``newsletter_packet``."""
    emb = _embodiment()
    if emb is not None and tcps is None:
        if read_packet_gripper_contact_force_n(emb._base) > 0.05:
            return True
    if tcps is None:
        tcps = tcp_points()
    if not tcps:
        return False
    pkt = objects["newsletter_packet"]
    d_pkt = min(box_distance(pkt, PACKET_SIZE, np.asarray(p, dtype=float)) for p in tcps)
    return bool(d_pkt <= GRIPPER_TOUCH_MAX_M)


def min_tcp_clearance_m(objects: dict[str, ObjectState], tcps: list[np.ndarray] | None = None) -> float:
    """Smallest distance (m) from any TCP to autofolder_housing, feed_table, or newsletter_packet; ``-inf`` if empty."""
    if tcps is None:
        tcps = tcp_points()
    if not tcps:
        return float("-inf")
    part_sizes = (
        ("autofolder_housing", FOLDER_SIZE),
        ("feed_table", FEED_TABLE_SIZE),
        ("newsletter_packet", PACKET_SIZE),
    )
    return float(
        min(box_distance(objects[name], size, np.asarray(p, dtype=float)) for name, size in part_sizes for p in tcps)
    )


def hands_clear(objects: dict[str, ObjectState], tcps: list[np.ndarray] | None = None) -> bool:
    """True when both grippers are at least GRIPPER_CLEAR (30 mm) from all task objects."""
    return min_tcp_clearance_m(objects, tcps) >= GRIPPER_CLEAR


def all_resting(objects: dict[str, ObjectState]) -> bool:
    """True when autofolder_housing, feed_table, and newsletter_packet are at rest."""
    return all(objects[k].resting() for k in ("autofolder_housing", "feed_table", "newsletter_packet"))


# -- Episode state & latched fault tracking ----------------------------------------------------------------------------
_S: dict[str, Any] = {}


def new_episode() -> None:
    _S.clear()
    _S.update(
        t=None,
        k=0,
        peak_folder_shift_mm=0.0,
        max_packet_z=PACKET_SPAWN_Z,
        max_unheld_packet_z=PACKET_SPAWN_Z,
        min_table_deg=0.0,
        fault=None,
    )


new_episode()


def packet_dropped_over_50mm(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
) -> bool:
    """True if newsletter_packet descends > 50 mm while neither gripper touches it, or falls off the station table."""
    if any_part_off_table(objects):
        return True
    if gripper_touches_packet(objects, tcps):
        return False
    pkt = objects["newsletter_packet"]
    pz = float(pkt.pos[2])
    unheld_peak_z = max(float(_S.get("max_unheld_packet_z", PACKET_SPAWN_Z)), pz)
    return bool((unheld_peak_z - pz) > PACKET_DROP_MAX_M)


def advance(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
) -> str | None:
    """Evaluate latched faults once per control step (purely observational; never modifies joints or drives)."""
    emb = _embodiment()
    if emb is not None and tcps is None:
        t = float(emb._t)
        if _S["t"] is not None and t < _S["t"]:
            new_episode()
        if t == _S["t"]:
            return _S["fault"]
        _S["t"] = t

    _S["k"] += 1
    st = _S["k"]

    f_shift = folder_shift_m(objects)
    f_tilt = tilt_deg(objects["autofolder_housing"])
    t_deg = feed_table_angle_deg(objects)
    pz = float(objects["newsletter_packet"].pos[2])
    touched = gripper_touches_packet(objects, tcps)

    _S["peak_folder_shift_mm"] = max(float(_S["peak_folder_shift_mm"]), f_shift * 1000.0)
    _S["max_packet_z"] = max(float(_S["max_packet_z"]), pz)
    if touched:
        _S["max_unheld_packet_z"] = pz
    else:
        _S["max_unheld_packet_z"] = max(float(_S["max_unheld_packet_z"]), pz)
    _S["min_table_deg"] = min(float(_S["min_table_deg"]), t_deg)

    # 1. Latched fault: folder moved > 15 mm (or tipped > 8 deg):
    if (f_shift > FOLDER_SHIFT_MAX_M or f_tilt > FOLDER_TILT_MAX_DEG) and _S["fault"] is None:
        _S["fault"] = "folder_moved"
        print(
            f"AUTOFOLDER_FAULT folder_moved step={st} shift_mm={f_shift * 1000.0:.2f} tilt_deg={f_tilt:.2f}",
            flush=True,
        )

    # 2. Latched fault: packet dropped > 50 mm while neither gripper touches it:
    if packet_dropped_over_50mm(objects, tcps) and _S["fault"] is None:
        _S["fault"] = "packet_dropped"
        drop_mm = (float(_S["max_unheld_packet_z"]) - pz) * 1000.0
        print(
            f"AUTOFOLDER_FAULT packet_dropped step={st} drop_mm={drop_mm:.1f} pz_rel={pz - TOP:.3f}",
            flush=True,
        )

    # 3. Latched fault: feed table pushed past its stop (< -16.5 deg):
    if t_deg < TABLE_STOP_DEG and _S["fault"] is None:
        _S["fault"] = "table_pushed_past_stop"
        print(
            f"AUTOFOLDER_FAULT table_pushed_past_stop step={st} table_deg={t_deg:.2f}",
            flush=True,
        )

    if emb is not None and (st == 1 or st % 10 == 0):
        pkt = objects["newsletter_packet"]
        stop_gap, lat_dx, deck_dz = packet_offsets_in_table_m(objects)
        w_gap = packet_wheel_gap_m(objects)
        clr_mm = min_tcp_clearance_m(objects) * 1000.0
        print(
            f"AUTOFOLDER_STEP step={st:3d} t={st / CONTROL_HZ:5.2f}s "
            f"table_deg={t_deg:+6.2f}(clamp={int(feed_table_clamped(objects))}) "
            f"pkt=({pkt.pos[0]:.3f},{pkt.pos[1]:+.3f},z_rel={pkt.pos[2] - TOP:.3f},"
            f"stop_mm={stop_gap*1000:.1f},lat_mm={lat_dx*1000:.1f},deck_mm={deck_dz*1000:.1f},"
            f"wheel_mm={w_gap*1000:.1f},yaw={packet_yaw_deg(objects):.1f}deg,"
            f"tilt={packet_table_tilt_deg(objects):.1f}deg) "
            f"fld_mm={f_shift*1000:.1f} tcp_clr_mm={clr_mm:5.1f} held={int(touched)} fault={_S['fault']}",
            flush=True,
        )
    return _S["fault"]


def score(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
) -> float:
    """1.0: newsletter_packet leading edge within 5 mm of the retarder stop, yaw <= 5 deg (and flat on the feed table
    within 4 deg), feed_table released and clamping the packet against the overhead feed wheel (-7.5 to -3.5 deg),
    both grippers >= 30 mm clear, and all objects at rest (held 1 s via SUCCESS_HOLD_S).
    0.5: folder stable on the table and either (a) newsletter_packet loaded at the retarder stop and clamped while a
    gripper is still within 30 mm, or (b) feed_table depressed (<= -12 deg) and newsletter_packet slid into the guide
    channel within 15 mm of the retarder stop (yaw <= 5 deg) before releasing the feed table.
    0.0 otherwise or after any latched fault."""
    if advance(objects, tcps) is not None:
        return 0.0
    if not folder_stable_on_table(objects):
        return 0.0
    if tcps is None:
        tcps = tcp_points()
    if not tcps:
        return 0.0

    loaded_clamped = packet_loaded_and_clamped(objects)
    if loaded_clamped and all_resting(objects) and hands_clear(objects, tcps):
        return 1.0

    if loaded_clamped and objects["newsletter_packet"].resting():
        return 0.5

    stop_gap, lat_dx, deck_dz = packet_offsets_in_table_m(objects)
    in_channel_near_stop = (
        stop_gap <= 0.015
        and lat_dx <= PACKET_LATERAL_MAX_M
        and deck_dz <= 0.008
        and packet_square_in_guides(objects)
    )
    if float(_S["min_table_deg"]) <= DEPRESS_CLEAR_DEG and in_channel_near_stop:
        return 0.5

    return 0.0


def success(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
) -> bool:
    """The 1.0 condition (held for SUCCESS_HOLD_S = 1.0 s by the runner)."""
    return bool(score(objects, tcps) == 1.0)


def distance(objects: dict[str, ObjectState]) -> float:
    """Remaining normalized distance to retarder-stop seating and feed-wheel clamping (0.0 at goal)."""
    stop_gap, lat_dx, deck_dz = packet_offsets_in_table_m(objects)
    d_pos = (
        max(0.0, stop_gap - STOP_GAP_MAX_M)
        + max(0.0, lat_dx - PACKET_LATERAL_MAX_M)
        + max(0.0, deck_dz - PACKET_DECK_GAP_MAX_M)
    ) / 0.18
    ang = feed_table_angle_deg(objects)
    if ang < CLAMP_ANGLE_MIN_DEG:
        d_ang = (CLAMP_ANGLE_MIN_DEG - ang) / 15.0
    elif ang > CLAMP_ANGLE_MAX_DEG:
        d_ang = (ang - CLAMP_ANGLE_MAX_DEG) / 10.0
    else:
        d_ang = 0.0
    d_wheel = max(0.0, packet_wheel_gap_m(objects) - WHEEL_NIP_GAP_MAX_M) / 0.03
    return float((d_pos + d_ang + d_wheel) / 3.0)


# -- Open-loop bimanual demo -------------------------------------------------------------------------------------------
# Left arm: pitched -25 deg about Y, rolled -15 deg about X, fingers closed (forming a 20 x 40 mm flat presser pad)
# to press down on feed_table's left-side amber depress paddle at (0.255, +0.175), holding feed_table at ~-14.0 deg
# (above the -16.5 deg stop) while the right arm loads newsletter_packet.
GRIP_TABLE_L = grip_rot(pitch_deg=-25.0, roll_deg=-15.0)

# Right arm: rolled +73 deg about X so the 34 x 40 mm flat finger pads close vertically along Z/table-normal across
# the 20.0 mm thickness of newsletter_packet inside the 104 mm-wide central tray cutout, while the approach axis points
# along +Y at -17 deg (matching the depressed feed_table slope).
GRIP_PACKET_R = grip_rot(pitch_deg=0.0, roll_deg=73.0)
GRIP_PARK_R = grip_rot(pitch_deg=-25.0, roll_deg=30.0)
FINGER_PACKET_GRIP = -0.0060

PADDLE_X = X_CENTER
PADDLE_Y = +0.1750
PADDLE_ABOVE_Z = Z_PIVOT + 0.0550
PADDLE_TOUCH_Z = Z_PIVOT + 0.0080
PADDLE_DEPRESS_Z = Z_PIVOT - 0.0310  # depresses feed_table to ~-14.0 deg (safely above the -16.5 deg stop)

PKT_PRE_Y = -0.1700
PKT_PICK_Y = -0.1430
PKT_LIFT_Y = -0.1300
PKT_MID_Y = -0.0500
PKT_SEAT_Y = +0.0300                 # seats 140 mm packet's +y leading edge against retarder_stop_lip at y = +0.130 m
PKT_RETRACT_Y = -0.0800

PKT_PRE_Z = Z_PIVOT + 0.0140
PKT_PICK_Z = Z_PIVOT + 0.0100
PKT_LIFT_Z = Z_PIVOT + 0.0220        # lifts packet 10 mm clear of staging apron (< 50 mm drop limit)
PKT_MID_Z = Z_PIVOT + 0.0000
PKT_SEAT_Z = Z_PIVOT - 0.0220

PARK_LEFT = f3((0.185, +0.290, TOP + 0.180))
PARK_RIGHT = f3((0.195, -0.268, TOP + 0.180))


def demo() -> dict[str, list[Waypoint]]:
    """Left arm presses down on feed_table's left-side paddle to hold the spring-elevated feed table at ~-14.0 deg,
    opening a ~39 mm vertical gap under the overhead rubber feed wheel. While the left hand holds the feed table down,
    the right hand approaches newsletter_packet from -y with rolled jaws, pinches its 20 mm thickness inside the
    central finger cutout, slides it between the 223 mm side paper guides under the overhead feed wheel until its
    leading edge contacts the retarder stop, opens its jaws and retracts, and then the left hand lifts off the paddle
    so the passive upward spring clamps the stack against the feed wheel at -5.5 deg."""
    left = [
        Waypoint(0.9, f3((PADDLE_X, PADDLE_Y, PADDLE_ABOVE_Z)), FINGER_CLOSED, "l_above_paddle", GRIP_TABLE_L),
        Waypoint(0.9, f3((PADDLE_X, PADDLE_Y, PADDLE_TOUCH_Z)), FINGER_CLOSED, "l_touch_paddle", GRIP_TABLE_L),
        Waypoint(1.0, f3((PADDLE_X, PADDLE_Y, PADDLE_DEPRESS_Z)), FINGER_CLOSED, "l_depress_table", GRIP_TABLE_L),
        # Hold feed_table depressed at ~-14.0 deg while right arm slides newsletter_packet to the retarder stop (4.8 s):
        Waypoint(4.8, f3((PADDLE_X, PADDLE_Y, PADDLE_DEPRESS_Z)), FINGER_CLOSED, "l_hold_depressed", GRIP_TABLE_L),
        Waypoint(1.0, f3((PADDLE_X, PADDLE_Y, PADDLE_TOUCH_Z + 0.007)), FINGER_CLOSED, "l_release_table", GRIP_TABLE_L),
        Waypoint(0.9, f3((PADDLE_X, PADDLE_Y, PADDLE_ABOVE_Z)), FINGER_CLOSED, "l_rise_clear", GRIP_TABLE_L),
        Waypoint(1.0, PARK_LEFT, FINGER_OPEN, "l_park", GRIP_TABLE_L),
        Waypoint(1.5, PARK_LEFT, FINGER_OPEN, "l_done", GRIP_TABLE_L),
    ]

    right = [
        Waypoint(1.0, f3((X_CENTER, PKT_PRE_Y, PKT_PRE_Z)), FINGER_OPEN, "r_approach_packet", GRIP_PACKET_R),
        Waypoint(0.8, f3((X_CENTER, PKT_PICK_Y, PKT_PICK_Z)), FINGER_OPEN, "r_around_packet", GRIP_PACKET_R),
        Waypoint(0.8, f3((X_CENTER, PKT_PICK_Y, PKT_PICK_Z)), FINGER_PACKET_GRIP, "r_grasp_packet", GRIP_PACKET_R),
        # At t = 2.8 s, left arm has depressed feed_table to ~-14.0 deg, opening the nip under the feed wheel:
        Waypoint(0.9, f3((X_CENTER, PKT_LIFT_Y, PKT_LIFT_Z)), FINGER_PACKET_GRIP, "r_lift_align", GRIP_PACKET_R),
        Waypoint(1.2, f3((X_CENTER, PKT_MID_Y, PKT_MID_Z)), FINGER_PACKET_GRIP, "r_slide_into_guides", GRIP_PACKET_R),
        Waypoint(1.3, f3((X_CENTER, PKT_SEAT_Y, PKT_SEAT_Z)), FINGER_PACKET_GRIP, "r_seat_at_stop", GRIP_PACKET_R),
        Waypoint(0.7, f3((X_CENTER, PKT_SEAT_Y, PKT_SEAT_Z)), FINGER_OPEN, "r_release_packet", GRIP_PACKET_R),
        Waypoint(0.9, f3((X_CENTER, PKT_RETRACT_Y, PKT_PRE_Z)), FINGER_OPEN, "r_retract_clear", GRIP_PACKET_R),
        Waypoint(1.2, PARK_RIGHT, FINGER_OPEN, "r_park", GRIP_PARK_R),
    ]
    left_s = sum(w.duration for w in left)
    right_s = sum(w.duration for w in right)
    right.append(Waypoint(round(left_s - right_s, 6), PARK_RIGHT, FINGER_OPEN, "r_done", GRIP_PARK_R))
    return {"left": left, "right": right}


if __name__ == "__main__":
    for side, plan in demo().items():
        q = None
        for w in plan:
            q, pe, re = ik.solve(
                np.array(w.pos),
                ik.matrix_to_quat_xyzw(_autofolder_grip_rotation(w.yaw)),
                q_init=q,
                base_pos=station.BASES[side],
                iters=400,
            )
            m = np.degrees(np.minimum(q - ik.LOWER, ik.UPPER - q)).min()
            flag = "BAD" if pe > 0.002 or re > 0.02 or m < 15.0 else ""
            print(
                f"{side:5s} {w.label:22s} pos=({w.pos[0]:.3f},{w.pos[1]:+.3f},{w.pos[2] - TOP:.3f}) "
                f"pos_err={pe * 1000:.2f}mm rot={re:.4f} margin={m:.1f}deg {flag}"
            )
        print(side, "plan s", round(sum(w.duration for w in plan), 2))
