"""At a care bed: a fixed mattress block (0.50 x 0.90 x 0.18 m) lies on the table with its near edge at x 0.44 m,
centred at y 0.0, with a fixed steel latch keeper bracket at y +0.255 m carrying a 20 deg entry ramp and locking catch
lip. In front of the mattress hangs a 3.0 kg half side rail, 0.60 m long along y and 0.30 m tall, hinged along y at
x 0.42 m, 0.11 m above the table. The rail has a round 30 mm top tube, a bottom bar and three vertical posts. It starts
fully lowered (100 deg down toward -x, resting on its lower stop) on a free gravity hinge and swings upward to 0 deg
(upright, top tube 0.23 m above the mattress top).

A red spring-loaded ramped latch assembly (80 g) rides on the +y end of the rail on a vertical 15 mm slide with an
upward preload spring. As the rail reaches full height, the keeper's entry ramp depresses the latch tooth downward
against its spring until the tooth clears the keeper lip and snaps upward into the catch pocket, holding the rail
upright by contact alone.

The goal: raise the bed's side rail to its top position so the latch catches and holds it upright after release. If the
rail is ever lifted more than 50 mm above its lower stop and dropped back onto the lower stop, the trial scores 0.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402

import yam_arena.demo as _demo  # noqa: E402
from yam_arena import YAM_URDF, ik, station, urdf  # noqa: E402
from yam_arena import analysis as _an  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, ObjectState  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix, rpy_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
CONTROL_HZ = 20.0

INSTRUCTION = "Put the bed's side rail up."


class Grip(float):
    """A 3x3 TCP rotation matrix carried in Waypoint.yaw (a float field)."""

    def __new__(cls, m):
        obj = super().__new__(cls, 1.0)
        obj.m = np.asarray(m, float)
        return obj


_orig_grip_rotation = getattr(_demo, "_orig_grip_rotation", _demo.grip_rotation)
_demo._orig_grip_rotation = _orig_grip_rotation


def _rail_grip_rotation(y=0.0) -> np.ndarray:
    if hasattr(y, "m"):
        return y.m
    if isinstance(y, np.ndarray):
        return y
    return _orig_grip_rotation(float(y))


_demo.grip_rotation = _rail_grip_rotation
DOWN = _orig_grip_rotation(0.0)


def rx_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(deg))


def ry_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(deg))


def rz_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(deg))


def grip_rot(pitch_deg: float, jaw_rot_deg: float = 0.0, roll_deg: float = 0.0) -> Grip:
    """TCP rotation pitched about world y.
    - `jaw_rot_deg = 0`: jaws close along world y.
    - `jaw_rot_deg = -90`: jaws close in the world x-z plane, across the 30 mm top tube.
    """
    return Grip(ry_mat(pitch_deg) @ rx_mat(roll_deg) @ DOWN @ rz_mat(jaw_rot_deg))


# -- Fast URDF forward kinematics for the arm chain (same results as urdf.forward_kinematics, fewer allocations) ------
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


# -- Geometry -----------------------------------------------------------------------------------------------------------
# Mattress: x 0.44 .. 0.94 m, y -0.45 .. +0.45 m, z TOP .. TOP + 0.18 m, plus fixed steel latch keeper at y = +0.255 m.
MATTRESS_NEAR_X = 0.44
MATTRESS_FAR_X = 0.94
MATTRESS_SIZE = (MATTRESS_FAR_X - MATTRESS_NEAR_X, 0.90, 0.18)
MATTRESS_POS = (0.5 * (MATTRESS_NEAR_X + MATTRESS_FAR_X), 0.0, TOP + 0.5 * MATTRESS_SIZE[2])
MATTRESS_COLOR = (0.76, 0.82, 0.90)
KEEPER_COLOR = (0.52, 0.56, 0.62)

# Hinge axis along y at x 0.42 m, 0.11 m above the table:
HINGE_X = 0.42
HINGE_Z_REL = 0.11
HINGE_WORLD = np.array([HINGE_X, 0.0, TOP + HINGE_Z_REL], dtype=float)

# Rail: 0.025 (x) x 0.60 (y) x 0.30 (z) m, 3.0 kg.
RAIL_THICK_X = 0.025
RAIL_LEN_Y = 0.60
RAIL_HEIGHT_Z = 0.30
RAIL_SIZE = (RAIL_THICK_X, RAIL_LEN_Y, RAIL_HEIGHT_Z)
RAIL_MASS = 3.0
RAIL_POS = (HINGE_X, 0.0, TOP + HINGE_Z_REL + 0.5 * RAIL_HEIGHT_Z)
RAIL_HINGE_ANCHOR = (0.0, 0.0, -0.5 * RAIL_HEIGHT_Z)  # hinge in the rail's local frame
# Center of mass shifted 6 mm toward -x so gravity always pulls an unlatched rail down toward its lower stop even at 0 deg:
RAIL_COM = (-0.006, 0.0, 0.0)

# The top bar is a round 30 mm tube (a 12-gon of six rotated boxes) so it can turn in the pads while the wrist keeps
# one orientation, as a tubular rail top does in a hand.
TOP_BAR_DIAM = 0.030
TOP_BAR_SIDES = 12
TOP_BAR_R_OUT = 0.5 * TOP_BAR_DIAM / math.cos(math.pi / TOP_BAR_SIDES)  # 15.5 mm to the 12-gon's corners
TOP_BAR_PIECE = (TOP_BAR_DIAM, 0.60, TOP_BAR_DIAM * math.tan(math.pi / TOP_BAR_SIDES))
TOP_BAR_PIECE_DEG = tuple(k * 360.0 / TOP_BAR_SIDES for k in range(TOP_BAR_SIDES // 2))
TOP_BAR_SIZE = (2 * TOP_BAR_R_OUT, 0.60, 2 * TOP_BAR_R_OUT)  # bounding box of the tube
TOP_BAR_Z_FROM_HINGE = 0.285
BOT_BAR_SIZE = (0.025, 0.60, 0.025)
BOT_BAR_Z_FROM_HINGE = 0.0125
POST_SIZE = (0.025, 0.025, 0.245)
POST_Z_FROM_HINGE = 0.1475
RAIL_COLOR = (0.82, 0.84, 0.88)
TOP_BAR_COLOR = (0.72, 0.76, 0.82)

# Keeper bracket geometry at y = +0.255 m (fixed on mattress, inside the open window between middle post y=0 and left post y=+0.2875):
# A single uninterrupted 32 mm ramp (tilted +20.0 deg about Y, bottom-right tip at x = 0.4495 m, z = TOP + 0.3565 m)
# paired with an upper locking lip whose bottom sits 3.5 mm above the ramp tip so no seam exists on the ramp underside.
KEEPER_Y = 0.255
KEEPER_WIDTH_Y = 0.024
KEEPER_RAMP_DEG = 20.0
KEEPER_RAMP_SIZE = (0.032, KEEPER_WIDTH_Y, 0.008)
_KEEPER_RAMP_TIP = np.array([0.4495, KEEPER_Y, TOP + 0.3565], dtype=float)
KEEPER_RAMP_CENTER_WORLD = _KEEPER_RAMP_TIP + ry_mat(KEEPER_RAMP_DEG) @ np.array([-0.016, 0.0, +0.004], dtype=float)
KEEPER_LIP_X = 0.4510  # vertical +x locking wall that catches the latch tooth once rail <= 0.6 deg
KEEPER_LIP_SIZE = (0.006, KEEPER_WIDTH_Y, 0.008)
KEEPER_LIP_CENTER_WORLD = np.array([0.4480, KEEPER_Y, TOP + 0.3640], dtype=float)
KEEPER_ROOF_SIZE = (0.032, KEEPER_WIDTH_Y, 0.008)
KEEPER_ROOF_CENTER_WORLD = np.array([0.4660, KEEPER_Y, TOP + 0.3720], dtype=float)
KEEPER_RISER_SIZE = (0.020, KEEPER_WIDTH_Y, 0.196)
KEEPER_RISER_CENTER_WORLD = np.array([0.4840, KEEPER_Y, TOP + 0.2780], dtype=float)

# Spring-loaded ramped latch assembly on the +y end of the rail:
# Hinge-relative centres of the 4 pieces at rest (dep = 0):
# 1. Front red release paddle: (-0.031, +0.285, +0.300), size (0.026, 0.015, 0.080) -> bounds [-0.044..0.047, 0.249..0.2925, 0.228..0.340]
# 2. Front bridge strap: (-0.028, +0.270, +0.234), size (0.016, 0.042, 0.012)
# 3. Rear catch arm: (+0.0055, +0.255, +0.234), size (0.083, 0.012, 0.012)
# 4. Single-box bevelled catch tooth tilted +24.0 deg (> 20.0 deg keeper ramp so contact is strictly face-to-face
#    with zero internal seam), top-left locking corner at (dx_h = +0.0340, dy_h = +0.255, dz_h = +0.2550).
LATCH_PADDLE_SIZE = (0.026, 0.015, 0.080)
LATCH_STRAP_SIZE = (0.016, 0.042, 0.012)
LATCH_ARM_SIZE = (0.083, 0.012, 0.012)
LATCH_TOOTH_DEG = 24.0
LATCH_TOOTH_SIZE = (0.012, 0.012, 0.018)
_TOOTH_TOP_LEFT_H = np.array([+0.0340, KEEPER_Y, +0.2550], dtype=float)
_TOOTH_CENTER_H = _TOOTH_TOP_LEFT_H + ry_mat(LATCH_TOOTH_DEG) @ np.array([+0.006, 0.0, -0.009], dtype=float)

# LATCH_OFFSET_FROM_HINGE_REST is the exact geometric bounding-box centre of the 4-piece latch assembly in the rail
# frame, so Arena's `get_centroid_w("latch")` equals the body origin `T_W_F` with zero local AABB offset.
LATCH_OFFSET_FROM_HINGE_REST = np.array([+0.0015, +0.27075, +0.2840], dtype=float)
LATCH_SIZE = (0.091, 0.0435, 0.112)
LATCH_MASS = 0.08
LATCH_POS = (
    float(HINGE_WORLD[0] + LATCH_OFFSET_FROM_HINGE_REST[0]),
    float(HINGE_WORLD[1] + LATCH_OFFSET_FROM_HINGE_REST[1]),
    float(HINGE_WORLD[2] + LATCH_OFFSET_FROM_HINGE_REST[2]),
)
LATCH_COLOR = (0.88, 0.26, 0.22)
LATCH_TOOTH_COLOR = (0.78, 0.22, 0.18)

LATCH_PADDLE_OFFSET = tuple(float(v) for v in (np.array([-0.031, +0.285, +0.300]) - LATCH_OFFSET_FROM_HINGE_REST))
LATCH_STRAP_OFFSET = tuple(float(v) for v in (np.array([-0.028, +0.270, +0.234]) - LATCH_OFFSET_FROM_HINGE_REST))
LATCH_ARM_OFFSET = tuple(float(v) for v in (np.array([+0.0055, +0.255, +0.234]) - LATCH_OFFSET_FROM_HINGE_REST))
LATCH_TOOTH_OFFSET = tuple(float(v) for v in (_TOOTH_CENTER_H - LATCH_OFFSET_FROM_HINGE_REST))
LATCH_TOOTH_LOCK_CORNER_LOCAL = _TOOTH_TOP_LEFT_H - LATCH_OFFSET_FROM_HINGE_REST

# Joints:
RAIL_DOWN_STOP_DEG = 100.0
RAIL_DAMPING = 0.0005          # N*m*s/deg: free gravity hinge, no spring drive, no runtime drive switching
LATCH_TRAVEL_MAX_M = 0.015     # latch slides up to 15 mm downward (-z in rail frame) when depressed by the keeper ramp
# Preloaded upward spring (target +15 mm above the upper stop): 1.2 N upward at rest (1.5x the 80 g latch weight),
# 2.4 N when depressed 15 mm, easily depressed by the 3.0 kg rail's top bar and snapping up cleanly into the catch pocket.
LATCH_STIFFNESS = 80.0         # N/m
LATCH_PRELOAD_M = 0.015
LATCH_DAMPING = 3.0            # N*s/m

MATTRESS_COMPOUND = rx.Compound(
    pieces=(
        rx.Piece(size=MATTRESS_SIZE, offset=(0.0, 0.0, 0.0), color=MATTRESS_COLOR),
        rx.Piece(
            size=KEEPER_RAMP_SIZE,
            offset=tuple(float(v) for v in (KEEPER_RAMP_CENTER_WORLD - np.asarray(MATTRESS_POS, float))),
            color=KEEPER_COLOR,
            rot_xyzw=(
                0.0,
                math.sin(math.radians(KEEPER_RAMP_DEG) / 2.0),
                0.0,
                math.cos(math.radians(KEEPER_RAMP_DEG) / 2.0),
            ),
        ),
        rx.Piece(
            size=KEEPER_LIP_SIZE,
            offset=tuple(float(v) for v in (KEEPER_LIP_CENTER_WORLD - np.asarray(MATTRESS_POS, float))),
            color=KEEPER_COLOR,
        ),
        rx.Piece(
            size=KEEPER_ROOF_SIZE,
            offset=tuple(float(v) for v in (KEEPER_ROOF_CENTER_WORLD - np.asarray(MATTRESS_POS, float))),
            color=KEEPER_COLOR,
        ),
        rx.Piece(
            size=KEEPER_RISER_SIZE,
            offset=tuple(float(v) for v in (KEEPER_RISER_CENTER_WORLD - np.asarray(MATTRESS_POS, float))),
            color=KEEPER_COLOR,
        ),
    )
)

# Rail compound pieces in the rail's local frame (z_local = z_from_hinge - 0.15):
RAIL_COMPOUND = rx.Compound(
    pieces=(
        *(
            rx.Piece(
                size=TOP_BAR_PIECE,
                offset=(0.0, 0.0, TOP_BAR_Z_FROM_HINGE - 0.15),
                color=TOP_BAR_COLOR,
                rot_xyzw=(0.0, math.sin(math.radians(d) / 2), 0.0, math.cos(math.radians(d) / 2)),
            )
            for d in TOP_BAR_PIECE_DEG
        ),
        rx.Piece(size=BOT_BAR_SIZE, offset=(0.0, 0.0, BOT_BAR_Z_FROM_HINGE - 0.15), color=RAIL_COLOR),
        rx.Piece(size=POST_SIZE, offset=(0.0, +0.2875, POST_Z_FROM_HINGE - 0.15), color=RAIL_COLOR),
        rx.Piece(size=POST_SIZE, offset=(0.0, 0.0, POST_Z_FROM_HINGE - 0.15), color=RAIL_COLOR),
        rx.Piece(size=POST_SIZE, offset=(0.0, -0.2875, POST_Z_FROM_HINGE - 0.15), color=RAIL_COLOR),
    ),
    com=RAIL_COM,
)

LATCH_COMPOUND = rx.Compound(
    pieces=(
        rx.Piece(size=LATCH_PADDLE_SIZE, offset=LATCH_PADDLE_OFFSET, color=LATCH_COLOR),
        rx.Piece(size=LATCH_STRAP_SIZE, offset=LATCH_STRAP_OFFSET, color=LATCH_COLOR),
        rx.Piece(size=LATCH_ARM_SIZE, offset=LATCH_ARM_OFFSET, color=LATCH_COLOR),
        rx.Piece(
            size=LATCH_TOOTH_SIZE,
            offset=LATCH_TOOTH_OFFSET,
            color=LATCH_TOOTH_COLOR,
            rot_xyzw=(
                0.0,
                math.sin(math.radians(LATCH_TOOTH_DEG) / 2.0),
                0.0,
                math.cos(math.radians(LATCH_TOOTH_DEG) / 2.0),
            ),
        ),
    )
)

MATTRESS_SPEC = Box("mattress", pos=MATTRESS_POS, size=MATTRESS_SIZE, color=MATTRESS_COLOR)
RAIL_SPEC = Box("rail", pos=RAIL_POS, size=RAIL_SIZE, mass=RAIL_MASS, color=RAIL_COLOR)
LATCH_SPEC = Box("latch", pos=LATCH_POS, size=LATCH_SIZE, mass=LATCH_MASS, color=LATCH_COLOR)

OBJECTS = [MATTRESS_SPEC, RAIL_SPEC, LATCH_SPEC]

# Lowered position is -100 deg about +Y; rail starts DOWN on its lower stop with stiffness = 0.0 (pure gravity hinge).
RAIL_JOINT = rx.Revolute(
    anchor=RAIL_HINGE_ANCHOR,
    axis="Y",
    lower=-RAIL_DOWN_STOP_DEG,
    upper=0.0,
    target=0.0,
    stiffness=0.0,
    damping=RAIL_DAMPING,
    initial=-RAIL_DOWN_STOP_DEG,
)

# Latch on the rail: slides vertically along Z in the rail frame (-0.015 m depressed .. 0.0 m rest).
LATCH_JOINT = rx.Prismatic(
    anchor=(0.0, 0.0, 0.0),
    axis="Z",
    lower=-LATCH_TRAVEL_MAX_M,
    upper=0.0,
    target=LATCH_PRELOAD_M,
    stiffness=LATCH_STIFFNESS,
    damping=LATCH_DAMPING,
    parent="rail",
    initial=0.0,
)

rx.install(
    {
        "mattress": MATTRESS_COMPOUND,
        "rail": (RAIL_COMPOUND, RAIL_JOINT),
        "latch": (LATCH_COMPOUND, LATCH_JOINT),
    }
)

# -- Scene patch: 20 Hz control, high grip friction on the rail top bar and low-friction latch/keeper contact ----------
_SCENE_MOD = "yam_arena.scene"


def _patch_scene_physics(scene_mod) -> None:
    if getattr(scene_mod, "_raise_bed_rail_patched", False):
        return
    import isaaclab.sim as sim_utils

    orig_task_object = scene_mod.task_object
    orig_build_task_env = scene_mod.build_task_environment

    grip_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        friction_combine_mode="max",
        restitution=0.0,
    )
    latch_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.02,
        dynamic_friction=0.02,
        friction_combine_mode="min",
        restitution=0.0,
    )

    def task_object(spec):
        obj = orig_task_object(spec)
        if spec.name == "rail":
            obj.object_cfg.spawn.physics_material = grip_mat
            obj.spawner_cfg.physics_material = grip_mat
        elif spec.name in ("latch", "mattress"):
            obj.object_cfg.spawn.physics_material = latch_mat
            obj.spawner_cfg.physics_material = latch_mat
        return obj

    def build_task_environment(embodiment, task, control_hz=None):
        return orig_build_task_env(embodiment, task, control_hz=control_hz or CONTROL_HZ)

    scene_mod.task_object = task_object
    scene_mod.build_task_environment = build_task_environment
    scene_mod._raise_bed_rail_patched = True


_orig_rx_patch = getattr(rx, "_orig_rx_patch", rx._patch)
rx._orig_rx_patch = _orig_rx_patch


def _raise_bed_rail_rx_patch(scene_mod) -> None:
    _orig_rx_patch(scene_mod)
    _patch_scene_physics(scene_mod)


rx._patch = _raise_bed_rail_rx_patch
if _SCENE_MOD in sys.modules:
    _raise_bed_rail_rx_patch(sys.modules[_SCENE_MOD])


# -- Scoring constants --------------------------------------------------------------------------------------------------
TOP_POS_TOL_M = 0.005          # 5 mm: rail top tube centreline within 5 mm of its 0 deg top position
UP_MAX_DEG = float(math.degrees(2.0 * math.asin(TOP_POS_TOL_M / (2.0 * TOP_BAR_Z_FROM_HINGE))))  # ~1.005 deg (5 mm arc)
DROP_HEIGHT_ARM_M = 0.050      # 50 mm: once the rail top bar has risen > 50 mm above its lower stop...
DROP_STOP_TOL_M = 0.005        # ...returning within 5 mm of the lower stop latches rail_dropped
STILL_WINDOW_STEPS = 20        # 1.0 s window at 20 Hz
STILL_ANGLE_TOL = 0.5          # deg; rail moved at most this much over the 1.0 s window
LATCH_CAUGHT_MAX_M = 0.003     # m; latch returned to within 3 mm of its upper stop inside the keeper pocket
GRIPPER_CLEAR = 0.040          # m; both TCPs at least 40 mm from every rail and latch collider


# -- State readout ------------------------------------------------------------------------------------------------------
def rail_angle_and_speed_deg(rail_state: ObjectState) -> tuple[float, float]:
    """`(rail_angle_deg, physx_angular_speed_deg_s)`; the angle is 0 deg up .. 100 deg down toward -x."""
    r = quat_xyzw_to_matrix(np.asarray(rail_state.quat_xyzw, dtype=float))
    # The rail's local +z at angle a toward -x is (-sin a, 0, cos a):
    ang_deg = float(math.degrees(math.atan2(-r[0, 2], r[2, 2])))
    omega_deg_s = float(math.degrees(abs(float(rail_state.ang_vel[1]))))
    return ang_deg, omega_deg_s


def top_bar_point_from_rail_state(rail_state: ObjectState, y: float = 0.0) -> np.ndarray:
    """World point on the top tube's centreline at coordinate `y` along the rail."""
    r_rail = quat_xyzw_to_matrix(np.asarray(rail_state.quat_xyzw, dtype=float))
    p_rail = np.asarray(rail_state.pos, dtype=float)
    local_bar = np.array([0.0, y, TOP_BAR_Z_FROM_HINGE - 0.15], dtype=float)
    return p_rail + r_rail @ local_bar


def top_bar_Nominal_point(rail_angle_deg: float, y: float = 0.0) -> np.ndarray:
    """World point on the top tube's centreline at a nominal rail angle (0 deg = upright top, 100 deg = lower stop)."""
    r_rail = ry_mat(-rail_angle_deg)
    return HINGE_WORLD + r_rail @ np.array([0.0, y, TOP_BAR_Z_FROM_HINGE], dtype=float)


def top_bar_error_from_up_m(rail_state: ObjectState) -> float:
    """3D Euclidean distance (m) of the rail's top tube centreline from its 0 deg full-up position."""
    return float(np.linalg.norm(top_bar_point_from_rail_state(rail_state, 0.0) - top_bar_Nominal_point(0.0, 0.0)))


def top_bar_height_above_lower_stop_m(rail_state: ObjectState) -> float:
    """Vertical height (m) of the rail's top tube centreline above its 100 deg lower stop."""
    z_now = float(top_bar_point_from_rail_state(rail_state, 0.0)[2])
    z_down = float(top_bar_Nominal_point(RAIL_DOWN_STOP_DEG, 0.0)[2])
    return z_now - z_down


def latch_depression_m(rail_state: ObjectState, latch_state: ObjectState) -> float:
    """Downward latch travel (m, along -z in the rail frame; 0 at engaged upper stop, up to 0.015 depressed)."""
    r_rail = quat_xyzw_to_matrix(np.asarray(rail_state.quat_xyzw, dtype=float))
    p_rail = np.asarray(rail_state.pos, dtype=float)
    p_latch = np.asarray(latch_state.pos, dtype=float)
    rest_in_rail = LATCH_OFFSET_FROM_HINGE_REST - np.array([0.0, 0.0, 0.15], dtype=float)
    rel_in_rail = r_rail.T @ (p_latch - p_rail)
    return float(max(0.0, -(rel_in_rail[2] - rest_in_rail[2])))


def latch_tooth_lock_x_world(rail_state: ObjectState, latch_state: ObjectState) -> float:
    """World x coordinate (m) of the latch catch tooth's top-left locking corner."""
    r_latch = quat_xyzw_to_matrix(np.asarray(latch_state.quat_xyzw, dtype=float))
    p_latch = np.asarray(latch_state.pos, dtype=float)
    return float((p_latch + r_latch @ LATCH_TOOTH_LOCK_CORNER_LOCAL)[0])


def bar_point_from_rail_state(rail_state: ObjectState) -> np.ndarray:
    """World point on the top tube's centreline at the right hand's grip (y = -0.05 m)."""
    return top_bar_point_from_rail_state(rail_state, BAR_Y_GRIP)


def scene_boxes_from_states(
    rail_state: ObjectState, latch_state: ObjectState
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """`{name: (centre, rotation, half_extents)}` of the mattress, keeper, every rail piece and latch piece in world."""
    r_rail = quat_xyzw_to_matrix(np.asarray(rail_state.quat_xyzw, dtype=float))
    p_rail = np.asarray(rail_state.pos, dtype=float)
    r_latch = quat_xyzw_to_matrix(np.asarray(latch_state.quat_xyzw, dtype=float))
    p_latch = np.asarray(latch_state.pos, dtype=float)
    return {
        "mattress": (np.asarray(MATTRESS_POS, float), np.eye(3), np.asarray(MATTRESS_SIZE, float) / 2.0),
        "keeper_ramp": (
            KEEPER_RAMP_CENTER_WORLD.copy(),
            ry_mat(KEEPER_RAMP_DEG),
            np.asarray(KEEPER_RAMP_SIZE, float) / 2.0,
        ),
        "keeper_lip": (KEEPER_LIP_CENTER_WORLD.copy(), np.eye(3), np.asarray(KEEPER_LIP_SIZE, float) / 2.0),
        "keeper_roof": (KEEPER_ROOF_CENTER_WORLD.copy(), np.eye(3), np.asarray(KEEPER_ROOF_SIZE, float) / 2.0),
        "keeper_riser": (KEEPER_RISER_CENTER_WORLD.copy(), np.eye(3), np.asarray(KEEPER_RISER_SIZE, float) / 2.0),
        **{
            f"rail_top_{i}": (
                p_rail + r_rail @ np.array([0.0, 0.0, TOP_BAR_Z_FROM_HINGE - 0.15]),
                r_rail @ ry_mat(d),
                np.asarray(TOP_BAR_PIECE, float) / 2.0,
            )
            for i, d in enumerate(TOP_BAR_PIECE_DEG)
        },
        "rail_bot": (
            p_rail + r_rail @ np.array([0.0, 0.0, BOT_BAR_Z_FROM_HINGE - 0.15]),
            r_rail,
            np.asarray(BOT_BAR_SIZE, float) / 2.0,
        ),
        "rail_post_l": (
            p_rail + r_rail @ np.array([0.0, +0.2875, POST_Z_FROM_HINGE - 0.15]),
            r_rail,
            np.asarray(POST_SIZE, float) / 2.0,
        ),
        "rail_post_m": (
            p_rail + r_rail @ np.array([0.0, 0.0, POST_Z_FROM_HINGE - 0.15]),
            r_rail,
            np.asarray(POST_SIZE, float) / 2.0,
        ),
        "rail_post_r": (
            p_rail + r_rail @ np.array([0.0, -0.2875, POST_Z_FROM_HINGE - 0.15]),
            r_rail,
            np.asarray(POST_SIZE, float) / 2.0,
        ),
        "latch": (
            p_latch + r_latch @ np.asarray(LATCH_PADDLE_OFFSET, float),
            r_latch,
            np.asarray(LATCH_PADDLE_SIZE, float) / 2.0,
        ),
        "latch_strap": (
            p_latch + r_latch @ np.asarray(LATCH_STRAP_OFFSET, float),
            r_latch,
            np.asarray(LATCH_STRAP_SIZE, float) / 2.0,
        ),
        "latch_arm": (
            p_latch + r_latch @ np.asarray(LATCH_ARM_OFFSET, float),
            r_latch,
            np.asarray(LATCH_ARM_SIZE, float) / 2.0,
        ),
        "latch_tooth": (
            p_latch + r_latch @ np.asarray(LATCH_TOOTH_OFFSET, float),
            r_latch @ ry_mat(LATCH_TOOTH_DEG),
            np.asarray(LATCH_TOOTH_SIZE, float) / 2.0,
        ),
    }


def _box_hull(c: np.ndarray, r: np.ndarray, h: np.ndarray):
    v = np.array([c + r @ (h * np.array([a, b, d])) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    n = np.vstack([r.T, -r.T])
    return _an.Hull(vertices=v, planes=np.c_[n, n @ c + np.concatenate([h, h])])


def min_arm_clearance_mm(
    robot_state: dict,
    rail_state: ObjectState,
    latch_state: ObjectState,
    left_gripping_latch: bool,
    right_gripping_bar: bool,
) -> tuple[float, str]:
    """Minimum signed distance (mm) between the arms' URDF collision hulls and the mattress, keeper, rail and latch,
    leaving out the fingertips on the latch (left) or the top tube (right) while they grip it."""
    boxes = {k: _box_hull(*v) for k, v in scene_boxes_from_states(rail_state, latch_state).items()}
    worst = (float("inf"), "")
    for side in ("left", "right"):
        q6, finger, _, _ = robot_state[side]
        joints = {n: float(v) for n, v in zip(ARM_JOINTS, q6, strict=True)}
        joints["joint7"] = joints["joint8"] = float(finger)
        hulls = _an.world_hulls(_an.fk_link_poses_world(joints, station.BASES[side]))
        for bname, bh in boxes.items():
            skip = (side == "left" and left_gripping_latch and bname.startswith("latch")) or (
                side == "right" and right_gripping_bar and bname.startswith("rail_top")
            )
            arm = [(n, h) for n, h in hulls if not (skip and n in ("tip_left", "tip_right"))]
            d, pair = _an.hull_set_distance(arm, [(bname, bh)])
            if d * 1000.0 < worst[0]:
                worst = (d * 1000.0, f"{side}.{pair[0]}/{bname}")
    return worst


def _embodiment():
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and getattr(s, "_base", None) is not None and hasattr(s, "_ids") and hasattr(s, "_t"):
            return s
        f = f.f_back
    return None


def _robot_state(emb) -> dict:
    """Per arm `(q6, mean finger joint, tcp_pos, tcp_rot)` read from the embodiment."""
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
    """World TCP positions of both arms this step; `[]` offline."""
    emb = _embodiment()
    if emb is None:
        return []
    rs = _robot_state(emb)
    return [rs["left"][2], rs["right"][2]]


def min_tcp_clearance_m(rail_state: ObjectState, latch_state: ObjectState, tcps: list[np.ndarray]) -> float:
    """Smallest distance (m) from any TCP to any rail or latch collider; `-inf` for an empty TCP list."""
    if not tcps:
        return float("-inf")
    pts = np.asarray(tcps, dtype=float).reshape(-1, 3)
    boxes = scene_boxes_from_states(rail_state, latch_state)
    return float(
        min(
            _an.box_signed_distance(pts, c, r, 2.0 * h).min()
            for name, (c, r, h) in boxes.items()
            if not name.startswith(("mattress", "keeper"))
        )
    )


def released(objects: dict[str, ObjectState], tcps: list[np.ndarray] | None = None) -> bool:
    """Both TCPs at least GRIPPER_CLEAR (40 mm) from every rail and latch collider. Empty TCP list = not released."""
    if tcps is None:
        tcps = tcp_points()
    return min_tcp_clearance_m(objects["rail"], objects["latch"], tcps) >= GRIPPER_CLEAR


# -- Episode state: latched fault, peak lift height, angle window, diagnostics -----------------------------------------
_S: dict = {}


def new_episode() -> None:
    _S.clear()
    _S.update(
        t=None,
        k=0,
        prev_angle_deg=None,
        peak_lift_m=0.0,
        lifted_above_50mm=False,
        fault=None,
        peak_omega_deg_s=0.0,
        peak_latch_depress_mm=0.0,
        bar_grip_ref_dist_m=None,
        max_bar_slip_mm=0.0,
        min_clearance_mm=float("inf"),
        min_clearance_what="",
        angle_history=[],
    )


new_episode()


def advance(objects: dict[str, ObjectState]) -> str | None:
    """Update the latched drop fault and the angle window once per control step (every call offline)."""
    emb = _embodiment()
    if emb is not None:
        t = float(emb._t)
        if _S["t"] is not None and t < _S["t"]:
            new_episode()
        if t == _S["t"]:
            return _S["fault"]
        _S["t"] = t

    _S["k"] += 1
    st = _S["k"]
    t_s = st / CONTROL_HZ
    rail = objects["rail"]
    latch = objects["latch"]

    ang_deg, omega_physx = rail_angle_and_speed_deg(rail)
    omega_fd = abs(ang_deg - _S["prev_angle_deg"]) * CONTROL_HZ if _S["prev_angle_deg"] is not None else 0.0
    _S["prev_angle_deg"] = ang_deg
    omega = max(omega_physx, omega_fd)
    _S["peak_omega_deg_s"] = max(_S["peak_omega_deg_s"], omega)

    lift_m = top_bar_height_above_lower_stop_m(rail)
    _S["peak_lift_m"] = max(_S["peak_lift_m"], lift_m)
    if lift_m > DROP_HEIGHT_ARM_M:
        _S["lifted_above_50mm"] = True
    elif _S["lifted_above_50mm"] and lift_m <= DROP_STOP_TOL_M and _S["fault"] is None:
        _S["fault"] = "rail_dropped"
        print(
            f"RAIL_FAULT rail_dropped step={st} t={t_s:.2f}s rail_deg={ang_deg:.2f} "
            f"lift_mm={lift_m * 1000.0:.1f} peak_lift_mm={_S['peak_lift_m'] * 1000.0:.1f}",
            flush=True,
        )

    depress_m = latch_depression_m(rail, latch)
    depress_mm = depress_m * 1000.0
    _S["peak_latch_depress_mm"] = max(_S["peak_latch_depress_mm"], depress_mm)

    hist: list[float] = _S["angle_history"]
    hist.append(ang_deg)
    if len(hist) > STILL_WINDOW_STEPS + 1:
        del hist[0 : len(hist) - (STILL_WINDOW_STEPS + 1)]

    if emb is not None:
        _log_diagnostics(emb, objects, st, t_s, ang_deg, omega, depress_mm, lift_m)
    return _S["fault"]


def _log_diagnostics(emb, objects, st, t_s, ang_deg, omega, depress_mm, lift_m) -> None:
    """Bar slip and arm-link clearance: logged only, not scored."""
    rail, latch = objects["rail"], objects["latch"]
    rs = _robot_state(emb)
    r_tcp, l_f, r_f = rs["right"][2], rs["left"][1], rs["right"][1]
    tcp_to_bar_m = float(np.linalg.norm(r_tcp - bar_point_from_rail_state(rail)))
    if 3.8 <= t_s <= 9.7:
        if _S["bar_grip_ref_dist_m"] is None:
            _S["bar_grip_ref_dist_m"] = tcp_to_bar_m
        slip_mm = abs(tcp_to_bar_m - _S["bar_grip_ref_dist_m"]) * 1000.0
        _S["max_bar_slip_mm"] = max(_S["max_bar_slip_mm"], slip_mm)
    else:
        slip_mm = _S["max_bar_slip_mm"]
    clr_mm, clr_what = min_arm_clearance_mm(rs, rail, latch, False, 2.0 <= t_s <= 11.1)
    if clr_mm < _S["min_clearance_mm"]:
        _S["min_clearance_mm"], _S["min_clearance_what"] = clr_mm, clr_what
    if st == 1 or st % 10 == 0:
        tcp_clr_mm = min_tcp_clearance_m(rail, latch, [rs["left"][2], r_tcp]) * 1000.0
        up_err_mm = top_bar_error_from_up_m(rail) * 1000.0
        caught = int(latch_caught(objects))
        tooth_x = latch_tooth_lock_x_world(rail, latch)
        print(
            f"RAIL_STEP step={st:3d} t={t_s:5.2f}s "
            f"rail_deg={ang_deg:6.2f} up_err_mm={up_err_mm:5.2f} lift_mm={lift_m * 1000.0:5.1f} "
            f"omega={omega:6.2f}(peak={_S['peak_omega_deg_s']:6.2f}) "
            f"latch_dep_mm={depress_mm:5.2f}(peak={_S['peak_latch_depress_mm']:5.2f},tooth_x={tooth_x:.4f},caught={caught}) "
            f"r_tcp_bar_mm={tcp_to_bar_m * 1000.0:6.2f} slip_mm={slip_mm:5.2f}(max={_S['max_bar_slip_mm']:5.2f}) "
            f"f=({l_f * 1000:+.1f},{r_f * 1000:+.1f})mm "
            f"clr={clr_mm:+5.1f}mm(min={_S['min_clearance_mm']:+5.1f}mm,{_S['min_clearance_what']}) "
            f"tcp_clr={tcp_clr_mm:5.1f}mm moved_1s={window_motion_deg():5.2f}deg fault={_S['fault']}",
            flush=True,
        )


def window_motion_deg() -> float:
    """Range of the rail angle (deg) over the last 1.0 s; `inf` until a full window exists."""
    hist: list[float] = _S["angle_history"]
    if len(hist) < STILL_WINDOW_STEPS + 1:
        return float("inf")
    w = hist[-(STILL_WINDOW_STEPS + 1) :]
    return float(max(w) - min(w))


def rail_up_and_still(objects: dict[str, ObjectState]) -> bool:
    """Rail top bar within 5 mm of its 0 deg top position and moved <= 0.5 deg over the last 1.0 s."""
    rail = objects["rail"]
    ang_deg, _ = rail_angle_and_speed_deg(rail)
    return (
        0.0 <= ang_deg <= UP_MAX_DEG
        and top_bar_error_from_up_m(rail) <= TOP_POS_TOL_M
        and window_motion_deg() <= STILL_ANGLE_TOL
    )


def latch_caught(objects: dict[str, ObjectState]) -> bool:
    """Latch tooth has cleared the keeper locking lip (x >= 0.4510 m) and sprung back up into the catch pocket."""
    rail, latch = objects["rail"], objects["latch"]
    return (
        latch_depression_m(rail, latch) <= LATCH_CAUGHT_MAX_M
        and latch_tooth_lock_x_world(rail, latch) >= KEEPER_LIP_X - 0.0005
    )


def success(objects: dict[str, ObjectState], tcps: list[np.ndarray] | None = None) -> bool:
    advance(objects)
    if _S["fault"] is not None:
        return False
    return rail_up_and_still(objects) and latch_caught(objects) and released(objects, tcps)


def score(objects: dict[str, ObjectState]) -> float:
    """1.0 comes only from the runner's success ending (after success() has held for 1.0 s); partial credit is 0.0."""
    advance(objects)
    return 0.0


def distance(objects: dict[str, ObjectState]) -> float:
    return top_bar_error_from_up_m(objects["rail"])


# -- Open-loop demo -----------------------------------------------------------------------------------------------------
GRIP_LEFT = grip_rot(-60.0, 0.0)
# Right arm: pitch -60 deg, jaws across the 30 mm top tube (jaw_rot -90 deg keeps joint6 well inside its limits).
GRIP_RIGHT = grip_rot(-60.0, -90.0)

FINGER_BAR_OPEN = -0.026   # 52 mm opening: 11 mm clear per side of the 30 mm tube
FINGER_BAR_CLOSE = -0.012  # 24 mm target opening on 30 mm tube (width minus 6 mm, partial close per YAM rule)

# High parks that stay clear of the whole 0..100 deg rail sweep and out of the top camera's view of the rail:
PARK_LEFT = (0.20, +0.29, TOP + 0.44)
PARK_RIGHT = (0.20, -0.30, TOP + 0.44)
MID_PARK_RIGHT = (0.24, -0.32, TOP + 0.30)

# -4 mm along tool z centres the tube in the pads.
BAR_Y_GRIP = -0.05
BAR_TOOL_Z_OFFSET = -0.004


def bar_grip_point_world(rail_angle_deg: float) -> np.ndarray:
    """World point on the top tube's centreline at y = -0.05 m for a nominal rail angle (deg)."""
    r_rail = ry_mat(-rail_angle_deg)
    return HINGE_WORLD + r_rail @ np.array([0.0, BAR_Y_GRIP, TOP_BAR_Z_FROM_HINGE], dtype=float)


def bar_tcp(
    rail_angle_deg: float, backoff_m: float = 0.0, extra_x: float = 0.0, extra_z: float = 0.0
) -> tuple[float, float, float]:
    """Right TCP on the top tube at a nominal rail angle, backed off along tool z by `backoff_m`."""
    p = (
        bar_grip_point_world(rail_angle_deg)
        + GRIP_RIGHT.m @ np.array([0.0, 0.0, BAR_TOOL_Z_OFFSET + backoff_m])
        + np.array([extra_x, 0.0, extra_z], dtype=float)
    )
    return (float(p[0]), float(p[1]), float(p[2]))


def demo() -> dict[str, list[Waypoint]]:
    """The right hand approaches the lowered top tube at 100 deg, closes partway (24 mm on the 30 mm tube), swings the
    rail smoothly up through its 100 -> 0 deg arc with a +15 mm/+4 mm seat push at 0 deg to overcome wrist lag until the
    ramped spring latch clicks into the keeper pocket, opens its jaws, backs away and parks high."""
    right = [
        Waypoint(1.0, MID_PARK_RIGHT, FINGER_BAR_OPEN, "r_mid_in", GRIP_RIGHT),
        Waypoint(1.0, bar_tcp(100.0, -0.055), FINGER_BAR_OPEN, "r_pre_bar", GRIP_RIGHT),
        Waypoint(0.8, bar_tcp(100.0, 0.0), FINGER_BAR_OPEN, "r_at_bar", GRIP_RIGHT),
        Waypoint(1.0, bar_tcp(100.0, 0.0), FINGER_BAR_CLOSE, "r_close_bar", GRIP_RIGHT),  # t = 3.8 s
        Waypoint(0.4, bar_tcp(90.0, 0.0), FINGER_BAR_CLOSE, "r_raise_90", GRIP_RIGHT),
        Waypoint(0.4, bar_tcp(80.0, 0.0), FINGER_BAR_CLOSE, "r_raise_80", GRIP_RIGHT),
        Waypoint(0.4, bar_tcp(70.0, 0.0), FINGER_BAR_CLOSE, "r_raise_70", GRIP_RIGHT),
        Waypoint(0.4, bar_tcp(60.0, 0.0), FINGER_BAR_CLOSE, "r_raise_60", GRIP_RIGHT),
        Waypoint(0.4, bar_tcp(50.0, 0.0), FINGER_BAR_CLOSE, "r_raise_50", GRIP_RIGHT),
        Waypoint(0.4, bar_tcp(40.0, 0.0), FINGER_BAR_CLOSE, "r_raise_40", GRIP_RIGHT),
        Waypoint(0.4, bar_tcp(30.0, 0.0), FINGER_BAR_CLOSE, "r_raise_30", GRIP_RIGHT),
        Waypoint(0.4, bar_tcp(20.0, 0.0), FINGER_BAR_CLOSE, "r_raise_20", GRIP_RIGHT),
        Waypoint(0.5, bar_tcp(10.0, 0.0, extra_x=0.004), FINGER_BAR_CLOSE, "r_raise_10", GRIP_RIGHT),
        Waypoint(0.6, bar_tcp(3.0, 0.0, extra_x=0.009, extra_z=0.002), FINGER_BAR_CLOSE, "r_raise_3", GRIP_RIGHT),
        Waypoint(0.8, bar_tcp(0.0, 0.0, extra_x=0.015, extra_z=0.004), FINGER_BAR_CLOSE, "r_seat_0", GRIP_RIGHT),  # t = 8.9 s
        Waypoint(0.8, bar_tcp(0.0, 0.0, extra_x=0.015, extra_z=0.004), FINGER_BAR_CLOSE, "r_hold_latch", GRIP_RIGHT),  # t = 9.7 s
        Waypoint(0.6, bar_tcp(0.0, 0.0, extra_x=0.002), FINGER_BAR_OPEN, "r_open_bar", GRIP_RIGHT),  # t = 10.3 s
        Waypoint(0.8, bar_tcp(0.0, -0.055), FINGER_BAR_OPEN, "r_retreat_bar", GRIP_RIGHT),  # t = 11.1 s
        Waypoint(1.0, PARK_RIGHT, FINGER_OPEN, "r_park", GRIP_RIGHT),  # t = 12.1 s
        Waypoint(1.5, PARK_RIGHT, FINGER_OPEN, "r_done", GRIP_RIGHT),  # t = 13.6 s
    ]
    total_dur = sum(w.duration for w in right)
    left = [
        Waypoint(1.2, PARK_LEFT, FINGER_OPEN, "l_park", GRIP_LEFT),
        Waypoint(total_dur - 1.2, PARK_LEFT, FINGER_OPEN, "l_done", GRIP_LEFT),
    ]
    return {"left": left, "right": right}
