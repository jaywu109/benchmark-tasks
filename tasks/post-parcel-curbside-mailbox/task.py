"""On the station table: a 1:1 Gibraltar Mailboxes Elite Medium curbside mailbox (Model E1100B00, 175 mm wide x 225 mm
tall x 509 mm deep, powder-coated galvanized steel, fixed to the table at x = 0.300..0.809 m, y = -0.1075..+0.0675 m)
stands with its front mouth at x = 0.300 m facing the robot and its back extending to x = 0.809 m.

Hinged along the bottom sill of the mouth (x = 0.296 m, z = TOP + 0.014 m) is the front carrier service door
(175 mm wide x 196 mm tall, 190 g, with a rolled top pull flange), which starts open at -82 deg and swings up to 0 deg
vertical where a top roller-catch detent holds it shut. Mounted on the right side wall (y = -0.122 m, when facing the
mailbox from the front, per USPS-STD-7C Section 3.6) is the fluorescent orange-red outgoing carrier signal flag (170 mm
long, 45 g), which pivots about a horizontal pin at (0.355, -0.122, TOP + 0.135 m) from 0 deg (horizontal, pointing
forward toward -x) up to +90 deg (vertical upright) into a detent keeper.

To the left (+y side) of the mailbox at (0.260, +0.210, TOP + 0.0685 m) stands an outgoing prepaid USPS Small Flat Rate
Box (8-5/8 x 5-3/8 x 1-5/8 in, 219 x 137 x 41 mm, 0.25 kg, rigid cardboard, standing on its 219 x 41 mm spine).

The goal: place the parcel fully inside the mailbox past the front sill (x >= 0.300 m), close and latch the front door
(<= 3 deg from closed), raise the carrier signal flag upright (90 +/- 6 deg), and move both grippers at least 25 mm
clear with everything at rest for 1 s. Only the end state is scored. Forcing the door against the parcel (contact > 15 N at
any door angle), or dropping the parcel off the table latches a fault and scores 0 for the rest of the trial.
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
from yam_arena.task_file import TABLE_TOP, Box, ObjectState  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix, rpy_matrix  # noqa: E402

SCENE = "station"
TOP = float(TABLE_TOP[SCENE])
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
CONTROL_HZ = 20.0

INSTRUCTION = "Leave this parcel in the mailbox for the carrier to collect."


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


def _mailbox_grip_rotation(y=0.0) -> np.ndarray:
    if hasattr(y, "m"):
        return y.m
    if isinstance(y, np.ndarray):
        return y
    return _orig_grip_rotation(float(y))


_demo.grip_rotation = _mailbox_grip_rotation
DOWN = _orig_grip_rotation(0.0)


def rx_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(deg))


def ry_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(deg))


def rz_mat(deg: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(deg))


def grip_rot(pitch_deg: float, jaw_rot_deg: float = 0.0, roll_deg: float = 0.0) -> Grip:
    """TCP rotation pitched about world y."""
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
MAILBOX_CHARCOAL = (0.18, 0.20, 0.22)  # Gibraltar E1100B00 textured black/charcoal powder-coated galvanized steel
MAILBOX_BASE_WOOD = (0.48, 0.36, 0.24)  # lower mounting board / curb inside base
STEEL_CATCH = (0.74, 0.76, 0.79)        # zinc-plated top roller catch bracket
DOOR_CHARCOAL = (0.22, 0.24, 0.27)      # front carrier service door leaf and pull flange
FLAG_RED = (0.92, 0.16, 0.10)           # USPS-STD-7C high-contrast fluorescent orange-red signal flag
PARCEL_CARDBOARD = (0.86, 0.80, 0.68)   # USPS Small Flat Rate Box cardboard


# -- 1. Gibraltar E1100B00 Curbside Mailbox Body (175 x 225 x 509 mm, fixed to table) ----------------------------------
# Full 1:1 dimensions: 20.05 in (509 mm) deep along x, 6.9 in (175 mm) wide along y, 8.87 in (225 mm) tall along z.
# Front mouth sill at X_SILL = 0.300 m; back extends to X_BACK = 0.809 m.
X_SILL = 0.300
MAILBOX_LX = 0.509
MAILBOX_LY = 0.175
MAILBOX_H = 0.225
X_BACK = X_SILL + MAILBOX_LX  # 0.809 m
Y_MAILBOX = -0.020
MAILBOX_Y0 = Y_MAILBOX - 0.5 * MAILBOX_LY  # -0.1075 m (right side when facing mailbox from front)
MAILBOX_Y1 = Y_MAILBOX + 0.5 * MAILBOX_LY  # +0.0675 m (left side when facing mailbox from front)
WALL_T = 0.008
FLOOR_H = 0.026
INTERIOR_Y0 = MAILBOX_Y0 + WALL_T  # -0.0995 m (159 mm interior width)
INTERIOR_Y1 = MAILBOX_Y1 - WALL_T  # +0.0595 m
INTERIOR_CEIL_H = 0.213

MAILBOX_SIZE = (MAILBOX_LX, MAILBOX_LY, MAILBOX_H)  # (0.509, 0.175, 0.225)
MAILBOX_POS = f3((X_SILL + 0.5 * MAILBOX_LX, Y_MAILBOX, TOP + 0.5 * MAILBOX_H))  # (0.5545, -0.020, TOP + 0.1125)


def _mb_piece(lo, hi, color=MAILBOX_CHARCOAL) -> rx.Piece:
    """A Piece of the mailbox body from world (x0, y0, z0_above_top)..(x1, y1, z1_above_top)."""
    lo_w = (lo[0], lo[1], TOP + lo[2])
    hi_w = (hi[0], hi[1], TOP + hi[2])
    return rx.Piece(
        size=f3(h - l for l, h in zip(lo_w, hi_w)),
        offset=f3(0.5 * (l + h) - c for l, h, c in zip(lo_w, hi_w, MAILBOX_POS)),
        color=color,
    )


MAILBOX_PIECE_BOUNDS = (
    # Lower mounting board & galvanized floor bed (z = 0.000..0.026 m):
    ("mounting_base", (X_SILL, MAILBOX_Y0, 0.000), (X_BACK, MAILBOX_Y1, 0.016), MAILBOX_BASE_WOOD),
    ("floor_bed", (X_SILL, MAILBOX_Y0, 0.016), (X_BACK, MAILBOX_Y1, FLOOR_H), MAILBOX_CHARCOAL),
    # Left side wall (+y side, 8 mm thick, z = 0.026..0.213 m):
    ("left_wall", (X_SILL, INTERIOR_Y1, FLOOR_H), (X_BACK, MAILBOX_Y1, INTERIOR_CEIL_H), MAILBOX_CHARCOAL),
    # Right side wall (-y side, 8 mm thick, z = 0.026..0.213 m):
    ("right_wall", (X_SILL, MAILBOX_Y0, FLOOR_H), (X_BACK, INTERIOR_Y0, INTERIOR_CEIL_H), MAILBOX_CHARCOAL),
    # Stepped arch roof reaching full 225 mm height (z = 0.213..0.225 m):
    ("roof_deck", (X_SILL, MAILBOX_Y0, INTERIOR_CEIL_H), (X_BACK, MAILBOX_Y1, 0.220), MAILBOX_CHARCOAL),
    ("roof_crown", (X_SILL, MAILBOX_Y0 + 0.022, 0.220), (X_BACK, MAILBOX_Y1 - 0.022, MAILBOX_H), MAILBOX_CHARCOAL),
    # Rear end wall closing the tunnel at x = 0.801..0.809 m:
    ("rear_wall", (X_BACK - WALL_T, INTERIOR_Y0, FLOOR_H), (X_BACK, INTERIOR_Y1, INTERIOR_CEIL_H), MAILBOX_CHARCOAL),
    # Top front roller-catch keeper bracket at the upper centre of the mouth:
    ("top_catch", (X_SILL, Y_MAILBOX - 0.020, 0.207), (X_SILL + 0.016, Y_MAILBOX + 0.020, INTERIOR_CEIL_H), STEEL_CATCH),
)

MAILBOX_COMPOUND = rx.Compound(pieces=tuple(_mb_piece(lo, hi, col) for _, lo, hi, col in MAILBOX_PIECE_BOUNDS))
MAILBOX = Box("mailbox", pos=MAILBOX_POS, size=MAILBOX_SIZE, color=MAILBOX_CHARCOAL)


# -- 2. Bottom-Hinged Carrier Service Door (door, loose Box + rx.Compound + rx.Revolute) -------------------------------
# Zero pose (0 deg) is closed vertical at the front mouth (x = 0.292..0.300 m, y = -0.1075..+0.0675 m,
# z = TOP + 0.018..TOP + 0.214 m) with a rolled top pull flange (x = 0.287..0.292 m, z = TOP + 0.200..TOP + 0.214 m).
# Hinged along Y at world (0.296, -0.020, TOP + 0.014 m); spawns open at DOOR_OPEN_INIT_DEG = -82.0 deg.
DOOR_SIZE = (0.013, 0.175, 0.196)
DOOR_ZERO_POS = f3((0.2935, Y_MAILBOX, TOP + 0.116))
DOOR_MASS = 0.190
DOOR_HINGE_WORLD = (0.296, Y_MAILBOX, TOP + 0.014)
DOOR_ANCHOR = f3(h - c for h, c in zip(DOOR_HINGE_WORLD, DOOR_ZERO_POS))  # (+0.0025, 0.0, -0.102)

DOOR_OPEN_INIT_DEG = -82.0          # spawns open at -82 deg (82 deg from closed)
DOOR_CATCH_ENGAGE_DEG = 4.0         # within 4 deg of closed (angle >= -4 deg), top roller catch snaps/holds to 0 deg
DOOR_OPEN_STIFFNESS = 0.003         # N*m/deg light hold toward -82 deg while open so door rests on its stop
DOOR_CATCH_STIFFNESS = 0.016        # N*m/deg top roller catch holding door closed at 0 deg
DOOR_DAMPING = 0.0010               # N*m*s/deg


def _door_piece(lo, hi, color=DOOR_CHARCOAL) -> rx.Piece:
    lo_w = (lo[0], lo[1], TOP + lo[2])
    hi_w = (hi[0], hi[1], TOP + hi[2])
    return rx.Piece(
        size=f3(h - l for l, h in zip(lo_w, hi_w)),
        offset=f3(0.5 * (l + h) - c for l, h, c in zip(lo_w, hi_w, DOOR_ZERO_POS)),
        color=color,
    )


DOOR_PIECE_BOUNDS = (
    ("door_leaf", (0.292, MAILBOX_Y0, 0.018), (0.300, MAILBOX_Y1, 0.214), DOOR_CHARCOAL),
    ("door_flange", (0.287, Y_MAILBOX - 0.030, 0.200), (0.292, Y_MAILBOX + 0.030, 0.214), STEEL_CATCH),
)

DOOR_COMPOUND = rx.Compound(pieces=tuple(_door_piece(lo, hi, col) for _, lo, hi, col in DOOR_PIECE_BOUNDS))
DOOR = Box("door", pos=DOOR_ZERO_POS, size=DOOR_SIZE, mass=DOOR_MASS, color=DOOR_CHARCOAL)
DOOR_JOINT = rx.Revolute(
    anchor=DOOR_ANCHOR,
    axis="Y",
    lower=DOOR_OPEN_INIT_DEG,
    upper=0.0,
    target=DOOR_OPEN_INIT_DEG,
    stiffness=DOOR_OPEN_STIFFNESS,
    damping=DOOR_DAMPING,
    parent=None,
    initial=DOOR_OPEN_INIT_DEG,
)


# -- 3. Right-Side Carrier Signal Flag (flag, loose Box + rx.Compound + rx.Revolute) -----------------------------------
# Mounted on the right side wall (-y side when facing the mailbox from the front, per USPS-STD-7C Section 3.6).
# Zero pose (0 deg) is horizontal down (pointing forward toward -x, spanning x = 0.195..0.365 m, y = -0.130..-0.114 m,
# z = TOP + 0.114..TOP + 0.156 m). Pivots about +Y at world (0.355, -0.122, TOP + 0.135 m) from 0 deg to +90 deg.
Y_FLAG = -0.122  # 6.5 mm clear outside the right wall at -0.1075 m
X_FLAG_PIVOT = 0.355
Z_FLAG_PIVOT = TOP + 0.135
FLAG_SIZE = (0.170, 0.016, 0.042)
FLAG_ZERO_POS = f3((0.280, Y_FLAG, Z_FLAG_PIVOT))
FLAG_MASS = 0.045
FLAG_PIVOT_WORLD = (X_FLAG_PIVOT, Y_FLAG, Z_FLAG_PIVOT)
FLAG_ANCHOR = f3(p - c for p, c in zip(FLAG_PIVOT_WORLD, FLAG_ZERO_POS))  # (+0.075, 0.0, 0.0)

FLAG_UP_TARGET_DEG = 90.0           # upright vertical detent target (+90 deg)
FLAG_DETENT_SNAP_DEG = 68.0         # past 68 deg, detent snaps and holds flag at +90 deg
FLAG_DETENT_STIFFNESS = 0.012       # N*m/deg
FLAG_DAMPING = 0.0008               # N*m*s/deg


def _flag_piece(lo, hi, color=FLAG_RED) -> rx.Piece:
    lo_w = (lo[0], lo[1], TOP + lo[2])
    hi_w = (hi[0], hi[1], TOP + hi[2])
    return rx.Piece(
        size=f3(h - l for l, h in zip(lo_w, hi_w)),
        offset=f3(0.5 * (l + h) - c for l, h, c in zip(lo_w, hi_w, FLAG_ZERO_POS)),
        color=color,
    )


FLAG_PIECE_BOUNDS = (
    ("flag_stem", (0.255, Y_FLAG - 0.007, 0.128), (0.365, Y_FLAG + 0.007, 0.142), FLAG_RED),
    ("flag_paddle", (0.195, Y_FLAG - 0.008, 0.114), (0.255, Y_FLAG + 0.008, 0.156), FLAG_RED),
)

FLAG_COMPOUND = rx.Compound(pieces=tuple(_flag_piece(lo, hi, col) for _, lo, hi, col in FLAG_PIECE_BOUNDS))
FLAG = Box("flag", pos=FLAG_ZERO_POS, size=FLAG_SIZE, mass=FLAG_MASS, color=FLAG_RED)
FLAG_JOINT = rx.Revolute(
    anchor=FLAG_ANCHOR,
    axis="Y",
    lower=0.0,
    upper=92.0,
    target=0.0,
    stiffness=FLAG_DETENT_STIFFNESS,
    damping=FLAG_DAMPING,
    parent=None,
    initial=0.0,
)


# -- 4. Outgoing USPS Small Flat Rate Box Parcel (parcel, 219 x 137 x 41 mm, 0.25 kg) ----------------------------------
# Real dimensions: 8-5/8 x 5-3/8 x 1-5/8 in (219 x 137 x 41 mm), 0.25 kg, standing on its 219 x 41 mm spine to the left
# (+y side) of the mailbox at (0.260, +0.210, TOP + 0.0685 m).
PARCEL_SIZE = (0.219, 0.041, 0.137)
PARCEL_MASS = 0.250
PARCEL_START_POS = f3((0.260, +0.210, TOP + 0.5 * PARCEL_SIZE[2]))  # (0.260, +0.210, TOP + 0.0685)
PARCEL = Box("parcel", pos=PARCEL_START_POS, size=PARCEL_SIZE, mass=PARCEL_MASS, color=PARCEL_CARDBOARD)

OBJECTS = [MAILBOX, DOOR, FLAG, PARCEL]

rx.install(
    {
        "mailbox": MAILBOX_COMPOUND,
        "door": (DOOR_COMPOUND, DOOR_JOINT),
        "flag": (FLAG_COMPOUND, FLAG_JOINT),
    }
)


# -- Scene physics patch: 20 Hz control, contact sensors on door & parcel, runtime joint drive handles -----------------
_SCENE_MOD = "yam_arena.scene"
_RUNTIME_DRIVES: dict[str, Any] = {}
_CONTACT_VIEWS: dict[str, Any] = {}


def _set_door_drive(latched: bool) -> None:
    drive = _RUNTIME_DRIVES.get("door_drive")
    if drive is not None:
        if latched:
            drive.GetTargetPositionAttr().Set(0.0)
            drive.GetStiffnessAttr().Set(float(DOOR_CATCH_STIFFNESS))
        else:
            drive.GetTargetPositionAttr().Set(float(DOOR_OPEN_INIT_DEG))
            drive.GetStiffnessAttr().Set(float(DOOR_OPEN_STIFFNESS))


def _set_flag_target(target_deg: float) -> None:
    drive = _RUNTIME_DRIVES.get("flag_drive")
    if drive is not None:
        drive.GetTargetPositionAttr().Set(float(target_deg))


def _patch_scene_physics(scene_mod) -> None:
    if getattr(scene_mod, "_mailbox_patched", False):
        return
    import isaaclab.sim as sim_utils

    orig_task_object = scene_mod.task_object
    orig_build_task_env = scene_mod.build_task_environment

    grip_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.9,
        dynamic_friction=0.8,
        friction_combine_mode="max",
        restitution=0.0,
    )
    floor_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.35,
        dynamic_friction=0.30,
        friction_combine_mode="min",
        restitution=0.0,
    )

    def task_object(spec):
        obj = orig_task_object(spec)
        if spec.name == "mailbox":
            obj.object_cfg.spawn.physics_material = floor_mat
            obj.spawner_cfg.physics_material = floor_mat
        elif spec.name in ("door", "flag", "parcel"):
            obj.object_cfg.spawn.physics_material = grip_mat
            obj.spawner_cfg.physics_material = grip_mat
            obj.object_cfg.spawn.activate_contact_sensors = True
            obj.spawner_cfg.activate_contact_sensors = True
            prev_spawn = obj.object_cfg.spawn.func
            part_name = spec.name

            def _spawn_part(prim_path, cfg, *args, _prev=prev_spawn, _name=part_name, **kwargs):
                prim = _prev(prim_path, cfg, *args, **kwargs)
                from pxr import PhysxSchema, UsdPhysics

                PhysxSchema.PhysxContactReportAPI.Apply(prim)
                if _name in ("door", "flag"):
                    stage = prim.GetStage()
                    env_dir = prim_path.rsplit("/", 1)[0]
                    j_prim = stage.GetPrimAtPath(f"{env_dir}/{_name}_joint")
                    if j_prim.IsValid():
                        _RUNTIME_DRIVES[f"{_name}_drive"] = UsdPhysics.DriveAPI.Get(j_prim, "angular")
                _CONTACT_VIEWS.clear()
                return prim

            obj.object_cfg.spawn.func = _spawn_part
            obj.spawner_cfg.func = _spawn_part
        return obj

    def build_task_environment(embodiment, task, control_hz=None):
        return orig_build_task_env(embodiment, task, control_hz=control_hz or CONTROL_HZ)

    scene_mod.task_object = task_object
    scene_mod.build_task_environment = build_task_environment
    scene_mod._mailbox_patched = True


_orig_rx_patch = getattr(rx, "_orig_rx_patch", rx._patch)
rx._orig_rx_patch = _orig_rx_patch


def _mailbox_rx_patch(scene_mod) -> None:
    _orig_rx_patch(scene_mod)
    _patch_scene_physics(scene_mod)


rx._patch = _mailbox_rx_patch
if _SCENE_MOD in sys.modules:
    _mailbox_rx_patch(sys.modules[_SCENE_MOD])


# -- Scoring constants -------------------------------------------------------------------------------------------------
DOOR_CLOSED_MAX_DEG = 3.0       # deg: door closed and latched when <= 3.0 deg from closed (0 deg)
FLAG_UP_TOL_DEG = 6.0           # deg: flag upright when within 90 +/- 6 deg (84..96 deg)
MAX_DOOR_PARCEL_FORCE_N = 15.0  # N: forcing door against parcel (> 15 N contact at any door angle) latches fault
GRIPPER_CLEAR = 0.025           # m (25 mm): both grippers must be >= 25 mm clear of mailbox, door, flag, and parcel


# -- State readout helpers ---------------------------------------------------------------------------------------------
def _rot_y_deg(state: ObjectState) -> float:
    """Signed rotation angle (deg) about +Y from the zero pose."""
    r = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, dtype=float))
    return float(math.degrees(math.atan2(float(r[0, 2]), float(r[0, 0]))))


def door_open_deg(objects: dict[str, ObjectState]) -> float:
    """Angle (deg, 0.0 when closed vertical .. 82.0 when fully open) of the front door from its closed 0 deg pose."""
    return float(max(0.0, -_rot_y_deg(objects["door"])))


def flag_angle_deg(objects: dict[str, ObjectState]) -> float:
    """Angle (deg, 0.0 when horizontal down .. +90.0 when upright vertical) of the signal flag about +Y."""
    return float(_rot_y_deg(objects["flag"]))


def parcel_off_table(objects: dict[str, ObjectState]) -> bool:
    """True if the parcel has fallen below or outside the station table surface."""
    p = objects["parcel"]
    px, py, pz = (float(v) for v in p.pos)
    low_z = float(p.low[2])
    return bool(pz < TOP or low_z < TOP - 0.005 or not (0.02 <= px <= 0.80 and abs(py) <= 0.52))


def parcel_inside(objects: dict[str, ObjectState]) -> bool:
    """True when the parcel is completely inside the mailbox tunnel past the front sill (low_x >= X_SILL)."""
    if parcel_off_table(objects):
        return False
    p = objects["parcel"]
    lo_x, hi_x = float(p.low[0]), float(p.high[0])
    lo_y, hi_y = float(p.low[1]), float(p.high[1])
    lo_z, hi_z = float(p.low[2]), float(p.high[2])
    return bool(
        lo_x >= X_SILL - 0.002
        and hi_x <= X_BACK + 0.002
        and lo_y >= MAILBOX_Y0 - 0.002
        and hi_y <= MAILBOX_Y1 + 0.002
        and lo_z >= TOP + FLOOR_H - 0.005
        and hi_z <= TOP + MAILBOX_H + 0.002
    )


def door_closed(objects: dict[str, ObjectState]) -> bool:
    """True when the front door is closed and latched within DOOR_CLOSED_MAX_DEG (3.0 deg)."""
    return door_open_deg(objects) <= DOOR_CLOSED_MAX_DEG


def flag_upright(objects: dict[str, ObjectState]) -> bool:
    """True when the signal flag is upright within 90 +/- 6 deg."""
    return abs(flag_angle_deg(objects) - FLAG_UP_TARGET_DEG) <= FLAG_UP_TOL_DEG


def scene_boxes_from_states(
    objects: dict[str, ObjectState],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{name: (centre, rotation, half_extents)}`` of all mailbox pieces, door pieces, flag pieces, and parcel."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    p_mb = np.asarray(objects["mailbox"].pos, dtype=float)
    r_mb = quat_xyzw_to_matrix(np.asarray(objects["mailbox"].quat_xyzw, dtype=float))
    for pname, lo, hi, _ in MAILBOX_PIECE_BOUNDS:
        lo_w = np.array([lo[0], lo[1], TOP + lo[2]], dtype=float)
        hi_w = np.array([hi[0], hi[1], TOP + hi[2]], dtype=float)
        offset = 0.5 * (lo_w + hi_w) - np.asarray(MAILBOX_POS, dtype=float)
        half = 0.5 * (hi_w - lo_w)
        out[f"mb_{pname}"] = (p_mb + r_mb @ offset, r_mb, half)

    p_door = np.asarray(objects["door"].pos, dtype=float)
    r_door = quat_xyzw_to_matrix(np.asarray(objects["door"].quat_xyzw, dtype=float))
    for pname, lo, hi, _ in DOOR_PIECE_BOUNDS:
        lo_w = np.array([lo[0], lo[1], TOP + lo[2]], dtype=float)
        hi_w = np.array([hi[0], hi[1], TOP + hi[2]], dtype=float)
        offset = 0.5 * (lo_w + hi_w) - np.asarray(DOOR_ZERO_POS, dtype=float)
        half = 0.5 * (hi_w - lo_w)
        out[f"door_{pname}"] = (p_door + r_door @ offset, r_door, half)

    p_flag = np.asarray(objects["flag"].pos, dtype=float)
    r_flag = quat_xyzw_to_matrix(np.asarray(objects["flag"].quat_xyzw, dtype=float))
    for pname, lo, hi, _ in FLAG_PIECE_BOUNDS:
        lo_w = np.array([lo[0], lo[1], TOP + lo[2]], dtype=float)
        hi_w = np.array([hi[0], hi[1], TOP + hi[2]], dtype=float)
        offset = 0.5 * (lo_w + hi_w) - np.asarray(FLAG_ZERO_POS, dtype=float)
        half = 0.5 * (hi_w - lo_w)
        out[f"flag_{pname}"] = (p_flag + r_flag @ offset, r_flag, half)

    p_par = np.asarray(objects["parcel"].pos, dtype=float)
    r_par = quat_xyzw_to_matrix(np.asarray(objects["parcel"].quat_xyzw, dtype=float))
    out["parcel"] = (p_par, r_par, np.asarray(PARCEL_SIZE, dtype=float) / 2.0)
    return out


# -- Embodiment & PhysX RigidContactView force readout -----------------------------------------------------------------
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
    """Build PhysX RigidContactView between ``door`` and ``parcel`` once on the first step."""
    if _CONTACT_VIEWS.get("ready") or _CONTACT_VIEWS.get("error"):
        return _CONTACT_VIEWS
    try:
        try:
            import carb.settings

            carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        except Exception:
            pass

        sim_view = base.sim.physics_sim_view
        door_parcel_view = sim_view.create_rigid_contact_view(
            ["/World/envs/env_0/door"],
            filter_patterns=[["/World/envs/env_0/parcel"]],
            max_contact_data_count=16,
        )
        _CONTACT_VIEWS["door_parcel_view"] = door_parcel_view
        _CONTACT_VIEWS["ready"] = True
    except Exception as exc:
        _CONTACT_VIEWS["error"] = f"{type(exc).__name__}: {exc}"
        print(f"MAILBOX_CONTACT_VIEW_ERROR: {_CONTACT_VIEWS['error']}", flush=True)
    return _CONTACT_VIEWS


def read_door_parcel_contact_force_n(base: Any) -> float:
    """Read the contact force magnitude (N) between the front door and the parcel."""
    if base is None:
        return 0.0
    views = _ensure_contact_views(base)
    if views.get("error"):
        raise RuntimeError(views["error"])
    if not views.get("ready"):
        return 0.0
    dt = float(base.sim.get_physics_dt())
    dv = views.get("door_parcel_view")
    if dv is None or getattr(dv, "filter_count", 0) <= 0:
        return 0.0
    mat = _to_numpy(dv.get_contact_force_matrix(dt)).reshape(dv.sensor_count, dv.filter_count, 3)
    return float(np.linalg.norm(mat[0, 0]))


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
    """Smallest distance (m) from any TCP to the mailbox, door, flag, or parcel; ``-inf`` if ``tcps`` is empty."""
    if tcps is None:
        tcps = tcp_points()
    if not tcps:
        return float("-inf")
    part_sizes = (
        ("mailbox", MAILBOX_SIZE),
        ("door", DOOR_SIZE),
        ("flag", FLAG_SIZE),
        ("parcel", PARCEL_SIZE),
    )
    return float(
        min(box_distance(objects[name], size, np.asarray(p, dtype=float)) for name, size in part_sizes for p in tcps)
    )


def hands_clear(objects: dict[str, ObjectState], tcps: list[np.ndarray] | None = None) -> bool:
    """True when both grippers are at least GRIPPER_CLEAR (25 mm) from the mailbox, door, flag, and parcel."""
    return min_tcp_clearance_m(objects, tcps) >= GRIPPER_CLEAR


def all_resting(objects: dict[str, ObjectState]) -> bool:
    """True when the parcel, door, and flag are at rest."""
    return all(objects[k].resting() for k in ("parcel", "door", "flag"))


# -- Episode state: door top catch detent, flag upright detent, and latched faults -------------------------------------
_S: dict[str, Any] = {}


def new_episode() -> None:
    _S.clear()
    _S.update(
        t=None,
        k=0,
        door_latched=False,
        flag_target_deg=0.0,
        peak_door_parcel_force_n=0.0,
        peak_flag_deg=0.0,
        fault=None,
    )
    _set_door_drive(False)
    _set_flag_target(0.0)


new_episode()


def advance(
    objects: dict[str, ObjectState],
    *,
    door_parcel_force_n: float | None = None,
) -> str | None:
    """Update door and flag detents and evaluate latched faults once per control step."""
    emb = _embodiment()
    if emb is not None and door_parcel_force_n is None:
        t = float(emb._t)
        if _S["t"] is not None and t < _S["t"]:
            new_episode()
        if t == _S["t"]:
            return _S["fault"]
        _S["t"] = t

    _S["k"] += 1
    st = _S["k"]

    d_open = door_open_deg(objects)
    f_deg = flag_angle_deg(objects)
    _S["peak_flag_deg"] = max(float(_S["peak_flag_deg"]), f_deg)
    inside = parcel_inside(objects)

    # 1. Door top roller-catch detent:
    if d_open <= DOOR_CATCH_ENGAGE_DEG:
        _S["door_latched"] = True
    elif d_open >= DOOR_CATCH_ENGAGE_DEG + 5.0:
        _S["door_latched"] = False
    _set_door_drive(bool(_S["door_latched"]))

    # 2. Flag upright detent keeper:
    if f_deg >= FLAG_DETENT_SNAP_DEG:
        _S["flag_target_deg"] = FLAG_UP_TARGET_DEG
    elif f_deg <= FLAG_DETENT_SNAP_DEG - 10.0:
        _S["flag_target_deg"] = 0.0
    _set_flag_target(float(_S["flag_target_deg"]))

    # 3. Latched fault: parcel off the table:
    if parcel_off_table(objects) and _S["fault"] is None:
        _S["fault"] = "parcel_off_table"
        print(f"MAILBOX_FAULT parcel_off_table step={st} pos={objects['parcel'].pos}", flush=True)

    # 4. Latched fault: door forced against the parcel (contact > 15 N at any door angle):
    base = emb._base if emb is not None else None
    f_n = (
        float(door_parcel_force_n)
        if door_parcel_force_n is not None
        else read_door_parcel_contact_force_n(base)
    )
    _S["peak_door_parcel_force_n"] = max(float(_S["peak_door_parcel_force_n"]), f_n)
    if f_n > MAX_DOOR_PARCEL_FORCE_N and _S["fault"] is None:
        _S["fault"] = "door_forced_on_parcel"
        print(
            f"MAILBOX_FAULT door_forced_on_parcel step={st} door_deg={d_open:.2f} force_n={f_n:.2f}",
            flush=True,
        )

    if emb is not None and (st == 1 or st % 10 == 0):
        p = objects["parcel"]
        clr_mm = min_tcp_clearance_m(objects) * 1000.0
        print(
            f"MAILBOX_STEP step={st:3d} t={st / CONTROL_HZ:5.2f}s "
            f"parcel=({p.pos[0]:.3f},{p.pos[1]:+.3f},{p.pos[2] - TOP:.3f},lo_x={p.low[0]:.3f},in={int(inside)}) "
            f"door_deg={d_open:5.2f}(shut={int(door_closed(objects))}) "
            f"flag_deg={f_deg:5.2f}(up={int(flag_upright(objects))}) "
            f"force_n={f_n:4.2f} peak_n={_S['peak_door_parcel_force_n']:4.2f} tcp_clr_mm={clr_mm:5.1f} "
            f"fault={_S['fault']}",
            flush=True,
        )
    return _S["fault"]


def score(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
    *,
    door_parcel_force_n: float | None = None,
) -> float:
    """1.0: parcel fully inside the box past the sill, door closed and latched (<= 3 deg), flag upright (90 +/- 6 deg),
    grippers >= 25 mm clear, at rest (held 1 s via SUCCESS_HOLD_S). 0.0 otherwise or after any latched fault
    (no partial credit: parcel in and door shut without the flag is the plain put-in-and-latch job)."""
    if advance(objects, door_parcel_force_n=door_parcel_force_n) is not None:
        return 0.0
    if not (parcel_inside(objects) and door_closed(objects) and all_resting(objects) and hands_clear(objects, tcps)):
        return 0.0
    return 1.0 if flag_upright(objects) else 0.0


def success(
    objects: dict[str, ObjectState],
    tcps: list[np.ndarray] | None = None,
    *,
    door_parcel_force_n: float | None = None,
) -> bool:
    """The 1.0 condition (held for SUCCESS_HOLD_S = 1.0 s by the runner)."""
    return bool(score(objects, tcps, door_parcel_force_n=door_parcel_force_n) == 1.0)


def distance(objects: dict[str, ObjectState]) -> float:
    """Normalized remaining progress across parcel insertion, door closing, and flag raising (0.0 at goal)."""
    lo_x = float(objects["parcel"].low[0])
    d_parcel = max(0.0, (X_SILL - lo_x) / 0.16)
    d_door = max(0.0, (door_open_deg(objects) - DOOR_CLOSED_MAX_DEG) / abs(DOOR_OPEN_INIT_DEG))
    d_flag = max(0.0, (FLAG_UP_TARGET_DEG - FLAG_UP_TOL_DEG - flag_angle_deg(objects)) / FLAG_UP_TARGET_DEG)
    return float((d_parcel + d_door + d_flag) / 3.0)


# -- Open-loop bimanual demo -------------------------------------------------------------------------------------------
GRIP_PICK_L = grip_rot(-25.0, 0.0)     # Left arm pitched 25 deg forward (jaws along world Y) for picking/carrying parcel
GRIP_PLACE_L = grip_rot(-30.0, 0.0)    # Left arm pitched 30 deg forward (5 deg nose-down) so parcel rear clears open door
GRIP_PUSH_L = grip_rot(-70.0, 0.0)     # Left arm forward-pitched 46 mm wide bar for pushing parcel past the sill
GRIP_DOOR_L = grip_rot(-50.0, 0.0)     # Left arm 46 mm wide bar along Y for sweeping the bottom-hinged door up to 0 deg
GRIP_FLAG_R = grip_rot(-55.0, 0.0)     # Right arm 46 mm wide bar along Y for sweeping the signal flag 0 -> 90 deg

# Partial close (target ~13.1 mm per finger on 20.5 mm half-width parcel) avoids dual-drive saturation drift (G0)
# and holds the 250 g cardboard box centred across its 41 mm spine:
FINGER_PARCEL_CLOSE = 0.72

PARK_LEFT = f3((0.18, +0.32, TOP + 0.14))
PARK_RIGHT = f3((0.20, -0.28, TOP + 0.14))

# Stage 1 (Left arm): Pick parcel at (0.235, +0.210, TOP + 0.115) (25 mm behind parcel centre 0.260 m), lift and back
# out to x = 0.155 m so the parcel's front face (x = 0.290 m) is behind the mailbox mouth (0.300 m), translate laterally
# to y = Y_MAILBOX = -0.020 m above the open door, and insert straight along +x with a 5 deg nose-down tilt (-30 deg
# grip) to TCP x = 0.278 m so the front half lands on the mailbox floor while the rear stays >= 12 mm above the open door.
PICK_X = 0.235
PICK_Y = +0.210
PICK_PRE_Z = TOP + 0.165
PICK_Z = TOP + 0.115
ALIGN_X = 0.155
CARRY_Z = TOP + 0.170
PLACE_X = 0.278
PLACE_Y = Y_MAILBOX
INSERT_Z = TOP + 0.155
PLACE_Z = TOP + 0.147

# Stage 2 (Left arm): Pitch wrist forward (-70 deg) behind parcel's -x end (at x = 0.1935 m), push +x to TCP x = 0.355 m
# (fingertip front at 0.369 m, sliding parcel's rear -x face ~65 mm past X_SILL = 0.300 m), back out of the mouth, then
# close the door with an open-jaw fork: jaws turned 90 deg so they open across the door's thickness, approach pitched
# to the door angle (pitch = -door_open_deg, pointing along the door toward its hinge), the door's top edge 30 mm deep
# between the pads. The fork slides in from beside the door (+y), and the lower pad lifts the door by its pull flange
# along the hinge circle to 28 deg. The fork cannot follow the door shut (its inner finger would hit the roof), so it
# slides up the door until only the lower fingertip holds the flange, rolls the wrist about that fingertip so the fingers
# point into the door, and presses the flange along the arc to 1 deg, past the 4 deg roller catch, then backs away.
PUSH_START_X = 0.155
PUSH_END_X = 0.355
PUSH_Z = TOP + 0.100
FORK_ZL = 0.170          # m from the hinge along the door to the TCP (fingertips at 0.157, door top edge at 0.200)
FORK_PAD_XL = -0.007     # door-normal offset of the lower pad face (flange outer face at -0.009, leaf at -0.004)
FORK_ENTRY_PAD_XL = -0.019  # 10 mm below the flange at the TCP while the fork slides in under the open door
FORK_SIDE_Y = 0.115      # beside the door (door edge y = +0.0675, finger half-width 0.034)


def door_fork(open_deg: float, finger: float, tilt: float = 0.0, zl: float = FORK_ZL, pad_xl: float = FORK_PAD_XL,
              y: float = Y_MAILBOX):
    """(TCP position, grip) of the open-jaw fork whose lower pad face sits at pad_xl (at the TCP) on a door open_deg
    from closed; tilt (deg, + = fingertips down) turns the fork off the door plane for joint margin."""
    th = math.radians(-open_deg)
    xl = pad_xl - finger  # TCP sits |finger| above the lower pad face along the door normal
    x = DOOR_HINGE_WORLD[0] + xl * math.cos(th) + zl * math.sin(th)
    z = DOOR_HINGE_WORLD[2] - xl * math.sin(th) + zl * math.cos(th)
    return f3((x, y, z)), grip_rot(-open_deg + tilt, 90.0)


def door_press(open_deg: float, roll: float, finger: float, tip_zl: float, pad_xl: float, y: float = Y_MAILBOX):
    """(TCP position, grip) with the lower fingertip at (tip_zl along the door, pad_xl along its outward normal) on a
    door open_deg from closed, the fingers rolled `roll` deg from pointing along the door (0, the fork) toward pointing
    into it. roll = 0 equals door_fork(open_deg, finger, 0, tip_zl + PAD_TIP_PAST_TCP, pad_xl)."""
    th, ph = math.radians(open_deg), math.radians(roll)
    along = np.array([-math.sin(th), 0.0, math.cos(th)])   # hinge -> door top
    normal = np.array([math.cos(th), 0.0, math.sin(th)])   # outer face -> mailbox
    approach = -math.cos(ph) * along + math.sin(ph) * normal
    jaw = math.sin(ph) * along + math.cos(ph) * normal      # lower pad -> upper pad
    tip = np.array(DOOR_HINGE_WORLD) + pad_xl * normal + tip_zl * along
    tcp = tip - PAD_TIP_PAST_TCP * approach - finger * jaw
    return f3((tcp[0], y, tcp[2])), grip_rot(-open_deg - roll, 90.0)


# Fork phase, (door angle deg, finger, tilt deg): fully open while the fork is near the table, then narrowed; the tilt
# keeps every joint >= 15 deg from its limit.
DOOR_ARC = ((82.0, FINGER_OPEN, 10.0), (70.0, FINGER_OPEN, 5.0), (58.0, FINGER_OPEN, 0.0), (46.0, -0.030, 0.0),
            (34.0, -0.020, 0.0), (28.0, -0.014, 0.0))
# Press phase, (door angle deg, roll deg, lower fingertip along the door m, its normal offset m): slide up to the top
# edge, roll the wrist, then press the flange (outer face at -0.009) 1 mm deep along the arc to 1 deg.
DOOR_PRESS_FINGER = -0.014
DOOR_PRESS = ((28.0, 0.0, 0.192, -0.007), (22.0, 20.0, 0.193, -0.008), (16.0, 35.0, 0.194, -0.008),
              (10.0, 50.0, 0.195, -0.008), (5.0, 50.0, 0.195, -0.008), (1.0, 50.0, 0.195, -0.008))
DOOR_PRESS_BACK = 0.040  # m backed off along the approach after the press

# Stage 3 (Right arm): Sweep flag at y = -0.136 m (staying >= 6.5 mm clear outside the mailbox right wall at -0.1075 m).
R_FLAG_Y = -0.136


def door_waypoints() -> list[Waypoint]:
    """Left arm: open fork beside the door, slide in under it, lift it along the hinge circle, roll onto the fingertip,
    press it past the catch, back away."""
    a0, f0, t0 = DOOR_ARC[0]
    entry, g0 = door_fork(a0, f0, t0, pad_xl=FORK_ENTRY_PAD_XL)
    beside = f3((entry[0], FORK_SIDE_Y, entry[2]))
    wps = [
        Waypoint(0.8, f3((entry[0], FORK_SIDE_Y, entry[2] + 0.060)), FINGER_OPEN, "l_fork_above_side", g0),
        Waypoint(0.6, beside, FINGER_OPEN, "l_fork_beside_door", g0),
        Waypoint(0.8, entry, FINGER_OPEN, "l_fork_slide_in", g0),
    ]
    for a, f, tilt in DOOR_ARC:
        pos, g = door_fork(a, f, tilt)
        wps.append(Waypoint(0.5 if a == a0 else 0.6, pos, f, f"l_door_{int(a)}deg", g))
    f = DOOR_PRESS_FINGER
    for a, roll, tip_zl, pad_xl in DOOR_PRESS:
        pos, g = door_press(a, roll, f, tip_zl, pad_xl)
        wps.append(Waypoint(0.5, pos, f, f"l_press_{int(a)}deg_roll{int(roll)}", g))
    wps.append(Waypoint(0.4, pos, f, "l_hold_past_catch", g))
    back = np.asarray(pos) - DOOR_PRESS_BACK * g.m[:, 2]
    wps.append(Waypoint(0.5, f3(back), f, "l_back_off_door", g))
    wps.append(Waypoint(0.6, f3((0.200, +0.160, TOP + 0.170)), FINGER_OPEN, "l_back_from_door", GRIP_PICK_L))
    return wps


def demo() -> dict[str, list[Waypoint]]:
    """The right arm sweeps the red signal flag 90 deg upright into its vertical detent and parks while the left arm
    picks the outgoing USPS Small Flat Rate Box. The left arm backs clear of the mailbox mouth, aligns with the tunnel
    and places the parcel's front half onto the mailbox floor above the open door, pitches its wrist to push the parcel
    completely inside past the front sill, then slides an open-jaw fork in from the side around the open door's top
    edge, carries the door up along its hinge circle, rolls onto its lower fingertip and presses the door past its top
    roller catch, and parks."""
    left = [
        Waypoint(0.8, f3((PICK_X, PICK_Y, PICK_PRE_Z)), FINGER_OPEN, "l_above_parcel", GRIP_PICK_L),
        Waypoint(0.8, f3((PICK_X, PICK_Y, PICK_Z)), FINGER_OPEN, "l_around_parcel", GRIP_PICK_L),
        Waypoint(1.1, f3((PICK_X, PICK_Y, PICK_Z)), FINGER_PARCEL_CLOSE, "l_grasp_parcel", GRIP_PICK_L),
        Waypoint(0.9, f3((ALIGN_X, PICK_Y, CARRY_Z)), FINGER_PARCEL_CLOSE, "l_lift_back", GRIP_PICK_L),
        Waypoint(0.9, f3((ALIGN_X, PLACE_Y, CARRY_Z)), FINGER_PARCEL_CLOSE, "l_align_mouth", GRIP_PICK_L),
        Waypoint(1.0, f3((PLACE_X, PLACE_Y, INSERT_Z)), FINGER_PARCEL_CLOSE, "l_insert_into_mouth", GRIP_PLACE_L),
        Waypoint(0.8, f3((PLACE_X, PLACE_Y, PLACE_Z)), FINGER_PARCEL_CLOSE, "l_set_on_floor", GRIP_PLACE_L),
        Waypoint(0.6, f3((PLACE_X, PLACE_Y, PLACE_Z)), FINGER_OPEN, "l_release_parcel", GRIP_PLACE_L),
        Waypoint(0.7, f3((0.220, PLACE_Y, CARRY_Z)), FINGER_OPEN, "l_rise_clear", GRIP_PICK_L),
        Waypoint(0.8, f3((PUSH_START_X, PLACE_Y, PUSH_Z)), FINGER_CLOSED, "l_behind_parcel", GRIP_PUSH_L),
        Waypoint(1.5, f3((PUSH_END_X, PLACE_Y, PUSH_Z)), FINGER_CLOSED, "l_push_past_sill", GRIP_PUSH_L),
        Waypoint(0.8, f3((0.150, PLACE_Y, PUSH_Z)), FINGER_CLOSED, "l_back_from_mouth", GRIP_PUSH_L),
        *door_waypoints(),
        Waypoint(0.7, PARK_LEFT, FINGER_OPEN, "l_park", GRIP_PICK_L),
        Waypoint(1.5, PARK_LEFT, FINGER_OPEN, "l_done", GRIP_PICK_L),
    ]
    right = [
        # Flag first, done and parked before the left arm swings the parcel over to the mouth (t = 3.6 s); the flag
        # only counts at the end. Come in beside the flag below it, then slide under it:
        Waypoint(0.9, f3((0.225, -0.200, TOP + 0.095)), FINGER_CLOSED, "r_beside_flag", GRIP_FLAG_R),
        Waypoint(0.5, f3((0.225, R_FLAG_Y, TOP + 0.095)), FINGER_CLOSED, "r_under_flag", GRIP_FLAG_R),
        # Sweep flag 0 -> 90 deg around (0.355, Y_FLAG, TOP + 0.135):
        Waypoint(0.6, f3((0.235, R_FLAG_Y, TOP + 0.165)), FINGER_CLOSED, "r_flag_30deg", GRIP_FLAG_R),
        Waypoint(0.6, f3((0.280, R_FLAG_Y, TOP + 0.225)), FINGER_CLOSED, "r_flag_65deg", GRIP_FLAG_R),
        Waypoint(0.7, f3((0.340, R_FLAG_Y, TOP + 0.240)), FINGER_CLOSED, "r_flag_90deg", GRIP_FLAG_R),
        Waypoint(0.2, f3((0.340, R_FLAG_Y, TOP + 0.240)), FINGER_CLOSED, "r_hold_flag_up", GRIP_FLAG_R),
        Waypoint(0.5, f3((0.300, -0.240, TOP + 0.240)), FINGER_OPEN, "r_back_from_flag", GRIP_FLAG_R),
        Waypoint(0.6, PARK_RIGHT, FINGER_OPEN, "r_park", GRIP_FLAG_R),
    ]
    left_s = sum(w.duration for w in left)
    right_s = sum(w.duration for w in right)
    right.append(Waypoint(round(left_s - right_s, 6), PARK_RIGHT, FINGER_OPEN, "r_done", GRIP_FLAG_R))
    return {"left": left, "right": right}


if __name__ == "__main__":
    for side, plan in demo().items():
        q = None
        for w in plan:
            q, pe, re = ik.solve(
                np.array(w.pos),
                ik.matrix_to_quat_xyzw(_mailbox_grip_rotation(w.yaw)),
                q_init=q,
                base_pos=station.BASES[side],
                iters=400,
            )
            m = np.degrees(np.minimum(q - ik.LOWER, ik.UPPER - q)).min()
            flag = "BAD" if pe > 0.002 or re > 0.02 or m < 5.0 else ""
            print(
                f"{side:5s} {w.label:20s} pos=({w.pos[0]:.3f},{w.pos[1]:+.3f},{w.pos[2] - TOP:.3f}) "
                f"pos_err={pe * 1000:.2f}mm rot={re:.4f} margin={m:.1f}deg {flag}"
            )
        print(side, "plan s", round(sum(w.duration for w in plan), 2))
