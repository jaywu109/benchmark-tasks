"""On the station table, between the two arms: two loose large hen's eggs (egg_l and egg_r, 57 mm long by 44 mm
equatorial diameter, 58 g each, cream shell colour) lie horizontally on fixed blue rubber two-rail cradles at
x = 0.22 m, y = +0.20 m and x = 0.22 m, y = -0.20 m, with their long axes along x and their small ends pointing
forward (+x). Forward at x = 0.34 m sits a fixed six-cup moulded-pulp carton row: two empty end cups (cup_l at
y = +0.20 m and cup_r at y = -0.20 m, each a 60 x 60 x 35 mm pulp block with a 25 mm deep conical cup tapering from
46 mm at the rim to a 24 mm flat seat floor), a 3 mm thick pulp plate bridging the tops of the cups from y = -0.17 m
to +0.17 m (x = 0.31 to 0.37 m), and four fixed inner cups at y = -0.105, -0.035, +0.035, and +0.105 m holding four
fixed eggs already standing small end down. Nothing else is on the table.

The goal: put the two loose eggs into the empty carton cups (cup_l and cup_r), small end down (small-end axis within
20 degrees of downward vertical -z, lowest point below the cup rim, centre within 4 mm of the cup axis), at rest, and
released with both grippers' TCPs at least 3 cm from every point of each egg's bounding box.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402

import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
REST_GAP = 0.001


class Grip(float):
    """A 3x3 TCP rotation matrix carried in Waypoint.yaw (a float field)."""

    def __new__(cls, m):
        obj = super().__new__(cls, 1.0)
        obj.m = np.asarray(m, float)
        return obj


_orig_grip_rotation = _demo.grip_rotation
_demo.grip_rotation = lambda y=0.0: y.m if hasattr(y, "m") else _orig_grip_rotation(y)
DOWN = _orig_grip_rotation(0.0)


def pitched_grip(deg: float) -> Grip:
    """Straight-down grip (fingers closing along world y) pitched by ``deg`` about world +Y."""
    return Grip(axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(deg)) @ DOWN)


# -- egg geometry (Hügelschäffer egg, L=57 mm, B=44 mm, w=3 mm, small end at local +z) -------------------------------
EGG_D = 0.044
EGG_R = EGG_D / 2.0  # 22 mm
EGG_L = 0.057
EGG_W = 0.003
EGG_SIZE = (EGG_D, EGG_D, EGG_L)  # 44 x 44 x 57 mm in local frame
EGG_MASS = 0.058
EGG_COLOR = (0.93, 0.87, 0.76)
EGG_MESH_PATH = os.path.join(HERE, "egg.obj")
EGG_NAMES = ("egg_l", "egg_r")

# Spawn quaternions (XYZW):
# Lying with local +z (small end) toward world +x: +90 deg about +Y
ROT_LYING_PLUS_X = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
# Standing with local +z (small end) toward world -z: 180 deg about +Y
ROT_SMALL_END_DOWN = (0.0, 1.0, 0.0, 0.0)

# -- cradle geometry (fixed, 2 rails per lying egg) -------------------------------------------------------------------
CRADLE_X = 0.22
LEFT_Y = +0.20
RIGHT_Y = -0.20
RAIL_LX = 0.040
RAIL_LY = 0.006
RAIL_H = 0.008
RAIL_DY = 0.012  # centres at egg_y +- 12 mm (24 mm apart; inner faces 18 mm apart at +-9 mm)
RAIL_COLOR = (0.25, 0.45, 0.80)
RAIL_TOP_Z = TOP + RAIL_H

# Lying egg resting centre height on the two rails (inner edge at |dy| = 9 mm):
EGG_REST_Z = RAIL_TOP_Z + math.sqrt(EGG_R**2 - (RAIL_DY - RAIL_LY / 2.0) ** 2)  # TOP + 0.028075 m
# Spawn 1 mm above the rail top even by full AABB (settles 2.9 mm into the cradle during reset's 0.5 s settle):
EGG_LYING_SPAWN_Z = RAIL_TOP_Z + EGG_R + REST_GAP  # TOP + 0.031 m

# -- carton geometry (two empty end cups, bridging top plate, and four fixed inner cups with fixed eggs) --------------
CUP_X = 0.34
CUP_LX = 0.060
CUP_LY = 0.060
CUP_H = 0.035
CUP_SIZE = (CUP_LX, CUP_LY, CUP_H)
CUP_TOP_D = 0.046
CUP_BOT_D = 0.024
CUP_DEPTH = 0.025
CUP_FLOOR_H = CUP_H - CUP_DEPTH  # 10 mm floor thickness
CUP_FLOOR_Z = TOP + CUP_FLOOR_H  # TOP + 0.010 m
CUP_TOP_Z = TOP + CUP_H  # TOP + 0.035 m
CUP_RIM_Z = CUP_TOP_Z
CUP_COLOR = (0.72, 0.68, 0.60)
CARTON_MESH_PATH = os.path.join(HERE, "carton.obj")
CUP_MESH_PATH = CARTON_MESH_PATH

PLATE_Y_SPAN = 0.340  # y from -0.17 to +0.17 m (bridging cup_r at y <= -0.17 and cup_l at y >= +0.17)
PLATE_H = 0.003  # 3 mm thick at the top of the 35 mm cup blocks (z from TOP + 0.032 to TOP + 0.035)
PLATE_Z = TOP + CUP_H - PLATE_H / 2.0
PLATE_SIZE = (CUP_LX, PLATE_Y_SPAN, PLATE_H)

FIXED_CUP_YS = (-0.105, -0.035, +0.035, +0.105)
EGG_STATIC_LOW_Z = CUP_FLOOR_Z + 0.002  # TOP + 0.012 m (2 mm clear of floor, >= 0.4 mm clear of conical wall)
EGG_STATIC_SPAWN_Z = EGG_STATIC_LOW_Z + EGG_L / 2.0  # TOP + 0.0405 m

EGG_L_SPEC = Box("egg_l", pos=(CRADLE_X, LEFT_Y, EGG_LYING_SPAWN_Z), size=EGG_SIZE, mass=EGG_MASS, color=EGG_COLOR)
EGG_R_SPEC = Box("egg_r", pos=(CRADLE_X, RIGHT_Y, EGG_LYING_SPAWN_Z), size=EGG_SIZE, mass=EGG_MASS, color=EGG_COLOR)

CRADLE_L = [
    Box("cradle_l_front", pos=(CRADLE_X, LEFT_Y - RAIL_DY, TOP + RAIL_H / 2.0), size=(RAIL_LX, RAIL_LY, RAIL_H), color=RAIL_COLOR),
    Box("cradle_l_back", pos=(CRADLE_X, LEFT_Y + RAIL_DY, TOP + RAIL_H / 2.0), size=(RAIL_LX, RAIL_LY, RAIL_H), color=RAIL_COLOR),
]
CRADLE_R = [
    Box("cradle_r_front", pos=(CRADLE_X, RIGHT_Y - RAIL_DY, TOP + RAIL_H / 2.0), size=(RAIL_LX, RAIL_LY, RAIL_H), color=RAIL_COLOR),
    Box("cradle_r_back", pos=(CRADLE_X, RIGHT_Y + RAIL_DY, TOP + RAIL_H / 2.0), size=(RAIL_LX, RAIL_LY, RAIL_H), color=RAIL_COLOR),
]

CUP_L = Box("cup_l", pos=(CUP_X, LEFT_Y, TOP + CUP_H / 2.0), size=CUP_SIZE, color=CUP_COLOR)
CUP_R = Box("cup_r", pos=(CUP_X, RIGHT_Y, TOP + CUP_H / 2.0), size=CUP_SIZE, color=CUP_COLOR)
CARTON_PLATE = Box("carton_plate", pos=(CUP_X, 0.0, PLATE_Z), size=PLATE_SIZE, color=CUP_COLOR)

FIXED_CUPS = [
    Box(f"cup_{i + 1}", pos=(CUP_X, y, TOP + CUP_H / 2.0), size=CUP_SIZE, color=CUP_COLOR)
    for i, y in enumerate(FIXED_CUP_YS)
]
FIXED_EGGS = [
    Box(f"fixed_egg_{i + 1}", pos=(CUP_X, y, EGG_STATIC_SPAWN_Z), size=EGG_SIZE, color=EGG_COLOR)
    for i, y in enumerate(FIXED_CUP_YS)
]

# Bridging plate segments across the 5 gaps between the 6 cup blocks in y in [-0.17, +0.17]:
_ALL_CUP_YS = (RIGHT_Y, *FIXED_CUP_YS, LEFT_Y)
PLATE_GAPS_Y = tuple(
    ( _ALL_CUP_YS[i] + CUP_LY / 2.0, _ALL_CUP_YS[i + 1] - CUP_LY / 2.0 )
    for i in range(len(_ALL_CUP_YS) - 1)
)
PLATE_COMPOUND = rx.Compound(
    pieces=tuple(
        rx.Piece(size=(CUP_LX, y1 - y0, PLATE_H), offset=(0.0, (y0 + y1) / 2.0, 0.0), color=CUP_COLOR)
        for y0, y1 in PLATE_GAPS_Y
    )
)

OBJECTS = [
    EGG_L_SPEC,
    EGG_R_SPEC,
    CUP_L,
    CUP_R,
    CARTON_PLATE,
    *FIXED_CUPS,
    *FIXED_EGGS,
    *CRADLE_L,
    *CRADLE_R,
]


def _mesh_with_rot(path: str, approximation: str, rot_xyzw: tuple[float, float, float, float] | None = None) -> rx.Mesh:
    m = rx.Mesh(path, approximation=approximation)
    object.__setattr__(m, "rot_xyzw", rot_xyzw)
    return m


SPAWN_ROT_XYZW: dict[str, tuple[float, float, float, float]] = {
    "egg_l": ROT_LYING_PLUS_X,
    "egg_r": ROT_LYING_PLUS_X,
    **{f"fixed_egg_{i + 1}": ROT_SMALL_END_DOWN for i in range(len(FIXED_CUP_YS))},
}

_orig_rx_patch = rx._patch


def _patch_with_rot(scene) -> None:
    _orig_rx_patch(scene)
    if getattr(scene, "_robocurve_rot", False):
        return
    prev = scene.task_object

    def task_object(spec):
        obj = prev(spec)
        rot = SPAWN_ROT_XYZW.get(spec.name)
        if rot is not None:
            from isaaclab_arena.utils.pose import Pose

            spawn_func = obj.object_cfg.spawn.func
            pos = tuple(float(v) for v in spec.pos)
            quat = tuple(float(v) for v in rot)
            obj.initial_pose = Pose(position_xyz=pos, rotation_xyzw=quat)
            obj.object_cfg = obj._init_object_cfg()
            obj.object_cfg.spawn.func = spawn_func
            obj.spawner_cfg = obj.object_cfg.spawn
            obj._pose_event_cfg = obj._build_reset_event()
            rx.SPAWN_POSE[spec.name] = (pos, quat)
        return obj

    scene.task_object = task_object
    scene._robocurve_rot = True


rx._patch = _patch_with_rot

rx.install({
    "egg_l": _mesh_with_rot(EGG_MESH_PATH, approximation="convexHull", rot_xyzw=ROT_LYING_PLUS_X),
    "egg_r": _mesh_with_rot(EGG_MESH_PATH, approximation="convexHull", rot_xyzw=ROT_LYING_PLUS_X),
    "cup_l": _mesh_with_rot(CARTON_MESH_PATH, approximation="none"),
    "cup_r": _mesh_with_rot(CARTON_MESH_PATH, approximation="none"),
    "carton_plate": PLATE_COMPOUND,
    **{f"cup_{i + 1}": _mesh_with_rot(CARTON_MESH_PATH, approximation="none") for i in range(len(FIXED_CUP_YS))},
    **{
        f"fixed_egg_{i + 1}": _mesh_with_rot(EGG_MESH_PATH, approximation="convexHull", rot_xyzw=ROT_SMALL_END_DOWN)
        for i in range(len(FIXED_CUP_YS))
    },
})

EMPTY_CUP_CENTRES = {
    "cup_l": (CUP_X, LEFT_Y),
    "cup_r": (CUP_X, RIGHT_Y),
}

# -- release check --------------------------------------------
GRIPPER_CLEAR = 0.030  # an egg counts only when both TCPs are at least 3 cm from its box (released)


def tcp_points() -> list:
    """World TCP positions of both arms this step, read from the embodiment up the call stack; [] offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            from yam_arena import ik, runtime, station

            scene = s._base.scene
            return [ik.tcp_pose(runtime.arm_q(scene[f"{a}_robot"]), station.BASES[a])[0] for a in ("left", "right")]
        f = f.f_back
    return []


def box_distance(state, size, point) -> float:
    """Distance (m) from ``point`` to the nearest point of a box of ``size`` at ``state``'s pose (0 inside it)."""
    import numpy as np
    from yam_arena.urdf import quat_xyzw_to_matrix

    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))
    local = rot.T @ (np.asarray(point, float) - np.asarray(state.pos, float))
    half = np.asarray(size, float) / 2
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def released(state, tcps) -> bool:
    """True when every TCP in ``tcps`` is at least GRIPPER_CLEAR from the nearest point of the egg."""
    return all(box_distance(state, EGG_SIZE, p) >= GRIPPER_CLEAR for p in tcps)


# -- scoring (SCORE_OVER = "end") -------------------------------------------------------------------------------------
PASS_XY_TOL = 0.004  # centre xy within 4 mm of an empty cup's axis
PASS_TILT_MAX_DEG = 20.0  # small-end axis (local +z) within 20 deg of world -z


def tilt_deg(state) -> float:
    """Angle (deg) between the egg's local +z (small end) and world -z; 0 means small end straight down."""
    rot = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))
    cos_a = max(-1.0, min(1.0, -float(rot[2, 2])))
    return math.degrees(math.acos(cos_a))


def egg_cup(state) -> str | None:
    """Name ('cup_l' or 'cup_r') of the empty cup whose axis is within 4 mm of ``state``'s centre xy, or None."""
    x, y = float(state.pos[0]), float(state.pos[1])
    for cname, (cx, cy) in EMPTY_CUP_CENTRES.items():
        if math.hypot(x - cx, y - cy) <= PASS_XY_TOL:
            return cname
    return None


def packed_cups(objects) -> dict[str, str]:
    """Mapping {cup_name: egg_name} for empty cups containing a packed egg (each empty cup counts once)."""
    tcps = tcp_points()
    out: dict[str, str] = {}
    for ename in EGG_NAMES:
        st = objects[ename]
        cname = egg_cup(st)
        if cname is None or cname in out:
            continue
        low_z = float(st.low[2])
        if (
            TOP - 0.002 <= low_z < CUP_RIM_Z
            and tilt_deg(st) <= PASS_TILT_MAX_DEG
            and bool(st.resting())
            and released(st, tcps)
        ):
            out[cname] = ename
    return out


_STEP = 0


def _format_egg(name: str, state, cup_xy: tuple[float, float], tcps: list) -> str:
    dx = float(state.pos[0]) - cup_xy[0]
    dy = float(state.pos[1]) - cup_xy[1]
    xy_mm = 1000.0 * math.hypot(dx, dy)
    low_mm = 1000.0 * (float(state.low[2]) - TOP)
    t_deg = tilt_deg(state)
    rest = int(bool(state.resting()))
    rel = int(released(state, tcps))
    pk = int(
        xy_mm <= PASS_XY_TOL * 1000.0
        and -2.0 <= low_mm < (CUP_RIM_Z - TOP) * 1000.0
        and t_deg <= PASS_TILT_MAX_DEG
        and bool(rest)
        and bool(rel)
    )
    return (
        f"{name}(pos=({float(state.pos[0]):.3f},{float(state.pos[1]):+.3f}) xy={xy_mm:4.1f}mm "
        f"low_z={low_mm:4.1f}mm tilt={t_deg:4.1f}deg rest={rest} rel={rel} packed={pk})"
    )


def score(objects) -> float:
    """0.5 x the number of packed eggs (each empty cup counts once)."""
    global _STEP
    tcps = tcp_points()
    if tcps:
        _STEP += 1
        if _STEP == 1 or _STEP % 5 == 0:
            print(
                f"eggs_step={_STEP:3d}: {_format_egg('egg_l', objects['egg_l'], EMPTY_CUP_CENTRES['cup_l'], tcps)} | "
                f"{_format_egg('egg_r', objects['egg_r'], EMPTY_CUP_CENTRES['cup_r'], tcps)}",
                flush=True,
            )
    return 0.5 * float(len(packed_cups(objects)))


def success(objects) -> bool:
    """True when both loose eggs are packed in the two empty cups."""
    return len(packed_cups(objects)) == 2


def distance(objects) -> float:
    """Sum over loose eggs of the xy distance (m) to the nearer empty cup, plus 0.001 per degree of tilt above 20 deg."""
    total = 0.0
    for ename in EGG_NAMES:
        st = objects[ename]
        x, y = float(st.pos[0]), float(st.pos[1])
        xy_d = min(math.hypot(x - cx, y - cy) for cx, cy in EMPTY_CUP_CENTRES.values())
        tilt_excess = max(0.0, tilt_deg(st) - PASS_TILT_MAX_DEG)
        total += xy_d + 0.001 * tilt_excess
    return float(total)


# -- demo: both arms run V2 (angled pick at -60 deg pitch + 75 deg wrist turn to +15 deg pitch) -----------------------
FINGER_GRIP = -0.015  # partial close on the 44 mm equator: -(w/2 - 7 mm)
FINGER_RELEASE = -0.026  # open to 52 mm apart (4 mm clearance per side on 44 mm egg)

HOVER_Z = TOP + 0.095
CARRY_Z = TOP + 0.110
PLACE_Z = TOP + 0.055  # pad tips >= 6.2 mm above the 35 mm cup block top

V2_PICK_X = CRADLE_X  # 0.220 m
V2_GRIP_Z = TOP + 0.0285  # TCP at the resting egg equator height; pad tips at TOP + 16.4 mm (8.4 mm above rail top)
V2_PICK_GRIP = pitched_grip(-60.0)
V2_PLACE_GRIP = pitched_grip(+15.0)
V2_PLACE_Z = PLACE_Z

PARK_LEFT = (0.20, +0.32, TOP + 0.120)
PARK_RIGHT = (0.20, -0.32, TOP + 0.120)


def _arm_waypoints(y: float, park: tuple[float, float, float], prefix: str) -> list[Waypoint]:
    return [
        Waypoint(1.8, (V2_PICK_X, y, HOVER_Z), FINGER_OPEN, f"{prefix} hover", V2_PICK_GRIP),
        Waypoint(1.5, (V2_PICK_X, y, V2_GRIP_Z), FINGER_OPEN, f"{prefix} down to grip", V2_PICK_GRIP),
        Waypoint(0.8, (V2_PICK_X, y, V2_GRIP_Z), FINGER_GRIP, f"{prefix} close", V2_PICK_GRIP),
        Waypoint(1.5, (V2_PICK_X, y, CARRY_Z), FINGER_GRIP, f"{prefix} lift", V2_PICK_GRIP),
        Waypoint(1.0, (V2_PICK_X, y, CARRY_Z), FINGER_GRIP, f"{prefix} hold 1.0s", V2_PICK_GRIP),
        Waypoint(0.5, (V2_PICK_X, y, CARRY_Z), FINGER_GRIP, f"{prefix} hold 0.5s", V2_PICK_GRIP),
        Waypoint(2.2, (CUP_X, y, CARRY_Z), FINGER_GRIP, f"{prefix} carry + 75deg turn", V2_PLACE_GRIP),
        Waypoint(2.0, (CUP_X, y, V2_PLACE_Z), FINGER_GRIP, f"{prefix} lower into cup", V2_PLACE_GRIP),
        Waypoint(0.8, (CUP_X, y, V2_PLACE_Z), FINGER_RELEASE, f"{prefix} open", V2_PLACE_GRIP),
        Waypoint(0.5, (CUP_X, y, CARRY_Z), FINGER_OPEN, f"{prefix} retreat", V2_PLACE_GRIP),
        Waypoint(0.6, park, FINGER_OPEN, f"{prefix} park", 0.0),
        Waypoint(2.8, park, FINGER_OPEN, f"{prefix} hold", 0.0),
    ]


def demo():
    return {
        "left": _arm_waypoints(LEFT_Y, PARK_LEFT, "left"),
        "right": _arm_waypoints(RIGHT_Y, PARK_RIGHT, "right"),
    }
