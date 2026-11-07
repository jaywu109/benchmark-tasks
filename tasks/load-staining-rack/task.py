"""On the station table, between the two arms: a fixed grey plastic slide drying rack sits on the left at x 0.31 m,
y +0.30 m, and a fixed white plastic staining rack sits on the right at x 0.31 m, y +0.12 m. Each rack is built on a
150 x 45 x 3 mm floor slab resting on the table, with six 45 mm long groove blocks on top whose gaps form five 4.0 mm
wide grooves at 30 mm pitch along x (groove centres at x 0.25, 0.28, 0.31, 0.34 and 0.37 m). The drying rack's groove
blocks are 10 mm tall; the staining rack's groove blocks are 12 mm tall with chamfered top mouths (opening to 8.0 mm at
the top and narrowing to the 4.0 mm groove body 4 mm down). Five loose soda-lime glass microscope slides,
75 x 25 x 1.0 mm and 5 g each, stand upright on a long edge along y in the drying rack's five grooves, overhanging the
45 mm groove blocks by 15 mm at each end. Nothing else is on the table.

The goal: move the five slides from the drying rack into the staining rack, one slide per slot, standing upright on the
rack slab and at rest, with both grippers' TCPs at least 3 cm from every point of it (released). A slide that ever tips past
60 degrees (lying flat) or falls off the table counts 0 for the rest of the trial, and two or more slides in the same
slot all count 0.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robocurve_ext as rx  # noqa: E402

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": the score is taken at the trial's last step; success (all five resting in the rack for 1 s) ends it with 1.0.
SCORE_OVER = "end"

REST_GAP = 0.001  # loose objects spawn 1 mm above their support and settle onto it
TABLE_X = (-0.19, 0.91)  # the station table's footprint (station.TABLE_CENTER +- TABLE_SIZE / 2)
TABLE_Y = (-0.65, 0.65)

# -- the racks: 150 (x) x 45 (y) x 3 mm floor slabs with six groove blocks forming five 4.0 mm grooves at 30 mm pitch
SLAB_X = 0.31
SLAB_L = 0.150  # along x
SLAB_W = 0.045  # along y: slides (75 mm) overhang by 15 mm on each side for end gripping
SLAB_H = 0.003
SLAB_TOP = TOP + SLAB_H
GROOVE_W = 0.004  # five 4.0 mm grooves at 30 mm pitch along x
PITCH = 0.030
SLOT_X = (0.25, 0.28, 0.31, 0.34, 0.37)

DRYING_Y = 0.30  # both racks in front of the left arm: across the midline slot 5 was out of reach
DRYING_BLOCK_H = 0.010
DRYING_COLOR = (0.55, 0.57, 0.60)

RACK_Y = 0.12
RACK_BLOCK_H = 0.012
RACK_COLOR = (0.92, 0.92, 0.90)
CHAMFER_DX = 0.002  # 2.0 mm horizontal chamfer per groove side -> 8.0 mm mouth at the top of each 4.0 mm groove
CHAMFER_DZ = 0.004  # 4.0 mm vertical chamfer height -> lower 8.0 mm of each rib has vertical walls (4.0 mm groove)
CHAMFER_T = 0.002  # thickness of the tilted chamfer strip


def _rack_boxes(prefix: str, y_c: float, block_h: float, color: tuple[float, float, float]) -> list[Box]:
    x0 = SLAB_X - SLAB_L / 2
    x1 = SLAB_X + SLAB_L / 2
    edges = [x0]
    for sx in SLOT_X:
        edges.extend([sx - GROOVE_W / 2, sx + GROOVE_W / 2])
    edges.append(x1)
    boxes = [
        Box(f"{prefix}_slab", pos=(SLAB_X, y_c, TOP + SLAB_H / 2), size=(SLAB_L, SLAB_W, SLAB_H), color=color)
    ]
    for i in range(6):
        bx0, bx1 = edges[2 * i], edges[2 * i + 1]
        boxes.append(
            Box(
                f"{prefix}_block_{i + 1}",
                pos=((bx0 + bx1) / 2, y_c, SLAB_TOP + block_h / 2),
                size=(bx1 - bx0, SLAB_W, block_h),
                color=color,
            )
        )
    return boxes


def _rib_compound(width: float, chamfer_left: bool, chamfer_right: bool, color: tuple[float, float, float]) -> rx.Compound:
    """Staining rack rib with 4.0 mm groove walls in the lower 8 mm and tilted chamfer strips in the top 4 mm."""
    base_h = RACK_BLOCK_H - CHAMFER_DZ
    x_l = -width / 2 + (CHAMFER_DX if chamfer_left else 0.0)
    x_r = width / 2 - (CHAMFER_DX if chamfer_right else 0.0)
    pieces = [
        rx.Piece(size=(width, SLAB_W, base_h), offset=(0.0, 0.0, -CHAMFER_DZ / 2), color=color),
        rx.Piece(size=(x_r - x_l, SLAB_W, RACK_BLOCK_H), offset=((x_l + x_r) / 2, 0.0, 0.0), color=color),
    ]
    slope_l = math.hypot(CHAMFER_DX, CHAMFER_DZ)
    for s, enabled in ((-1, chamfer_left), (1, chamfer_right)):
        if not enabled:
            continue
        alpha = -s * math.atan2(CHAMFER_DX, CHAMFER_DZ)
        quat = (0.0, math.sin(alpha / 2), 0.0, math.cos(alpha / 2))
        mid_x = s * (width / 2 - CHAMFER_DX / 2)
        mid_z = RACK_BLOCK_H / 2 - CHAMFER_DZ / 2
        cx = mid_x - s * (CHAMFER_T / 2) * (CHAMFER_DZ / slope_l)
        cz = mid_z - (CHAMFER_T / 2) * (CHAMFER_DX / slope_l)
        pieces.append(rx.Piece(size=(CHAMFER_T, SLAB_W, slope_l), offset=(cx, 0.0, cz), color=color, rot_xyzw=quat))
    return rx.Compound(pieces=tuple(pieces))


DRYING_RACK = _rack_boxes("drying", DRYING_Y, DRYING_BLOCK_H, DRYING_COLOR)
STAINING_RACK = _rack_boxes("rack", RACK_Y, RACK_BLOCK_H, RACK_COLOR)
COMPOUNDS = {
    b.name: _rib_compound(b.size[0], chamfer_left=(i > 0), chamfer_right=(i < 5), color=RACK_COLOR)
    for i, b in enumerate(STAINING_RACK[1:])
}
rx.install(COMPOUNDS)

# -- the five microscope slides: 75 (y) x 1.0 (x) x 25 (z) mm soda-lime glass, 5 g each -----------------------------
SLIDE_T = 0.001
SLIDE_L = 0.075
SLIDE_H = 0.025
SLIDE_SIZE = (SLIDE_T, SLIDE_L, SLIDE_H)
SLIDE_MASS = 0.005
SLIDE_COLOR = (0.80, 0.88, 0.90)
SLIDE_NAMES = ("slide_1", "slide_2", "slide_3", "slide_4", "slide_5")

SLIDES = [
    Box(
        n,
        pos=(x, DRYING_Y, SLAB_TOP + REST_GAP + SLIDE_H / 2),
        size=SLIDE_SIZE,
        mass=SLIDE_MASS,
        color=SLIDE_COLOR,
    )
    for n, x in zip(SLIDE_NAMES, SLOT_X, strict=True)
]

OBJECTS = SLIDES + DRYING_RACK + STAINING_RACK

# -- scoring ----------------------------------------------------------------------------------------------------------
SLOT_X_TOL = 0.002  # centre x within 2.0 mm of slot k's centre
SLOT_Y_TOL = 0.010  # centre y within 10 mm of the staining rack's centre y
SLOT_Z_TOL = 0.003  # lowest point within 3 mm of the rack slab's top
TILT_MAX_DEG = 10.0  # upright in a slot: tilt under 10 deg
FALLEN_TILT_DEG = 60.0  # lying flat: tilt over 60 deg latches "<slide> fell"


def tilt(state) -> float:
    """Angle (deg) between the slide's local z axis (25 mm side) and world z."""
    x, y, _, _ = (float(v) for v in state.quat_xyzw)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


tilt_deg = tilt


def off_table(state) -> bool:
    """Centre below the table top or outside the station table's footprint."""
    x, y, z = (float(v) for v in state.pos)
    on_footprint = TABLE_X[0] <= x <= TABLE_X[1] and TABLE_Y[0] <= y <= TABLE_Y[1]
    return bool(z < TOP or not on_footprint)


def fell(state) -> bool:
    """True when a slide has tipped over 60 deg (lying flat) or gone off the table."""
    return bool(tilt(state) > FALLEN_TILT_DEG or off_table(state))


def in_slot(state, k: int) -> bool:
    """True when the slide stands in staining rack slot k (0-indexed)."""
    x, y = float(state.pos[0]), float(state.pos[1])
    low_z = float(state.low[2])
    return bool(
        abs(x - SLOT_X[k]) <= SLOT_X_TOL
        and abs(y - RACK_Y) <= SLOT_Y_TOL
        and SLAB_TOP - SLOT_Z_TOL <= low_z <= SLAB_TOP + SLOT_Z_TOL
        and tilt(state) < TILT_MAX_DEG
        and not off_table(state)
    )


def slide_slot(state) -> int | None:
    """Index 0..4 of the staining rack slot holding ``state``, or None."""
    for k in range(len(SLOT_X)):
        if in_slot(state, k):
            return k
    return None


# -- latched violations -----------------------------------------------------------------------------------------------
_EPISODE = {"t": None, "violations": set()}


GRIPPER_CLEAR = 0.030  # a slide counts only when both TCPs are at least 3 cm from its box (released)


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
    """True when every TCP in ``tcps`` is at least GRIPPER_CLEAR from the nearest point of the slide."""
    return all(box_distance(state, SLIDE_SIZE, p) >= GRIPPER_CLEAR for p in tcps)


def new_episode():
    _EPISODE["t"] = None
    _EPISODE["violations"] = set()


def _episode_time():
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_t") and hasattr(s, "_held"):
            return float(s._t)
        f = f.f_back
    return None


def violations(objects) -> set[str]:
    """This episode's latched violations: '<slide> fell' for any slide that tilted over 60 deg or left the table."""
    t = _episode_time()
    if t is not None:
        if _EPISODE["t"] is not None and t < _EPISODE["t"]:
            new_episode()
        _EPISODE["t"] = t
    for n in SLIDE_NAMES:
        if fell(objects[n]):
            _EPISODE["violations"].add(f"{n} fell")
    return _EPISODE["violations"]


def counted_slides(objects) -> list[str]:
    """Slides currently in a rack slot, at rest, released by both grippers, not fallen this episode, and alone in their
    slot."""
    v = violations(objects)
    tcps = tcp_points()
    slots = {n: slide_slot(objects[n]) for n in SLIDE_NAMES}
    counts: dict[int, int] = {}
    for s in slots.values():
        if s is not None:
            counts[s] = counts.get(s, 0) + 1
    return [
        n
        for n in SLIDE_NAMES
        if slots[n] is not None
        and counts[slots[n]] == 1
        and f"{n} fell" not in v
        and objects[n].resting()
        and released(objects[n], tcps)
    ]


def score(objects) -> float:
    """0.2 per slide at rest in a rack slot, both TCPs at least 3 cm from it, not fallen this trial, and alone in its slot
    (up to 1.0)."""
    return round(0.2 * len(counted_slides(objects)), 10)


def success(objects) -> bool:
    """All five slides counted and every slide resting(); held for 1 s it ends the trial."""
    return len(counted_slides(objects)) == len(SLIDE_NAMES) and all(objects[n].resting() for n in SLIDE_NAMES)


def distance(objects) -> float:
    """Sum over slides not counted of the horizontal distance (m) from the slide centre to the nearest free rack slot."""
    counted = set(counted_slides(objects))
    if len(counted) == len(SLIDE_NAMES):
        return 0.0
    occupied = {slide_slot(objects[n]) for n in SLIDE_NAMES} - {None}
    free = [k for k in range(len(SLOT_X)) if k not in occupied]
    if not free:
        used_by_counted = {slide_slot(objects[n]) for n in counted}
        free = [k for k in range(len(SLOT_X)) if k not in used_by_counted]
    total = 0.0
    for n in SLIDE_NAMES:
        if n not in counted:
            sx, sy = float(objects[n].pos[0]), float(objects[n].pos[1])
            total += min(math.hypot(sx - SLOT_X[k], sy - RACK_Y) for k in free)
    return float(total)


# -- demo: left arm moves slides 1..5 from drying rack to staining rack; right arm waits parked ----------------------
GRIP_YAW = 0.0  # fingers close along y on the slide's two 25 x 1 mm end faces
GRIP_FINGER = -(SLIDE_L / 2 - 0.0017)  # -0.0358 m: ~7 N clamp on the 75 mm ends without pushing into the hull seam
RELEASE_FINGER = -(SLIDE_L / 2 + 0.006)  # -0.0435 m: 6 mm off each end while rising clear of the slide
PICK_X_OFFSET = -0.0012  # pick with TCP ~1.9 mm to -x of the slide so both ends sit on the flat single-hull half-pads
PLACE_X_OFFSET = -0.0006  # compensate for the additional -0.6 mm TCP x lag at RACK_Y so the slide aims at SLOT_X[k]
GRIP_H = 0.020  # TCP height above the slab top (pad tips 7 mm above it, pads on the slide's upper 18 mm)
HOVER_H = 0.080  # 8 cm above the table top
CARRY_CLEAR = 0.040  # slide bottom 4 cm above the 12 mm staining-rack groove tops (4.2 cm above drying-rack tops)
RELEASE_CLEAR = 0.002  # slide bottom 2 mm above the rack slab top at release
RISE_H = 0.030  # pad clearance above the 25 mm slide top before opening fully to FINGER_OPEN

HOVER_Z = TOP + HOVER_H
GRIP_Z = SLAB_TOP + GRIP_H
CARRY_Z = SLAB_TOP + RACK_BLOCK_H + CARRY_CLEAR + GRIP_H
PLACE_Z = SLAB_TOP + RELEASE_CLEAR + GRIP_H
RISE_Z = SLAB_TOP + SLIDE_H + RISE_H + PAD_TIP_PAST_TCP

PARK_LEFT = (0.20, 0.40, TOP + 0.12)
PARK_RIGHT = (0.20, -0.32, TOP + 0.12)
HOLD_S = 2.0

APPROACH_S = 1.5
DESCEND_S = 1.5
CLOSE_S = 0.8
LIFT_S = 2.0
CARRY_S = 2.0
LOWER_S = 3.0
OPEN_S = 0.8
RISE_S = 1.0
PARK_S = 2.0


def demo():
    """Left arm loads slides 1..5 from the drying rack into the staining rack one by one, then after slide_5 rises
    straight up and parks at (0.20, +0.40, TOP + 0.12) within ~0.85 s before holding still. Right arm waits at
    (0.20, -0.32, TOP + 0.12) with fingers open."""
    left: list[Waypoint] = []
    for k, x in enumerate(SLOT_X, 1):
        px = round(x + PICK_X_OFFSET, 6)
        qx = round(x + PLACE_X_OFFSET, 6)
        open_s = 0.25 if k == 5 else OPEN_S
        rise_s = 0.35 if k == 5 else RISE_S
        left.extend(
            [
                Waypoint(APPROACH_S, (px, DRYING_Y, HOVER_Z), FINGER_OPEN, f"over slide_{k}", GRIP_YAW),
                Waypoint(DESCEND_S, (px, DRYING_Y, GRIP_Z), FINGER_OPEN, f"descend to slide_{k}", GRIP_YAW),
                Waypoint(CLOSE_S, (px, DRYING_Y, GRIP_Z), GRIP_FINGER, f"close on slide_{k}", GRIP_YAW),
                Waypoint(LIFT_S, (px, DRYING_Y, CARRY_Z), GRIP_FINGER, f"lift slide_{k}", GRIP_YAW),
                Waypoint(CARRY_S, (qx, RACK_Y, CARRY_Z), GRIP_FINGER, f"carry slide_{k}", GRIP_YAW),
                Waypoint(LOWER_S, (qx, RACK_Y, PLACE_Z), GRIP_FINGER, f"lower slide_{k}", GRIP_YAW),
                Waypoint(open_s, (qx, RACK_Y, PLACE_Z), RELEASE_FINGER, f"open on slide_{k}", GRIP_YAW),
                Waypoint(rise_s, (qx, RACK_Y, RISE_Z), RELEASE_FINGER, f"rise from slide_{k}", GRIP_YAW),
            ]
        )
    left.append(Waypoint(0.50, PARK_LEFT, FINGER_OPEN, "park", GRIP_YAW))
    left.append(Waypoint(3.70, PARK_LEFT, FINGER_OPEN, "hold still", GRIP_YAW))
    right = [Waypoint(w.duration, PARK_RIGHT, FINGER_OPEN, "wait", GRIP_YAW) for w in left]
    return {"left": left, "right": right}
