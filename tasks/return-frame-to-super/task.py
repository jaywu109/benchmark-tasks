"""On the station table, centred on the midline in front of the arms: an empty-floored 10-frame Langstroth shallow
super (a pine box with no top or bottom, inside 375 x 467 mm, 146 mm tall, 19 mm walls) stands with its long sides
across the table (inside x 0.245 to 0.620 m, |y| <= 0.2335 m). Along the top inside edge of each short end wall
runs the frame rest: a 16 mm wide rebate whose floor is 9.5 mm below the rim (136.5 mm above the table), leaving a
3 mm lip at full height on the outside.

Seven drawn shallow frames (beeswax comb in a pine frame) hang in the super by their lugs on the two rebates, pushed
together against the far wall: their centres are at x 0.6025, 0.5675, ... 0.3925 m (35 mm apart, Hoffman shoulders
touching). The nearest frame's shoulders face the robot at x 0.375 m, leaving a 130 mm space along the near wall.

An eighth, identical frame hangs on a frame perch hooked over the super's near wall: two 16 mm bars at the rebates'
height (|y| 0.2335 to 0.2495 m, x 0.135 to 0.226 m) with a 3 mm lip at their outer ends. Its centre is at
x 0.170 m.

Each frame: a top bar 448 x 27 x 19 mm with a 17.5 mm lug at each end (27 mm wide, 9.5 mm thick, flush with the top
bar's top), 483 mm over the lugs; two Hoffman end bars 9.5 mm thick, 35 mm wide (the shoulders) for the top 61 mm
under the top bar and 27 mm wide below; a 20 x 429 x 9.5 mm bottom bar; a 25 mm thick comb between them; 137 mm
tall overall; 0.35 kg. Hanging by its lugs, a frame's top is level with the rim and its bottom 9 mm off the table.

The goal: the perch frame hangs by both lugs on the super's rebates, level and square, pushed against the other
frames (no gap over 2 mm between any two adjacent frames' shoulders), the seven other frames still hang and none of them left
its rebates during the trial, nothing is moving, and both grippers' TCPs are at least 3 cm from the frame
(released).
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robocurve_ext as rx  # noqa: E402

import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"

PINE = (0.80, 0.68, 0.48)
WAX = (0.86, 0.66, 0.22)
PERCH_COLOR = (0.35, 0.36, 0.38)

# -- the super (fixed) ----------------------------------------------------------------------------------------------
IN_X0, IN_X1 = 0.245, 0.620  # inside faces of the near and far walls
IN_Y = 0.2335  # inside faces of the end walls at |y| = IN_Y
WALL_T = 0.019
BOX_H = 0.146
REBATE_W = 0.016
REBATE_Z = 0.1365  # rebate floor above the table: frames hang here by their lugs
LIP_Y = IN_Y + REBATE_W  # 0.2495: the 3 mm outer lip runs from here to the outside face
OUT_Y = IN_Y + WALL_T  # 0.2525
SUPER_X0, SUPER_X1 = IN_X0 - WALL_T, IN_X1 + WALL_T
SUPER_SIZE = (SUPER_X1 - SUPER_X0, 2 * OUT_Y, BOX_H)
SUPER_C = ((SUPER_X0 + SUPER_X1) / 2, 0.0, BOX_H / 2)  # bounding-box centre, z above TOP


def _fixed_piece(centre, x0, x1, y0, y1, z0, z1, color) -> rx.Piece:
    """An axis-aligned piece from world x/y bounds and z bounds above TOP, in a fixed part's frame at ``centre``."""
    return rx.Piece(size=(x1 - x0, y1 - y0, z1 - z0),
                    offset=((x0 + x1) / 2 - centre[0], (y0 + y1) / 2 - centre[1], (z0 + z1) / 2 - centre[2]),
                    color=color)


def _super_compound() -> rx.Compound:
    p = lambda *b: _fixed_piece(SUPER_C, *b, PINE)  # noqa: E731
    pieces = [
        p(SUPER_X0, IN_X0, -OUT_Y, OUT_Y, 0.0, BOX_H),  # near wall
        p(IN_X1, SUPER_X1, -OUT_Y, OUT_Y, 0.0, BOX_H),  # far wall
    ]
    for s in (+1, -1):
        lo, hi = sorted((s * IN_Y, s * LIP_Y))
        pieces.append(p(IN_X0, IN_X1, lo, hi, 0.0, REBATE_Z))  # end wall under the rebate
        lo, hi = sorted((s * LIP_Y, s * OUT_Y))
        pieces.append(p(IN_X0, IN_X1, lo, hi, 0.0, BOX_H))  # the lip outside the rebate
    return rx.Compound(pieces=tuple(pieces))


SUPER = Box("super", pos=(SUPER_C[0], 0.0, TOP + SUPER_C[2]), size=SUPER_SIZE, color=PINE)

# -- the perch (fixed): two bars at rebate height hooked over the near wall ------------------------------------------
PERCH_X0, PERCH_X1 = 0.135, SUPER_X0
PERCH_BAR_H = 0.010
PERCH_LIP_H = 0.010
PERCH_SIZE = (PERCH_X1 - PERCH_X0, 2 * OUT_Y, PERCH_BAR_H + PERCH_LIP_H)
PERCH_C = ((PERCH_X0 + PERCH_X1) / 2, 0.0, REBATE_Z - PERCH_BAR_H + PERCH_SIZE[2] / 2)


def _perch_compound() -> rx.Compound:
    p = lambda *b: _fixed_piece(PERCH_C, *b, PERCH_COLOR)  # noqa: E731
    pieces = []
    for s in (+1, -1):
        lo, hi = sorted((s * IN_Y, s * LIP_Y))
        pieces.append(p(PERCH_X0, PERCH_X1, lo, hi, REBATE_Z - PERCH_BAR_H, REBATE_Z))
        lo, hi = sorted((s * LIP_Y, s * OUT_Y))
        pieces.append(p(PERCH_X0, PERCH_X1, lo, hi, REBATE_Z - PERCH_BAR_H, REBATE_Z + PERCH_LIP_H))
    return rx.Compound(pieces=tuple(pieces))


PERCH = Box("perch", pos=(PERCH_C[0], 0.0, TOP + PERCH_C[2]), size=PERCH_SIZE, color=PERCH_COLOR)

# -- a frame (loose): origin at the top bar's top centre; x across the comb, y along the top bar, z up -----------------
TOPBAR_W, TOPBAR_T = 0.027, 0.019
LUG_T = 0.0095
END_IN_Y, END_OUT_Y = 0.2145, 0.224  # end bars, 9.5 mm thick; the top bar spans |y| <= END_OUT_Y
LUG_Y = 0.2415  # lug ends: 483 mm over the lugs
SHOULDER_W = 0.035  # Hoffman shoulders: frames touching here sit 35 mm apart
SHOULDER_Z = -0.080
FRAME_H = 0.137
BOTTOM_W, BOTTOM_T = 0.020, 0.0095
COMB_T = 0.025
FRAME_MASS = 0.35  # kg, assumed for an extracted drawn shallow comb (see the fidelity note)
FRAME_SIZE = (SHOULDER_W, 2 * LUG_Y, FRAME_H)
FRAME_BBOX_C = np.array([0.0, 0.0, -FRAME_H / 2])  # the bounding-box centre in the frame's origin frame
HANG_Z = REBATE_Z + LUG_T  # 0.146: frame origin height above TOP when its lugs rest on the rebates


def _frame_piece(x0, x1, y0, y1, z0, z1, color=PINE) -> rx.Piece:
    c = FRAME_BBOX_C
    return rx.Piece(size=(x1 - x0, y1 - y0, z1 - z0),
                    offset=((x0 + x1) / 2 - c[0], (y0 + y1) / 2 - c[1], (z0 + z1) / 2 - c[2]), color=color)


def _frame_compound() -> rx.Compound:
    hw, sw, bw = TOPBAR_W / 2, SHOULDER_W / 2, BOTTOM_W / 2
    pieces = [
        _frame_piece(-hw, hw, -END_OUT_Y, END_OUT_Y, -TOPBAR_T, 0.0),  # top bar
        _frame_piece(-hw, hw, END_OUT_Y, LUG_Y, -LUG_T, 0.0),  # lugs
        _frame_piece(-hw, hw, -LUG_Y, -END_OUT_Y, -LUG_T, 0.0),
        _frame_piece(-bw, bw, -END_IN_Y, END_IN_Y, -FRAME_H, -FRAME_H + BOTTOM_T),  # bottom bar
        _frame_piece(-COMB_T / 2, COMB_T / 2, -END_IN_Y, END_IN_Y, -FRAME_H + BOTTOM_T, -TOPBAR_T, WAX),  # comb
    ]
    for s in (+1, -1):
        lo, hi = sorted((s * END_IN_Y, s * END_OUT_Y))
        pieces.append(_frame_piece(-sw, sw, lo, hi, SHOULDER_Z, -TOPBAR_T))  # shoulder (wide) part
        pieces.append(_frame_piece(-hw, hw, lo, hi, -FRAME_H + BOTTOM_T, SHOULDER_Z))  # narrow part
    return rx.Compound(pieces=tuple(pieces))


SPAWN_GAP = 0.001  # frames spawn with their lugs 1 mm over their rests and settle
PERCH_FRAME_X = 0.170  # 56 mm in front of the near wall: room for the open fingers
KEPT_X = tuple(round(IN_X1 - SHOULDER_W / 2 - SHOULDER_W * k, 4) for k in range(7))  # 0.6025 ... 0.3925
NEIGHBOUR_FACE_X = KEPT_X[-1] - SHOULDER_W / 2  # 0.375: the nearest kept frame's shoulders
GOAL_X = NEIGHBOUR_FACE_X - SHOULDER_W / 2  # 0.3575: the returned frame touching it


def _frame_box(name: str, x: float) -> Box:
    z = TOP + HANG_Z + SPAWN_GAP + FRAME_BBOX_C[2]
    return Box(name, pos=(float(x), 0.0, float(z)), size=FRAME_SIZE, mass=FRAME_MASS, color=PINE)


FRAME = _frame_box("frame", PERCH_FRAME_X)
KEPT = [_frame_box(f"comb_{k + 1}", x) for k, x in enumerate(KEPT_X)]
KEPT_NAMES = tuple(b.name for b in KEPT)

rx.install({"super": _super_compound(), "perch": _perch_compound(), "frame": _frame_compound(),
            **{n: _frame_compound() for n in KEPT_NAMES}})

OBJECTS = [SUPER, PERCH, FRAME, *KEPT]

# -- measures ---------------------------------------------------------------------------------------------------------
HANG_DZ_MM = 2.0
TILT_DEG = 2.0
YAW_DEG = 3.0
CENTRE_Y_MM = 5.0
TOUCH_GAP_MM = 2.0
GRIPPER_CLEAR = 0.030
KNOCK_DZ_MM, KNOCK_TILT_DEG = 5.0, 10.0  # a kept frame this far off its hanging pose has left its rebates (latched)


def _rot(state):
    from yam_arena.urdf import quat_xyzw_to_matrix

    return quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))


def origin(state) -> np.ndarray:
    """World position of a frame's origin (the top bar's top centre)."""
    return np.asarray(state.pos, float) + _rot(state) @ (-FRAME_BBOX_C)


def tilt_deg(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(_rot(state)[2, 2])))))


def yaw_deg(state) -> float:
    """Turn of the frame's long axis (body y) from world y, folded into [0, 90]: a frame is the same either way round."""
    r = _rot(state)
    a = abs(math.degrees(math.atan2(-float(r[0, 1]), float(r[1, 1]))))
    return min(a, 180.0 - a)


def in_super(state) -> bool:
    """True when the frame's centre is over the super's inside, between the near and far walls."""
    x = origin(state)[0]
    return bool(IN_X0 < x < IN_X1)


def hung(state) -> bool:
    """Hangs by both lugs on rebates at the super's (and perch's) height: level, square and centred along y."""
    o = origin(state)
    return bool(abs((o[2] - TOP - HANG_Z) * 1000.0) <= HANG_DZ_MM and tilt_deg(state) <= TILT_DEG
                and yaw_deg(state) <= YAW_DEG and abs(o[1]) * 1000.0 <= CENTRE_Y_MM)


def gap_mm(frame, neighbour) -> float:
    """Space between the returned frame's far shoulders and the next frame's near shoulders, along x (mm)."""
    return float((origin(neighbour)[0] - origin(frame)[0] - SHOULDER_W) * 1000.0)


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
    local = _rot(state).T @ (np.asarray(point, float) - np.asarray(state.pos, float))
    half = np.asarray(size, float) / 2
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def released(state, tcps) -> bool:
    return all(box_distance(state, FRAME_SIZE, p) >= GRIPPER_CLEAR for p in tcps)


def episode_t():
    """The embodiment's episode clock (s) from up the call stack; None offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            return float(s._t)
        f = f.f_back
    return None


_KNOCK = {"t": None, "knocked": False}


def knocked(kept) -> bool:
    """True once any kept frame has left its rebates this episode (dropped 5 mm, tilted 10 deg or out of the super);
    stays true for the rest of the episode, so hanging it back does not undo it. Offline (no clock) it is not latched."""
    t = episode_t()
    if t is not None and _KNOCK["t"] is not None and t < _KNOCK["t"]:
        _KNOCK["knocked"] = False  # the clock went back: a new episode
    off = any(abs((origin(k)[2] - TOP - HANG_Z) * 1000.0) > KNOCK_DZ_MM or tilt_deg(k) > KNOCK_TILT_DEG
              or not in_super(k) for k in kept)
    if t is None:
        return off
    _KNOCK["t"] = t
    _KNOCK["knocked"] = _KNOCK["knocked"] or off
    return _KNOCK["knocked"]


def _parts(objects):
    """(score, hung in the super, widest gap between adjacent frames mm, kept frames all hung and never knocked off, released, resting)."""
    f = objects["frame"]
    kept = [objects[n] for n in KEPT_NAMES]
    row = sorted([f] + kept, key=lambda k: origin(k)[0])
    h = hung(f) and in_super(f)
    gap = max(gap_mm(a, b) for a, b in zip(row, row[1:]))  # widest gap anywhere: a kept frame dragged out counts
    kept_ok = not knocked(kept) and all(hung(k) and in_super(k) for k in kept)
    rel = released(f, tcp_points())
    rest = f.resting() and all(k.resting() for k in kept)
    if not (h and kept_ok and rel and rest):
        val = 0.0
    else:
        val = 1.0 if gap <= TOUCH_GAP_MM else 0.5
    return val, h, gap, kept_ok, rel, rest


_STEP = {"n": 0}


def score(objects) -> float:
    """1.0: the frame hangs on the super's rebates, no gap over 2 mm between any two adjacent frames, the seven others
    still hang, all resting, released. 0.5: the same but a gap is over 2 mm. 0 otherwise."""
    val, h, gap, kept_ok, rel, rest = _parts(objects)
    if tcp_points():
        _STEP["n"] += 1
        if _STEP["n"] == 1 or _STEP["n"] % 10 == 0:
            o = origin(objects["frame"])
            print(f"FRAME_STEP step={_STEP['n']} x={o[0]:.4f} y={o[1]:+.4f} z={(o[2] - TOP) * 1000:.1f}mm "
                  f"tilt={tilt_deg(objects['frame']):.2f} yaw={yaw_deg(objects['frame']):.2f} gap={gap:.1f}mm "
                  f"hung={int(h)} kept={int(kept_ok)} rel={int(rel)} rest={int(rest)} score={val:.1f}", flush=True)
    return val


def success(objects) -> bool:
    return _parts(objects)[0] == 1.0


def distance(objects) -> float:
    """Metres from the goal pose: the frame origin's offset from (GOAL_X, 0, hanging height) plus 1 cm per degree of
    tilt."""
    f = objects["frame"]
    o = origin(f)
    return float(np.linalg.norm(o - np.array([GOAL_X, 0.0, TOP + HANG_Z])) + 0.01 * tilt_deg(f))


# -- demo -------------------------------------------------------------------------------------------------------------
class Grip(float):
    """A 3x3 TCP rotation matrix carried in Waypoint.yaw (a float field)."""

    def __new__(cls, m):
        obj = super().__new__(cls, 1.0)
        obj.m = np.asarray(m, float)
        return obj


_orig_grip_rotation = _demo.grip_rotation
_demo.grip_rotation = lambda y=0.0: y.m if hasattr(y, "m") else _orig_grip_rotation(y)

TILT = 50.0  # deg: the tool tipped toward the midline about world x (offline IK: 0.30 m reach needs 45-60)
_R_RIGHT = axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(TILT)) @ _orig_grip_rotation(math.pi / 2)
# The left grip is the right one mirrored across the midline, turned 180 deg about its approach (the plain mirror
# needs joint angles past the arm's limits). Both close along world x.
_R_LEFT = axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(-TILT)) @ _orig_grip_rotation(-math.pi / 2)
GRIPS = {"right": Grip(_R_RIGHT), "left": Grip(_R_LEFT)}
HAND_Y = 0.08  # each hand holds the top bar 8 cm from its middle
TCP_DZ = -TOPBAR_T / 2  # TCP at the top bar's mid-thickness
FINGER_HOLD = -(TOPBAR_W / 2 - 0.008)  # partial close on the 27 mm top bar (w/2 - 8 mm)
LIFT_Z = 0.300  # top bar's top above TOP while carried: the frame's bottom 17 mm over the rim
SET_X = 0.300  # where the frame is lowered, 57.5 mm short of touching
PUSH_BACK = 0.020  # closed fingers start this far behind the top bar's near face before pushing
FINGER_HALF = 0.0080  # closed fingertips' lead past the TCP along +x (fitted to the URDF hulls in test_task.py)
PUSH_OVER = 0.004  # aim the push 4 mm past touching: the arms lag under the lugs' friction
PUSH_DZ = 0.005  # push with the fingertips on the top bar's upper part; the finger bodies stay over the near wall
FINGER_AJAR = -(TOPBAR_W / 2 + 0.005)  # pads 5 mm off the top bar: fully open, the 24 mm thick fingers hit the walls


def _pair(duration, x, z_top, finger, label):
    """Both hands' TCPs at (x, +-HAND_Y), level with the top bar of a frame whose top is z_top above TOP."""
    return {side: Waypoint(duration, (float(x), float(s * HAND_Y), float(TOP + z_top + TCP_DZ)), finger, label,
                           GRIPS[side])
            for side, s in (("left", 1.0), ("right", -1.0))}


def demo():
    """Grip the perch frame's top bar at y +-0.08, lift it to 0.30 m, carry it over the super, lower it onto the
    rebates at x 0.300, let go, close the fingers behind the top bar and push it 57.5 mm against the next frame."""
    near_face = lambda x: x - TOPBAR_W / 2  # noqa: E731
    steps = [
        _pair(2.0, PERCH_FRAME_X, HANG_Z + 0.040, FINGER_AJAR, "over the perch frame"),
        _pair(1.5, PERCH_FRAME_X, HANG_Z, FINGER_AJAR, "down to the top bar"),
        _pair(1.2, PERCH_FRAME_X, HANG_Z, FINGER_HOLD, "close"),
        _pair(0.5, PERCH_FRAME_X, HANG_Z, FINGER_HOLD, "hold"),
        _pair(2.5, PERCH_FRAME_X, LIFT_Z, FINGER_HOLD, "lift off the perch"),
        _pair(2.0, SET_X, LIFT_Z, FINGER_HOLD, "over the super"),
        _pair(2.5, SET_X, HANG_Z + 0.002, FINGER_HOLD, "lower onto the rebates"),
        _pair(0.6, SET_X, HANG_Z, FINGER_HOLD, "seat"),
        _pair(1.0, SET_X, HANG_Z, FINGER_AJAR, "let go"),
        _pair(1.0, SET_X, HANG_Z + 0.040, FINGER_AJAR, "rise"),
        _pair(1.0, near_face(SET_X) - PUSH_BACK, HANG_Z + 0.040, FINGER_CLOSED, "behind the top bar, close"),
        _pair(1.0, near_face(SET_X) - PUSH_BACK, HANG_Z + PUSH_DZ, FINGER_CLOSED, "down beside the top bar"),
        _pair(3.0, near_face(GOAL_X) - FINGER_HALF + PUSH_OVER, HANG_Z + PUSH_DZ, FINGER_CLOSED, "push against the next frame"),
        _pair(0.5, near_face(GOAL_X) - FINGER_HALF + PUSH_OVER, HANG_Z + PUSH_DZ, FINGER_CLOSED, "hold"),
        _pair(1.0, near_face(GOAL_X) - PUSH_BACK, HANG_Z + 0.080, FINGER_CLOSED, "back off and up"),
    ]
    # park high over the super, 10 cm above the frames (a straight-down park this high is past joint 4's limit)
    park = {"left": (0.22, +0.12, TOP + 0.25), "right": (0.22, -0.12, TOP + 0.25)}
    out = {}
    for side in ("left", "right"):
        out[side] = [st[side] for st in steps] + [
            Waypoint(1.5, park[side], FINGER_CLOSED, "park", GRIPS[side]),
            Waypoint(2.0, park[side], FINGER_CLOSED, "hold still", GRIPS[side]),
        ]
    return out
