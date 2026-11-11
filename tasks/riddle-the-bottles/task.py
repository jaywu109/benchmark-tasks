"""On the station table: one row of a riddling rack (pupitre) for sparkling wine. A wooden board (25 mm thick, 52 cm
wide) stands across the table, leaning back 15 deg from vertical. Four 37.5 cl bottles sit neck-down in its row of
holes at y -0.195, -0.065, +0.065, +0.195 (13 cm apart, as on a 60-hole French pupitre), each at 45 deg below the
horizontal, its neck in a 31.6 mm square hole 40 mm deep and its shoulder resting on the hole's mouth; the base ends
face the robot, about 24 cm above the table. Each bottle (Saverglass "Couronne Champ 29": 243.5 mm tall, 70.6 mm
body; 875 g full) has a white chalk stripe across its base, from the centre out, as riddlers mark them.

The goal: each bottle has been given an eighth turn about its own axis (30-60 deg either way from where it started,
read from the stripe), is still seated (base-end centre within 5 mm, axis within 3 deg of its start), and is let go
(both TCPs at least 3 cm from it), with nothing moving. 0.25 per bottle turned. 0 for everything if any bottle was
pulled out of its hole (base-end centre more than 30 mm, or axis more than 15 deg, from its start) at any time
during the trial (latched until the next episode).
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


def f3(v):
    return tuple(float(x) for x in v)


def qy(t):
    """Quaternion (x, y, z, w) of a turn by ``t`` rad about +y."""
    return (0.0, float(math.sin(t / 2)), 0.0, float(math.cos(t / 2)))


# -- bottle (bottle.obj, written by make_bottle.py already tilted) ----------------------------------------------------
A = math.radians(45.0)  # neck-down angle below the horizontal
D = np.array([math.cos(A), 0.0, -math.sin(A)])  # the bottle's axis, base to neck
E = np.array([math.sin(A), 0.0, math.cos(A)])  # square to it, the upper side; the chalk stripe points this way
Y = np.array([0.0, 1.0, 0.0])
LENGTH, R_BODY, R_NECK = 0.2435, 0.0353, 0.0148
S0 = LENGTH / 2  # body origin on the axis, this far from the base
MASS = 0.875  # 500 g of glass (Saverglass) + 375 ml of wine
BOTTLE_SIZE = f3(2 * (S0 * np.abs(D) + R_BODY * np.abs(E)) + 2 * R_BODY * Y)  # world box of the tilted bottle
BOTTLE_COLOR = (0.10, 0.25, 0.12)
CHALK = rx.Piece(size=(0.001, 0.004, 0.030), offset=f3(-(S0 + 0.0005) * D + 0.018 * E), rot_xyzw=qy(A),
                 color=(0.95, 0.95, 0.95))

# -- the rack ---------------------------------------------------------------------------------------------------------
YS = (-0.195, -0.065, 0.065, 0.195)
NAMES = ("bottle_1", "bottle_2", "bottle_3", "bottle_4")  # right to left
BASE_X, BASE_Z = 0.32, 0.24  # base-end centre of a seated bottle, by the hull (it rests ~12 mm further out)
S_MOUTH = 0.1993  # the shoulder is 15.8 mm in radius here: it rests on the hole's mouth
BETA = math.radians(15.0)  # the board leans back
HW, T_W, T_L, PT = 0.0158, 0.006, 0.040, 0.025  # hole half-width, its wall, its depth, the board's thickness
DELTA = math.atan(2 * (HW - R_NECK) / T_L)  # the hole runs this much steeper: the resting neck lies on D
AT = A + DELTA
DT = np.array([math.cos(AT), 0.0, -math.sin(AT)])
ET = np.array([math.sin(AT), 0.0, math.cos(AT)])
EV = np.array([math.sin(BETA), 0.0, math.cos(BETA)])  # up the board's face
NB = np.array([math.cos(BETA), 0.0, -math.sin(BETA)])  # into the board, away from the robot
BOARD_W = 0.52
WOOD = (0.55, 0.38, 0.22)


def base_design(i):
    return np.array([BASE_X, YS[i], TOP + BASE_Z])


MOUTH0 = base_design(0) + S_MOUTH * D - YS[0] * Y  # mouth of the hole, at y 0 (the rack's origin)


def _rack_compound() -> rx.Compound:
    """Board with a window round each hole, and each hole's four walls, in the rack's frame (origin at MOUTH0)."""
    pieces = []
    tc = T_L / 2 * DT
    u_lo, u_hi, win_v, win_y = -(MOUTH0[2] - TOP) / math.cos(BETA) + 0.003, 0.25, 0.042, 0.026
    bc = 0.6 * PT * NB  # the board's mid-plane, just behind the mouths
    edges = [-BOARD_W / 2] + [e for y in YS for e in (y - win_y, y + win_y)] + [BOARD_W / 2]
    for y0, y1 in zip(edges[0::2], edges[1::2]):  # full-height strips between the windows
        pieces.append(rx.Piece(size=f3((PT, y1 - y0, u_hi - u_lo)), offset=f3(bc + (u_lo + u_hi) / 2 * EV + (y0 + y1) / 2 * Y),
                               rot_xyzw=qy(BETA), color=WOOD))
    for y in YS:
        for u0, u1 in ((win_v, u_hi), (u_lo, -win_v)):  # above and below each window
            pieces.append(rx.Piece(size=f3((PT, 2 * win_y, u1 - u0)), offset=f3(bc + (u0 + u1) / 2 * EV + y * Y),
                                   rot_xyzw=qy(BETA), color=WOOD))
        h = y * Y + tc
        pieces += [
            rx.Piece(size=f3((T_L, 2 * (HW + T_W), T_W)), offset=f3(h + (HW + T_W / 2) * ET), rot_xyzw=qy(AT)),
            rx.Piece(size=f3((T_L, 2 * (HW + T_W), T_W)), offset=f3(h - (HW + T_W / 2) * ET), rot_xyzw=qy(AT)),
            rx.Piece(size=f3((T_L, T_W, 2 * HW)), offset=f3(h + (HW + T_W / 2) * Y), rot_xyzw=qy(AT)),
            rx.Piece(size=f3((T_L, T_W, 2 * HW)), offset=f3(h - (HW + T_W / 2) * Y), rot_xyzw=qy(AT)),
        ]
    return rx.Compound(pieces=tuple(pieces))


RACK = Box("pupitre", pos=f3(MOUTH0), size=(0.10, BOARD_W, 0.30), color=(0.40, 0.40, 0.40))
SPAWN_OUT = 0.002  # spawned 2 mm out of the seat; it settles in


def _bottle_box(i) -> Box:
    origin = base_design(i) + (S0 - SPAWN_OUT) * D
    return Box(NAMES[i], pos=f3(origin), size=BOTTLE_SIZE, mass=MASS, color=BOTTLE_COLOR)


BOTTLES = [_bottle_box(i) for i in range(4)]
_BOTTLE = rx.Compound(pieces=(rx.Mesh(os.path.join(HERE, "bottle.obj"), approximation="convexHull"), CHALK))
rx.install({"pupitre": _rack_compound(), **{n: _BOTTLE for n in NAMES}})

OBJECTS = [RACK, *BOTTLES]

# -- measures ---------------------------------------------------------------------------------------------------------
TURN_MIN, TURN_MAX = 30.0, 60.0  # an eighth turn is 45 deg
SEAT_MM, AXIS_DEG = 5.0, 3.0
OUT_MM, OUT_DEG = 30.0, 15.0  # pulled out of the hole
GRIPPER_CLEAR = 0.030


def _rot(state):
    return quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))


def axis_of(state) -> np.ndarray:
    return _rot(state) @ D


def base_centre(state) -> np.ndarray:
    """World position of the centre of the bottle's base (the end that faces the robot)."""
    return np.asarray(state.pos, float) - S0 * axis_of(state)


def roll_deg(state, r0) -> float:
    """Signed turn (deg) of the bottle about its own axis since orientation ``r0``, read from the chalk stripe:
    positive is anticlockwise as seen looking at the base."""
    d, v = axis_of(state), _rot(state) @ E
    ref = r0 @ E
    ref = ref - (ref @ d) * d
    ref /= np.linalg.norm(ref)
    return math.degrees(math.atan2(float(np.cross(v, ref) @ d), float(ref @ v)))


def axis_change_deg(state, r0) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(axis_of(state) @ (r0 @ D))))))


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


def episode_t():
    """The embodiment's episode clock (s) from up the call stack; None offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            return float(s._t)
        f = f.f_back
    return None


def bottle_gap(state, point) -> float:
    """Distance (m) from ``point`` to the bottle, taken as a 70.6 mm cylinder over its whole length (0 inside)."""
    b, d = base_centre(state), axis_of(state)
    v = np.asarray(point, float) - b
    s = float(v @ d)
    radial = float(np.linalg.norm(v - s * d))
    out_r = max(0.0, radial - R_BODY)
    out_s = max(0.0, -s, s - LENGTH)
    return math.hypot(out_r, out_s)


def released(state, tcps) -> bool:
    return all(bottle_gap(state, p) >= GRIPPER_CLEAR for p in tcps)


def spawn_start():
    """Each bottle's start (rotation, base centre) as spawned: offline reference."""
    return {n: (np.eye(3), base_design(i) - SPAWN_OUT * D) for i, n in enumerate(NAMES)}


_EP = {"t": None, "start": None, "out": False}


def start_and_latch(objects):
    """(start poses, pulled-out latch). The start is each bottle's pose at the episode's first scored step (seated,
    before any hand arrives); the latch is set once any bottle leaves its hole and stays set until the clock goes back
    (a new episode). Offline (no clock): the spawn pose, no latch."""
    t = episode_t()
    if t is None:
        start = spawn_start()
        return start, any(_pulled(objects[n], *start[n]) for n in NAMES)
    if _EP["t"] is None or t < _EP["t"] or _EP["start"] is None:
        _EP["start"] = {n: (_rot(objects[n]), base_centre(objects[n])) for n in NAMES}
        _EP["out"] = False
    _EP["t"] = t
    _EP["out"] = _EP["out"] or any(_pulled(objects[n], *_EP["start"][n]) for n in NAMES)
    return _EP["start"], _EP["out"]


def _pulled(state, r0, b0) -> bool:
    return bool(np.linalg.norm(base_centre(state) - b0) * 1000 > OUT_MM or axis_change_deg(state, r0) > OUT_DEG)


def turned(state, r0, b0) -> bool:
    """Given an eighth turn (30-60 deg either way) and still seated."""
    return bool(TURN_MIN <= abs(roll_deg(state, r0)) <= TURN_MAX
                and np.linalg.norm(base_centre(state) - b0) * 1000 <= SEAT_MM
                and axis_change_deg(state, r0) <= AXIS_DEG)


def _parts(objects):
    """(score, bottles turned, pulled out, released, resting)."""
    start, out = start_and_latch(objects)
    tcps = tcp_points()
    bs = [objects[n] for n in NAMES]
    n_turned = sum(turned(b, *start[n]) for b, n in zip(bs, NAMES))
    rel = all(released(b, tcps) for b in bs)
    rest = all(b.resting() for b in bs)
    val = 0.0 if out or not (rel and rest) else 0.25 * n_turned
    return val, n_turned, out, rel, rest, start


_STEP = {"n": 0}


def score(objects) -> float:
    """0.25 per bottle given an eighth turn and still seated, once all are let go and still; 0 if any was pulled out."""
    val, n_turned, out, rel, rest, start = _parts(objects)
    if tcp_points():
        _STEP["n"] += 1
        if _STEP["n"] == 1 or _STEP["n"] % 10 == 0:
            parts = []
            for n in NAMES:
                b, (r0, b0) = objects[n], start[n]
                parts.append(f"{n[-1]}=({roll_deg(b, r0):+.1f}deg,{np.linalg.norm(base_centre(b) - b0) * 1000:.1f}mm,"
                             f"{axis_change_deg(b, r0):.2f}deg)")
            print(f"RIDDLE_STEP step={_STEP['n']} {' '.join(parts)} turned={n_turned} out={int(out)} rel={int(rel)} "
                  f"rest={int(rest)} score={val:.2f}", flush=True)
    return val


def success(objects) -> bool:
    return _parts(objects)[0] == 1.0


def distance(objects) -> float:
    """How far from an eighth turn: the sum over bottles of |45 - |turn|| in rad, plus seat drift in m."""
    start, _ = start_and_latch(objects)
    d = 0.0
    for n in NAMES:
        b, (r0, b0) = objects[n], start[n]
        d += abs(45.0 - abs(roll_deg(b, r0))) * math.pi / 180 + float(np.linalg.norm(base_centre(b) - b0))
    return d


# -- demo: each arm turns its two bottles; the left takes its inner one after the right hand has parked --------------
class Grip(float):
    """A 3x3 TCP rotation matrix carried in Waypoint.yaw (a float field)."""

    def __new__(cls, m):
        obj = super().__new__(cls, 1.0)
        obj.m = np.asarray(m, float)
        return obj


_orig_grip_rotation = _demo.grip_rotation
_demo.grip_rotation = lambda y=0.0: y.m if hasattr(y, "m") else _orig_grip_rotation(y)
G0 = axis_angle_matrix(Y, -(math.pi / 2 - A)) @ _orig_grip_rotation(0.0)  # fingers along D, jaws along y


def grip(roll):
    return Grip(axis_angle_matrix(D, math.radians(roll)) @ G0)


SEAT_OUT = 0.012  # the bottle rests this much further out than the hull estimate (GPU spike run 2)
GRIP_IN = 0.030  # TCP this far in from the base
CLOSE = -(R_BODY - 0.008)  # partial close on the 70.6 mm body
ROLL = 52.0  # commanded; the bottle lags ~6 deg (spike run 2: 45 commanded, 38.7 turned)
BACK, CLEAR = 0.10, 0.14  # back out this far along the axis; move between bottles this far back


def _bottle_steps(i, sign, last=False):
    p = base_design(i) - SEAT_OUT * D + GRIP_IN * D
    r = sign * ROLL
    n = NAMES[i]
    w = lambda d, pos, f, roll, lab: Waypoint(d, f3(pos), f, f"{n}: {lab}", grip(roll))  # noqa: E731
    return [
        w(2.0, p - CLEAR * D, FINGER_OPEN, 0.0, "behind the base"),
        w(1.0, p - BACK * D, FINGER_OPEN, 0.0, "line up on the axis"),
        w(1.2, p, FINGER_OPEN, 0.0, "in round the base"),
        w(1.2, p, CLOSE, 0.0, "close"),
        w(0.3, p, CLOSE, 0.0, "hold"),
        w(2.0, p, CLOSE, r, "an eighth turn"),
        w(0.4, p, CLOSE, r, "hold"),
        w(1.0, p, FINGER_OPEN, r, "let go"),
    ] + ([w(0.4, p - 0.07 * D, FINGER_OPEN, r, "back out")] if last else  # then straight to park, off the row
         [w(1.2, p - CLEAR * D, FINGER_OPEN, r, "back out")])


def _plan_s(steps):
    return sum(s.duration for s in steps)


def demo():
    """Right: bottle 1 then 2, rolling clockwise as it sees them; left: bottle 4 at the same time, then 3 once the
    right hand has parked (the inner bottles are only 13 cm apart)."""
    rpark = f3((0.22, -0.32, TOP + 0.22))
    right = _bottle_steps(0, -1) + _bottle_steps(1, -1) + [Waypoint(2.0, rpark, FINGER_OPEN, "park", grip(0.0))]
    left = _bottle_steps(3, +1)
    lwait = left[-1]
    left.append(Waypoint(_plan_s(right) - _plan_s(left), lwait.pos, FINGER_OPEN, "wait for the right hand", lwait.yaw))
    lpark = f3((0.22, 0.32, TOP + 0.22))
    left += _bottle_steps(2, +1, last=True) + [Waypoint(1.8, lpark, FINGER_OPEN, "park", grip(0.0)),
                                    Waypoint(1.0, lpark, FINGER_OPEN, "hold still", grip(0.0))]
    right.append(Waypoint(_plan_s(left) - _plan_s(right), rpark, FINGER_OPEN, "hold still", grip(0.0)))
    return {"left": left, "right": right}
