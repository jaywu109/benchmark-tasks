"""On the station table: a record deck with its platter stopped and an LP on it, five tracks of unequal length, the
lead-in gaps between them a lighter ring than the grooved bands. The tonearm (pivot at the deck's right rear) lies in
its rest clip beside the record; a finger lift stands up from the side of its headshell.
The goal (cueing a record before air): lift the arm by its finger lift, swing it over the record and lower the stylus
by hand into the gap before track 3 (counted from the edge), then let go. The stylus must end at rest on the record
inside that gap with no hand at the finger lift. Landing on the record faster than 5 cm/s is a dropped arm and
scores 0 for the episode.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"


def f3(v):
    return tuple(float(x) for x in v)


BLACK, GROOVE, LAND, LABEL = (0.03, 0.03, 0.03), (0.07, 0.07, 0.08), (0.30, 0.30, 0.32), (0.70, 0.12, 0.10)
SILVER, WOOD = (0.75, 0.75, 0.78), (0.40, 0.26, 0.15)

# -- the deck and the record ------------------------------------------------------------------------------------------
C = np.array([0.40, 0.06])  # the spindle (x, y)
PLINTH_H, PLATTER_H, RECORD_H = 0.040, 0.010, 0.002
Z_REC = TOP + PLINTH_H + PLATTER_H + RECORD_H  # the record's playing surface
R_REC = 0.150
# rings from the edge in: (outer radius, inner radius, kind); "gap" = a lead-in gap (smooth, lighter), "band" = a track
RINGS = (
    (0.150, 0.147, "edge"),
    (0.147, 0.129, "band"),  # track 1
    (0.129, 0.125, "gap"),   # before track 2
    (0.125, 0.113, "band"),  # track 2
    (0.113, 0.109, "gap"),   # before track 3: the target
    (0.109, 0.087, "band"),  # track 3
    (0.087, 0.083, "gap"),   # before track 4
    (0.083, 0.074, "band"),  # track 4
    (0.074, 0.070, "gap"),   # before track 5
    (0.070, 0.060, "band"),  # track 5
    (0.060, 0.050, "runout"),
)
GAPS = tuple((o, i) for o, i, k in RINGS if k == "gap")
TARGET = GAPS[1]  # the gap before track 3
STEP_H = 0.00005  # each ring's disc is this much higher than the one outside it, so its top shows


N_SIDES = 96  # disc meshes: a 96-sided prism, under 0.1 mm from a true circle at 15 cm


def disc_obj(r, h) -> str:
    """OBJ text of a ``N_SIDES``-sided prism of radius ``r`` and height ``h``, centred on the origin."""
    ring = [(r * math.cos(2 * math.pi * k / N_SIDES), r * math.sin(2 * math.pi * k / N_SIDES)) for k in range(N_SIDES)]
    v = [f"v {x:.6f} {y:.6f} {z:.6f}" for z in (-h / 2, h / 2) for x, y in ring]
    v += [f"v 0 0 {-h / 2:.6f}", f"v 0 0 {h / 2:.6f}"]
    n, f = N_SIDES, []
    for k in range(n):
        a, b = k + 1, (k + 1) % n + 1
        f += [f"f {a} {b} {b + n} {a + n}", f"f {2 * n + 1} {b} {a}", f"f {2 * n + 2} {a + n} {b + n}"]
    return "\n".join(v + f) + "\n"


def _disc(name, r, h, z_bottom, color):
    """A fixed disc on the spindle axis, its mesh written next to this file (``disc_<name>.obj``)."""
    path = os.path.join(HERE, f"disc_{name}.obj")
    text = disc_obj(r, h)
    if not os.path.exists(path) or open(path).read() != text:
        with open(path, "w") as fh:
            fh.write(text)
    rx.install({name: rx.Mesh(path, approximation="none")})
    return Box(name, pos=f3((C[0], C[1], z_bottom + h / 2)), size=(2 * r, 2 * r, h), color=color)


def _ring_discs():
    """The record as stacked discs, each ring's disc STEP_H higher than the one outside it, so its top shows."""
    out, z = [], Z_REC - RECORD_H
    for k, (r_out, _, kind) in enumerate(RINGS):
        out.append(_disc(f"ring_{k}", r_out, RECORD_H + k * STEP_H, z, GROOVE if kind == "band" else LAND))
    out.append(_disc("label", 0.050, RECORD_H + len(RINGS) * STEP_H, z, LABEL))
    return out


PIVOT = np.array([C[0] + 0.13, C[1] - 0.17, Z_REC + 0.020])  # the arm's yaw and pitch hinges cross here
PLINTH = Box("plinth", pos=f3((0.40 + 0.03, C[1] - 0.02, TOP + PLINTH_H / 2)), size=(0.44, 0.38, PLINTH_H), color=WOOD)
PLATTER = _disc("platter", 0.152, PLATTER_H, TOP + PLINTH_H, SILVER)

# -- the tonearm: arm frame = pitch hinge at the pivot, the arm pointing -x at rest (zero pose), z up ----------------
L_EFF = 0.229  # pivot to stylus tip (a 9-inch arm)
TIP = np.array([-L_EFF, 0.0, -0.020])  # stylus tip: at the record's surface when the arm is level
FIN = np.array([-0.222, -0.0335, 0.023])  # where the hand grips the finger lift
CW = 0.035  # counterweight cube edge: the arm nose-heavy by about 2.4 g at the stylus (test_task)
ARM_PIECES = (
    rx.Piece(size=(0.200, 0.008, 0.008), offset=(-0.105, 0.0, 0.0), color=SILVER),          # arm tube
    rx.Piece(size=(0.030, 0.018, 0.008), offset=(-0.222, 0.0, -0.004), color=BLACK),        # headshell
    rx.Piece(size=(0.018, 0.012, 0.010), offset=(-0.226, 0.0, -0.013), color=(0.15, 0.25, 0.60)),  # cartridge
    rx.Piece(size=(0.003, 0.003, 0.002), offset=(-0.229, 0.0, -0.019), color=SILVER),       # stylus
    rx.Piece(size=(0.006, 0.022, 0.003), offset=(-0.222, -0.020, -0.0015), color=SILVER),   # finger-lift bar
    rx.Piece(size=(0.012, 0.005, 0.030), offset=(-0.222, -0.0335, 0.013), color=SILVER),    # finger-lift fin
    rx.Piece(size=(0.050, 0.008, 0.008), offset=(0.025, 0.0, 0.0), color=SILVER),           # stub
    rx.Piece(size=(CW, CW, CW), offset=(0.050 + CW / 2, 0.0, 0.0), color=(0.20, 0.20, 0.22)),  # counterweight
)
LO = np.min([np.array(p.offset) - np.array(p.size) / 2 for p in ARM_PIECES], axis=0)
HI = np.max([np.array(p.offset) + np.array(p.size) / 2 for p in ARM_PIECES], axis=0)
BOX_C = (LO + HI) / 2  # the reported pos is the centre of the pieces' bounds, in the arm frame
ARM_MASS = 0.12
YAW_LIM, PITCH_LIM = (-45.0, 5.0), (-3.0, 12.0)

YOKE = Box("yoke", pos=f3(PIVOT), size=(0.02, 0.02, 0.024), mass=0.05, color=SILVER)
YOKE_PARTS = rx.Compound(pieces=(rx.Piece(size=(0.020, 0.020, 0.022), offset=(0.0, 0.0, -0.016)),))  # bearing post
ARM = Box("arm", pos=f3(PIVOT), size=tuple(float(v) for v in HI - LO), mass=ARM_MASS, color=SILVER)

# rest post with its clip: the tube lies on it 15 cm out from the pivot, between two clip fingers
POST_X = -0.150
REST = Box("rest", pos=f3((PIVOT[0] + POST_X, PIVOT[1], (TOP + PLINTH_H + PIVOT[2] - 0.004) / 2)),
           size=(0.012, 0.022, PIVOT[2] - 0.004 - TOP - PLINTH_H), color=BLACK)
_post_h = PIVOT[2] - 0.004 - TOP - PLINTH_H
REST_PARTS = rx.Compound(pieces=(
    rx.Piece(size=(0.012, 0.012, _post_h)),
    rx.Piece(size=(0.012, 0.004, 0.008), offset=(0.0, +0.0075, _post_h / 2 + 0.004)),
    rx.Piece(size=(0.012, 0.004, 0.008), offset=(0.0, -0.0075, _post_h / 2 + 0.004)),
))

rx.install({"yoke": (YOKE_PARTS, rx.Revolute(anchor=(0.0, 0.0, 0.0), axis="Z", lower=YAW_LIM[0], upper=YAW_LIM[1],
                                             damping=2e-4)),
            "arm": (rx.Compound(pieces=ARM_PIECES),
                    rx.Revolute(anchor=(0.0, 0.0, 0.0), axis="Y", lower=PITCH_LIM[0], upper=PITCH_LIM[1],
                                damping=2e-5, parent="yoke")),
            "rest": REST_PARTS})
OBJECTS = [PLINTH, PLATTER, *_ring_discs(), REST, YOKE, ARM]


# -- reading the arm --------------------------------------------------------------------------------------------------
def arm_point(state, local) -> np.ndarray:
    """A point given in the arm frame, in the world."""
    r = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))
    return np.asarray(state.pos, float) + r @ (np.asarray(local, float) - BOX_C)


def tip_of(state) -> np.ndarray:
    return arm_point(state, TIP)


def radius(p) -> float:
    return float(np.hypot(p[0] - C[0], p[1] - C[1]))


def on_record(tip) -> bool:
    return radius(tip) <= R_REC and tip[2] <= Z_REC + 0.0015


def in_gap(tip, gap) -> bool:
    return gap[1] <= radius(tip) <= gap[0]


def tcp_points() -> list:
    """World TCP positions of both arms this step, read from the embodiment up the call stack; [] offline."""
    emb = _embodiment()
    if emb is None:
        return []
    from yam_arena import ik, runtime, station

    scene = emb._base.scene
    return [ik.tcp_pose(runtime.arm_q(scene[f"{a}_robot"]), station.BASES[a])[0] for a in ("left", "right")]


def _embodiment():
    """The yam_arena embodiment up the call stack (it has the arms and the episode clock); None offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            return s
        f = f.f_back
    return None


# -- scoring ----------------------------------------------------------------------------------------------------------
DROP_SPEED = 0.05  # m/s: the stylus meeting the record faster than this is a dropped arm
HAND_CLEAR = 0.08  # no TCP within this of the finger lift at the end: hands off the deck
LINE_CLEAR = 0.05  # and none within this of the arm's line from the pivot to the stylus
HOLD_STEPS = 20  # the cued state must hold this many control steps (1 s at 20 Hz) before it counts
SCRATCH = 0.003  # m: the stylus sliding this far across the vinyl while it touches is a scratch
_S = {"t": None, "tip": None, "landed": False, "dropped": False, "touch": None, "n": 0, "k": 0, "ok_since": None}


def new_episode() -> None:
    _S.update(t=None, tip=None, landed=False, dropped=False, touch=None, n=0, k=0, ok_since=None)


def advance(objects, dt=0.05) -> bool:
    """Update the drop/scratch latch once per control step (every call offline, ``dt`` apart): True once the stylus
    met the record faster than DROP_SPEED or slid more than SCRATCH across it from where it touched down."""
    emb = _embodiment()
    if emb is not None:
        t = float(emb._t)
        if _S["t"] is not None and t < _S["t"]:
            new_episode()
        if t == _S["t"]:
            return _S["dropped"]
        if _S["t"] is not None:
            dt = t - _S["t"]
        _S["t"] = t
    _S["k"] += 1
    tip = tip_of(objects["arm"])
    if _S["tip"] is not None and dt > 0:
        speed = float(np.linalg.norm(tip - _S["tip"])) / dt
        touching = on_record(tip)
        if touching and not _S["landed"] and speed > DROP_SPEED:
            _S["dropped"] = True
        _S["landed"] = touching
    else:
        _S["landed"] = on_record(tip)
    if not _S["landed"]:
        _S["touch"] = None
    elif _S["touch"] is None:
        _S["touch"] = tip[:2]
    elif np.linalg.norm(tip[:2] - _S["touch"]) > SCRATCH:
        _S["dropped"] = True
    _S["tip"] = tip
    if emb is not None:
        _S["n"] += 1
        if _S["n"] == 1 or _S["n"] % 10 == 0:
            st = objects["arm"]
            print(f"CUE_STEP step={_S['n']} tip=({tip[0]:.4f},{tip[1]:+.4f},{(tip[2] - Z_REC) * 1000:+.1f}mm) "
                  f"r={radius(tip) * 1000:.1f}mm landed={int(_S['landed'])} dropped={int(_S['dropped'])} "
                  f"pos=({st.pos[0]:.4f},{st.pos[1]:+.4f},{st.pos[2]:.4f}) "
                  f"yoke=({objects['yoke'].pos[0]:.4f},{objects['yoke'].pos[1]:+.4f},{objects['yoke'].pos[2]:.4f})",
                  flush=True)
    return _S["dropped"]


def seg_dist(p, a, b) -> float:
    """Distance from point ``p`` to the segment ``a``-``b``."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    u = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + u * (b - a))))


def hands_clear(objects) -> bool:
    """No TCP within HAND_CLEAR of the finger lift or within LINE_CLEAR of the arm from its pivot to the stylus."""
    arm = objects["arm"]
    fin, pivot, tip = arm_point(arm, FIN), arm_point(arm, np.zeros(3)), tip_of(arm)
    return all(np.linalg.norm(np.asarray(p) - fin) >= HAND_CLEAR and seg_dist(p, pivot, tip) >= LINE_CLEAR
               for p in tcp_points())


def score(objects) -> float:
    """1.0: the stylus at rest on the record in the gap before track 3, hands off the arm, held so for 1 s. 0.5: the same in
    another gap. 0: on a grooved band, off the record, held, still moving, or dropped or dragged across the vinyl at
    any time this episode."""
    if advance(objects):
        return 0.0
    arm = objects["arm"]
    tip = tip_of(arm)
    s = 0.0
    if on_record(tip) and arm.resting(0.005, 0.05) and hands_clear(objects):
        s = 1.0 if in_gap(tip, TARGET) else 0.5 if any(in_gap(tip, g) for g in GAPS) else 0.0
    if s == 0.0 or _S["ok_since"] is None or _S["ok_since"][1] != s:
        _S["ok_since"] = None if s == 0.0 else (_S["k"], s)
    return s if _S["ok_since"] is not None and _S["k"] - _S["ok_since"][0] >= HOLD_STEPS else 0.0


def success(objects) -> bool:
    return score(objects) == 1.0


def distance(objects) -> float:
    """From the stylus tip to the nearest point of the middle of the gap before track 3, on the record (m)."""
    tip = tip_of(objects["arm"])
    r_mid = sum(TARGET) / 2
    d = tip[:2] - C
    n = np.linalg.norm(d)
    goal = C + (d / n if n > 1e-9 else np.array([1.0, 0.0])) * r_mid
    return float(np.linalg.norm(np.array([*goal, Z_REC]) - tip))


# -- demo: the right hand lifts the arm by its finger lift, swings it over the gap, lowers it, lets go ------------------
def rz(t):
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def ry(t):
    return np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])


def pose_point(local, yaw_deg, pitch_deg) -> np.ndarray:
    """An arm-frame point in the world with the arm at (yaw, pitch) from its rest pose."""
    return PIVOT + rz(math.radians(yaw_deg)) @ ry(math.radians(pitch_deg)) @ np.asarray(local, float)


def yaw_for_radius(r) -> float:
    """The arm's yaw (deg) that puts the stylus tip over radius ``r`` of the record."""
    lo, hi = YAW_LIM[0], 0.0
    for _ in range(60):  # the tip's radius falls as the arm swings in (yaw more negative)
        mid = (lo + hi) / 2
        if radius(pose_point(TIP, mid, 0.0)) > r:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


AIM_R = sum(TARGET) / 2  # demo knob: aim at the gap's middle
LIFT = 5.0  # deg of pitch: the stylus 20 mm up, the tube 13 mm above the clip
PRESS = -0.3  # deg: the hand sets the stylus down this far past touching, so it is on the record before the release
SWING_STEP = 4.0  # deg between swing waypoints (a 0.2 mm sag from the true arc)


def _grip(yaw, pitch, dur, label, finger=FINGER_CLOSED, up=0.0):
    p = pose_point(FIN, yaw, pitch) + np.array([0.0, 0.0, up])
    return Waypoint(dur, f3(p), finger, label, math.radians(yaw))


def demo():
    open_ = -0.014  # partly open (14 mm a side): pads clear the headshell and the fin
    y3 = yaw_for_radius(AIM_R)
    right = [_grip(0.0, 0.0, 2.0, "above the finger lift", open_, up=0.05),
             _grip(0.0, 0.0, 1.2, "down to it", open_),
             _grip(0.0, 0.0, 1.0, "close"),
             _grip(0.0, 0.0, 0.3, "hold"),
             _grip(0.0, LIFT, 1.2, "lift out of the clip")]
    n = max(1, math.ceil(abs(y3) / SWING_STEP))
    right += [_grip(y3 * k / n, LIFT, 0.5, f"swing {k}/{n}") for k in range(1, n + 1)]
    right += [_grip(y3, LIFT, 0.5, "settle over the gap"),
              _grip(y3, 1.0, 1.5, "lower"),
              _grip(y3, PRESS, 1.5, "set down slowly"),
              _grip(y3, PRESS, 0.5, "hold"),
              _grip(y3, PRESS, 0.8, "let go", open_),
              _grip(y3, PRESS, 0.8, "lift off", open_, up=0.04),
              Waypoint(1.2, (0.25, -0.24, TOP + 0.12), open_, "away")]
    left = [Waypoint(sum(w.duration for w in right), (0.22, 0.30, TOP + 0.12), open_, "wait", 0.0)]
    return {"left": left, "right": right}


if __name__ == "__main__":  # offline: IK for every waypoint
    from yam_arena import ik, station
    from yam_arena.demo import grip_rotation
    for side, plan in demo().items():
        q = None
        for w in plan:
            q, pe, re = ik.solve(np.array(w.pos), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q_init=q,
                                 base_pos=station.BASES[side], iters=400)
            m = np.degrees(np.minimum(q - ik.LOWER, ik.UPPER - q)).min()
            flag = "BAD" if pe > 0.002 or re > 0.02 or m < 5 else ""
            print(f"{side:5s} {w.label:24s} pos=({w.pos[0]:.3f},{w.pos[1]:+.3f},{w.pos[2] - TOP:.3f}) "
                  f"yaw={math.degrees(w.yaw):6.1f} pos_err={pe * 1000:.1f}mm rot={re:.3f} margin={m:.1f} {flag}")
        print(side, "plan s", round(sum(w.duration for w in plan), 1))
