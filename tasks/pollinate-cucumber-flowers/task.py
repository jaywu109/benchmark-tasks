"""On the station table: a cucumber vine trained along a support wire 12 cm above the bench, across the table at
x 0.45, with four open flowers on it facing the robot, from right to left: a female, a male, a female, a male. A
female flower has a miniature cucumber (35 mm, green) between its petals and the vine; a male sits on a very short stalk.
A pollination brush (178 mm, 7 mm handle) lies on a rest to the right, tip toward the vine.
The goal (hand pollination): load the brush with pollen from a male flower's anthers, then brush each female flower's
stigma. A stroke: the brush tip stays on a stigma or anther column (within 4 mm of its front centre and no more than
1 mm in front of its face) while it moves at least 5 mm across it. A female counts once her stigma is stroked at any
time after a male's anthers were stroked (pollen on the brush). 0.5 per female.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"


def f3(v):
    return tuple(float(x) for x in v)


def qx(t):
    """Quaternion (x, y, z, w) of a turn by ``t`` rad about +x."""
    return (float(math.sin(t / 2)), 0.0, 0.0, float(math.cos(t / 2)))


YELLOW, ORANGE, GREEN, DARK_GREEN, LEAF = (0.98, 0.82, 0.10), (0.95, 0.55, 0.05), (0.35, 0.55, 0.20), (0.12, 0.35, 0.12), (0.20, 0.45, 0.18)

# -- the vine and its flowers -----------------------------------------------------------------------------------------
X_FACE = 0.40  # the front of each flower's stigma or anther column
X_VINE, Z_WIRE = 0.45, TOP + 0.12
FLOWERS = (  # (name, y, height of the flower's centre above the table, female?)
    ("female_1", -0.12, 0.110, True),
    ("male_1", -0.04, 0.125, False),
    ("female_2", 0.04, 0.110, True),
    ("male_2", 0.12, 0.125, False),
)
X_FRONT = {True: X_FACE, False: X_FACE + 0.022}  # males sit on very short stalks, females on longer ones (UC 21242)
FEMALES = tuple(n for n, *_, fem in FLOWERS if fem)
MALES = tuple(n for n, *_, fem in FLOWERS if not fem)
PETAL_R, PETAL_L = 0.011, 0.016  # petal centres this far out, each 16 mm long: a 40 mm flower


def _flower(female: bool, z: float):
    """Pieces in the flower's frame: origin at the front centre of its stigma (female) or anther column (male); the
    flower faces -x. The stalk runs back (+x) to the vine; it bends down or up to the wire at its end."""
    p = [rx.Piece(size=(0.0015, 0.010, 0.010), offset=(0.0070, 0.0, 0.0))]  # the corolla's base
    for k in range(5):
        a = 2 * math.pi * k / 5 + math.pi / 2
        p.append(rx.Piece(size=(0.0015, 0.012, PETAL_L), offset=(0.0060, PETAL_R * math.cos(a), PETAL_R * math.sin(a)),
                          rot_xyzw=qx(a - math.pi / 2)))
    if female:
        p.append(rx.Piece(size=(0.006, 0.008, 0.008), offset=(0.0030, 0.0, 0.0), color=ORANGE))  # three-lobed stigma
        p.append(rx.Piece(size=(0.035, 0.009, 0.009), offset=(0.0255, 0.0, 0.0), color=DARK_GREEN))  # the ovary
        stalk0 = 0.043
    else:
        p.append(rx.Piece(size=(0.007, 0.005, 0.005), offset=(0.0035, 0.0, 0.0), color=ORANGE))  # anther column
        stalk0 = 0.008
    x1 = X_VINE - X_FRONT[female]
    if not female:  # the rest of the cluster: two closed buds on the node
        for dy, dz in ((0.010, 0.012), (-0.011, 0.016)):
            p.append(rx.Piece(size=(0.016, 0.006, 0.006), offset=(x1 - 0.010, dy, dz), color=(0.75, 0.80, 0.25)))
    p.append(rx.Piece(size=(x1 - stalk0, 0.003, 0.003), offset=((stalk0 + x1) / 2, 0.0, 0.0), color=GREEN))
    dz = Z_WIRE - (TOP + z)
    p.append(rx.Piece(size=(0.003, 0.003, abs(dz) + 0.003), offset=(x1, 0.0, dz / 2), color=GREEN))
    return rx.Compound(pieces=tuple(p))


FLOWER_BOXES = [Box(n, pos=f3((X_FRONT[fem], y, TOP + z)), size=(0.055, 0.040, 0.040), color=YELLOW) for n, y, z, fem in FLOWERS]


def _vine():
    """Wire, two posts, the vine along the wire and five leaves behind it (frame origin at the wire's middle)."""
    p = [rx.Piece(size=(0.002, 0.44, 0.002), offset=(0.004, 0.0, 0.0), color=(0.6, 0.6, 0.6))]  # support wire
    for y in (-0.22, 0.22):
        p.append(rx.Piece(size=(0.012, 0.012, Z_WIRE - TOP + 0.03), offset=(0.004, y, (0.03 - (Z_WIRE - TOP)) / 2 + 0.0),
                          color=(0.55, 0.45, 0.30)))
    p.append(rx.Piece(size=(0.007, 0.36, 0.007), offset=(0.0, 0.0, 0.0), color=GREEN))  # the vine, tied along the wire
    for y, z in ((-0.17, 0.05), (-0.08, 0.065), (0.0, 0.045), (0.08, 0.065), (0.17, 0.05)):  # leaves above, behind
        p.append(rx.Piece(size=(0.002, 0.065, 0.065), offset=(0.022, y, z), rot_xyzw=qx(math.pi / 4), color=LEAF))
    return rx.Compound(pieces=tuple(p))


VINE = Box("vine", pos=f3((X_VINE, 0.0, Z_WIRE)), size=(0.03, 0.46, 0.20), color=GREEN)

# -- the brush on its rest --------------------------------------------------------------------------------------------
BRUSH_L = 0.178
TIP = BRUSH_L / 2  # the tuft's tip, from the brush's centre along its +x
BRUSH_C = np.array([0.20, -0.20, TOP + 0.030 + 0.0035])
BRUSH = Box("brush", pos=f3(BRUSH_C), size=(BRUSH_L, 0.008, 0.008), mass=0.012, color=(0.75, 0.12, 0.10))
BRUSH_PARTS = rx.Compound(pieces=(
    rx.Piece(size=(0.140, 0.007, 0.007), offset=(-0.019, 0.0, 0.0)),  # lacquered handle
    rx.Piece(size=(0.020, 0.008, 0.008), offset=(0.061, 0.0, 0.0), color=(0.75, 0.75, 0.78)),  # ferrule
    rx.Piece(size=(0.018, 0.006, 0.006), offset=(0.080, 0.0, 0.0), color=(0.93, 0.88, 0.75)),  # tuft
))
REST = Box("rest", pos=f3((BRUSH_C[0], BRUSH_C[1], TOP + 0.015)), size=(0.12, 0.03, 0.03), color=(0.30, 0.30, 0.32))
REST_PARTS = rx.Compound(pieces=(rx.Piece(size=(0.012, 0.030, 0.030), offset=(-0.060, 0.0, 0.0)),
                                 rx.Piece(size=(0.012, 0.030, 0.030), offset=(0.045, 0.0, 0.0))))

rx.install({"vine": _vine(), "brush": BRUSH_PARTS, "rest": REST_PARTS,
            **{n: _flower(fem, z) for n, _, z, fem in FLOWERS}})
OBJECTS = [VINE, REST, BRUSH, *FLOWER_BOXES]

# -- scoring ----------------------------------------------------------------------------------------------------------
TOUCH = 0.004  # the tip within this of a stigma's or anther column's front centre
FRONT = 0.001  # and no more than this in front of its face: a tip hovering in the air does not count
STROKE = 0.005  # a brush stroke: the tip moves this far across the face while it stays on it
CENTRES = {n: np.array([X_FRONT[fem], y, TOP + z]) for n, y, z, fem in FLOWERS}


def tip_of(state) -> np.ndarray:
    r = quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))
    return np.asarray(state.pos, float) + r @ np.array([TIP, 0.0, 0.0])


def touching(state) -> str | None:
    """The flower whose stigma or anthers the brush tip is on, else None."""
    tip = tip_of(state)
    for n, c in CENTRES.items():
        if np.linalg.norm(tip - c) <= TOUCH and tip[0] >= c[0] - FRONT:
            return n
    return None


def _embodiment():
    """The yam_arena embodiment up the call stack (it has the arms and the episode clock); None offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            return s
        f = f.f_back
    return None


_S = {"t": None, "loaded": False, "start": {}, "done": set(), "n": 0}


def new_episode() -> None:
    _S.update(t=None, loaded=False, start={}, done=set(), n=0)


def stroked(on, tip) -> str | None:
    """The flower the tip has just brushed: it has stayed on that flower's face since touching it and has moved at
    least STROKE across it (in y-z) from where it first touched."""
    for n in CENTRES:
        if n != on:
            _S["start"].pop(n, None)
    if on is None:
        return None
    start = _S["start"].setdefault(on, tip[1:].copy())
    return on if np.linalg.norm(tip[1:] - start) >= STROKE else None


def advance(objects) -> set:
    """Update the latch once per control step (or every call offline): returns the females pollinated so far."""
    emb = _embodiment()
    if emb is not None:
        t = float(emb._t)
        if _S["t"] is not None and t < _S["t"]:
            new_episode()
        if t == _S["t"]:
            return _S["done"]
        _S["t"] = t
    on = touching(objects["brush"])
    hit = stroked(on, tip_of(objects["brush"]))
    if hit in FEMALES and _S["loaded"]:
        _S["done"].add(hit)
    if hit in MALES:
        _S["loaded"] = True
    if emb is not None:
        _S["n"] += 1
        if _S["n"] == 1 or _S["n"] % 10 == 0:
            tip = tip_of(objects["brush"])
            d = {n: np.linalg.norm(tip - c) * 1000 for n, c in CENTRES.items()}
            near = min(d, key=d.get)
            print(f"POLL_STEP step={_S['n']} tip=({tip[0]:.3f},{tip[1]:+.3f},{tip[2] - TOP:.3f}) nearest={near} "
                  f"{d[near]:.1f}mm loaded={int(_S['loaded'])} done={sorted(_S['done'])}", flush=True)
    return _S["done"]


def score(objects) -> float:
    """0.5 per female flower whose stigma the brush tip stroked after it had stroked a male's anthers."""
    return 0.5 * len(advance(objects))


def success(objects) -> bool:
    return score(objects) == 1.0


def distance(objects) -> float:
    """From the brush tip to the next thing it needs: a male's anthers while unloaded, else the nearest unpollinated
    female's stigma (m)."""
    tip = tip_of(objects["brush"])
    goals = MALES if not _S["loaded"] else [n for n in FEMALES if n not in _S["done"]] or FEMALES
    return min(float(np.linalg.norm(tip - CENTRES[n])) for n in goals)


# -- demo: the right hand takes the brush, loads it on male 1, then brushes female 1 and female 2 ---------------------
GRIP_BACK = 0.099  # grip the handle this far behind the tip
PRESS = 0.001  # the tip held 1 mm into the flower's centre
AIM_DY = {"male_1": -0.003, "female_1": -0.003, "female_2": -0.004}  # the tip landed 3-4 mm left of its aim in GPU runs 1-3
SWEEP = 0.003  # each stroke runs from 3 mm right of the centre to 3 mm left of it (6 mm across the face)


def _touch(name, label, hold=0.8):
    c = CENTRES[name]
    y = c[1] + AIM_DY.get(name, 0.0)
    x_on, x_off = c[0] - GRIP_BACK + PRESS, c[0] - GRIP_BACK - 0.03
    return [Waypoint(1.5, f3((x_off, y - SWEEP, c[2])), FINGER_CLOSED, f"{label}: in front"),
            Waypoint(1.0, f3((x_on, y - SWEEP, c[2])), FINGER_CLOSED, f"{label}: tip on"),
            Waypoint(hold, f3((x_on, y + SWEEP, c[2])), FINGER_CLOSED, f"{label}: brush across"),
            Waypoint(0.5, f3((x_on, y + SWEEP, c[2])), FINGER_CLOSED, f"{label}: settle"),  # run 4: female 2's tip lagged 2 mm
            Waypoint(0.8, f3((x_off, y + SWEEP, c[2])), FINGER_CLOSED, f"{label}: back")]


def demo():
    g = (BRUSH_C[0] + TIP - GRIP_BACK, BRUSH_C[1], BRUSH_C[2])
    right = [Waypoint(1.5, (g[0], g[1], g[2] + 0.06), FINGER_OPEN, "above the brush"),
             Waypoint(1.0, g, FINGER_OPEN, "down"),
             Waypoint(1.2, g, FINGER_CLOSED, "close"),
             Waypoint(0.3, g, FINGER_CLOSED, "hold"),
             Waypoint(1.2, (g[0], g[1], g[2] + 0.06), FINGER_CLOSED, "lift")]
    right += _touch("male_1", "male 1 (load)") + _touch("female_1", "female 1") + _touch("female_2", "female 2")
    right.append(Waypoint(2.0, (0.22, -0.20, TOP + 0.10), FINGER_CLOSED, "away"))
    left = [Waypoint(sum(w.duration for w in right), (0.22, 0.30, TOP + 0.12), FINGER_OPEN, "wait", 0.0)]
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
            print(f"{side:5s} {w.label:28s} pos=({w.pos[0]:.3f},{w.pos[1]:+.3f},{w.pos[2] - TOP:.3f}) pos_err={pe * 1000:.1f}mm rot={re:.3f} margin={m:.1f} {flag}")
        print(side, "plan s", round(sum(w.duration for w in plan), 1))
