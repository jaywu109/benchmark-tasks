"""On the station table: a radio desk with eight long-throw faders side by side, sliding toward and away from the robot
(up = away). Fader 1 (left) carries the song on air and stands at its 0 mark; fader 8 (right) holds the next song,
cued, its fader closed; the green lamps above faders 1 and 8 show both channels switched on; faders 2 to 7 are
closed and dark.
The goal (a crossfade as the song ends): bring fader 1 down to closed while pushing fader 8 up to its 0 mark, so
the next song takes over without a silent gap. Both faders in their bottom quarter at the same moment is dead air,
and both past halfway at once is two songs at full over each other; either scores 0 for the episode.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"


def f3(v):
    return tuple(float(x) for x in v)


BLACK, PANEL, CAP, WHITE = (0.03, 0.03, 0.03), (0.22, 0.23, 0.25), (0.80, 0.80, 0.82), (0.95, 0.95, 0.95)
LAMP_ON, LAMP_OFF = (0.10, 0.85, 0.20), (0.12, 0.13, 0.12)

# -- the desk ---------------------------------------------------------------------------------------------------------
DESK_H = 0.040
Z_P = TOP + DESK_H  # the panel's top
X0 = 0.300  # a cap's centre with its fader closed
TRAVEL = 0.100  # long-throw fader
MARK = 0.075  # the 0 mark, this far up from closed
N_CH = 8
CH_Y = tuple(round(0.14 - 0.04 * k, 4) for k in range(N_CH))  # channels 1-8, left to right from the operator's seat
ON_AIR, CUED = 0, 7  # channel indices: 1 on air, 8 cued (28 cm apart: room for both wrists)
START = (MARK,) + (0.0,) * (N_CH - 1)
CAP_SIZE = (0.020, 0.012, 0.016)  # along the travel, across, tall
CAP_Z = Z_P + 0.002 + CAP_SIZE[2] / 2  # the cap rides 2 mm above the panel on its (unmodelled) stem
DAMPING = 2.0  # N*s/m: the fader stays where it is left

DESK = Box("desk", pos=f3((0.38, 0.0, TOP + DESK_H / 2)), size=(0.20, 0.36, DESK_H), color=PANEL)


def _strip(k):
    y = CH_Y[k]
    lit = k in (ON_AIR, CUED)
    return [Box(f"slot{k + 1}", pos=f3((X0 + TRAVEL / 2, y, Z_P + 0.0005)), size=(TRAVEL + 0.010, 0.004, 0.001),
                color=BLACK),
            Box(f"mark{k + 1}", pos=f3((X0 + MARK, y + 0.011, Z_P + 0.0005)), size=(0.002, 0.010, 0.001), color=WHITE),
            Box(f"lamp{k + 1}", pos=f3((X0 + TRAVEL + 0.020, y, Z_P + 0.002)), size=(0.010, 0.010, 0.004),
                color=LAMP_ON if lit else LAMP_OFF)]


FADERS = [Box(f"fader{k + 1}", pos=f3((X0 + START[k], CH_Y[k], CAP_Z)), size=CAP_SIZE, mass=0.010, color=CAP)
          for k in range(N_CH)]
OBJECTS = [DESK, *[b for k in range(N_CH) for b in _strip(k)], *FADERS]
rx.install({f"fader{k + 1}": rx.Prismatic(anchor=(0.0, 0.0, 0.0), axis="X", lower=-START[k],
                                          upper=TRAVEL - START[k], damping=DAMPING) for k in range(N_CH)})


def level(state) -> float:
    """How far up a fader is from closed (m)."""
    return float(state.pos[0] - X0)


# -- the arms, read up the call stack (offline: none) -----------------------------------------------------------------
def _embodiment():
    """The yam_arena embodiment up the call stack (it has the arms and the episode clock); None offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            return s
        f = f.f_back
    return None


def tcp_points() -> list:
    """World TCP positions of both arms this step; [] offline."""
    emb = _embodiment()
    if emb is None:
        return []
    from yam_arena import ik, runtime, station

    scene = emb._base.scene
    return [ik.tcp_pose(runtime.arm_q(scene[f"{a}_robot"]), station.BASES[a])[0] for a in ("left", "right")]


# -- scoring ----------------------------------------------------------------------------------------------------------
QUIET = 0.025  # a fader in its bottom quarter is close to silent
LOUD = 0.050  # past halfway it plays near full level
MARK_TOL = 0.004  # the cued fader within 4 mm of its 0 mark
CLOSED_TOL = 0.003  # the outgoing fader within 3 mm of closed
HAND_CLEAR = 0.04  # no TCP within this of either cap at the end: hands off the faders
HOLD_STEPS = 20  # the end state must hold this many control steps (1 s at 20 Hz) before it counts
_S = {"t": None, "fault": None, "n": 0, "k": 0, "ok_since": None}


def new_episode() -> None:
    _S.update(t=None, fault=None, n=0, k=0, ok_since=None)


def advance(objects):
    """Update the latch once per control step (every call offline): the fault ("dead air" / "both up") or None."""
    emb = _embodiment()
    if emb is not None:
        t = float(emb._t)
        if _S["t"] is not None and t < _S["t"]:
            new_episode()
        if t == _S["t"]:
            return _S["fault"]
        _S["t"] = t
    _S["k"] += 1
    a, b = level(objects[f"fader{ON_AIR + 1}"]), level(objects[f"fader{CUED + 1}"])
    if _S["fault"] is None:
        if a < QUIET and b < QUIET:
            _S["fault"] = "dead air"
        elif a > LOUD and b > LOUD:
            _S["fault"] = "both up"
    if emb is not None:
        _S["n"] += 1
        if _S["n"] == 1 or _S["n"] % 10 == 0:
            print(f"XF_STEP step={_S['n']} on_air={a * 1000:.1f}mm cued={b * 1000:.1f}mm "
                  f"others_max={max(level(objects[f'fader{k + 1}']) for k in range(1, N_CH - 1)) * 1000:.1f}mm "
                  f"fault={_S['fault']}", flush=True)
    return _S["fault"]


def hands_clear(objects) -> bool:
    caps = [np.asarray(objects[f"fader{k + 1}"].pos) for k in (ON_AIR, CUED)]
    return all(np.linalg.norm(np.asarray(p) - c) >= HAND_CLEAR for p in tcp_points() for c in caps)


def score(objects) -> float:
    """1.0: fader 8 at its 0 mark, fader 1 closed, both still, hands off them, held so for 1 s, and never dead air or
    both up this episode. 0 otherwise."""
    if advance(objects) is not None:
        return 0.0
    out, cue = objects[f"fader{ON_AIR + 1}"], objects[f"fader{CUED + 1}"]
    done = (abs(level(cue) - MARK) <= MARK_TOL and level(out) <= CLOSED_TOL
            and out.resting(0.005) and cue.resting(0.005) and hands_clear(objects))
    if not done:
        _S["ok_since"] = None
    elif _S["ok_since"] is None:
        _S["ok_since"] = _S["k"]
    return 1.0 if done and _S["k"] - _S["ok_since"] >= HOLD_STEPS else 0.0


def success(objects) -> bool:
    return score(objects) == 1.0


def distance(objects) -> float:
    """How far the two faders are from the end of the crossfade (m, summed)."""
    return abs(level(objects[f"fader{CUED + 1}"]) - MARK) + max(0.0, level(objects[f"fader{ON_AIR + 1}"]))


# -- demo: each hand pinches its fader's cap fore-and-aft, and both move together -------------------------------------
GRIP_Z = CAP_Z + 0.010  # TCP height: the pads (13 mm long) cover the cap's upper part
OPEN = -0.018  # partly open: 36 mm between the pads, over a 20 mm cap
PINCH = -(CAP_SIZE[0] / 2 - 0.008)  # the partial close (w/2 - 8 mm)
YAW = {"left": -math.pi / 2, "right": math.pi / 2}  # fingers close along x, in the fader's own lane (+-90: joint room)
FADE_S = 4.0  # the crossfade's length


def _at(k, travel, z=GRIP_Z):
    return f3((X0 + travel, CH_Y[k], z))


def demo():
    lo, hi = GRIP_Z + 0.05, GRIP_Z
    YL, YR = YAW["left"], YAW["right"]
    n = 8
    left = [Waypoint(2.0, _at(ON_AIR, MARK, lo), OPEN, "above fader 1", YL),
            Waypoint(1.2, _at(ON_AIR, MARK, hi), OPEN, "down to its cap", YL),
            Waypoint(1.0, _at(ON_AIR, MARK, hi), PINCH, "pinch", YL),
            Waypoint(0.3, _at(ON_AIR, MARK, hi), PINCH, "hold", YL)]
    right = [Waypoint(2.0, _at(CUED, 0.0, lo), OPEN, "above fader 8", YR),
             Waypoint(1.2, _at(CUED, 0.0, hi), OPEN, "down to its cap", YR),
             Waypoint(1.0, _at(CUED, 0.0, hi), PINCH, "pinch", YR),
             Waypoint(0.3, _at(CUED, 0.0, hi), PINCH, "hold", YR)]
    for i in range(1, n + 1):  # together: fader 1 down as fader 8 comes up, their sum kept at the 0 mark
        f = i / n
        left.append(Waypoint(FADE_S / n, _at(ON_AIR, MARK * (1 - f) + PULL_PAST * f), PINCH, f"fade {i}/{n}", YL))
        right.append(Waypoint(FADE_S / n, _at(CUED, MARK * f + PUSH_PAST * f), PINCH, f"fade {i}/{n}", YR))
    for plan, k, x, yw in ((left, ON_AIR, PULL_PAST, YL), (right, CUED, MARK + PUSH_PAST, YR)):
        plan += [Waypoint(0.5, _at(k, x), PINCH, "settle", yw),
                 Waypoint(0.8, _at(k, x), OPEN, "let go", yw),
                 Waypoint(0.8, _at(k, x, hi + 0.05), OPEN, "lift off", yw)]
    left.append(Waypoint(1.0, (0.22, 0.30, TOP + 0.12), OPEN, "away", 0.0))
    right.append(Waypoint(1.0, (0.22, -0.30, TOP + 0.12), OPEN, "away", 0.0))
    return {"left": left, "right": right}


PULL_PAST = -0.002  # demo knob: aim the outgoing cap 2 mm past closed (the slot end stops it)
PUSH_PAST = 0.0  # demo knob: aim offset for the cued cap at its mark


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
            print(f"{side:5s} {w.label:18s} pos=({w.pos[0]:.3f},{w.pos[1]:+.3f},{w.pos[2] - TOP:.3f}) "
                  f"yaw={math.degrees(w.yaw):6.1f} pos_err={pe * 1000:.1f}mm rot={re:.3f} margin={m:.1f} {flag}")
        print(side, "plan s", round(sum(w.duration for w in plan), 1))
