"""On the station table, right of the midline: a 430 x 210 x 18 mm wood mounting plate (four screw heads) carries a
vehicle bracket for a 2.5 lb fire extinguisher, installed horizontally with its latch facing up. A red Rusoh-size
extinguisher (175 x 410 x 167 mm, 3.155 kg) lies nested in the bracket's cradle along y, its bottom rim against the
bracket's end tab, its head toward +y. A quick-release strap (rigid band) runs around the extinguisher's neck; its
latch lever (yellow, 70 mm, hinged on the strap's free end) stands raised about 100 deg, and the latch's wire loop
already hangs on the bracket's hook. The strap's tension acts on the lever through the loop, so the lever, once
pushed down past its over-centre point, is held closed against its stop; raised, it stays raised.

The goal: the lever is down within 10 deg of its closed stop (past the catch), the loop is still on the hook, the
extinguisher is still seated in the bracket and the bracket has not moved, both hands are at least 3 cm clear, and the
lever stays down for 1 s after release. Loop off the hook, extinguisher leaving its seat, or the bracket shifting
more than 10 mm latches the trial to 0.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402

from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = float(TABLE_TOP[SCENE])
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
INSTRUCTION = "Secure the fire extinguisher in its vehicle bracket."

WOOD = (0.62, 0.47, 0.30)
BRACKET_GREY = (0.10, 0.10, 0.11)
STRAP_BLACK = (0.04, 0.04, 0.05)
LEVER_YELLOW = (0.93, 0.76, 0.08)
STEEL = (0.62, 0.64, 0.66)
EXT_RED = (0.78, 0.07, 0.06)

# -- layout (m): x absolute; "s" runs along +y from the extinguisher's bottom end at Y0; z above TOP -----------------
XA = 0.245  # extinguisher axis x
Y0 = -0.433  # extinguisher bottom end y
PLATE_SIZE = (0.210, 0.430, 0.018)
PLATE_T = PLATE_SIZE[2]
FLOOR_T = 0.008
R_BODY, R_NECK = 0.050, 0.035
ZA = PLATE_T + FLOOR_T + R_BODY  # extinguisher axis height
EXT_SIZE = (0.175, 0.410, 0.167)
EXT_MASS = 3.155
EXT_DZ = EXT_SIZE[2] / 2 - R_BODY  # bbox centre above the axis (the head stands up)
BRACKET_MASS = 0.340
BRACKET_S = (-0.009, 0.268)  # 277 mm from the end tab to the strap's end wall
HOOP_S = (0.240, 0.266)  # strap band
STRAP_S = 0.253  # band centre

# the latch (x relative to XA, z relative to the axis): pivot P on the strap's free end, loop pin Q on the lever's ear,
# hook H on the bracket; the loop is a rigid link of length LINK between Q and H, and strap tension pulls P toward -x
PZ = 0.0515
QN = -0.012  # pin 12 mm below the pivot when the lever lies closed
HX, HZ = 0.052, 0.060
LINK = 0.022
LEVER_LEN = 0.070
STRAP_PULL_N = 24.0  # strap tension at the free end, N
K_STRAP = 600.0  # N/m: stiffness of the strap's stretch
U_TARGET = -STRAP_PULL_N / K_STRAP  # drive target (m from the closed position) giving STRAP_PULL_N at u = 0

OPEN_DEG = 100.0  # the raised lever's stop
CATCH_DEG = 32.0  # over-centre point (set by the geometry above; checked in the tests)
CLOSED_DEG = 10.0  # "within 10 deg of the closed stop"
PARTIAL_DEG = OPEN_DEG - 45.0  # 45 deg of the closing arc
START_MIN_DEG, START_MAX_DEG = 90.0, 105.0
CLEAR_M = 0.030
LOOP_OFF_M = 0.012  # loop's hole centre off the hook by more than this: off the hook
LOOP_OFF_Y = 0.010
EXT_SHIFT_M = 0.015
EXT_LIFT_M = 0.008
BRACKET_SHIFT_M = 0.010
TILT_DEG = 5.0
JOINT_BREAK_M = 0.010
LEVER_STILL_DEG = 0.5  # the lever's angle may vary by less than this across the 1 s hold


def mech_px(theta_deg: float) -> float:
    """x (m from the extinguisher axis) of the pivot P with the lever at ``theta_deg`` and the loop taut."""
    t = math.radians(theta_deg)
    off = (QN * math.sin(t), QN * math.cos(t))
    dz = PZ + off[1] - HZ
    return HX - off[0] - math.sqrt(LINK ** 2 - dz ** 2)


def mech_q(theta_deg: float):
    t = math.radians(theta_deg)
    return mech_px(theta_deg) + QN * math.sin(t), PZ + QN * math.cos(t)


def link_beta_deg(theta_deg: float) -> float:
    """Direction of the loop (pin to hook), degrees from straight up toward +x."""
    qx, qz = mech_q(theta_deg)
    return math.degrees(math.atan2(HX - qx, HZ - qz))


PX0 = mech_px(0.0)
BETA0 = link_beta_deg(0.0)
CARRIAGE_INITIAL = mech_px(OPEN_DEG) - PX0
RING_INITIAL = (link_beta_deg(OPEN_DEG) - BETA0) - OPEN_DEG  # loop's angle relative to the lever at spawn


def _b(x0, x1, s0, s1, z0, z1, centre, color=None):
    """A box piece from bounds (x absolute, s from Y0, z above TOP) for a part whose frame origin is ``centre`` = (x, s, z)."""
    return rx.Piece(size=(x1 - x0, s1 - s0, z1 - z0),
                    offset=((x0 + x1) / 2 - centre[0], (s0 + s1) / 2 - centre[1], (z0 + z1) / 2 - centre[2]), color=color)


# -- plate (fixed): wood board and four screw heads -------------------------------------------------------------------
PLATE_C = (XA, 0.1295, PLATE_T / 2)  # centred under the bracket
_SCREWS = [(XA + dx, PLATE_C[1] + dy) for dx in (-0.085, 0.085) for dy in (-0.190, 0.190)]


def _plate_compound() -> rx.Compound:
    p = [rx.Piece(size=PLATE_SIZE)]
    for x, s in _SCREWS:
        p.append(_b(x - 0.006, x + 0.006, s - 0.006, s + 0.006, PLATE_T, PLATE_T + 0.003, PLATE_C))
    return rx.Compound(pieces=tuple(p))


# -- bracket (loose, 0.34 kg, sits on the plate): cradle, end tab, strap's fixed half and the hook ---------------------
Z_TOP_BRACKET = ZA + 0.063
BRACKET_C = (XA, (BRACKET_S[0] + BRACKET_S[1]) / 2, (PLATE_T + Z_TOP_BRACKET) / 2)
BRACKET_SIZE = (0.146, BRACKET_S[1] - BRACKET_S[0], Z_TOP_BRACKET - PLATE_T)
HOOK = (XA + HX, 0.2285, ZA + HZ)  # peg centre
HOOK_REL = tuple(h - c for h, c in zip(HOOK, BRACKET_C))


def _bracket_compound() -> rx.Compound:
    c = BRACKET_C
    p = lambda *a, col=BRACKET_GREY: _b(*a, c, color=col)  # noqa: E731
    zf = PLATE_T + FLOOR_T
    pieces = [
        p(XA - 0.073, XA + 0.073, *BRACKET_S, PLATE_T, zf),  # floor
        p(XA - 0.060, XA + 0.060, BRACKET_S[0], -0.001, zf, 0.080),  # end tab behind the rim
        p(XA - 0.073, XA - 0.0535, 0.010, 0.200, zf, zf + 0.030),  # cradle cheeks
        p(XA + 0.0535, XA + 0.073, 0.010, 0.200, zf, zf + 0.030),
        p(XA - 0.050, XA - 0.045, *HOOP_S, zf, ZA + 0.0455, col=STRAP_BLACK),  # strap's fixed half: wall, arm over the neck
        p(XA - 0.050, XA - 0.012, *HOOP_S, ZA + 0.0425, ZA + 0.0455, col=STRAP_BLACK),
        p(XA + 0.049, XA + 0.064, 0.213, 0.221, zf, ZA + 0.063),  # hook post
    ]
    pieces.append(rx.Mesh(os.path.join(HERE, "hook_peg.obj"), approximation="convexHull"))  # round hook peg, 6 mm
    return rx.Compound(pieces=tuple(pieces))


# -- carriage = the strap's free end (slides along x, pulled toward -x by the strap's tension) ---------------------------
P0 = (XA + PX0, STRAP_S, ZA + PZ)  # pivot with the lever closed
CARRIAGE_SIZE = (0.069, 0.026, 0.0605)


def _carriage_compound() -> rx.Compound:
    c = P0
    p = lambda x0, x1, s0, s1, z0, z1: _b(P0[0] + x0, P0[0] + x1, s0, s1, ZA + z0, ZA + z1, c)  # noqa: E731
    return rx.Compound(pieces=(
        p(-0.045, 0.012, *HOOP_S, 0.0425, 0.0455),  # band over the neck
        p(0.007, 0.012, *HOOP_S, -0.015, 0.0425),  # tail down the neck's right side
    ))


# -- lever (hinge about y at P; arm lies toward -x when closed, ear carries the loop's pin) ----------------------------
LEVER_SIZE = (0.075, 0.022, 0.010)
LEVER_ARM_CENTRE = (-0.0325, 0.0, 0.0)  # arm box centre in the lever frame (x -70 .. +5 mm)


def _lever_compound() -> rx.Compound:
    c = P0
    p = lambda x0, x1, s0, s1, z0, z1: _b(P0[0] + x0, P0[0] + x1, s0, s1, P0[2] + z0, P0[2] + z1, c)  # noqa: E731
    return rx.Compound(pieces=(
        p(-LEVER_LEN, 0.005, STRAP_S - 0.011, STRAP_S + 0.011, -0.005, 0.005),
        p(-0.006, 0.006, 0.231, 0.239, QN - 0.006, 0.0),  # ear
    ))


# -- loop (rigid wire loop on the lever's pin; its hole hangs over the hook's peg) -------------------------------------
Q0 = (P0[0], 0.227, ZA + PZ + QN)
RING_PEG_LOCAL = (LINK * math.sin(math.radians(BETA0)), 0.0, LINK * math.cos(math.radians(BETA0)))  # peg point in the ring frame


def _ring_compound() -> rx.Compound:
    b = math.radians(BETA0)
    q = (0.0, math.sin(b / 2), 0.0, math.cos(b / 2))
    pieces = []
    for x0, x1, z0, z1 in [(-0.010, 0.010, -0.006, 0.011), (-0.010, -0.005, 0.011, 0.031), (0.005, 0.010, 0.011, 0.031),
                           (-0.010, 0.010, 0.025, 0.031)]:  # base, two sides, top bar around a 10 x 14 mm hole
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        pieces.append(rx.Piece(size=(x1 - x0, 0.008, z1 - z0),
                               offset=(cx * math.cos(b) + cz * math.sin(b), 0.0, -cx * math.sin(b) + cz * math.cos(b)),
                               rot_xyzw=q))
    return rx.Compound(pieces=tuple(pieces))


# -- extinguisher (loose): Mesh from extinguisher.obj, bbox centre -------------------------------------------------------
EXT_C = (XA, 0.205, ZA + EXT_DZ)
# the powder fills the body, the head is hollow: centre of mass on the axis 160 mm from the bottom (40 % of the height)
EXT_COM = (0.0, 0.160 - EXT_C[1], -EXT_DZ)  # in the mesh frame (origin at the bounding-box centre)


def f3(v):
    return tuple(float(x) for x in v)


LIFT = 0.0005  # parts on the plate spawn this far above their supports (no spawn-time interpenetration)


def _pos(c, lift=LIFT):
    return f3((c[0], Y0 + c[1], TOP + c[2] + lift))


PLATE = Box("plate", pos=_pos(PLATE_C, 0.0), size=PLATE_SIZE, color=WOOD)
BRACKET = Box("bracket", pos=_pos(BRACKET_C), size=BRACKET_SIZE, mass=BRACKET_MASS, color=BRACKET_GREY)
CARRIAGE = Box("strap_end", pos=_pos(P0), size=CARRIAGE_SIZE, mass=0.050, color=STRAP_BLACK)
LEVER = Box("lever", pos=_pos(P0), size=LEVER_SIZE, mass=0.040, color=LEVER_YELLOW)
RING = Box("loop", pos=_pos(Q0), size=(0.020, 0.008, 0.037), mass=0.020, color=STEEL)
EXTINGUISHER = Box("extinguisher", pos=_pos(EXT_C, 2 * LIFT), size=EXT_SIZE, mass=EXT_MASS, color=EXT_RED)

STRAP = rx.Prismatic(anchor=(0.0, 0.0, 0.0), axis="X", lower=-0.006, upper=0.008, target=U_TARGET, stiffness=K_STRAP,
                     damping=12.0, parent="bracket", initial=CARRIAGE_INITIAL)
HINGE = rx.Revolute(anchor=(0.0, 0.0, 0.0), axis="Y", lower=0.0, upper=OPEN_DEG, damping=0.0002,
                    parent="strap_end", initial=OPEN_DEG)
PIN = rx.Revolute(anchor=(0.0, 0.0, 0.0), axis="Y", lower=-150.0, upper=30.0, damping=0.00005, parent="lever",
                  initial=RING_INITIAL)
rx.install({
    "plate": _plate_compound(),
    "bracket": (_bracket_compound(), rx.Friction(0.08, 0.08)),  # steel parts: the hook contact is metal on metal
    "strap_end": (_carriage_compound(), STRAP, rx.Friction(0.15, 0.15)),
    "lever": (_lever_compound(), HINGE, rx.Friction(0.15, 0.15)),
    "loop": (_ring_compound(), PIN, rx.Friction(0.05, 0.05)),
    "extinguisher": (rx.Mesh(os.path.join(HERE, "extinguisher.obj"), approximation="convexHull"),
                     rx.CenterOfMass(EXT_COM), rx.Friction(0.30, 0.30)),  # painted steel on the bracket's steel
})
OBJECTS = [PLATE, BRACKET, CARRIAGE, LEVER, RING, EXTINGUISHER]


# -- measures ---------------------------------------------------------------------------------------------------------
def _rot(state):
    return quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))


def _bounds_centre(compound) -> np.ndarray:
    """Centre of a part's local geometry bounds (body frame): yam_arena reports this, not the body origin, as ``pos``."""
    pts = []
    for p in compound.pieces:
        if isinstance(p, rx.Mesh):
            pts += [np.asarray(v, float) for v in rx.read_obj(p.path)[0]]
            continue
        r = quat_xyzw_to_matrix(np.asarray(p.rot_xyzw, float))
        h = np.asarray(p.size, float) / 2
        pts += [r @ (h * np.array(c)) + np.asarray(p.offset) for c in
                [(a, b, d) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]]
    a = np.asarray(pts)
    return (a.max(0) + a.min(0)) / 2


PART_C = {"bracket": _bounds_centre(_bracket_compound()), "strap_end": _bounds_centre(_carriage_compound()),
          "lever": _bounds_centre(_lever_compound()), "loop": _bounds_centre(_ring_compound()),
          "extinguisher": _bounds_centre(rx.Compound(pieces=(rx.Mesh(os.path.join(HERE, "extinguisher.obj")),)))}


def origin_of(name, state) -> np.ndarray:
    """Body origin of a part from its reported ``pos`` (the centre of its geometry bounds)."""
    return np.asarray(state.pos, float) - _rot(state) @ PART_C[name]


def centroid_of(name, origin, quat) -> np.ndarray:
    """The ``pos`` yam_arena reports for a part whose body origin is at ``origin`` with orientation ``quat``."""
    return np.asarray(origin, float) + quat_xyzw_to_matrix(np.asarray(quat, float)) @ PART_C[name]


def tilt_deg(state) -> float:
    return math.degrees(math.acos(float(np.clip(_rot(state)[2, 2], -1.0, 1.0))))


def lever_angle_deg(state) -> float:
    """Degrees the lever is raised from its closed stop (rotation about +y)."""
    r = _rot(state)
    return math.degrees(math.atan2(-float(r[2, 0]), float(r[0, 0])))


def _bracket_at(objects):
    b = objects["bracket"]
    return origin_of("bracket", b), _rot(b)


def hook_world(objects) -> np.ndarray:
    p, r = _bracket_at(objects)
    return p + r @ np.asarray(HOOK_REL)


def loop_on_hook(objects) -> bool:
    """The loop's hole is around the peg: its peg point within 12 mm of the hook in x/z and 10 mm along y."""
    ring = objects["loop"]
    d = origin_of("loop", ring) + _rot(ring) @ np.asarray(RING_PEG_LOCAL) - hook_world(objects)
    return bool(math.hypot(d[0], d[2]) <= LOOP_OFF_M and abs(d[1] - (Q0[1] - HOOK[1])) <= LOOP_OFF_Y)


def extinguisher_seated(objects) -> bool:
    e = objects["extinguisher"]
    d = origin_of("extinguisher", e) - np.asarray(EXTINGUISHER.pos)
    return bool(math.hypot(d[0], d[1]) <= EXT_SHIFT_M and -EXT_LIFT_M * 0.5 <= d[2] <= EXT_LIFT_M and tilt_deg(e) <= TILT_DEG)


def bracket_in_place(objects) -> bool:
    b = objects["bracket"]
    d = origin_of("bracket", b) - np.asarray(BRACKET.pos)
    return bool(math.hypot(d[0], d[1]) <= BRACKET_SHIFT_M and abs(d[2]) <= BRACKET_SHIFT_M and tilt_deg(b) <= TILT_DEG)


def latch_intact(objects) -> bool:
    """Strap end, lever and loop are still joined (a broken joint would show as a displaced part)."""
    b, c, lv, rg = (objects[n] for n in ("bracket", "strap_end", "lever", "loop"))
    o = {n: origin_of(n, s) for n, s in (("bracket", b), ("strap_end", c), ("lever", lv), ("loop", rg))}
    local = _rot(b).T @ (o["strap_end"] - o["bracket"]) - (np.asarray(_pos(P0)) - np.asarray(_pos(BRACKET_C)))
    ok_c = abs(local[1]) <= JOINT_BREAK_M and abs(local[2]) <= JOINT_BREAK_M and -0.010 <= local[0] <= 0.012
    ok_l = np.linalg.norm(o["lever"] - o["strap_end"]) <= JOINT_BREAK_M
    q = o["lever"] + _rot(lv) @ np.array([0.0, Q0[1] - STRAP_S, QN])
    ok_r = np.linalg.norm(o["loop"] - q) <= JOINT_BREAK_M
    return bool(ok_c and ok_l and ok_r)


def tcp_points() -> list:
    """World TCP positions of both arms this step, read from the embodiment up the call stack; [] offline."""
    emb = _embodiment()
    if emb is None:
        return []
    from yam_arena import runtime

    return [ik.tcp_pose(runtime.arm_q(emb._base.scene[f"{a}_robot"]), station.BASES[a])[0] for a in ("left", "right")]


def _embodiment():
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and getattr(s, "_base", None) is not None and hasattr(s, "_t") and hasattr(s, "_held"):
            return s
        f = f.f_back
    return None


def box_distance(state, size, point, centre=(0.0, 0.0, 0.0), origin=None) -> float:
    """Distance from ``point`` to a box of ``size`` whose centre is ``centre`` in the part's frame (0 inside)."""
    origin = np.asarray(state.pos, float) if origin is None else origin
    local = _rot(state).T @ (np.asarray(point, float) - origin) - np.asarray(centre)
    half = np.asarray(size, float) / 2
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def hands_clear(objects, tcps) -> bool:
    lv, e = objects["lever"], objects["extinguisher"]
    lo, eo = origin_of("lever", lv), origin_of("extinguisher", e)
    return all(box_distance(lv, LEVER_SIZE, p, LEVER_ARM_CENTRE, lo) >= CLEAR_M
               and box_distance(e, EXT_SIZE, p, origin=eo) >= CLEAR_M for p in tcps)


_STATE = {"t": None, "fault": False, "started": False, "min_angle": 180.0, "ok_since": None, "step": 0, "hist": []}


def new_episode():
    _STATE.update(t=None, fault=False, started=False, min_angle=180.0, ok_since=None, step=0, hist=[])


def _grade(objects) -> float:
    emb = _embodiment()
    t = None if emb is None else float(emb._t)
    fresh = t is not None and (_STATE["t"] is None or t < _STATE["t"])
    if fresh:
        new_episode()
    if t is not None:
        _STATE["t"] = t
    lever = objects["lever"]
    angle = lever_angle_deg(lever)
    if fresh:
        _STATE["started"] = START_MIN_DEG <= angle <= START_MAX_DEG
    bad = (not loop_on_hook(objects) or not extinguisher_seated(objects) or not bracket_in_place(objects)
           or not latch_intact(objects) or (t is not None and not _STATE["started"]))
    _STATE["fault"] = (_STATE["fault"] or bad) if t is not None else bad
    _STATE["min_angle"] = min(_STATE["min_angle"], angle) if t is not None else angle
    tcps = tcp_points()
    clear = hands_clear(objects, tcps)
    down = angle <= CLOSED_DEG and clear
    if t is not None:
        if down and _STATE["ok_since"] is None:
            _STATE["ok_since"] = t
        elif not down:
            _STATE["ok_since"] = None
    held_s = math.inf if t is None else (0.0 if _STATE["ok_since"] is None else t - _STATE["ok_since"])
    # the lever sits on its hard stop under strap tension, where the joint-limit solve jitters its reported velocity;
    # so it is judged still by its angle over the last 1 s, the extinguisher by its velocity
    if t is not None:
        _STATE["hist"] = [(ht, a) for ht, a in _STATE["hist"] if ht >= t - SUCCESS_HOLD_S] + [(t, angle)]
    angles = [a for _, a in _STATE["hist"]] or [angle]
    span = max(angles) - min(angles)
    at_rest = span < LEVER_STILL_DEG and objects["extinguisher"].resting()
    if _STATE["fault"]:
        val = 0.0
    elif down and held_s >= SUCCESS_HOLD_S and at_rest:
        val = 1.0
    elif _STATE["min_angle"] <= PARTIAL_DEG:
        val = 0.5
    else:
        val = 0.0
    if t is not None:
        _STATE["step"] += 1
        if _STATE["step"] == 1 or _STATE["step"] % 10 == 0:
            e, ring = objects["extinguisher"], objects["loop"]
            dh = origin_of("loop", ring) + _rot(ring) @ np.asarray(RING_PEG_LOCAL) - hook_world(objects)
            eo, lo = origin_of("extinguisher", e), origin_of("lever", lever)
            tcp = [tuple(round(float(v), 4) for v in p) for p in tcps[1:]]  # right arm
            sp = " ".join(f"{n}_v={1000 * np.linalg.norm(objects[n].lin_vel):.2f} {n}_w={np.linalg.norm(objects[n].ang_vel):.3f}"
                          for n in ("extinguisher", "lever", "strap_end", "loop"))
            jv = math.degrees(float(lever.ang_vel[1] - objects["strap_end"].ang_vel[1]))
            print(f"EXT_STEP step={_STATE['step']} t={t:.2f} lever_deg={angle:.2f} min_deg={_STATE['min_angle']:.1f} "
                  f"loop_off_mm={1000 * math.hypot(dh[0], dh[2]):.1f} hook={int(loop_on_hook(objects))} "
                  f"seated={int(extinguisher_seated(objects))} bracket_ok={int(bracket_in_place(objects))} "
                  f"intact={int(latch_intact(objects))} clear={int(clear)} held_s={held_s:.2f} "
                  f"fault={int(_STATE['fault'])} score={val:.1f} ext_d_mm={np.round(1000 * (eo - np.asarray(EXTINGUISHER.pos)), 1).tolist()} "
                  f"ext_tilt={tilt_deg(e):.1f} lever_d_mm={np.round(1000 * (lo - np.asarray(configuration(OPEN_DEG)['lever'][0])), 1).tolist()} "
                  f"loop_d_mm={np.round(1000 * (origin_of('loop', ring) - np.asarray(configuration(OPEN_DEG)['loop'][0])), 1).tolist()} "
                  f"carr_d_mm={np.round(1000 * (origin_of('strap_end', objects['strap_end']) - np.asarray(configuration(OPEN_DEG)['strap_end'][0])), 1).tolist()} "
                  f"{sp} hinge_dps={jv:.2f} lever_rest={int(lever.resting())} ext_rest={int(e.resting())} "
                  f"span_deg={span:.3f} at_rest={int(at_rest)} rtcp={tcp}", flush=True)
    return val


def score(objects) -> float:
    """1.0: lever within 10 deg of closed, loop on hook, extinguisher seated, bracket in place, hands clear, held 1 s at
    rest. 0.5: the lever was pushed through 45 deg of its arc with no fault. Any fault latches 0."""
    return _grade(objects)


def success(objects) -> bool:
    return _grade(objects) == 1.0


def distance(objects) -> float:
    return max(0.0, lever_angle_deg(objects["lever"])) * math.pi / 180.0


# -- configuration of the latch at any lever angle (spawn and goal poses; used by the tests) ---------------------------
def configuration(theta_deg: float) -> dict:
    """World (origin, quat_xyzw) of strap end, lever and loop with the lever at ``theta_deg`` and the loop taut."""
    u = mech_px(theta_deg) - PX0

    def qy(deg):
        h = math.radians(deg) / 2
        return (0.0, math.sin(h), 0.0, math.cos(h))

    p = (P0[0] + u, Y0 + P0[1], TOP + P0[2] + LIFT)
    qx, qz = mech_q(theta_deg)
    q = (XA + qx, Y0 + Q0[1], TOP + ZA + qz + LIFT)
    return {"strap_end": (p, (0.0, 0.0, 0.0, 1.0)), "lever": (p, qy(theta_deg)),
            "loop": (q, qy(link_beta_deg(theta_deg) - BETA0))}


# -- demo (right arm; the left stays at READY) ------------------------------------------------------------------------
PUSH_R = 0.045  # contact point along the lever from the pivot
PAD_TIP = 0.013  # pad tips are 13 mm below the TCP
PAD_HALF = 0.006  # half of the closed pair of pads along the closing axis (the fingers close along x at yaw 90 deg)
YAW = math.pi / 2


def _contact(theta_deg: float, lead: float = 0.002) -> tuple:
    """TCP for the closed pads against the lever's push face (the outer face) at ``theta_deg``, ``lead`` m into it."""
    t = math.radians(theta_deg)
    px, pz = XA + mech_px(theta_deg), ZA + PZ
    d = 0.005 + PAD_HALF - lead + (PAD_TIP - PAD_HALF) * max(0.0, math.cos(t)) ** 2  # side of the pads -> their tips
    x = px - PUSH_R * math.cos(t) + d * math.sin(t)
    z = pz + LIFT + PUSH_R * math.sin(t) + d * math.cos(t)
    return (float(x), float(Y0 + STRAP_S), float(TOP + z))


def demo():
    """Right hand comes in from +x at the raised lever, pushes it down with a flat sweep of the closed pads, presses
    it onto its stop, lifts clear and parks off to the +x side."""
    w = lambda d, p, lab: Waypoint(d, tuple(float(v) for v in p), FINGER_CLOSED, lab, YAW)  # noqa: E731
    start = _contact(OPEN_DEG, lead=0.0)
    pre = (start[0] + 0.05, start[1], start[2])
    seq = [w(2.5, pre, "above and beside the raised lever"), w(1.5, start, "pads touch the lever")]
    for deg in (90, 80, 70, 60, 50, 42, 35, 28, 20, 12, 4):
        seq.append(w(0.55, _contact(float(deg)), f"sweep the lever down: {deg} deg"))
    end = _contact(0.0, lead=0.001)
    seq += [w(0.8, end, "press the lever onto its stop"), w(0.5, end, "hold")]
    up = (end[0], end[1], end[2] + 0.025)  # pad tips 12 mm over the lever, 30 mm over the hook peg
    park = (XA + 0.130, end[1], TOP + 0.10)  # beyond the extinguisher's +x side by more than 3 cm
    seq += [w(1.2, up, "lift off the lever"), w(1.5, (park[0], park[1], up[2]), "away to +x"), w(1.0, park, "lower clear"),
            w(3.0, park, "hold still")]
    return {"right": seq}
