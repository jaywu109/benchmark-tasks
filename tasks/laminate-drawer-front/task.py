"""On the station table, in front of the right arm: a fixed 160 x 240 x 18 mm MDF drawer-front blank, centred
at x 0.30 m, y 0.0 m, whose top face, like the laminate's underside, is coated with dry contact cement. A laminate sheet, cut oversize to 176 x 260 mm
(8 mm past the blank on each x edge, 10 mm past each y edge), lies on three 12 mm round spacer dowels resting on the
blank along y at x 0.241, 0.300 and 0.359 m. Each dowel is 300 mm long and sticks out 50 mm past the blank's -y edge.
The sheet is three 58 x 260 x 4 mm panels (40 g each, 1 mm apart) joined along y by spring hinges, so it sags where a
dowel is taken from under it.

Contact cement bonds on first contact: a panel bonds the first moment its lowest corner is within 1 mm of the blank's
top and it is within 3 degrees of flat, and from then on it is held where it lies.

The goal: pull the dowels out so that all three panels bond flat to the blank while the sheet still overhangs every
edge of the blank by at least 2 mm, with both grippers' TCPs at least 3 cm from every point of each counted panel.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402

from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": the score is taken at the trial's last step; success (all three panels counted for 1 s) ends it with 1.0.
SCORE_OVER = "end"

# -- scene -------------------------------------------------------------------------------------------------------
BOARD_SIZE = (0.160, 0.240, 0.018)
BOARD_XY = (0.300, 0.000)
BOARD_TOP = TOP + BOARD_SIZE[2]
DOWEL_R = 0.006
DOWEL_LEN = 0.300
DOWEL_Y = (-0.170 + 0.130) / 2  # from y -0.17 to +0.13: 50 mm out past the blank's -y edge
DOWEL_Z = BOARD_TOP + DOWEL_R
PANEL_SIZE = (0.058, 0.260, 0.004)
PANEL_MASS = 0.040
SHEET_Y0 = -PANEL_SIZE[1] / 2  # the sheet's -y edge
PANEL_X = {"m": 0.300, "n": 0.300 - 0.059, "f": 0.300 + 0.059}  # middle, near (toward the bases), far; 1 mm gaps
PANEL_Z = BOARD_TOP + 2 * DOWEL_R + PANEL_SIZE[2] / 2
# A soft bending spring: the middle panel's half weight on a 29.5 mm lever must bend each hinge about 0.41 rad for it
# to reach the blank; at 0.004 N m/rad that leaves 0.0042 of its 0.0058 N m to land it.
HINGE_K = 0.004 * math.pi / 180  # N m/deg
HINGE_C = 0.002 * math.pi / 180  # N m s/deg

BOARD = Box("blank", pos=(*BOARD_XY, TOP + BOARD_SIZE[2] / 2), size=BOARD_SIZE, color=(0.50, 0.36, 0.24))
PANELS = {
    k: Box(f"panel_{k}", pos=(x, 0.0, PANEL_Z), size=PANEL_SIZE, mass=PANEL_MASS, color=(0.45, 0.55, 0.65))
    for k, x in PANEL_X.items()
}
DOWELS = {
    k: Box(f"dowel_{k}", pos=(x, DOWEL_Y, DOWEL_Z), size=(2 * DOWEL_R, DOWEL_LEN, 2 * DOWEL_R), mass=0.060,
           color=(0.86, 0.72, 0.50))
    for k, x in PANEL_X.items()
}
OBJECTS = [BOARD, PANELS["m"], PANELS["n"], PANELS["f"], *DOWELS.values()]
PANEL_NAMES = tuple(p.name for p in PANELS.values())

GAP_HALF = (PANEL_X["m"] - PANEL_X["n"]) / 2
rx.install({
    "panel_n": rx.Revolute(anchor=(+GAP_HALF, 0.0, 0.0), axis="Y", lower=-60.0, upper=60.0, stiffness=HINGE_K,
                           damping=HINGE_C, parent="panel_m"),
    "panel_f": rx.Revolute(anchor=(-GAP_HALF, 0.0, 0.0), axis="Y", lower=-60.0, upper=60.0, stiffness=HINGE_K,
                           damping=HINGE_C, parent="panel_m"),
    **{d.name: rx.Mesh(os.path.join(HERE, "dowel.obj"), approximation="convexHull") for d in DOWELS.values()},
})
rx.GLUE.update(PANEL_NAMES)

# -- bonding -------------------------------------------------------------------------------------------------------
GLUE_GAP = 0.001  # lowest corner within 1 mm of the blank's top
GLUE_TILT_DEG = 3.0  # and within 3 deg of flat
_BOND: dict = {"t": None, "bond": {}}


def tilt_deg(state) -> float:
    from yam_arena.urdf import quat_xyzw_to_matrix

    rot = quat_xyzw_to_matrix(state.quat_xyzw)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


def glue_ready(state) -> bool:
    return float(state.low[2]) - BOARD_TOP <= GLUE_GAP and tilt_deg(state) <= GLUE_TILT_DEG


def _embodiment():
    """The yam_arena embodiment up the call stack (it has the arms and the episode clock); None offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            return s
        f = f.f_back
    return None


def _stage():
    try:
        import omni.usd
    except ImportError:
        return None  # offline
    return omni.usd.get_context().get_stage()


def _set_glue(name, state=None) -> None:
    """Switch a panel's glue joint on at the panel's pose now (``state``), or off (``state`` None)."""
    stage = _stage()
    if stage is None or name not in rx.GLUE_PATHS:
        return
    from pxr import Gf, UsdPhysics

    j = UsdPhysics.FixedJoint(stage.GetPrimAtPath(rx.GLUE_PATHS[name]))
    if state is not None:
        x, y, z, w = (float(v) for v in state.quat_xyzw)
        j.GetLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in state.pos)))
        j.GetLocalRot0Attr().Set(Gf.Quatf(w, x, y, z))
    j.GetJointEnabledAttr().Set(state is not None)


def new_episode() -> None:
    """Unbond every panel (joints off, record cleared)."""
    for name in PANEL_NAMES:
        _set_glue(name, None)
    _BOND["t"] = None
    _BOND["bond"] = {}


def _wrap_reset(emb) -> None:
    """Unbond before the embodiment resets, so no glue joint fights the panels being put back."""
    if getattr(emb, "_laminate_reset_wrapped", False):
        return
    original = emb.reset

    def reset(*args, **kwargs):
        new_episode()
        return original(*args, **kwargs)

    emb.reset = reset
    emb._laminate_reset_wrapped = True


def bonds(objects) -> dict:
    """{panel name: (x, y) where it bonded}; bonds new panels once per control step."""
    emb = _embodiment()
    if emb is not None:
        _wrap_reset(emb)
        t = float(emb._t)
        if _BOND["t"] is not None and t < _BOND["t"]:
            new_episode()
        if t == _BOND["t"]:
            return _BOND["bond"]
        _BOND["t"] = t
    for name in PANEL_NAMES:
        s = objects[name]
        if name not in _BOND["bond"] and glue_ready(s):
            _set_glue(name, s)
            _BOND["bond"][name] = (float(s.pos[0]), float(s.pos[1]))
    return _BOND["bond"]


# -- scoring -------------------------------------------------------------------------------------------------------
REG_X = 0.0060  # 8 mm set overhang - 2 mm
REG_Y = 0.0080  # 10 mm set overhang - 2 mm
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
    """True when every TCP in ``tcps`` is at least GRIPPER_CLEAR from the nearest point of the panel."""
    return all(box_distance(state, PANEL_SIZE, p) >= GRIPPER_CLEAR for p in tcps)


def misregistration(objects) -> float:
    """How far (m) the worst panel is past the registration window (0 when the sheet overhangs every edge by 2 mm)."""
    worst = 0.0
    for k, x in PANEL_X.items():
        s = objects[f"panel_{k}"]
        worst = max(worst, abs(float(s.pos[0]) - x) - REG_X, abs(float(s.pos[1])) - REG_Y)
    return max(0.0, worst)


def counted_panels(objects) -> list[str]:
    """Panels bonded, with the sheet registered, and released by both grippers."""
    b = bonds(objects)
    if misregistration(objects) > 0.0:
        return []
    tcps = tcp_points()
    return [n for n in PANEL_NAMES if n in b and released(objects[n], tcps)]


def score(objects) -> float:
    """1/3 per panel bonded flat to the blank, released, with the sheet overhanging every edge by 2 mm (up to 1.0)."""
    return round(len(counted_panels(objects)) / 3, 10)


def success(objects) -> bool:
    """All three panels counted; held for 1 s it ends the trial."""
    return len(counted_panels(objects)) == len(PANEL_NAMES)


def distance(objects) -> float:
    """Sum over unbonded panels of the lowest corner's height over the blank (m), plus the misregistration (m)."""
    b = bonds(objects)
    lift = sum(max(0.0, float(objects[n].low[2]) - BOARD_TOP) for n in PANEL_NAMES if n not in b)
    return float(lift + misregistration(objects))


# -- demo: right arm pulls the middle, near and far dowels in turn; the left arm stays at READY ---------------------
YAW = math.pi / 2  # jaws close along x, across the dowel
GRIP_Y = -0.155  # 35 mm out past the blank's -y edge
PULL_Y = -0.430  # dowel's +y end to y -0.145, 15 mm clear of the sheet
HOVER_Z = TOP + 0.090
SET_Z = TOP + DOWEL_R + 0.010  # pad tips are 13 mm below the TCP: set the dowel down 10 mm over the table
APPROACH = -0.020  # 40 mm between the pads: fully open (94 mm) would hit the neighbouring dowel ends
RELEASE = -0.020
ORDER = ("m", "n", "f")
PARK = (0.20, -0.32, TOP + 0.12)


def demo():
    w = lambda d, p, f, lab, y=YAW: Waypoint(d, tuple(float(v) for v in p), f, lab, y)  # noqa: E731
    right = []
    for i, k in enumerate(ORDER):
        x = PANEL_X[k]
        right += [
            w(2.0 if i == 0 else 1.5, (x, GRIP_Y, HOVER_Z), APPROACH, f"{k} hover"),
            w(1.0, (x, GRIP_Y, DOWEL_Z), APPROACH, f"{k} down"),
            w(0.8, (x, GRIP_Y, DOWEL_Z), FINGER_CLOSED, f"{k} close"),
            w(3.0, (x, PULL_Y, DOWEL_Z), FINGER_CLOSED, f"{k} pull"),
            w(0.6, (x, PULL_Y, SET_Z), FINGER_CLOSED, f"{k} set down"),
            w(0.5, (x, PULL_Y, SET_Z), RELEASE, f"{k} open"),
            w(0.8, (x, PULL_Y, HOVER_Z), RELEASE, f"{k} rise"),
            w(1.0, (x, PULL_Y, HOVER_Z), RELEASE, f"{k} wait"),
        ]
    right += [w(2.0, PARK, RELEASE, "park", 0.0), w(3.0, PARK, RELEASE, "hold", 0.0)]
    return {"right": right}
