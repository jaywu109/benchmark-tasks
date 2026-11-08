"""On the station table, centred on the midline in front of the arms: a fixed stainless-grey small-animal cage body
(140 x 240 x 130 mm) stands at x 0.44 m, y 0.0 m with its front face at x 0.37 m, framed in front (x 0.36 to 0.37 m) by
a hinge-side post on the left (y +0.077 to +0.120 m, 130 mm tall), a latch-side post on the right (y -0.120 to
-0.077 m, 130 mm tall), and an 8 mm tall bottom sill between them (y -0.077 to +0.077 m). In the 154 mm wide front
opening hangs a dark green powder-coated cage door (8 x 150 x 110 mm, 150 g) on a vertical revolute hinge along its
front hinge-side edge at x 0.360 m, y +0.075 m; a weak hinge spring pulls the door outward toward 35 degrees open,
where it starts at rest. Riding on the front of the latch-side post at x 0.349 m, z 0.065 m above the table is a zinc
grey horizontal slide bolt (20 x 40 x 14 mm, 20 g) on a prismatic slide along y (travel 0 to 25 mm toward +y), starting
retracted at y -0.117 to -0.077 m with its tip 2 mm short of the door's free edge. Nothing else is on the table.

The goal: hold the cage door shut against the frame (within 1.5 degrees of closed) and slide the bolt at least 15 mm
across the door's front face so the bolt keeps the door shut after both grippers release and move at least 3 cm clear.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": the score is taken at the trial's last step; success (door shut, bolt thrown >= 15 mm, released and resting for 1 s) ends it.
SCORE_OVER = "end"

STAINLESS_COLOR = (0.62, 0.64, 0.66)

# -- cage body (fixed Box): 140 (x) x 240 (y) x 130 (z) mm at x 0.44, y 0.0 ------------------------------------------
CAGE_X = 0.44
CAGE_Y = 0.0
CAGE_LX = 0.140
CAGE_LY = 0.240
CAGE_H = 0.130
CAGE_SIZE = (CAGE_LX, CAGE_LY, CAGE_H)

CAGE_BODY = Box(
    "cage_body",
    pos=(CAGE_X, CAGE_Y, TOP + CAGE_H / 2),
    size=CAGE_SIZE,
    color=STAINLESS_COLOR,
)

# -- front frame (fixed Boxes), each 10 mm deep in x (x 0.360 to 0.370) ----------------------------------------------
FRAME_X = 0.365
FRAME_LX = 0.010
POST_Y_INNER = 0.077
POST_Y_OUTER = 0.120
POST_LY = POST_Y_OUTER - POST_Y_INNER  # 0.043 m
POST_Y_CENTRE = (POST_Y_INNER + POST_Y_OUTER) / 2  # 0.0985 m
POST_H = 0.130
SILL_LY = 2 * POST_Y_INNER  # 0.154 m (-0.077 to +0.077)
SILL_H = 0.008

HINGE_POST = Box(
    "hinge_post",
    pos=(FRAME_X, +POST_Y_CENTRE, TOP + POST_H / 2),
    size=(FRAME_LX, POST_LY, POST_H),
    color=STAINLESS_COLOR,
)

LATCH_POST = Box(
    "latch_post",
    pos=(FRAME_X, -POST_Y_CENTRE, TOP + POST_H / 2),
    size=(FRAME_LX, POST_LY, POST_H),
    color=STAINLESS_COLOR,
)

SILL = Box(
    "sill",
    pos=(FRAME_X, 0.0, TOP + SILL_H / 2),
    size=(FRAME_LX, SILL_LY, SILL_H),
    color=STAINLESS_COLOR,
)

FRAME = [HINGE_POST, LATCH_POST, SILL]

# -- door (loose Box + rx.Revolute): 8 (x) x 150 (y) x 110 (z) mm, 150 g, powder-coated dark green -------------------
DOOR_LX = 0.008
DOOR_LY = 0.150
DOOR_H = 0.110
DOOR_SIZE = (DOOR_LX, DOOR_LY, DOOR_H)
DOOR_MASS = 0.150
DOOR_COLOR = (0.15, 0.35, 0.22)

DOOR_ZERO_X = 0.364  # front face at x 0.360 flush with frame front; back face at x 0.368
DOOR_ZERO_Y = 0.0
DOOR_BOTTOM_Z = TOP + 0.010  # 2 mm above the 8 mm sill
DOOR_ZERO_Z = DOOR_BOTTOM_Z + DOOR_H / 2  # TOP + 0.065
DOOR_TOP_Z = DOOR_BOTTOM_Z + DOOR_H  # TOP + 0.120

HINGE_X = 0.360
HINGE_Y = 0.075
DOOR_ANCHOR = (-DOOR_LX / 2, +DOOR_LY / 2, 0.0)  # (-0.004, +0.075, 0.0) -> world (0.360, +0.075, DOOR_ZERO_Z)
DOOR_LOWER_DEG = -100.0
DOOR_UPPER_DEG = 0.0
DOOR_OPEN_DEG = -35.0
DOOR_STIFFNESS = 0.0004  # N*m/deg pulling toward -35 deg (35 deg open outward)
DOOR_DAMPING = 0.00003  # N*m*s/deg light viscous damping

DOOR_JOINT = rx.Revolute(
    anchor=DOOR_ANCHOR,
    axis="Z",
    lower=DOOR_LOWER_DEG,
    upper=DOOR_UPPER_DEG,
    initial=DOOR_OPEN_DEG,
    target=DOOR_OPEN_DEG,
    stiffness=DOOR_STIFFNESS,
    damping=DOOR_DAMPING,
)

DOOR = Box(
    "door",
    pos=(DOOR_ZERO_X, DOOR_ZERO_Y, DOOR_ZERO_Z),
    size=DOOR_SIZE,
    mass=DOOR_MASS,
    color=DOOR_COLOR,
)

# -- slide bolt (loose Box + rx.Prismatic along world y): 20 (x) x 40 (y) x 14 (z) mm, 20 g, zinc grey ---------------
BOLT_LX = 0.020
BOLT_LY = 0.040
BOLT_H = 0.014
BOLT_SIZE = (BOLT_LX, BOLT_LY, BOLT_H)
BOLT_MASS = 0.020
BOLT_COLOR = (0.70, 0.70, 0.72)

BOLT_X = 0.349  # back (+x) face at 0.359, 1 mm in front of the door's closed front face at 0.360
BOLT_SPAWN_Y = -0.097  # spans y -0.117 to -0.077 (tip 2 mm short of the door's free edge at y -0.075)
BOLT_Z = TOP + 0.065  # spans z TOP + 0.058 to TOP + 0.072
BOLT_MAX_TRAVEL = 0.025  # 0 to 25 mm toward +y
BOLT_DAMPING = 2.0  # N*s/m so the bolt stays put when not pushed

BOLT_JOINT = rx.Prismatic(
    anchor=(0.0, 0.0, 0.0),
    axis="Y",
    lower=0.0,
    upper=BOLT_MAX_TRAVEL,
    damping=BOLT_DAMPING,
)

BOLT = Box(
    "bolt",
    pos=(BOLT_X, BOLT_SPAWN_Y, BOLT_Z),
    size=BOLT_SIZE,
    mass=BOLT_MASS,
    color=BOLT_COLOR,
)

OBJECTS = [DOOR, BOLT, CAGE_BODY, *FRAME]

rx.install({
    "door": DOOR_JOINT,
    "bolt": BOLT_JOINT,
})

# -- scoring ----------------------------------------------------------------------------------------------------------
DOOR_SHUT_DEG = 1.5
BOLT_FULL_MM = 15.0
BOLT_PARTIAL_MM = 5.0
GRIPPER_CLEAR = 0.030  # a part counts only when both TCPs are at least 3 cm from its box (released)


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


def released(state, tcps, size=None) -> bool:
    """True when every TCP in ``tcps`` is at least GRIPPER_CLEAR from the nearest point of ``state``'s box."""
    if size is None:
        size = DOOR_SIZE if float(state.high[2] - state.low[2]) > 0.05 else BOLT_SIZE
    return all(box_distance(state, size, p) >= GRIPPER_CLEAR for p in tcps)


def door_angle_deg(state) -> float:
    """The door's rotation from its closed zero pose, 2 * acos(|w|) in degrees."""
    w = abs(float(state.quat_xyzw[3]))
    return math.degrees(2.0 * math.acos(min(1.0, w)))


def bolt_throw_mm(state) -> float:
    """The bolt's +y travel from its spawn pose, in mm (bolt centre y - spawn centre y)."""
    return (float(state.pos[1]) - BOLT_SPAWN_Y) * 1000.0


def door_shut(objects) -> bool:
    """True when door_angle_deg <= DOOR_SHUT_DEG = 1.5."""
    return bool(door_angle_deg(objects["door"]) <= DOOR_SHUT_DEG)


def score(objects) -> float:
    """1.0 when the door is shut, bolt_throw_mm >= 15, both are released, and both are resting();
    0.5 when the door is shut, 5 <= bolt_throw_mm < 15, released and resting; 0 otherwise."""
    door = objects["door"]
    bolt = objects["bolt"]
    tcps = tcp_points()
    if not (
        door_shut(objects)
        and released(door, tcps, DOOR_SIZE)
        and released(bolt, tcps, BOLT_SIZE)
        and door.resting()
        and bolt.resting()
    ):
        return 0.0
    throw = bolt_throw_mm(bolt)
    if throw >= BOLT_FULL_MM:
        return 1.0
    if throw >= BOLT_PARTIAL_MM:
        return 0.5
    return 0.0


def success(objects) -> bool:
    """The 1.0 condition (held for 1 s by the runner)."""
    return bool(score(objects) == 1.0)


def distance(objects) -> float:
    """door_angle_deg / 35 + max(0, 15 - bolt_throw_mm) / 15."""
    angle = door_angle_deg(objects["door"])
    throw = bolt_throw_mm(objects["bolt"])
    return float(angle / 35.0 + max(0.0, BOLT_FULL_MM - throw) / BOLT_FULL_MM)


# -- demo (both arms; 12.0 s total; left holds door shut while right slides bolt across) ------------------------------
# Left arm: yaw = -pi/2 turns the closed fingers so tip_left's 26.4 mm broad face pushes in +x.
# At z = TOP + 0.105, tip_left's +x face at the door's top edge (z = TOP + 0.120, rel_z = +15 mm) is +8.72 mm ahead of
# the TCP, so TCP x = 0.3545 presses 3.2 mm past the closed door's front face (x = 0.360) while keeping link4 high
# enough to clear the right arm's link4 by > 7 mm during the bolt throw. (GPU 2026-11-07: x = 0.3520 held the door at
# 0.6 deg, its free edge 1.6 mm proud, and the bolt tip hit it; the joint stop at 0 deg takes the extra push.)
LEFT_YAW = -math.pi / 2
LEFT_WAIT = (0.20, +0.25, TOP + 0.12)
LEFT_PUSH_Z = TOP + 0.105
LEFT_PRE_PUSH = (0.25, 0.0, LEFT_PUSH_Z)
LEFT_PUSH = (0.3545, 0.0, LEFT_PUSH_Z)
LEFT_BACK = (0.3020, 0.0, LEFT_PUSH_Z)  # back off -x by 5 cm
LEFT_PARK = (0.20, +0.40, TOP + 0.12)

# Right arm: yaw = 0.0 with fingers closed. At TCP x = 0.323, z = TOP + 0.025 (lowest pad tip at TOP + 12.1 mm), the
# entire gripper stays >= 3.0 mm in front of the latch-side post (x = 0.360), while at the bolt's height (z = TOP +
# 58..72 mm, rel_z = +33..47 mm) the finger body spans x = 0.294..0.3526 m (overlapping 13.6 mm of the bolt's -y end
# face x = 0.339..0.359 m) with its +y face at rel_y = +19.7 mm. Starting at y = -0.137 (finger +y face at -0.1173,
# 0.3 mm before the bolt's -y end at -0.117) and pushing +y by 22 mm to y = -0.115 throws the bolt to y_end ~ -0.0953
# (bolt_throw_mm ~ 21.7 mm).
RIGHT_YAW = 0.0
RIGHT_WAIT = (0.20, -0.30, TOP + 0.12)
RIGHT_BOLT_X = 0.323
RIGHT_BOLT_Z = TOP + 0.025
RIGHT_APPROACH_Y = -0.220
RIGHT_BOLT_START_Y = -0.137
RIGHT_BOLT_END_Y = -0.115  # +22 mm push in +y
RIGHT_BOLT_BACK_Y = -0.145  # 3 cm back-off in -y
RIGHT_UP_Z = TOP + 0.080
RIGHT_PARK = (0.20, -0.40, TOP + 0.12)
HOLD_S = 2.0


def demo():
    """Left arm moves in front of the open door at (0.25, 0.0, TOP + 0.105) with yaw = -pi/2 and closed fingers,
    pushes +x over 1.5 s to x = 0.3545 so its broad finger face holds the door shut against its opening spring, and
    holds there while the right arm approaches the bolt's -y end at (0.323, -0.137, TOP + 0.025), pushes +y 22 mm over
    1.5 s to slide the bolt ~21 mm across the shut door, backs off 3 cm in -y and rises; then the left arm backs off
    5 cm in -x so the door settles ~0.4 deg against the bolt, and both arms park at (0.20, +-0.40, TOP + 0.12)."""
    left = [
        Waypoint(1.2, LEFT_WAIT, FINGER_CLOSED, "wait", LEFT_YAW),
        Waypoint(1.5, LEFT_PRE_PUSH, FINGER_CLOSED, "in front of open door", LEFT_YAW),
        Waypoint(1.5, LEFT_PUSH, FINGER_CLOSED, "push door shut", LEFT_YAW),
        Waypoint(1.2, LEFT_PUSH, FINGER_CLOSED, "hold door for right approach", LEFT_YAW),
        Waypoint(1.0, LEFT_PUSH, FINGER_CLOSED, "hold door for right to bolt", LEFT_YAW),
        Waypoint(1.5, LEFT_PUSH, FINGER_CLOSED, "hold door while bolt thrown", LEFT_YAW),
        Waypoint(0.5, LEFT_PUSH, FINGER_CLOSED, "hold door while right backs off", LEFT_YAW),
        Waypoint(0.5, LEFT_PUSH, FINGER_CLOSED, "hold door while right rises", LEFT_YAW),
        Waypoint(0.4, LEFT_BACK, FINGER_CLOSED, "back off door", LEFT_YAW),
        Waypoint(0.7, LEFT_PARK, FINGER_CLOSED, "park", LEFT_YAW),
        Waypoint(HOLD_S, LEFT_PARK, FINGER_CLOSED, "hold still", LEFT_YAW),
    ]
    right = [
        Waypoint(1.2, RIGHT_WAIT, FINGER_CLOSED, "wait", RIGHT_YAW),
        Waypoint(1.5, RIGHT_WAIT, FINGER_CLOSED, "wait for left pre-push", RIGHT_YAW),
        Waypoint(1.5, RIGHT_WAIT, FINGER_CLOSED, "wait for door shut", RIGHT_YAW),
        Waypoint(1.2, (RIGHT_BOLT_X, RIGHT_APPROACH_Y, RIGHT_BOLT_Z), FINGER_CLOSED, "lower beside bolt", RIGHT_YAW),
        Waypoint(1.0, (RIGHT_BOLT_X, RIGHT_BOLT_START_Y, RIGHT_BOLT_Z), FINGER_CLOSED, "to bolt end", RIGHT_YAW),
        Waypoint(1.5, (RIGHT_BOLT_X, RIGHT_BOLT_END_Y, RIGHT_BOLT_Z), FINGER_CLOSED, "throw bolt", RIGHT_YAW),
        Waypoint(0.5, (RIGHT_BOLT_X, RIGHT_BOLT_BACK_Y, RIGHT_BOLT_Z), FINGER_CLOSED, "back off in -y", RIGHT_YAW),
        Waypoint(0.5, (RIGHT_BOLT_X, RIGHT_APPROACH_Y, RIGHT_UP_Z), FINGER_CLOSED, "up and out", RIGHT_YAW),
        Waypoint(0.4, RIGHT_PARK, FINGER_CLOSED, "to park", RIGHT_YAW),
        Waypoint(0.7, RIGHT_PARK, FINGER_CLOSED, "park", RIGHT_YAW),
        Waypoint(HOLD_S, RIGHT_PARK, FINGER_CLOSED, "hold still", RIGHT_YAW),
    ]
    return {"left": left, "right": right}
