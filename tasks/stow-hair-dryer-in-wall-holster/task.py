"""An unplugged hair dryer lies on a bathroom vanity in front of its wall-mounted tapered stainless-steel spiral
holder. A porcelain tile carries the holder. The dryer's barrel points toward the holder and its handle lies to the
near side. Stow the dryer barrel-first into the coil and release it once gravity and the coil hold it. The coil is
four rigid rings (no spring, no cord); the dryer ends wedged barrel-down in the 70 mm ring, not hanging by its handle.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402

import yam_arena.demo as _demo  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
INSTRUCTION = "Stow the unplugged hair dryer in the wall holder so it remains supported."

# Published dimensions: Smedbo HK323, 155 x 129 x 100 mm, 90 / 60 mm openings,
# 302 g. Panasonic EH-NA2C, 154 x 140 x 79 mm, 368 g without its cord/nozzle.
TILE_SIZE = (0.009, 0.298, 0.298)
HOLSTER_SIZE = (0.129, 0.100, 0.155)
HOLSTER_MASS_KG = 0.302  # fixed by wall mounting
DRYER_SIZE = (0.154, 0.140, 0.079)
DRYER_MASS_KG = 0.368
BARREL_DIAMETER_M = 0.070
TOP_OPENING_M = 0.090
LOWER_OPENING_M = 0.060
ENTRY_CLEARANCE_M = (TOP_OPENING_M - BARREL_DIAMETER_M) / 2
TOP_TURN_Z = 0.257
LOWER_TURN_Z = 0.106
HOLSTER_XY = (0.325, -0.125)
HOLSTER_C = (HOLSTER_XY[0], HOLSTER_XY[1], TOP + 0.1815)
HOLSTER_REPORTED_C = (HOLSTER_C[0] + 0.0151, HOLSTER_C[1], HOLSTER_C[2])  # mesh bounds centre
TILE_C = (0.410, HOLSTER_XY[1], TOP + TILE_SIZE[2] / 2)
DRYER_C = (0.190, -0.220, TOP + DRYER_SIZE[2] / 2 + 0.0005)

TILE = Box("wall_tile", pos=tuple(float(v) for v in TILE_C), size=TILE_SIZE, color=(0.84, 0.86, 0.85))
HOLSTER = Box("holster", pos=tuple(float(v) for v in HOLSTER_C), size=HOLSTER_SIZE, color=(0.66, 0.70, 0.72))
DRYER = Box("dryer", pos=tuple(float(v) for v in DRYER_C), size=DRYER_SIZE,
            mass=DRYER_MASS_KG, color=(0.15, 0.33, 0.48))
DRYER_VERTICES = np.asarray(rx.read_obj(os.path.join(HERE, "dryer.obj"))[0], float)
rx.install({"holster": rx.Mesh(os.path.join(HERE, "holster.obj"), approximation="convexHull"),
            "dryer": rx.Mesh(os.path.join(HERE, "dryer.obj"), approximation="convexHull")})
OBJECTS = [TILE, HOLSTER, DRYER]

PARTIAL_DEPTH_M = 0.020
HOLDER_MAX_MOVE_M = 0.010
FALL_LOW_Z = TOP - 0.010
RETENTION_S = 1.0
CONTACT_EPS_N = 1e-3  # numeric noise floor: any robot link touching the dryer counts as contact
STILL_M_S = 0.012
STILL_RAD_S = 0.15
FOOTPRINT_TOL_M = 0.012


def _rot(s):
    return quat_xyzw_to_matrix(np.asarray(s.quat_xyzw, float))


def barrel_axis(s):
    return _rot(s)[:, 0]


def barrel_tip(s):
    """Front (narrow) end of the dryer's 154 mm barrel, in world coordinates."""
    return np.asarray(s.pos, float) + _rot(s) @ np.array([-.077, .0305, 0.0])


def barrel_center(s):
    return np.asarray(s.pos, float) + _rot(s) @ np.array([0.0, .0305, 0.0])


def _embodiment():
    f = sys._getframe(1)
    while f is not None:
        v = f.f_locals.get("self")
        if v is not None and getattr(v, "_base", None) is not None and hasattr(v, "_t") and hasattr(v, "_ids"):
            return v
        f = f.f_back
    return None


_VIEWS = {"base": None, "coil": None, "hands": None}


def _to_numpy(v):
    if hasattr(v, "torch"):
        return v.torch.cpu().numpy()
    if hasattr(v, "numpy"):
        return v.numpy()
    return np.asarray(v)


def contact_forces(base):
    """(coil N, all robot links N), or None if PhysX has not supplied a contact readout."""
    if base is None:
        return None
    if _VIEWS["base"] is not base:
        _VIEWS.update(base=base, coil=None, hands=None)
    if _VIEWS["coil"] is None:
        try:
            import carb.settings
            from pxr import Usd, UsdPhysics

            carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
            stage = base.scene.stage
            links = []
            for arm in ("LeftArm", "RightArm"):
                prim = stage.GetPrimAtPath(f"/World/envs/env_0/{arm}")
                if prim.IsValid():
                    links.extend(p.GetPath().pathString for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                                 if p.HasAPI(UsdPhysics.RigidBodyAPI))
            # The fixed holster is a static collider: filter on its collision shapes, not its Xform (run 4 read 0 N).
            coil = [p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/holster"))
                    if p.HasAPI(UsdPhysics.CollisionAPI)]
            print(f"HOLSTER_CONTACT_SHAPES {len(coil)}", flush=True)
            sim = base.sim.physics_sim_view
            sensor = ["/World/envs/env_0/dryer"]
            _VIEWS["coil"] = sim.create_rigid_contact_view(sensor, filter_patterns=[coil],
                                                             max_contact_data_count=512)
            _VIEWS["hands"] = sim.create_rigid_contact_view(sensor, filter_patterns=[links],
                                                              max_contact_data_count=128)
        except Exception as exc:
            print(f"HOLSTER_CONTACT_ERROR {type(exc).__name__}: {exc}", flush=True)
            return None
    try:
        dt = float(base.sim.get_physics_dt())
        vals = []
        for key in ("coil", "hands"):
            v = _VIEWS[key]
            if v.filter_count <= 0:
                return None
            matrix = _to_numpy(v.get_contact_force_matrix(dt)).reshape(v.sensor_count, v.filter_count, 3)
            vals.append(float(np.linalg.norm(matrix[0], axis=1).sum()))
        return tuple(vals)
    except Exception as exc:
        print(f"HOLSTER_CONTACT_READ_ERROR {type(exc).__name__}: {exc}", flush=True)
        return None


_S = {"t": None, "step": 0, "fault": False, "clear_since": None, "logged": False}


def reset_episode():
    _S.update(t=None, step=0, fault=False, clear_since=None, logged=False)


def grade(objects, *, t=None, forces=None):
    """Shared scorer, with injectable physical readings for the offline checks."""
    d, h = objects["dryer"], objects["holster"]
    if t is not None and _S["t"] is not None and t < _S["t"]:
        reset_episode()
    if t is not None and _S["t"] == t:
        return _S.get("score", 0.0)
    _S["t"] = t
    _S["step"] += 1
    holder_move = float(np.linalg.norm(np.asarray(h.pos, float) - np.asarray(HOLSTER_REPORTED_C, float)))
    world_vertices = DRYER_VERTICES @ _rot(d).T + np.asarray(d.pos, float)
    fell = (float(world_vertices[:, 2].min()) < FALL_LOW_Z or world_vertices[:, 0].max() < 0.0
            or world_vertices[:, 0].min() > 0.7 or world_vertices[:, 1].min() > 0.45
            or world_vertices[:, 1].max() < -0.45)
    fault_now = fell or holder_move > HOLDER_MAX_MOVE_M
    _S["fault"] = _S["fault"] or fault_now
    tip, centre, axis = barrel_tip(d), barrel_center(d), barrel_axis(d)
    within = math.hypot(centre[0]-HOLSTER_XY[0], centre[1]-HOLSTER_XY[1]) <= FOOTPRINT_TOL_M
    entered = tip[2] <= TOP + TOP_TURN_Z - PARTIAL_DEPTH_M and axis[2] > 0.85 and within
    coil_n, hand_n = forces if forces is not None else (0.0, math.inf)
    coil_touch = coil_n > CONTACT_EPS_N
    released = hand_n <= CONTACT_EPS_N  # no robot link touches the dryer
    seated = (entered and coil_touch and TOP + LOWER_TURN_Z - .015 <= tip[2]
              <= TOP + TOP_TURN_Z - PARTIAL_DEPTH_M and within)
    still = float(np.linalg.norm(d.lin_vel)) <= STILL_M_S and float(np.linalg.norm(d.ang_vel)) <= STILL_RAD_S
    if t is not None:
        if seated and released and still:
            if _S["clear_since"] is None:
                _S["clear_since"] = t
        else:
            _S["clear_since"] = None
    hold = math.inf if t is None and seated and released and still else (
        0.0 if _S["clear_since"] is None or t is None else t - _S["clear_since"])
    score = 0.0 if _S["fault"] else (1.0 if hold >= RETENTION_S else (0.5 if entered and coil_touch else 0.0))
    _S["score"] = score
    if t is not None and (_S["step"] == 1 or _S["step"] % 10 == 0):
        q = [round(float(v), 4) for v in d.quat_xyzw]
        print(f"HOLSTER_STEP step={_S['step']} t={t:.2f} dryer_pos={np.round(d.pos,4).tolist()} "
              f"dryer_quat={q} tip_z={tip[2]-TOP:.4f} axis_z={axis[2]:.3f} "
              f"barrel_xy_err_mm={1000*math.hypot(centre[0]-HOLSTER_XY[0],centre[1]-HOLSTER_XY[1]):.1f} "
              f"tip_xy_err_mm={1000*math.hypot(tip[0]-HOLSTER_XY[0],tip[1]-HOLSTER_XY[1]):.1f} "
              f"right_tcp={_S.get('tcp')} "
              f"coil_n={coil_n:.2f} hand_n={hand_n:.2f} "
              f"still={int(still)} held_s={hold:.2f} holder_move_mm={1000*holder_move:.1f} "
              f"fault={int(_S['fault'])} score={score:.1f}", flush=True)
    return score


def _right_tcp(emb):
    try:
        from yam_arena import runtime
        return np.round(ik.tcp_pose(runtime.arm_q(emb._base.scene["right_robot"]), station.BASES["right"])[0], 4).tolist()
    except Exception as exc:  # diagnostics only
        return f"err:{type(exc).__name__}"


def score(objects):
    emb = _embodiment()
    _S["tcp"] = None if emb is None else _right_tcp(emb)
    return grade(objects, t=None if emb is None else float(emb._t),
                 forces=None if emb is None else contact_forces(emb._base))


def success(objects):
    return score(objects) == 1.0


def distance(objects):
    d = objects["dryer"]
    return float(np.linalg.norm(barrel_center(d) - np.array([*HOLSTER_XY, TOP + .195])))


class Grip(float):
    def __new__(cls, matrix):
        self = super().__new__(cls, 1.0)
        self.matrix = np.asarray(matrix, float)
        return self


_default_grip_rotation = _demo.grip_rotation
_demo.grip_rotation = lambda a=0.0: a.matrix if hasattr(a, "matrix") else _default_grip_rotation(a)

# Rest pose of the dryer after spawn settle (GPU run 3, no jitter): it rolls 13 deg onto its handle tip.
REST_POS = np.array((0.1885, -0.2283, 0.7784))
REST_ROT = quat_xyzw_to_matrix(np.array((0.1123, -0.0210, 0.0021, 0.9934)))
UPRIGHT_ROT = axis_angle_matrix(np.array([0., 1., 0.]), -math.pi / 2)  # barrel axis up, tip down, handle -y
# Gripper in the dryer frame: fingers close across the 25 mm handle (dryer x), approach tilted 35 deg from
# straight down toward the body so the 43 mm pads lie along the handle and resist the dryer's turning moment.
APPROACH_TILT = math.radians(35)  # 48 deg from vertical once the 13 deg roll is added
GRIP_IN_DRYER = axis_angle_matrix(np.array([1., 0., 0.]), APPROACH_TILT) @ _default_grip_rotation(math.pi / 2)
GRIP_POINT = np.array((0.0535, -0.040, 0.0))  # handle centre line, 31 mm out from the rear housing
APPROACH = GRIP_IN_DRYER[:, 2]
HANDLE_W = 0.025
HANDLE_CLOSE = -(HANDLE_W / 2 - 0.008)  # partial close: 4.5 mm per finger, about 40 N per pad
HANDLE_TO_TIP = 0.077 + 0.041  # handle underside to barrel tip along the axis
RING_TOP_Z = TOP_TURN_Z + 0.002


def _tcp(dryer_pos, dryer_rot, rise=0.0):
    p = np.asarray(dryer_pos, float) + dryer_rot @ GRIP_POINT + np.array((0.0, 0.0, rise))
    return tuple(float(v) for v in p)


# Measured in GPU run 4: holding the dryer upright, the TCP rode 22 mm low and about 4 mm short in x.
CARRY_SAG = np.array((0.004, 0.003, 0.022))


def _upright_at(tip_z, sag=1.0):
    """Commanded dryer pose for its barrel on the coil axis with its tip ``tip_z`` above the table."""
    return np.array((HOLSTER_XY[0], HOLSTER_XY[1] - 0.0305, TOP + tip_z + 0.077)) + sag * CARRY_SAG


def demo():
    """Right hand takes the handle root, lifts, turns the barrel down high above the coil and lowers it until the
    coil bears the dryer (barrel wedged in the 70 mm ring), opens, and backs away along the handle. Tuned to the
    measured rest pose and carry sag; one passing run, robustness not checked."""
    pick, up = Grip(REST_ROT @ GRIP_IN_DRYER), Grip(UPRIGHT_ROT @ GRIP_IN_DRYER)
    lifted = REST_POS + np.array((0.0, 0.0, 0.110))
    turned = np.array((0.240, -0.190, TOP + 0.300 + 0.077)) + CARRY_SAG  # tip 41 mm over the 259 mm coil top
    over = _upright_at(0.300)
    hover = _upright_at(RING_TOP_Z + 0.011)  # tip 11 mm above the top turn
    seat = _upright_at(RING_TOP_Z - HANDLE_TO_TIP - 0.004, sag=0.5)  # handle 4-15 mm into the ring: coil bears it
    tcp_seat = np.asarray(_tcp(seat, UPRIGHT_ROT))
    retreat = tcp_seat - 0.080 * (UPRIGHT_ROT @ APPROACH)
    park = (0.280, -0.380, TOP + 0.300)
    w = Waypoint
    right = [
        w(2.0, _tcp(REST_POS, REST_ROT, 0.060), FINGER_OPEN, "above handle root", pick),
        w(1.5, _tcp(REST_POS, REST_ROT), FINGER_OPEN, "pads along handle", pick),
        w(1.2, _tcp(REST_POS, REST_ROT), HANDLE_CLOSE, "close on handle", pick),
        w(0.4, _tcp(REST_POS, REST_ROT), HANDLE_CLOSE, "secure grip", pick),
        w(1.5, _tcp(lifted, REST_ROT), HANDLE_CLOSE, "lift dryer clear", pick),
        w(3.0, _tcp(turned, UPRIGHT_ROT), HANDLE_CLOSE, "turn barrel down", up),
        w(3.0, _tcp(over, UPRIGHT_ROT), HANDLE_CLOSE, "over spiral", up),
        w(1.5, _tcp(hover, UPRIGHT_ROT), HANDLE_CLOSE, "tip at top turn", up),
        w(0.8, _tcp(hover, UPRIGHT_ROT), HANDLE_CLOSE, "let carry sag settle", up),
        w(4.5, _tcp(seat, UPRIGHT_ROT), HANDLE_CLOSE, "lower until coil bears dryer", up),
        w(0.5, _tcp(seat, UPRIGHT_ROT), HANDLE_CLOSE, "settle on coil", up),
        w(1.2, _tcp(seat, UPRIGHT_ROT), FINGER_OPEN, "open both fingers", up),
        w(1.2, tuple(float(v) for v in retreat), FINGER_OPEN, "slide off handle", up),
        w(1.5, park, FINGER_OPEN, "park clear", up),
        w(2.5, park, FINGER_OPEN, "hold still", up),
    ]
    return {"right": right}
