"""An empty, full-size tapered polypropylene shopping basket stands on the table with its single bow handle upright. The 430 by 300 by 225 mm basket weighs 620 g including its hinged handle, which pivots on the far long wall."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import robocurve_ext as rx
import yam_arena.demo as _demo
from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box, ObjectState
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix

SCENE = 'station'
TOP = TABLE_TOP[SCENE]
SCORE_OVER = 'end'
SUCCESS_HOLD_S = 1.0
CONTROL_HZ = 20.0
INSTRUCTION = "Fold the shopping basket's handle flat for storage."

# Full-size Intech EK-4330-EG envelope. Long dimension runs left/right on the table.
BODY_SIZE = (0.300, 0.430, 0.225)
BODY_MASS = 0.540
HANDLE_MASS = 0.080
BODY_CENTER = (0.280, 0.0, float(TOP + 0.1125))
HINGE_REL_BODY = (0.140, 0.0, 0.1225)
HINGE_WORLD = np.asarray(BODY_CENTER) + np.asarray(HINGE_REL_BODY)
HANDLE_RISE = 0.120
HANDLE_SECTION = 0.020
HANDLE_SIZE = (0.020, 0.430, 0.140)
HANDLE_CENTER = (float(HINGE_WORLD[0]), 0.0, float(HINGE_WORLD[2] + 0.060))
HANDLE_ANCHOR = (0.0, 0.0, -0.060)
RAISED_STOP_DEG = -5.0
FOLDED_JOINT_DEG = 90.0
FOLDED_STOP_DEG = FOLDED_JOINT_DEG - RAISED_STOP_DEG
PARTIAL_TRAVEL_DEG = 60.0
FULL_STOP_TOL_DEG = 5.0
TIP_FAULT_DEG = 30.0
RELEASE_CLEAR_M = 0.030
DETACH_TOL_M = 0.020
STILL_STEPS = 20
STILL_TOL_DEG = 0.5
REST_Z_TOL_M = 0.003
LIFT_FAULT_M = 0.020  # basket base more than 2 cm above the table at any time latches 0
TABLE_MIN_X = station.TABLE_CENTER[0] - station.TABLE_SIZE[0]/2
TABLE_MAX_X = station.TABLE_CENTER[0] + station.TABLE_SIZE[0]/2
TABLE_MIN_Y = station.TABLE_CENTER[1] - station.TABLE_SIZE[1]/2
TABLE_MAX_Y = station.TABLE_CENTER[1] + station.TABLE_SIZE[1]/2

OBJECTS = [
    Box('basket', pos=BODY_CENTER, size=BODY_SIZE, mass=BODY_MASS, color=(0.15, 0.42, 0.67)),
    Box('handle', pos=HANDLE_CENTER, size=HANDLE_SIZE, mass=HANDLE_MASS, color=(0.10, 0.26, 0.45)),
]
# A three-sided bow. The center section is its actual top contact surface.
BOW = rx.Compound(pieces=(
    rx.Piece(size=(.020,.390,.020), offset=(0,0,.060)),
    rx.Piece(size=(.020,.020,.120), offset=(0,+.205,0)),
    rx.Piece(size=(.020,.020,.120), offset=(0,-.205,0)),
))
HINGE = rx.Revolute(anchor=HANDLE_ANCHOR, axis='Y', lower=RAISED_STOP_DEG, upper=FOLDED_JOINT_DEG,
                    stiffness=0.0, damping=0.0005, friction=0.015, initial=RAISED_STOP_DEG, parent='basket')
rx.install({'basket': rx.Mesh(os.path.join(HERE, 'basket.obj'), approximation='convexHull'),
            'handle': (BOW, HINGE)})

# Basket body friction is set before any demo tuning. The handle's contact is also polypropylene.
_SCENE_MOD = 'yam_arena.scene'
_orig_patch = rx._patch

def _patch(scene_mod):
    _orig_patch(scene_mod)
    if getattr(scene_mod, '_basket_mat_patched', False):
        return
    import isaaclab.sim as sim_utils
    orig = scene_mod.task_object
    polypropylene = sim_utils.RigidBodyMaterialCfg(static_friction=0.4, dynamic_friction=0.4,
                                                    friction_combine_mode='max', restitution=0.0)
    def task_object(spec):
        obj = orig(spec)
        if spec.name in ('basket', 'handle'):
            obj.object_cfg.spawn.physics_material = polypropylene
            obj.spawner_cfg.physics_material = polypropylene
        return obj
    scene_mod.task_object = task_object
    scene_mod._basket_mat_patched = True
rx._patch = _patch
if _SCENE_MOD in sys.modules:
    _patch(sys.modules[_SCENE_MOD])

# Pitched, open flat fingers sweep the top of the bow. The float wrapper lets ArmPlan use a 3-D grip rotation.
class Grip(float):
    def __new__(cls, matrix):
        val = super().__new__(cls, 0.0)
        val.matrix = np.asarray(matrix, float)
        return val

_orig_grip_rotation = getattr(_demo, '_orig_grip_rotation', _demo.grip_rotation)
_demo._orig_grip_rotation = _orig_grip_rotation

def _rotation(yaw=0.0):
    return yaw.matrix if hasattr(yaw, 'matrix') else _orig_grip_rotation(float(yaw))
_demo.grip_rotation = _rotation
PUSH_GRIP = Grip(axis_angle_matrix(np.array([0.,1.,0.]), math.radians(-60.)) @ _orig_grip_rotation(0.0))


def _embodiment():
    frame = sys._getframe(1)
    while frame is not None:
        s = frame.f_locals.get('self')
        if s is not None and getattr(s, '_base', None) is not None and hasattr(s, '_ids') and hasattr(s, '_t'):
            return s
        frame = frame.f_back
    return None


def tcp_points():
    emb = _embodiment()
    if emb is None:
        return []
    out = []
    for side in ('left', 'right'):
        arm_ids, _ = emb._ids[f'{side}_robot']
        q = emb._base.scene[f'{side}_robot'].data.joint_pos.torch[0].cpu().numpy()[arm_ids]
        out.append(ik.tcp_pose(q.astype(float), station.BASES[side])[0])
    return out


def handle_angle_deg(objects):
    # Relative to the body, so moving the whole basket cannot fake a folded handle.
    rb = quat_xyzw_to_matrix(np.asarray(objects['basket'].quat_xyzw, float))
    rh = quat_xyzw_to_matrix(np.asarray(objects['handle'].quat_xyzw, float))
    r = rb.T @ rh
    return float(math.degrees(math.atan2(r[0,2], r[2,2])) - RAISED_STOP_DEG)


def tip_deg(objects):
    r = quat_xyzw_to_matrix(np.asarray(objects['basket'].quat_xyzw, float))
    return float(math.degrees(math.acos(float(np.clip(r[2,2], -1.0, 1.0)))))


def bow_top(objects):
    h = objects['handle']
    r = quat_xyzw_to_matrix(np.asarray(h.quat_xyzw, float))
    return np.asarray(h.pos, float) + r @ np.array([0., 0., .060])


def hinge_error_m(objects):
    b, h = objects['basket'], objects['handle']
    rb = quat_xyzw_to_matrix(np.asarray(b.quat_xyzw, float))
    rh = quat_xyzw_to_matrix(np.asarray(h.quat_xyzw, float))
    p0 = np.asarray(b.pos, float) + rb @ np.asarray(HINGE_REL_BODY)
    p1 = np.asarray(h.pos, float) + rh @ np.asarray(HANDLE_ANCHOR)
    return float(np.linalg.norm(p0-p1))


def released(objects, tcps=None):
    if tcps is None:
        tcps = tcp_points()
    if len(tcps) != 2:
        return False
    # Nearest point to the three bow members, allowing the two arms to steady the rim earlier.
    h = objects['handle']
    r = quat_xyzw_to_matrix(np.asarray(h.quat_xyzw, float))
    p = np.asarray(h.pos, float)
    centers = [p + r @ np.array([0.,0.,.060]),
               p + r @ np.array([0.,+.205,0.]), p + r @ np.array([0.,-.205,0.])]
    halves = [np.array([.010,.195,.010]), np.array([.010,.010,.060]), np.array([.010,.010,.060])]
    for tcp in tcps:
        local = r.T @ (np.asarray(tcp, float) - p)
        for center, half in zip(centers, halves):
            lc = r.T @ (center - p)
            if np.linalg.norm(np.maximum(np.abs(local-lc)-half, 0.0)) < RELEASE_CLEAR_M:
                return False
    return True


_S = {}
def new_episode():
    _S.clear()
    _S.update(t=None, step=0, fault=None, angle_history=[], max_angle=0.0, rest_steps=0)
new_episode()


def advance(objects):
    emb = _embodiment()
    if emb is not None:
        t = float(emb._t)
        if _S['t'] is not None and t < _S['t']:
            new_episode()
        if _S['t'] == t:
            return
        _S['t'] = t
    _S['step'] += 1
    angle = handle_angle_deg(objects)
    _S['max_angle'] = max(_S['max_angle'], angle)
    hist = _S['angle_history']
    hist.append(angle)
    if len(hist) > 41:
        del hist[:-41]
    b = objects['basket']
    _S['rest_steps'] = _S['rest_steps']+1 if on_table(b) and b.resting() else 0
    # Require the whole upright shell to remain supported by the table footprint.
    if _S['fault'] is None:
        if (b.low[0] < TABLE_MIN_X - .005 or b.high[0] > TABLE_MAX_X + .005 or
            b.low[1] < TABLE_MIN_Y - .005 or b.high[1] > TABLE_MAX_Y + .005 or
            b.pos[2] < TOP + .050 or b.low[2] > TOP + LIFT_FAULT_M):
            _S['fault'] = 'basket_off_table'
        elif tip_deg(objects) > TIP_FAULT_DEG:
            _S['fault'] = 'basket_tipped'
        elif hinge_error_m(objects) > DETACH_TOL_M:
            _S['fault'] = 'handle_detached'
        if _S['fault']:
            print(f"BASKET_FAULT step={_S['step']} reason={_S['fault']} angle_deg={angle:.2f}", flush=True)
    if emb is not None and (_S['step'] == 1 or _S['step'] % 5 == 0):
        print(f"BASKET_STEP step={_S['step']} t={_S['step']/CONTROL_HZ:.2f}s angle_deg={angle:.2f} "
              f"max_angle_deg={_S['max_angle']:.2f} tip_deg={tip_deg(objects):.2f} "
              f"hinge_error_mm={hinge_error_m(objects)*1000:.2f} released={int(released(objects))} "
              f"fault={_S['fault']}", flush=True)


def on_table(b):
    return bool(b.low[2] <= TOP + REST_Z_TOL_M)


def basket_clear(objects, tcps=None):
    # No gripper point within RELEASE_CLEAR_M of the basket's bounding box.
    if tcps is None:
        tcps = tcp_points()
    b = objects['basket']
    return len(tcps) == 2 and all(
        np.linalg.norm(np.maximum(np.maximum(b.low-np.asarray(t, float), np.asarray(t, float)-b.high), 0.0)) >= RELEASE_CLEAR_M
        for t in tcps)


def _still():
    h = _S['angle_history']
    return len(h) >= STILL_STEPS+1 and max(h[-STILL_STEPS-1:])-min(h[-STILL_STEPS-1:]) <= STILL_TOL_DEG


def success(objects, tcps=None):
    advance(objects)
    return (_S['fault'] is None and handle_angle_deg(objects) >= FOLDED_STOP_DEG-FULL_STOP_TOL_DEG
            and tip_deg(objects) <= TIP_FAULT_DEG and _still() and _S['rest_steps'] >= STILL_STEPS
            and released(objects, tcps) and basket_clear(objects, tcps))


def score(objects, tcps=None):
    advance(objects)
    if _S['fault'] is not None:
        return 0.0
    if success(objects, tcps):
        return 1.0
    if _S['max_angle'] >= PARTIAL_TRAVEL_DEG - 1e-6 and tip_deg(objects) <= TIP_FAULT_DEG \
            and on_table(objects['basket']):
        return 0.5
    return 0.0


def distance(objects):
    return max(0.0, (FOLDED_STOP_DEG-handle_angle_deg(objects))/FOLDED_STOP_DEG)


def demo():
    # TCP crosses the top of the raised bow from the near side, then sweeps forward and down.
    # Open jaws contact with their flat lower surfaces; neither arm grips the bow.
    x, y, z = float(HINGE_WORLD[0]), -0.065, float(HINGE_WORLD[2] + HANDLE_RISE)
    pushing = [
        Waypoint(1.2, (x-.065,y,z+.030), FINGER_OPEN, 'approach', PUSH_GRIP),
        Waypoint(0.8, (x-.018,y,z+.018), FINGER_OPEN, 'above_bow', PUSH_GRIP),
        Waypoint(0.7, (x+.025,y,z-.005), FINGER_OPEN, 'push_early', PUSH_GRIP),
        Waypoint(0.7, (x+.075,y,z-.055), FINGER_OPEN, 'push_middle', PUSH_GRIP),
        Waypoint(0.9, (x+.120,y,z-.105), FINGER_OPEN, 'push_flat', PUSH_GRIP),
        Waypoint(0.5, (x+.120,y,z-.105), FINGER_OPEN, 'seat', PUSH_GRIP),
        Waypoint(0.7, (x+.120,y,z-.045), FINGER_OPEN, 'lift_clear', PUSH_GRIP),
        Waypoint(0.9, (x-.070,y,z+.015), FINGER_OPEN, 'park', PUSH_GRIP),
        Waypoint(1.8, (x-.070,y,z+.015), FINGER_OPEN, 'hold', PUSH_GRIP),
    ]
    total = sum(w.duration for w in pushing)
    left = [Waypoint(total, (.20,.30,TOP+.30), FINGER_OPEN, 'steady_ready', PUSH_GRIP)]
    return {'left':left,'right':pushing}
