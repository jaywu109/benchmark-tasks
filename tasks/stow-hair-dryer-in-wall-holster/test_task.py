"""Offline geometry, scorer and 20 Hz reach checks."""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

if os.environ.get("YAM_ARENA"):
    sys.path.insert(0, os.path.expanduser(os.environ["YAM_ARENA"]))

from yam_arena import ik, station  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module


def state(pos, size, rot=None, vel=None):
    p = np.asarray(pos, float)
    r = np.eye(3) if rot is None else rot
    half = np.abs(r) @ (np.asarray(size, float) / 2)
    return ObjectState(p, np.asarray(ik.matrix_to_quat_xyzw(r)),
                       np.zeros(3) if vel is None else np.asarray(vel, float),
                       np.zeros(3), p-half, p+half)


def scene(dryer_z=.200, holder_offset=0.0):
    r = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    return {"dryer": state((M.HOLSTER_XY[0], M.HOLSTER_XY[1]-.0305, M.TOP+dryer_z), M.DRYER_SIZE, r),
            "holster": state((M.HOLSTER_REPORTED_C[0]+holder_offset, M.HOLSTER_REPORTED_C[1], M.HOLSTER_REPORTED_C[2]), M.HOLSTER_SIZE),
            "wall_tile": state(M.TILE_C, M.TILE_SIZE)}


def test_load_and_real_dimensions():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert M.INSTRUCTION == "Stow the unplugged hair dryer in the wall holder so it remains supported."
    assert [o.name for o in TASK.objects] == ["wall_tile", "holster", "dryer"]
    assert M.DRYER_SIZE == (.154, .140, .079) and M.DRYER_MASS_KG == .368
    assert M.HOLSTER_SIZE == (.129, .100, .155) and M.HOLSTER_MASS_KG == .302
    assert M.TOP_OPENING_M == .090 and M.LOWER_OPENING_M == .060
    assert M.BARREL_DIAMETER_M == .070 and M.ENTRY_CLEARANCE_M >= .005
    assert all(type(v) is float for o in TASK.objects for v in o.pos)
    meta = json.loads((HERE / "task.json").read_text())
    assert meta["instruction"] == M.INSTRUCTION
    assert len(meta["title"]) <= 120 and len(meta["context"]) <= 1000


def test_meshes_are_full_size_and_disjoint_on_spawn():
    for name, wanted in (("dryer", M.DRYER_SIZE), ("holster", M.HOLSTER_SIZE)):
        pts, _ = M.rx.read_obj(str(HERE / f"{name}.obj"))
        xyz = np.asarray(pts, float)
        assert np.allclose(xyz.max(0)-xyz.min(0), wanted, atol=.0015)
    assert M.DRYER.pos[2]-M.DRYER_SIZE[2]/2 >= M.TOP
    assert M.DRYER.pos[2]+M.DRYER_SIZE[2]/2 < M.HOLSTER.pos[2]-M.HOLSTER_SIZE[2]/2
    assert M.DRYER.pos[0]+M.DRYER_SIZE[0]/2 < M.TILE.pos[0]-M.TILE_SIZE[0]/2
    assert .025 + .010 < 2*abs(M.FINGER_OPEN)  # 25 mm handle and 10 mm grasp margin


def test_scoring_and_shortcuts():
    M.reset_episode()
    goal = scene()
    assert M.grade(goal, forces=(1.0, 0.0), opened=True) == 1.0
    assert M.success(goal) is False  # no live contact and opening data means no full score
    M.reset_episode()
    assert M.grade(goal, forces=(1.0, 1.0), opened=False) == .5  # held in goal
    M.reset_episode()
    assert M.grade(goal, forces=(0.0, 0.0), opened=True) == 0.0  # mere pose cannot score
    M.reset_episode()
    partial = scene(dryer_z=.305)
    assert M.grade(partial, forces=(1.0, 0.0), opened=False) == .5
    M.reset_episode()
    moved = scene(holder_offset=.011)
    assert M.grade(moved, forces=(1.0, 0.0), opened=True) == 0.0
    assert M.grade(goal, forces=(1.0, 0.0), opened=True) == 0.0  # fault latched
    M.reset_episode()
    fallen = scene(dryer_z=-.100)
    assert M.grade(fallen, forces=(1.0, 0.0), opened=True) == 0.0


def test_retention_clock():
    M.reset_episode()
    goal = scene()
    for t in (0.0, .5, .95):
        assert M.grade(goal, t=t, forces=(1.0, 0.0), opened=True) == .5
    assert M.grade(goal, t=1.0, forces=(1.0, 0.0), opened=True) == 1.0
    M.reset_episode()


def test_demo_ik_20_hz():
    plans = TASK.demo()
    assert set(plans) == {"right"}
    for side, waypoints in plans.items():
        plan = ArmPlan(waypoints, station.BASES[side])
        q = []
        for step in range(math.ceil(plan.duration*20)):
            action, _ = plan.action7((step+1)/20)
            q.append(action[:6])
        joints = np.asarray(q)
        assert plan.worst_ik_mm < .5, plan.worst_ik_mm
        assert plan.worst_ik_deg < .1, plan.worst_ik_deg
        assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER)
        assert np.max(np.abs(np.diff(joints, axis=0))) < .25
