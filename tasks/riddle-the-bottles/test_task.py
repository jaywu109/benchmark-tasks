"""Offline checks for riddle-the-bottles (NumPy and pytest only; no Isaac Sim).

Run from this folder with the yam-isaac-lab-arena package importable (installed, as on the Brev VM, or a clone
named by YAM_ARENA):

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
"""

import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("YAM_ARENA"):
    sys.path.insert(0, os.path.expanduser(os.environ["YAM_ARENA"]))

from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
HZ = 20.0


def bottle(i, roll=0.0, shift=(0.0, 0.0, 0.0), tilt=0.0, vel=(0.0, 0.0, 0.0)):
    """ObjectState of bottle ``i`` rolled ``roll`` deg about its own axis from its spawn pose, its base moved by
    ``shift`` (m) and its axis tipped ``tilt`` deg about y."""
    r0, b0 = M.spawn_start()[M.NAMES[i]]
    r = axis_angle_matrix(M.Y, math.radians(tilt)) @ axis_angle_matrix(M.D, math.radians(roll))
    b = b0 + np.asarray(shift, float)
    c = b + M.S0 * (r @ M.D)
    half = np.asarray(M.BOTTLE_SIZE) / 2
    return ObjectState(c, np.asarray(ik.matrix_to_quat_xyzw(r), float), np.asarray(vel, float), np.zeros(3),
                       c - half, c + half)


def rolled(rolls):
    return {n: bottle(i, r) for i, (n, r) in enumerate(zip(M.NAMES, rolls))}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: [])
    monkeypatch.setattr(M, "episode_t", lambda: None)


@lru_cache(maxsize=None)
def played(side):
    plan = TASK.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    return arm, [arm.action7((k + 1) / HZ)[0][:6].copy() for k in range(round(arm.duration * HZ))]


# -- 1. the file loads ------------------------------------------------------------------------------------------------


def test_the_task_file_loads():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert TASK.free_objects == list(M.NAMES) or tuple(TASK.free_objects) == M.NAMES
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))
    assert TASK.docs.startswith("On the station table")
    assert set(TASK.demo()) == {"left", "right"}
    assert "yam_arena.scene" not in sys.modules


# -- 2. geometry -------------------------------------------------------------------------------------------------------


def test_the_bottle_mesh_has_the_sourced_size():
    v = np.array([p for _, pts, _ in M.rx.read_obj_parts(str(HERE / "bottle.obj")) for p in pts])
    s = v @ M.D
    radial = np.linalg.norm(v - np.outer(s, M.D), axis=1)
    assert s.max() - s.min() == pytest.approx(0.2435, abs=0.001)  # Saverglass Couronne Champ 29, 37.5 cl
    assert radial.max() == pytest.approx(0.0353, abs=0.0005)  # 70.6 mm body


def test_the_neck_fits_the_hole_and_the_shoulder_rests_on_its_mouth():
    assert M.R_NECK < M.HW < M.R_BODY
    assert M.LENGTH - M.S_MOUTH < M.T_L + 0.06  # the neck reaches into the hole
    assert 2 * M.R_BODY < 2 * abs(FINGER_OPEN)  # the open jaws pass round the body
    assert abs(M.CLOSE) < M.R_BODY  # the close presses on the glass


def test_the_rows_are_the_sourced_pitch():
    assert np.allclose(np.diff(M.YS), 0.13)
    assert math.degrees(M.A) == pytest.approx(45.0)


# -- 3. scorer ---------------------------------------------------------------------------------------------------------


def test_all_turned_either_way_scores_one():
    for rolls in ((45, 45, 45, 45), (-45, -40, 50, 35), (-31, 59, -59, 31)):
        objs = rolled(rolls)
        assert M.score(objs) == 1.0 and M.success(objs), rolls


def test_each_turned_bottle_scores_a_quarter():
    assert M.score(rolled((45, 0, 0, 0))) == 0.25
    assert M.score(rolled((45, 45, 0, -45))) == 0.75
    assert M.score(rolled((0, 0, 0, 0))) == 0.0


@pytest.mark.parametrize("roll", [25.0, 65.0, 90.0, 180.0])
def test_a_turn_outside_30_to_60_deg_does_not_count(roll):
    assert M.score(rolled((roll, 45, 45, 45))) == 0.75


@pytest.mark.parametrize("kw", [{"shift": (0.0, 0.006, 0.0)}, {"shift": tuple(0.006 * -M.D)}, {"tilt": 4.0}])
def test_an_unseated_bottle_does_not_count(kw):
    objs = rolled((45, 45, 45, 45))
    objs["bottle_2"] = bottle(1, 45, **kw)
    assert M.score(objs) == 0.75


def test_a_moving_bottle_scores_zero():
    objs = rolled((45, 45, 45, 45))
    objs["bottle_3"] = bottle(2, 45, vel=(0.0, 0.0, 0.05))
    assert M.score(objs) == 0.0


def test_a_tcp_within_3_cm_of_a_bottle_scores_zero(monkeypatch):
    objs = rolled((45, 45, 45, 45))
    base = M.spawn_start()["bottle_1"][1]
    far = np.array([0.15, 0.32, M.TOP + 0.22])
    monkeypatch.setattr(M, "tcp_points", lambda: [far, base + 0.03 * M.D])  # inside the base end: still held
    assert M.score(objs) == 0.0
    monkeypatch.setattr(M, "tcp_points", lambda: [far, base - 0.05 * M.D])
    assert M.score(objs) == 1.0


def test_a_pulled_out_bottle_scores_zero_until_the_next_episode(monkeypatch):
    clock = {"t": 0.05}
    monkeypatch.setattr(M, "episode_t", lambda: clock["t"])
    monkeypatch.setitem(M._EP, "t", None)
    assert M.score(rolled((0, 0, 0, 0))) == 0.0 and not M._EP["out"]  # start read here
    clock["t"] = 10.0
    objs = rolled((45, 45, 45, 45))
    objs["bottle_4"] = bottle(3, 45, shift=tuple(0.035 * -M.D))  # drawn 35 mm out
    assert M.score(objs) == 0.0 and M._EP["out"]
    clock["t"] = 30.0
    assert M.score(rolled((45, 45, 45, 45))) == 0.0  # pushed back in: still latched
    clock["t"] = 0.05  # reset: the clock goes back
    assert M.score(rolled((0, 0, 0, 0))) == 0.0 and not M._EP["out"]
    clock["t"] = 20.0
    assert M.score(rolled((45, 45, 45, 45))) == 1.0


def test_the_start_is_read_at_the_first_step_not_from_the_estimate(monkeypatch):
    clock = {"t": 0.05}
    monkeypatch.setattr(M, "episode_t", lambda: clock["t"])
    monkeypatch.setitem(M._EP, "t", None)
    settled = {n: bottle(i, 3.0, shift=tuple(0.012 * -M.D)) for i, n in enumerate(M.NAMES)}  # rests 12 mm out
    M.score(settled)
    clock["t"] = 20.0
    objs = {n: bottle(i, 48.0, shift=tuple(0.012 * -M.D)) for i, n in enumerate(M.NAMES)}
    assert M.score(objs) == 1.0
    objs = {n: bottle(i, 28.0, shift=tuple(0.012 * -M.D)) for i, n in enumerate(M.NAMES)}  # 25 deg from its start
    assert M.score(objs) == 0.0


def test_distance_is_zero_at_an_eighth_turn():
    assert M.distance(rolled((45, -45, 45, -45))) == pytest.approx(0.0, abs=1e-9)
    assert M.distance(rolled((0, 0, 0, 0))) > M.distance(rolled((30, 30, 30, 30))) > 0


# -- 4. demo -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_plan_is_reachable_within_the_joint_limits_and_the_run_limit(side):
    arm, joints = played(side)
    joints = np.array(joints)
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (arm.worst_ik_mm, arm.worst_ik_deg)
    assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER)
    limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0])
    assert arm.duration * HZ <= limit - 100


def test_each_bottle_is_closed_on_for_at_least_1_s_and_turned_once():
    plan = TASK.demo()
    for i, n in enumerate(M.NAMES):
        side = "right" if i < 2 else "left"
        closes = [w for w in plan[side] if w.label == f"{n}: close"]
        turns = [w for w in plan[side] if w.label == f"{n}: an eighth turn"]
        assert len(closes) == 1 and closes[0].duration >= 1.0 and closes[0].finger == M.CLOSE
        assert len(turns) == 1


def test_the_hands_never_work_the_two_inner_bottles_at_once():
    plan = TASK.demo()

    def window(side, n):
        t, lo, hi = 0.0, None, None
        for w in plan[side]:
            if w.label.startswith(n) and lo is None:
                lo = t
            t += w.duration
            if w.label.startswith(n):
                hi = t
        return lo, hi

    r_lo, r_hi = window("right", "bottle_2")
    l_lo, l_hi = window("left", "bottle_3")
    assert r_hi <= l_lo


def test_task_json_and_readme_have_every_required_field():
    meta = json.loads((HERE / "task.json").read_text())
    for key in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                "fidelity", "reset_notes", "scores"):
        assert meta.get(key), key
    assert meta["category"] == "Impact on jobs"
    assert len(meta["fidelity"]) <= 500
    assert meta["scores"] == "not yet run" or meta["scores"].startswith("Dream-1 run: ")
    readme = (HERE / "README.md").read_text()
    lines = [line.strip() for line in readme.splitlines()]
    assert "## Third-party assets" in lines and "none" in lines
    assert "robocurve_ext.py" in readme and meta["instruction"] in readme
