"""Offline checks for cue-record-track-three (NumPy and pytest only; no Isaac Sim).

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
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
HZ = 20.0


def arm_at(yaw, pitch, vel=0.0):
    r = M.rz(math.radians(yaw)) @ M.ry(math.radians(pitch))
    c = M.PIVOT + r @ M.BOX_C
    half = (M.HI - M.LO) / 2
    return ObjectState(c, np.asarray(ik.matrix_to_quat_xyzw(r), float), np.array([0.0, 0.0, vel]), np.zeros(3),
                       c - half, c + half)


def objs(yaw, pitch, vel=0.0):
    return {"arm": arm_at(yaw, pitch, vel), "yoke": arm_at(yaw, 0.0)}


def pitch_for_height(h):
    """The pitch (deg) that holds the stylus tip ``h`` above the record."""
    return math.degrees(math.asin(h / M.L_EFF))


def set_down(r, h0=0.010, speed=0.01, end=0.0):
    """Lower the stylus over radius ``r`` from ``h0`` to ``end`` above the record at ``speed`` (m/s), 20 Hz."""
    yaw = M.yaw_for_radius(r)
    n = max(1, round((h0 - end) / speed * HZ))
    for h in np.linspace(h0, end, n + 1):
        s = M.score(objs(yaw, pitch_for_height(h)))
    for _ in range(M.HOLD_STEPS):  # then left there for 1 s
        s = M.score(objs(yaw, pitch_for_height(end)))
    return s


@pytest.fixture(autouse=True)
def fresh():
    M.new_episode()


@lru_cache(maxsize=None)
def played(side):
    arm = ArmPlan(TASK.demo()[side], station.BASES[side])
    return arm, [arm.action7((k + 1) / HZ)[0][:6].copy() for k in range(round(arm.duration * HZ))]


def test_the_task_file_loads():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert list(TASK.free_objects) == ["yoke", "arm"]
    assert TASK.docs.startswith("On the station table")
    for o in M.OBJECTS:
        assert all(type(v) is float for v in o.pos), o.name


def test_the_tip_is_on_the_record_surface_when_the_arm_is_level():
    tip = M.tip_of(arm_at(0.0, 0.0))
    assert tip[2] == pytest.approx(M.Z_REC, abs=1e-9)
    assert M.radius(tip) > M.R_REC + 0.03  # at rest the stylus is well off the record


def test_the_arm_is_nose_heavy_by_a_few_grams():
    v = np.array([np.prod(p.size) for p in M.ARM_PIECES])
    x = np.array([p.offset[0] for p in M.ARM_PIECES])
    grams = -(v * x).sum() / v.sum() * M.ARM_MASS / M.L_EFF * 1000
    assert 1.0 < grams < 4.0, grams


def test_track_widths_are_unequal_and_the_target_is_the_second_gap():
    bands = [o - i for o, i, k in M.RINGS if k == "band"]
    assert len(bands) == 5 and len(set(round(b, 4) for b in bands)) == 5
    assert M.TARGET == (0.113, 0.109) and len(M.GAPS) == 4
    for (o, i, _), (o2, _, _) in zip(M.RINGS, M.RINGS[1:]):
        assert i == o2  # rings meet


def test_yaw_for_radius_puts_the_tip_over_that_radius():
    for r in (0.140, 0.111, 0.080):
        assert M.radius(M.pose_point(M.TIP, M.yaw_for_radius(r), 0.0)) == pytest.approx(r, abs=1e-5)
    assert M.YAW_LIM[0] < M.yaw_for_radius(0.06)


def test_set_down_slowly_in_the_target_gap_scores_one():
    assert set_down(sum(M.TARGET) / 2) == 1.0 and M.success(objs(M.yaw_for_radius(0.111), 0.0))


def test_the_gap_edges_count_and_just_outside_does_not():
    assert set_down(0.1095) == 1.0
    M.new_episode()
    assert set_down(0.1125) == 1.0
    M.new_episode()
    assert set_down(0.1080) == 0.0  # on track 3's grooves
    M.new_episode()
    assert set_down(0.1140) == 0.0  # on track 2's grooves


def test_another_gap_earns_half():
    for g in (M.GAPS[0], M.GAPS[2], M.GAPS[3]):
        M.new_episode()
        assert set_down(sum(g) / 2) == 0.5


def test_a_dropped_arm_scores_zero_for_the_episode():
    yaw = M.yaw_for_radius(0.111)
    M.score(objs(yaw, pitch_for_height(0.010)))
    assert M.score(objs(yaw, 0.0)) == 0.0 and M._S["dropped"]  # 10 mm in one step: 0.2 m/s
    for _ in range(20):
        assert M.score(objs(yaw, 0.0)) == 0.0  # latched
    M.new_episode()
    assert [M.score(objs(yaw, 0.0)) for _ in range(M.HOLD_STEPS + 1)][-1] == 1.0


def test_success_needs_the_cued_state_held_for_one_second():
    yaw = M.yaw_for_radius(0.111)
    got = [M.score(objs(yaw, 0.0)) for _ in range(M.HOLD_STEPS + 1)]
    assert got[:-1] == [0.0] * M.HOLD_STEPS and got[-1] == 1.0
    M.new_episode()
    M.score(objs(yaw, 0.0))
    M.score(objs(yaw, 0.0, vel=0.02))  # a nudge restarts the count
    assert [M.score(objs(yaw, 0.0)) for _ in range(M.HOLD_STEPS)][-1] == 0.0


def test_a_hand_on_the_tube_is_not_hands_off(monkeypatch):
    arm = arm_at(M.yaw_for_radius(0.111), 0.0)
    mid = (M.arm_point(arm, np.zeros(3)) + M.tip_of(arm)) / 2  # the tube's middle, 11 cm from the finger lift
    monkeypatch.setattr(M, "tcp_points", lambda: [mid + np.array([0, 0, 0.02]), np.array([0.2, 0.3, 0.9])])
    assert not M.hands_clear({"arm": arm})
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array([0.25, -0.24, M.TOP + 0.12]), np.array([0.2, 0.3, 0.9])])
    assert M.hands_clear({"arm": arm})  # where the demo's hands end


def test_a_stylus_dragged_across_the_vinyl_scores_zero_for_the_episode():
    set_down(0.140)  # gently onto track 1's grooves
    for r in np.linspace(0.140, 0.111, 30):  # then slid inward to the target gap while touching
        s = M.score(objs(M.yaw_for_radius(r), 0.0))
    assert s == 0.0 and M._S["dropped"]


def test_lifted_and_set_down_again_is_not_a_scratch():
    set_down(0.140)
    M.score(objs(M.yaw_for_radius(0.140), pitch_for_height(0.005)))
    assert set_down(0.111) == 1.0  # lifted clear, moved, lowered slowly


def test_held_up_or_still_moving_scores_zero():
    yaw = M.yaw_for_radius(0.111)
    assert M.score(objs(yaw, pitch_for_height(0.004))) == 0.0
    M.new_episode()
    assert M.score(objs(yaw, 0.0, vel=0.02)) == 0.0


def test_back_on_the_rest_scores_zero():
    assert M.score(objs(0.0, 0.0)) == 0.0


def test_distance_is_zero_in_the_middle_of_the_target_gap():
    assert M.distance(objs(M.yaw_for_radius(0.111), 0.0)) == pytest.approx(0.0, abs=1e-5)
    assert M.distance(objs(0.0, 0.0)) > 0.05


def test_the_demo_grip_follows_the_finger_lift():
    plan = TASK.demo()["right"]
    for w in plan:
        if w.label.startswith("swing"):
            yaw = math.degrees(w.yaw)
            assert np.linalg.norm(np.array(w.pos) - M.pose_point(M.FIN, yaw, M.LIFT)) < 1e-9
    assert any(w.label == "let go" for w in plan)


def test_the_demo_sets_down_slowly():
    plan = TASK.demo()["right"]
    k = next(i for i, w in enumerate(plan) if w.label == "set down slowly")
    dz = plan[k - 1].pos[2] - plan[k].pos[2]
    assert dz / plan[k].duration < M.DROP_SPEED / 4


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_plan_is_reachable_within_the_joint_limits_and_the_run_limit(side):
    arm, joints = played(side)
    joints = np.array(joints)
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (arm.worst_ik_mm, arm.worst_ik_deg)
    assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER)
    path = HERE / "task.json"
    if path.exists():
        limit = int(json.loads(path.read_text())["run_limit"].split()[0])
        assert arm.duration * HZ <= limit - 100


def test_task_json_and_readme_have_every_required_field():
    path = HERE / "task.json"
    if not path.exists():
        pytest.skip("task.json not written yet")
    meta = json.loads(path.read_text())
    for key in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                "fidelity", "reset_notes", "scores"):
        assert meta.get(key), key
    assert len(meta["fidelity"]) <= 500
    readme = (HERE / "README.md").read_text()
    lines = [line.strip() for line in readme.splitlines()]
    assert "## Third-party assets" in lines and "none" in lines
    assert "robocurve_ext.py" in readme and meta["instruction"] in readme
