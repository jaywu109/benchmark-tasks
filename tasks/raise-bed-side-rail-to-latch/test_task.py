"""Offline checks for raise-bed-side-rail-to-latch (NumPy and pytest only; no Isaac Sim).

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

for _c in (os.environ.get("YAM_ARENA"), "~/yam-isaac-lab-arena"):
    if _c and Path(_c).expanduser().is_dir():
        sys.path.insert(0, str(Path(_c).expanduser()))
        break

from yam_arena import analysis, ik, station  # noqa: E402
from yam_arena.constants import ARM_JOINTS, READY  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
TOP = TASK.table_top
HZ = 20.0
FAR = [np.array([0.20, 0.29, TOP + 0.44]), np.array([0.20, -0.30, TOP + 0.44])]  # TCPs at the parks


def st(pos, quat, size, ang_vel=(0.0, 0.0, 0.0)):
    p = np.asarray(pos, float)
    h = np.asarray(size, float) / 2
    return ObjectState(p, np.asarray(quat, float), np.zeros(3), np.asarray(ang_vel, float), p - h, p + h)


def scene(ang_deg=100.0, depress_m=0.0, omega_deg_s=0.0):
    """Rail at `ang_deg` (0 up .. 100 down toward -x) turning at `omega_deg_s`, latch depressed downward `depress_m`."""
    a = math.radians(-ang_deg)
    q = (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
    r = M.ry_mat(-ang_deg)
    p_rail = M.HINGE_WORLD + r @ np.array([0.0, 0.0, 0.15])
    p_latch = M.HINGE_WORLD + r @ (M.LATCH_OFFSET_FROM_HINGE_REST + np.array([0.0, 0.0, -depress_m]))
    return {
        "mattress": st(M.MATTRESS_POS, (0, 0, 0, 1), M.MATTRESS_SIZE),
        "rail": st(p_rail, q, M.RAIL_SIZE, (0.0, -math.radians(omega_deg_s), 0.0)),
        "latch": st(p_latch, q, M.LATCH_SIZE),
    }


def adv(o, n=1):
    for _ in range(n):
        M.advance(o)


def corners(c, r, h):
    return np.array([c + r @ (h * np.array([a, b, d])) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])


@pytest.fixture(autouse=True)
def fresh():
    M.new_episode()


# -- constants, task.json limits, and passive contact geometry ---------------------------------------------------------
def test_constants_and_instruction():
    assert M.INSTRUCTION == "Put the bed's side rail up."
    meta = json.loads((HERE / "task.json").read_text())
    assert meta["instruction"] == M.INSTRUCTION
    assert len(meta["title"]) <= 120
    assert len(meta["context"]) <= 1000
    assert (M.TOP_POS_TOL_M, M.DROP_HEIGHT_ARM_M, M.DROP_STOP_TOL_M) == (0.005, 0.050, 0.005)
    assert (M.LATCH_CAUGHT_MAX_M, M.GRIPPER_CLEAR) == (0.003, 0.040)
    assert (M.STILL_WINDOW_STEPS, M.STILL_ANGLE_TOL) == (20, 0.5)
    assert M.SUCCESS_HOLD_S == 1.0 and M.SCORE_OVER == "end" and M.CONTROL_HZ == 20.0
    assert TASK.scene == "station" and [o.name for o in M.OBJECTS] == ["mattress", "rail", "latch"]
    for o in M.OBJECTS:
        assert all(isinstance(v, float) for v in o.pos), o.name
    assert M.RAIL_SIZE == (0.025, 0.60, 0.30) and M.RAIL_MASS == 3.0
    assert math.isclose(M.TOP_BAR_DIAM, 0.030) and M.RAIL_DOWN_STOP_DEG == 100.0
    # Rail hinge is passive gravity-only (stiffness 0.0), spawning down at -100 deg:
    assert M.RAIL_JOINT.stiffness == 0.0 and M.RAIL_JOINT.initial == -100.0
    assert M.LATCH_JOINT.axis == "Z" and M.LATCH_JOINT.stiffness == 80.0 and M.LATCH_JOINT.lower == -0.015


def test_angle_height_and_latch_readout():
    for ang in (0.0, 0.5, 20.0, 60.0, 100.0):
        for dep in (0.0, 0.009, 0.015):
            o = scene(ang, dep, 35.0)
            a, w = M.rail_angle_and_speed_deg(o["rail"])
            assert a == pytest.approx(ang, abs=1e-4) and w == pytest.approx(35.0, abs=1e-4)
            assert M.latch_depression_m(o["rail"], o["latch"]) == pytest.approx(dep, abs=1e-5)
    assert M.top_bar_error_from_up_m(scene(0.0)["rail"]) == pytest.approx(0.0, abs=1e-6)
    assert M.top_bar_error_from_up_m(scene(0.5)["rail"]) < 0.003
    assert M.top_bar_error_from_up_m(scene(1.5)["rail"]) > 0.007
    assert M.top_bar_height_above_lower_stop_m(scene(100.0)["rail"]) == pytest.approx(0.0, abs=1e-6)
    assert M.top_bar_height_above_lower_stop_m(scene(85.0)["rail"]) > 0.070


def test_ramped_keeper_contact_profile_and_self_clearance():
    # 1. No self-overlap between rail pieces and latch pieces across the entire 0..15 mm slide:
    for dep in (0.0, 0.008, 0.015):
        o = scene(0.0, dep)
        boxes = {k: M._box_hull(*v) for k, v in M.scene_boxes_from_states(o["rail"], o["latch"]).items()}
        rail_hulls = [(k, h) for k, h in boxes.items() if k.startswith("rail_")]
        latch_hulls = [(k, h) for k, h in boxes.items() if k.startswith("latch")]
        d, pair = analysis.hull_set_distance(rail_hulls, latch_hulls)
        assert d * 1000.0 >= 3.0, (dep, d * 1000.0, pair)

    # 2. During the swing up, keeper ramp interferes with an undepressed latch at 4 deg, accepts a 6.5 mm depressed
    #    latch at 4 deg, and accepts an undepressed (caught) latch at 0.3 deg behind the keeper lip:
    keeper_names = ("keeper_ramp", "keeper_lip", "keeper_roof", "keeper_riser")
    o_4_undep = scene(4.0, 0.0)
    b_4_undep = {k: M._box_hull(*v) for k, v in M.scene_boxes_from_states(o_4_undep["rail"], o_4_undep["latch"]).items()}
    d_interf, _ = analysis.hull_set_distance(
        [(k, b_4_undep[k]) for k in keeper_names],
        [(k, h) for k, h in b_4_undep.items() if k.startswith("latch")],
    )
    assert d_interf <= 0.0  # keeper ramp blocks undepressed tooth at 4 deg, forcing it down

    o_4_dep = scene(4.0, 0.0065)
    b_4_dep = {k: M._box_hull(*v) for k, v in M.scene_boxes_from_states(o_4_dep["rail"], o_4_dep["latch"]).items()}
    d_clear_dep, _ = analysis.hull_set_distance(
        [(k, b_4_dep[k]) for k in keeper_names],
        [(k, h) for k, h in b_4_dep.items() if k.startswith("latch")],
    )
    assert d_clear_dep > 0.0  # depressing the latch 6.5 mm clears the ramp

    o_caught = scene(0.3, 0.0)
    b_caught = {k: M._box_hull(*v) for k, v in M.scene_boxes_from_states(o_caught["rail"], o_caught["latch"]).items()}
    d_pocket, _ = analysis.hull_set_distance(
        [(k, b_caught[k]) for k in keeper_names],
        [(k, h) for k, h in b_caught.items() if k.startswith("latch")],
    )
    assert d_pocket > 0.0  # inside the keeper pocket at 0.3 deg with 0 depression, tooth sits cleanly behind the lip
    assert M.latch_caught(o_caught)


@pytest.mark.parametrize("ang", [0.0, 100.0])
def test_rail_and_latch_clear_the_table(ang):
    o = scene(ang)
    for name, box in M.scene_boxes_from_states(o["rail"], o["latch"]).items():
        if name.startswith(("mattress", "keeper")):
            continue
        assert corners(*box)[:, 2].min() >= TOP + 0.003, (ang, name)


def _clearance_mm(q_left, q_right, ang):
    o = scene(ang)
    rs = {
        "left": (np.asarray(q_left, float), -0.04695, None, None),
        "right": (np.asarray(q_right, float), -0.04695, None, None),
    }
    return M.min_arm_clearance_mm(rs, o["rail"], o["latch"], False, False)


@pytest.mark.parametrize("ang", [0.0, 100.0])
def test_rail_clears_the_parked_arms(ang, played):
    end_q = {side: steps[-1][0] for side, (_, steps) in played.items()}
    for ql, qr, what in ((READY, READY, "ready"), (end_q["left"], end_q["right"], "demo park")):
        clr, pair = _clearance_mm(ql, qr, ang)
        assert clr >= 5.0, (ang, what, clr, pair)


# -- latched drop fault ------------------------------------------------------------------------------------------------
def test_rail_dropped_from_above_50mm_latches():
    adv(scene(100.0), n=3)
    assert M._S["fault"] is None
    # Small lift < 50 mm (e.g. 92 deg -> lift ~40 mm) and back to 100 deg does NOT trigger fault:
    adv(scene(92.0), n=2)
    assert not M._S["lifted_above_50mm"]
    adv(scene(100.0), n=2)
    assert M._S["fault"] is None

    # Lift > 50 mm (e.g. 80 deg -> lift ~95 mm) and then drop back to 100 deg latches rail_dropped:
    adv(scene(80.0), n=2)
    assert M._S["lifted_above_50mm"] and M._S["fault"] is None
    adv(scene(99.5))
    assert M._S["fault"] == "rail_dropped"
    # Even if later raised to 0.3 deg and latched for > 1 s, score stays 0 / success False:
    adv(scene(0.3, 0.0), n=M.STILL_WINDOW_STEPS + 5)
    assert M._S["fault"] == "rail_dropped"
    assert not M.success(scene(0.3, 0.0), FAR)


# -- end state and "try to beat it" checks -----------------------------------------------------------------------------
def test_success_true_up_latched_still_released():
    o = scene(0.4, 0.0)
    adv(o, n=M.STILL_WINDOW_STEPS)
    assert M.success(o, FAR)


def test_try_to_beat_it_held_in_goal_pose_scores_zero():
    o = scene(0.4, 0.0)
    adv(o, n=M.STILL_WINDOW_STEPS)
    hand_on_tube = M.bar_point_from_rail_state(o["rail"])
    assert not M.success(o, [FAR[0], hand_on_tube])
    # Gripper 30 mm from the rail (< 40 mm required clearance) also scores 0:
    near_tube = hand_on_tube + np.array([-0.042, 0.0, 0.0])  # ~26.5 mm from 15.5 mm tube surface
    assert not M.success(o, [FAR[0], near_tube])


def test_try_to_beat_it_unlatched_or_short_of_top_scores_zero():
    # Rail at top angle (0.4 deg) with hands clear, but latch held/stuck depressed (not caught in pocket):
    o_unlatched = scene(0.4, 0.006)
    adv(o_unlatched, n=M.STILL_WINDOW_STEPS)
    assert not M.latch_caught(o_unlatched)
    assert not M.success(o_unlatched, FAR)

    # Rail at 2.0 deg (~10 mm short of top, latch tooth outside keeper lip):
    M.new_episode()
    o_short = scene(2.0, 0.0)
    adv(o_short, n=M.STILL_WINDOW_STEPS)
    assert not M.latch_caught(o_short)
    assert not M.rail_up_and_still(o_short)
    assert not M.success(o_short, FAR)


def test_success_false_just_arrived_or_drifting():
    # Just arrived at top after rising from 5 deg (not still for 1.0 s yet):
    adv(scene(5.0, 0.006), n=5)
    o = scene(0.4, 0.0)
    adv(o, n=10)
    assert not M.success(o, FAR)

    # Drifting down over the 1.0 s window (> 0.5 deg motion):
    M.new_episode()
    for k in range(M.STILL_WINDOW_STEPS + 1):
        adv(scene(0.1 + 0.04 * k, 0.0))
    assert not M.success(scene(0.9, 0.0), FAR)


def test_success_false_with_empty_tcp_list():
    o = scene(0.4, 0.0)
    adv(o, n=M.STILL_WINDOW_STEPS)
    assert not M.released(o, [])
    assert not M.success(o, [])
    assert not M.success(o)  # offline there are no TCPs


def test_score_is_zero_and_reset_clears():
    assert M.score(scene(0.4, 0.0)) == 0.0
    adv(scene(70.0))
    adv(scene(100.0))
    assert M._S["fault"] == "rail_dropped"
    M.new_episode()
    assert M._S["fault"] is None and not M._S["lifted_above_50mm"] and M._S["angle_history"] == []


# -- the demo ----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def played():
    """Per arm: the plan, and per control step (joints, finger, TCP target, label, grip_action)."""
    out = {}
    for side, waypoints in TASK.demo().items():
        arm = ArmPlan(waypoints, station.BASES[side])
        steps = []
        for k in range(round(arm.duration * HZ)):
            t = (k + 1) / HZ
            pos, _, finger, label = arm.target(t)
            action, _ = arm.action7(t)
            steps.append((action[:6].copy(), finger, pos, label, float(action[6])))
        out[side] = (arm, steps)
    return out


def test_demo_reachable_within_joint_limits_and_partial_grip(played):
    assert set(played) == {"left", "right"}
    assert played["left"][0].duration == pytest.approx(played["right"][0].duration)
    # Partial close rule: 30 mm top tube gripped at 24 mm opening (FINGER_BAR_CLOSE = -0.012 m, never full close 0.0)
    assert -0.014 <= M.FINGER_BAR_CLOSE <= -0.011
    for side, (arm, steps) in played.items():
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        margin = min(float(np.minimum(q - ik.LOWER, ik.UPPER - q).min()) for q, *_ in steps)
        assert margin >= 0.20, (side, margin)


def test_demo_arm_links_clear_of_table(played):
    for side, (_, steps) in played.items():
        low = 1e9
        for q, finger, *_ in steps[::2]:
            joints = {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)}
            joints["joint7"] = joints["joint8"] = float(finger)
            hulls = analysis.world_hulls(analysis.fk_link_poses_world(joints, station.BASES[side]))
            low = min(low, min(h.vertices[:, 2].min() for n, h in hulls if n != "base"))
        assert low >= TOP + 0.005, (side, low - TOP)


def _nominal_rail(t):
    """The demo's commanded rail angle (deg) and latch depression (m) at time t."""
    if t <= 3.8:
        ang = 100.0
    elif t <= 7.0:
        ang = 100.0 - 80.0 * (t - 3.8) / 3.2
    elif t <= 7.5:
        ang = 20.0 - 10.0 * (t - 7.0) / 0.5
    elif t <= 8.1:
        ang = 10.0 - 7.0 * (t - 7.5) / 0.6
    elif t <= 8.9:
        ang = 3.0 - 3.0 * (t - 8.1) / 0.8
    else:
        ang = 0.0
    dep = 0.009 if 0.7 <= ang <= 7.5 else 0.0
    return ang, dep


def test_demo_arms_clear_the_moving_rail(played):
    (_, ls), (_, rs) = played["left"], played["right"]
    worst = (1e9, "")
    for k, (lstep, rstep) in enumerate(zip(ls, rs, strict=True)):
        t = (k + 1) / HZ
        ang, dep = _nominal_rail(t)
        o = scene(ang, dep)
        fl = -lstep[4] * 0.04695
        fr = min(-rstep[4] * 0.04695, -0.015)
        state = {"left": (lstep[0], fl, None, None), "right": (rstep[0], fr, None, None)}
        clr, what = M.min_arm_clearance_mm(state, o["rail"], o["latch"], False, 2.0 <= t <= 11.1)
        if clr < worst[0]:
            worst = (clr, f"t={t:.2f}s {what}")
    assert worst[0] >= 4.0, worst


def test_demo_ends_parked_clear_and_still(played):
    for side, (_, steps) in played.items():
        assert steps[-1][3] in ("l_done", "r_done", "done")
        assert np.linalg.norm(steps[-1][2] - steps[-21][2]) < 1e-9  # last 1.0 s still
    o = scene(0.4, 0.0)
    tcps = [ik.tcp_pose(played[s][1][-1][0], station.BASES[s])[0] for s in ("left", "right")]
    assert M.min_tcp_clearance_m(o["rail"], o["latch"], tcps) >= M.GRIPPER_CLEAR + 0.05
