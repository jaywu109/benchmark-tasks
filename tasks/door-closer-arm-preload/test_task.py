"""Offline checks for door-closer-arm-preload (NumPy and pytest only; no Isaac Sim).

Run from the repo root with yam-isaac-lab-arena importable:

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q tasks/door-closer-arm-preload/test_task.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
YAM = os.environ.get("YAM_ARENA") or os.environ.get("YAM_ARENA_DIR")
if YAM and os.path.isdir(YAM) and YAM not in sys.path:
    sys.path.insert(0, YAM)

from yam_arena import analysis, checks, ik, station  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

TABLE_X = (0.05, 0.75)
TABLE_Y = (-0.60, 0.60)

TASK_PY = str(HERE / "task.py")
T = load(TASK_PY)

_SPEC = importlib.util.spec_from_file_location("door_closer_arm_preload_module", TASK_PY)
assert _SPEC and _SPEC.loader
M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M)


def _state_at_pose(
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ObjectState:
    p = np.asarray(pos, dtype=float)
    r = quat_xyzw_to_matrix(np.asarray(quat_xyzw, dtype=float))
    half = 0.5 * (np.abs(r) @ np.asarray(size, dtype=float))
    return ObjectState(
        pos=tuple(float(v) for v in p),
        quat_xyzw=tuple(float(v) for v in quat_xyzw),
        low=tuple(float(v) for v in (p - half)),
        high=tuple(float(v) for v in (p + half)),
        lin_vel=lin_vel,
        ang_vel=ang_vel,
    )


def _linkage_states(
    *,
    door_deg: float = M.DOOR_INITIAL_DEG,
    shoe_swivel_deg: float = M.FOREARM_SPAWN_SWIVEL_DEG,
    forearm_slide_z_m: float = 0.0,
    main_pull_deg: float = 0.0,
    lock_cw_deg: float = 0.0,
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, ObjectState]:
    """Compose exact world ObjectStates for all 6 parts through rx._moved_pose and rx._compose."""
    zero_header = (M.HEADER_POS, (0.0, 0.0, 0.0, 1.0))

    zero_door = (M.DOOR_LEAF_ZERO_POS, (0.0, 0.0, 0.0, 1.0))
    j_door = M.rx.Revolute(M.DOOR_HINGE_ANCHOR, "Z", 0.0, 30.0, initial=door_deg)
    pose_door = M.rx._moved_pose(zero_door, j_door) if door_deg else zero_door

    zero_shoe = (M.DOOR_SHOE_ZERO_POS, (0.0, 0.0, 0.0, 1.0))
    shoe_total_deg = M.FOREARM_ZERO_YAW_IN_DOOR_DEG + shoe_swivel_deg
    j_shoe = M.rx.Revolute((0.0, 0.0, 0.0), "Z", 46.0, 74.0, parent="door_leaf", initial=shoe_total_deg)
    pose_shoe_local = M.rx._moved_pose(zero_shoe, j_shoe) if shoe_total_deg else zero_shoe
    pose_shoe = M.rx._compose(M.rx._compose(pose_door, M.rx._inverse(zero_door)), pose_shoe_local)

    zero_fore = (M.FOREARM_ZERO_POS, (0.0, 0.0, 0.0, 1.0))
    j_fore = M.rx.Prismatic(M.FOREARM_ANCHOR, "Z", 0.0, 0.028, parent="door_shoe", initial=forearm_slide_z_m)
    pose_fore_local = M.rx._moved_pose(zero_fore, j_fore) if forearm_slide_z_m else zero_fore
    pose_fore = M.rx._compose(M.rx._compose(pose_shoe, M.rx._inverse(zero_shoe)), pose_fore_local)

    zero_main = (M.MAIN_ARM_ZERO_POS, (0.0, 0.0, 0.0, 1.0))
    main_total_deg = M.MAIN_ARM_SLACK_YAW_DEG - main_pull_deg
    j_main = M.rx.Revolute(M.MAIN_ARM_ANCHOR, "Z", -90.5, -48.0, initial=main_total_deg)
    pose_main = M.rx._moved_pose(zero_main, j_main) if main_total_deg else zero_main

    zero_lock = (M.ELBOW_LOCK_ZERO_POS, (0.0, 0.0, 0.0, 1.0))
    j_lock = M.rx.Revolute((0.0, 0.0, 0.0), "Z", -100.0, 5.0, parent="main_arm", initial=-lock_cw_deg)
    pose_lock_local = M.rx._moved_pose(zero_lock, j_lock) if lock_cw_deg else zero_lock
    pose_lock = M.rx._compose(M.rx._compose(pose_main, M.rx._inverse(zero_main)), pose_lock_local)

    return {
        "header_frame": _state_at_pose(zero_header[0], M.HEADER_SIZE, zero_header[1]),
        "door_leaf": _state_at_pose(pose_door[0], M.DOOR_LEAF_BOX_SIZE, pose_door[1], lin_vel=lin_vel, ang_vel=ang_vel),
        "door_shoe": _state_at_pose(pose_shoe[0], M.DOOR_SHOE_SIZE, pose_shoe[1], lin_vel=lin_vel, ang_vel=ang_vel),
        "forearm_rod": _state_at_pose(pose_fore[0], M.FOREARM_SIZE, pose_fore[1], lin_vel=lin_vel, ang_vel=ang_vel),
        "main_arm": _state_at_pose(pose_main[0], M.MAIN_ARM_SIZE, pose_main[1], lin_vel=lin_vel, ang_vel=ang_vel),
        "elbow_lock": _state_at_pose(pose_lock[0], M.ELBOW_LOCK_SIZE, pose_lock[1], lin_vel=lin_vel, ang_vel=ang_vel),
    }


def test_task_file_contract_and_1to1_dimensions():
    assert T.scene == "station"
    assert T.score_over == "end"
    assert M.SUCCESS_HOLD_S == pytest.approx(1.0)
    assert M.CONTROL_HZ == pytest.approx(20.0)
    assert M.INSTRUCTION == "Reconnect this door closer's arm so that it pulls the door shut against its stop."

    names = [o.name for o in M.OBJECTS]
    assert names == ["header_frame", "door_leaf", "door_shoe", "forearm_rod", "main_arm", "elbow_lock", "elbow_detent"]
    for obj in M.OBJECTS[1:]:
        assert all(type(v) is float for v in obj.pos), (obj.name, obj.pos)
        if hasattr(obj, "size"):
            assert all(type(v) is float for v in obj.size), (obj.name, obj.size)
        else:
            assert type(obj.radius) is float and obj.radius > 0.0
        assert type(obj.mass) is float, (obj.name, obj.mass)

    # Assumed arm lengths and the pin/eye fit:
    assert M.MAIN_ARM_LENGTH_M == pytest.approx(0.228)
    assert M.FOREARM_LENGTH_M == pytest.approx(0.215)
    assert M.STUD_DIAMETER_M == pytest.approx(0.014)
    assert M.CLEVIS_BORE_M == pytest.approx(0.018)
    assert M.CLEVIS_RADIAL_CLEARANCE_M == pytest.approx(0.002)
    assert M.CLEVIS_CHAMFER_M == pytest.approx(0.004)
    assert M.STUD_CHAMFER_M == pytest.approx(0.005)

    # Physical spring return on main_arm, and ZERO spring drive on door_leaf:
    assert M.MAIN_ARM_JOINT.stiffness > 0.0
    assert M.MAIN_ARM_JOINT.target == pytest.approx(M.MAIN_ARM_SLACK_YAW_DEG)
    assert M.DOOR_LEAF_JOINT.stiffness == pytest.approx(0.0)

    # Grasp widths fit comfortably inside the 93.9 mm YAM parallel jaw opening:
    for w_m in (M.MAIN_ARM_GRIP_WIDTH_M, M.FOREARM_GRIP_WIDTH_M, M.ELBOW_LOCK_WIDTH_M):
        assert 0.012 <= w_m <= 0.030
        assert M.YAM_MAX_OPENING_M - w_m >= 0.060


def test_compound_aabb_symmetry_and_contact_blocking_geometry():
    # Every moving compound part has zero XY AABB centroid shift across all yaw angles:
    for name, pieces, size in (
        ("door_leaf", M.DOOR_LEAF_PIECES, M.DOOR_LEAF_BOX_SIZE),
        ("forearm_rod", M.FOREARM_PIECES, M.FOREARM_SIZE),
        ("main_arm", M.MAIN_ARM_PIECES, M.MAIN_ARM_SIZE),
        ("elbow_lock", M.ELBOW_LOCK_PIECES, M.ELBOW_LOCK_SIZE),
    ):
        for yaw_deg in (0.0, 18.0, 42.0, 65.0, 90.0, 137.0):
            rad = math.radians(yaw_deg)
            rot = np.array(
                [[math.cos(rad), -math.sin(rad), 0.0], [math.sin(rad), math.cos(rad), 0.0], [0.0, 0.0, 1.0]]
            )
            pts = []
            for p in pieces:
                h = 0.5 * np.asarray(p.size, dtype=float)
                c = np.asarray(p.offset, dtype=float)
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        for sz in (-1.0, 1.0):
                            pts.append(rot @ (c + h * np.array([sx, sy, sz])))
            lo = np.min(pts, axis=0)
            hi = np.max(pts, axis=0)
            ctr = 0.5 * (lo + hi)
            assert np.allclose(ctr, 0.0, atol=1e-12), f"{name} centroid shift at {yaw_deg} deg: {ctr}"
            if yaw_deg == 0.0:
                assert np.allclose(hi - lo, size, atol=1e-12), f"{name} size {hi - lo} != {size}"

    # Verify elbow_lock physical clearance at 0 deg (unlocked) and physical overlap over the clevis/stud at 45..90 deg CW:
    # At 42 deg preload, stud S is at x = 0.2650 m, clevis outer wall spans x = 0.2510..0.2790 m, and lock pivot is at x = 0.2920 m.
    clevis_x_hi = M.STUD_PRELOAD_XY[0] + 0.0140  # 0.2790 m (28 mm outer clevis width)
    lock_x_lo_unlocked = M.LOCK_PIVOT_PRELOAD_XY[0] - 0.5 * M.ELBOW_LOCK_WIDTH_M  # 0.2860 m
    assert lock_x_lo_unlocked > clevis_x_hi  # 7.0 mm clear gap when unlocked at 0 deg!

    lock_x_lo_locked = M.LOCK_PIVOT_PRELOAD_XY[0] - 0.5 * M.ELBOW_LOCK_CAM_LEN_M  # 0.2660 m (reaches stud center!)
    assert clevis_x_hi - lock_x_lo_locked >= 0.013  # 13.0 mm solid steel overhang over clevis eye when locked!


def test_spring_ball_detent_contacts_notch_at_locked_angle():
    ball = M.ELBOW_DETENT
    joint = M.ELBOW_DETENT_JOINT
    assert ball in M.OBJECTS
    assert M.rx.JOINTS[ball.name] == joint
    assert M.rx.JOINTS["elbow_lock"][0].pieces == M.ELBOW_LOCK_PIECES
    assert isinstance(joint, M.rx.Prismatic)
    assert joint.parent == "main_arm" and joint.axis == "Z"
    assert joint.stiffness > 0.0 and joint.damping > 0.0
    assert joint.target > joint.upper  # constant mechanical spring preload
    assert M.ELBOW_LOCK_JOINT.stiffness == 0.0
    assert M.ELBOW_LOCK_JOINT.damping == 0.0004
    assert (M.ELBOW_LOCK_JOINT.lower, M.ELBOW_LOCK_JOINT.upper) == (-100.0, 5.0)

    def ball_center_in_cam(deg, slide):
        delta = np.asarray(ball.pos) - np.asarray(M.ELBOW_LOCK_ZERO_POS)
        delta[2] += slide
        return M.rz_mat(deg) @ delta  # inverse of the clockwise cam rotation

    def gaps(deg, slide):
        center = ball_center_in_cam(deg, slide)
        return np.array([
            np.linalg.norm(np.maximum(np.abs(center - p.offset) - 0.5 * np.asarray(p.size), 0.0))
            - ball.radius
            for p in M.ELBOW_LOCK_PIECES
        ])

    # Exact sphere/box contact: both notch corners touch the ball at 90 deg,
    # while the web above remains clear. No invisible angular drive holds it.
    rise = ball.radius - math.sqrt(ball.radius**2 - M.DETENT_NOTCH_HALF_WIDTH_M**2)
    seated_slide = rise - M.DETENT_PROTRUSION_M
    assert joint.lower < seated_slide < joint.upper
    locked_gaps = gaps(90.0, seated_slide)
    assert np.min(locked_gaps) == pytest.approx(0.0, abs=1e-12)
    assert np.count_nonzero(np.abs(locked_gaps) < 1e-12) == 2
    assert locked_gaps[0] > 0.0005  # roof of the groove
    assert gaps(0.0, joint.initial).min() > 0.002  # unlocked spawn is clear

    def highest_clear_slide(deg):
        lo, hi = joint.lower, joint.upper
        assert gaps(deg, lo).min() >= 0.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if gaps(deg, mid).min() >= 0.0:
                lo = mid
            else:
                hi = mid
        return lo

    locked_height = highest_clear_slide(90.0)
    assert locked_height == pytest.approx(seated_slide, abs=1e-10)
    for deg in (75.0, 80.0, 85.0, 95.0, 100.0):
        displaced_height = highest_clear_slide(deg)
        assert locked_height - displaced_height > 0.00005, deg
        # Moving out of the notch compresses a real spring and raises energy.
        assert 0.5 * joint.stiffness * (joint.target - displaced_height)**2 > (
            0.5 * joint.stiffness * (joint.target - locked_height)**2
        )


def test_four_bar_linkage_tiers_and_end_state_scoring():
    clear_tcps = [np.array(M.PARK_LEFT, dtype=float), np.array(M.PARK_RIGHT, dtype=float)]

    # 1. Spawn state (door 18 deg ajar, main_arm 0 deg slack, forearm +25 deg swivel):
    M.new_episode()
    s0 = _linkage_states()
    assert M.door_angle_deg(s0) == pytest.approx(18.0, abs=0.05)
    assert M.main_arm_deg(s0) == pytest.approx(0.0, abs=0.05)
    assert M.elbow_lock_deg(s0) == pytest.approx(0.0, abs=0.05)
    assert not M.clevis_seated(s0)
    assert M.score(s0, tcps=clear_tcps) == pytest.approx(0.0)
    assert not M.success(s0, tcps=clear_tcps)
    assert M.distance(s0) > 0.80

    # 2. Tier 0.35: main_arm pulled to +42.0 deg preload (>= 20.0 deg) and forearm clevis seated on stud (swivel = 0 deg):
    M.new_episode()
    s_coupled = _linkage_states(door_deg=18.0, shoe_swivel_deg=0.0, forearm_slide_z_m=0.0, main_pull_deg=42.0, lock_cw_deg=0.0)
    xy_err, z_err = M.clevis_seat_errors(s_coupled)
    assert xy_err < 1e-4 and z_err < 1e-4
    assert M.clevis_seated(s_coupled)
    assert M.score(s_coupled, tcps=clear_tcps) == pytest.approx(0.35)
    assert not M.success(s_coupled, tcps=clear_tcps)

    # 3. Tier 0.70: elbow_lock turned 90.0 deg CW (>= 75.0 deg) while clevis is seated and door is still 18.0 deg ajar:
    s_locked_ajar = _linkage_states(
        door_deg=18.0, shoe_swivel_deg=0.0, forearm_slide_z_m=0.0, main_pull_deg=42.0, lock_cw_deg=90.0
    )
    assert M.linkage_locked(s_locked_ajar)
    assert M.score(s_locked_ajar, tcps=clear_tcps) == pytest.approx(0.70)
    assert not M.success(s_locked_ajar, tcps=clear_tcps)

    # 4. Tier 1.00: preloaded closer arm pulls coupled forearm to close door_leaf to 0.0 deg (<= 2.5 deg),
    #    where main_arm_deg = 26.80 deg and shoe_swivel_deg = +4.18 deg keep ||S - B|| == 185.0 mm:
    s_closed = _linkage_states(
        door_deg=0.0, shoe_swivel_deg=4.18, forearm_slide_z_m=0.0, main_pull_deg=26.80, lock_cw_deg=90.0
    )
    xy_c, z_c = M.clevis_seat_errors(s_closed)
    assert xy_c < 0.0005 and z_c < 0.0005
    assert M.clevis_seated(s_closed)
    assert M.linkage_locked(s_closed)
    assert M.door_closed(s_closed)
    assert M.hands_clear(s_closed, clear_tcps)
    assert M.score(s_closed, tcps=clear_tcps) == pytest.approx(1.0)
    assert M.success(s_closed, tcps=clear_tcps)
    assert M.distance(s_closed) == pytest.approx(0.0)


def test_try_to_beat_it_and_latched_faults():
    clear_tcps = [np.array(M.PARK_LEFT, dtype=float), np.array(M.PARK_RIGHT, dtype=float)]

    # 1. Pushing door_leaf shut (0.0 deg) WITHOUT coupling the forearm clevis onto the main_arm stud scores 0.0:
    M.new_episode()
    s_unconnected_shut = _linkage_states(door_deg=0.0, shoe_swivel_deg=25.0, main_pull_deg=0.0, lock_cw_deg=0.0)
    assert M.door_closed(s_unconnected_shut)
    assert not M.clevis_seated(s_unconnected_shut)
    assert M.score(s_unconnected_shut, tcps=clear_tcps) == pytest.approx(0.0)
    assert not M.success(s_unconnected_shut, tcps=clear_tcps)

    # 2. Coupling clevis and closing door without turning elbow_lock (< 75 deg) cannot score 1.0 or 0.70:
    M.new_episode()
    M.score(_linkage_states(door_deg=18.0, shoe_swivel_deg=0.0, main_pull_deg=42.0, lock_cw_deg=0.0), tcps=clear_tcps)
    s_unlocked_shut = _linkage_states(door_deg=0.0, shoe_swivel_deg=4.18, main_pull_deg=26.80, lock_cw_deg=20.0)
    assert M.score(s_unlocked_shut, tcps=clear_tcps) == pytest.approx(0.35)
    assert not M.success(s_unlocked_shut, tcps=clear_tcps)

    # 2b. Door shut first, eye seated at the 26.8 deg the linkage allows, never pulled past 36 deg: still 0.35;
    #     the same seated eye with the main arm below 20 deg scores 0:
    M.new_episode()
    assert M.score(s_unlocked_shut, tcps=clear_tcps) == pytest.approx(0.35)
    M.new_episode()
    s_slack = _linkage_states(door_deg=18.0, shoe_swivel_deg=0.0, main_pull_deg=19.0, lock_cw_deg=0.0)
    s42 = _linkage_states(door_deg=18.0, shoe_swivel_deg=0.0, main_pull_deg=42.0, lock_cw_deg=0.0)
    shift = M.stud_world_pos(s_slack) - M.stud_world_pos(s42)  # carry the seated eye onto the 19 deg stud
    f = s42["forearm_rod"]
    s_slack["forearm_rod"] = _state_at_pose(tuple(np.asarray(f.pos) + shift), M.FOREARM_SIZE, f.quat_xyzw)
    assert M.clevis_seated(s_slack)
    assert M.main_arm_deg(s_slack) < M.MAIN_ARM_COUPLED_MIN_DEG
    assert M.score(s_slack, tcps=clear_tcps) == pytest.approx(0.0)

    # 3. Holding the coupled, locked, closed assembly with a gripper (< 40 mm clearance) cannot succeed:
    M.new_episode()
    s_goal = _linkage_states(door_deg=0.0, shoe_swivel_deg=4.18, main_pull_deg=26.80, lock_cw_deg=90.0)
    near_tcp = [np.asarray(s_goal["main_arm"].pos, dtype=float), np.array(M.PARK_RIGHT, dtype=float)]
    assert not M.hands_clear(s_goal, near_tcp)
    assert M.score(s_goal, tcps=near_tcp) == pytest.approx(0.70)
    assert not M.success(s_goal, tcps=near_tcp)

    # 4. Moving assembly (lin_vel >= 0.05 m/s) cannot succeed until at rest:
    s_moving = _linkage_states(
        door_deg=0.0, shoe_swivel_deg=4.18, main_pull_deg=26.80, lock_cw_deg=90.0, lin_vel=(0.08, 0.0, 0.0)
    )
    assert M.score(s_moving, tcps=clear_tcps) == pytest.approx(0.70)
    assert not M.success(s_moving, tcps=clear_tcps)

    # 5. Latched fault: turning elbow_lock > 45 deg BEFORE clevis is seated permanently latches 0.0:
    M.new_episode()
    s_early_lock = _linkage_states(door_deg=18.0, shoe_swivel_deg=25.0, main_pull_deg=42.0, lock_cw_deg=55.0)
    assert M.score(s_early_lock, tcps=clear_tcps) == pytest.approx(0.0)
    assert M._S["fault"] == "lock_turned_before_clevis_seated"
    # Even if later placed in the ideal 1.0 goal state, the latched fault keeps score at 0.0:
    assert M.score(s_goal, tcps=clear_tcps) == pytest.approx(0.0)
    assert not M.success(s_goal, tcps=clear_tcps)

    # 6. Latched fault: forcing door_leaf past its stop (> 25.0 N) permanently latches 0.0:
    M.new_episode()
    assert M.score(s_goal, tcps=clear_tcps, door_stop_force_n=28.5) == pytest.approx(0.0)
    assert M._S["fault"] == "door_forced_past_stop"
    assert M.score(s_goal, tcps=clear_tcps, door_stop_force_n=0.0) == pytest.approx(0.0)
    assert not M.success(s_goal, tcps=clear_tcps, door_stop_force_n=0.0)


def test_demo_ik_reachability_and_bimanual_clearance():
    plans_wp = T.demo()
    assert set(plans_wp.keys()) == {"left", "right"}

    arm_plans = {}
    for side, wps in plans_wp.items():
        base = station.BASES[side]
        q = None
        for w in wps:
            x, y, z = w.pos
            assert TABLE_X[0] <= x <= TABLE_X[1], f"{side} {w.label}: x={x} outside table"
            assert TABLE_Y[0] <= y <= TABLE_Y[1], f"{side} {w.label}: y={y} outside table"
            assert z >= M.TOP + 0.08, f"{side} {w.label}: z={z} too low"

            rot = M._closer_grip_rotation(w.yaw)
            q, pos_err, rot_err = ik.solve(
                np.array(w.pos),
                ik.matrix_to_quat_xyzw(rot),
                q_init=q,
                base_pos=base,
                iters=400,
            )
            margin_deg = float(np.degrees(np.minimum(q - ik.LOWER, ik.UPPER - q)).min())
            assert pos_err < 0.001, f"{side} {w.label}: IK pos_err={pos_err * 1000:.2f} mm"
            assert rot_err < 0.01, f"{side} {w.label}: IK rot_err={rot_err:.4f} rad"
            assert margin_deg >= 15.0, f"{side} {w.label}: joint margin {margin_deg:.1f} deg < 15 deg"

        plan = ArmPlan(wps, base, finger_start=FINGER_OPEN)
        steps = round(plan.duration * M.CONTROL_HZ)
        arr = np.array([plan.action7((k + 1) / M.CONTROL_HZ)[0] for k in range(steps)])
        max_dj = float(np.max(np.abs(np.diff(arr[:, :6], axis=0))))
        assert plan.worst_ik_mm < 0.5
        assert plan.worst_ik_deg < 0.1
        assert max_dj < 0.15
        arm_plans[side] = plan

    # Check bimanual TCP separation >= 100 mm across the entire 23.2 s trajectory:
    total_steps = round(max(arm_plans["left"].duration, arm_plans["right"].duration) * M.CONTROL_HZ)
    min_sep_m = min(
        float(
            np.linalg.norm(
                np.asarray(arm_plans["left"].target((k + 1) / M.CONTROL_HZ)[0])
                - np.asarray(arm_plans["right"].target((k + 1) / M.CONTROL_HZ)[0])
            )
        )
        for k in range(total_steps)
    )
    assert min_sep_m >= 0.100, f"Bimanual TCP separation {min_sep_m * 1000:.1f} mm < 100 mm"


def test_metadata_and_readme():
    meta = json.loads((HERE / "task.json").read_text())
    for key in (
        "title",
        "category",
        "keywords",
        "instruction",
        "context",
        "setup",
        "scoring",
        "run_limit",
        "fidelity",
        "reset_notes",
        "scores",
    ):
        assert key in meta and meta[key], f"Missing or empty key in task.json: {key}"
    assert "media" not in meta
    assert len(meta["title"]) <= 120
    assert len(meta["context"]) <= 1000
    assert meta["category"] == "Impact on jobs"
    assert meta["instruction"] == M.INSTRUCTION
    assert meta["scores"] == "not yet run" or meta["scores"].startswith("Dream-1 run:")
    assert "49-9071" in meta["context"] and "1,529,700" in meta["context"]

    readme = (HERE / "README.md").read_text()
    lines = [line.strip() for line in readme.splitlines()]
    assert "## Third-party assets" in lines and "none" in lines
    assert "robocurve_ext.py" in readme and meta["instruction"] in readme
    assert "feeler gauge" in readme and "protractor" in readme
