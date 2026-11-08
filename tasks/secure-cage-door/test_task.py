"""Offline checks for secure-cage-door (NumPy and pytest only; no Isaac Sim).

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

import yam_arena  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.analysis import box_signed_distance, hull_set_distance, hulls_to_box, world_hulls  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import forward_kinematics  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
TOP = TASK.table_top
HZ = 20.0
ARM_URDF = Path(yam_arena.__file__).resolve().parent / "assets" / "yam" / "yam_arm.urdf"


# -- helpers -----------------------------------------------------------------------------------------------------------


def _rot_matrix_from_quat_xyzw(quat_xyzw):
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], float)


def _aabb_for_rotated_box(centre, size, rot):
    centre = np.asarray(centre, float)
    half = np.asarray(size, float) / 2.0
    ext = np.abs(rot) @ half
    return centre - ext, centre + ext


def door_state(joint_deg: float = 0.0, vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> ObjectState:
    """Door pose at ``joint_deg`` about its vertical Z hinge from its closed zero pose (negative = open outward)."""
    joint = M.rx.Revolute(
        anchor=M.DOOR_JOINT.anchor,
        axis=M.DOOR_JOINT.axis,
        lower=M.DOOR_JOINT.lower,
        upper=M.DOOR_JOINT.upper,
        initial=float(joint_deg),
        target=M.DOOR_JOINT.target,
        stiffness=M.DOOR_JOINT.stiffness,
        damping=M.DOOR_JOINT.damping,
    )
    zero = (tuple(M.DOOR.pos), (0.0, 0.0, 0.0, 1.0))
    pos, quat = M.rx._moved_pose(zero, joint)
    pos = np.asarray(pos, float)
    quat = np.asarray(quat, float)
    rot = _rot_matrix_from_quat_xyzw(quat)
    lo, hi = _aabb_for_rotated_box(pos, M.DOOR_SIZE, rot)
    return ObjectState(pos, quat, np.asarray(vel, float), np.asarray(ang_vel, float), lo, hi)


def bolt_state(throw_mm: float = 0.0, vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> ObjectState:
    """Bolt pose translated by ``throw_mm`` (mm) in +y from its retracted spawn pose."""
    joint = M.rx.Prismatic(
        anchor=M.BOLT_JOINT.anchor,
        axis=M.BOLT_JOINT.axis,
        lower=M.BOLT_JOINT.lower,
        upper=M.BOLT_JOINT.upper,
        initial=float(throw_mm) / 1000.0,
        damping=M.BOLT_JOINT.damping,
    )
    zero = (tuple(M.BOLT.pos), (0.0, 0.0, 0.0, 1.0))
    pos, quat = M.rx._moved_pose(zero, joint)
    pos = np.asarray(pos, float)
    quat = np.asarray(quat, float)
    rot = _rot_matrix_from_quat_xyzw(quat)
    lo, hi = _aabb_for_rotated_box(pos, M.BOLT_SIZE, rot)
    return ObjectState(pos, quat, np.asarray(vel, float), np.asarray(ang_vel, float), lo, hi)


def make_objects(
    door_deg: float = 0.0,
    throw_mm: float = 20.0,
    door_vel=(0.0, 0.0, 0.0),
    bolt_vel=(0.0, 0.0, 0.0),
):
    return {
        "door": door_state(door_deg, vel=door_vel),
        "bolt": bolt_state(throw_mm, vel=bolt_vel),
    }


def blocked_door_angle_deg(throw_mm: float) -> float:
    """Exact opening angle (deg > 0, where joint angle is -deg) at which a closing door's back face (u = 8 mm) hits a
    slide bolt thrown by ``throw_mm`` (mm) whose back face is at x = 0.359 m and +y tip is at y = -0.077 + d."""
    d = throw_mm / 1000.0
    y_tip = -0.077 + d
    y_min = -0.117 + d
    x_bolt_lo, x_bolt_hi = M.BOLT_X - M.BOLT_LX / 2.0, M.BOLT_X + M.BOLT_LX / 2.0
    for k in range(35000, 0, -1):
        alpha_deg = k / 1000.0
        a = math.radians(alpha_deg)
        ca, sa = math.cos(a), math.sin(a)
        for u in (0.0, M.DOOR_LX):
            for r in (M.DOOR_LY, (M.HINGE_Y - y_tip - u * sa) / ca):
                if 0.0 <= r <= M.DOOR_LY:
                    x = M.HINGE_X - r * sa + u * ca
                    y = M.HINGE_Y - r * ca - u * sa
                    if x_bolt_lo - 1e-9 <= x <= x_bolt_hi + 1e-9 and y_min - 1e-9 <= y <= y_tip + 1e-9:
                        return alpha_deg
    return 0.0


def _poses(q6, finger, side):
    joints = {n: float(v) for n, v in zip(ARM_JOINTS, q6, strict=True)}
    joints["joint7"] = joints["joint8"] = float(finger)
    fk = forward_kinematics(ARM_URDF, joints)
    base = np.eye(4)
    base[:3, 3] = station.BASES[side]
    return {k: base @ v for k, v in fk.items() if k != "base"}


def _hulls(side, q, finger):
    return world_hulls(_poses(q, finger, side))


@lru_cache(maxsize=None)
def played(side):
    """Every control step of one arm's plan at 20 Hz: (t, joints, finger opening in m, label)."""
    plan = TASK.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    out = []
    for k in range(round(arm.duration * HZ)):
        t = (k + 1) / HZ
        action, label = arm.action7(t)
        out.append((t, action[:6].copy(), FINGER_OPEN * action[6], label))
    return arm, out


# -- 1. the file loads ------------------------------------------------------------------------------------------------


def test_the_task_file_loads_with_its_objects_and_hooks():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert TASK.free_objects == ("door", "bolt")
    assert len(TASK.objects) == 6  # door, bolt, cage_body, hinge_post, latch_post, sill
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))
    assert TASK.docs.startswith("On the station table")
    assert set(TASK.demo()) == {"left", "right"}
    assert "yam_arena.scene" not in sys.modules


# -- 2. both plans are reachable at 20 Hz and clear each other --------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_every_arm_plan_is_reachable_within_the_joint_limits_at_20_hz(side):
    arm, steps = played(side)
    joints = np.array([s[1] for s in steps])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (side, arm.worst_ik_mm, arm.worst_ik_deg)
    assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER), side
    assert max(w.pos[2] for w in TASK.demo()[side]) - TOP <= 0.14 + 1e-9


def test_both_arms_share_durations_and_stay_clear_of_each_other():
    plan = TASK.demo()
    left_wps, right_wps = plan["left"], plan["right"]
    assert [w.duration for w in left_wps] == [w.duration for w in right_wps]
    assert sum(w.duration for w in left_wps) <= 16.0 + 1e-9
    (_, left_steps), (_, right_steps) = played("left"), played("right")
    for (t, ql, fl, ll), (_, qr, fr, rl) in zip(left_steps, right_steps, strict=True):
        d, pair = hull_set_distance(_hulls("left", ql, fl), _hulls("right", qr, fr))
        assert d > 0.005, (t, ll, rl, d, pair)


# -- 3. hinge sign and bolt-to-door geometry --------------------------------------------------------------------------


def test_revolute_sign_negative_swings_free_edge_outward_toward_minus_x():
    """About +Z through (0.360, +0.075), negative angles swing the door's free (-y) edge outward toward -x, whereas
    positive angles swing it into the cage (+x)."""
    assert M.DOOR_JOINT.lower == -100.0 and M.DOOR_JOINT.upper == 0.0
    assert M.DOOR_JOINT.initial == -35.0 and M.DOOR_JOINT.target == -35.0
    outward = door_state(-35.0)
    inward = door_state(+35.0)
    assert outward.pos[0] < M.DOOR_ZERO_X - 0.035  # moves toward the robot (-x)
    assert inward.pos[0] > M.DOOR_ZERO_X + 0.035  # moves into the cage (+x)
    # At -35 deg open, the door's back face crosses y = 0 at x ~ 0.3172 m (and front face at x ~ 0.3075 m)
    x_front_y0 = M.HINGE_X - M.HINGE_Y * math.tan(math.radians(35.0))
    x_back_y0 = x_front_y0 + M.DOOR_LX / math.cos(math.radians(35.0))
    assert x_back_y0 == pytest.approx(0.3172, abs=5e-4)


def test_retracted_bolt_clears_door_sweep_and_thrown_bolt_overlaps_door_front_face():
    """Retracted (0 mm), the bolt tip at y = -0.077 sits 2 mm outside the door's swept disk (radius 150 mm about
    y = +0.075, reaching y = -0.075 at closed). Thrown >= 5 mm, the bolt tip extends past y = -0.075 in front of the
    closed door's front face (x = 0.360) and blocks the door within ~0.38 deg."""
    b_retracted = bolt_state(0.0)
    assert b_retracted.high[1] == pytest.approx(-0.077)
    # Every point of the door on its hinge has distance <= hypot(0.150, 0.008) from (0.360, +0.075); for all angles in
    # [-100, 0] deg, the door's minimum y occurs near closed (y = -0.075), so y <= -0.077 never overlaps the door.
    for deg in np.linspace(-100.0, 0.0, 201):
        d = door_state(float(deg))
        rot = _rot_matrix_from_quat_xyzw(d.quat_xyzw)
        # sample the 8 corners of the door box against the retracted bolt box
        corners = np.array([
            [sx * M.DOOR_LX / 2, sy * M.DOOR_LY / 2, sz * M.DOOR_H / 2]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]) @ rot.T + d.pos
        assert box_signed_distance(corners, b_retracted.pos, np.eye(3), np.asarray(M.BOLT_SIZE)).min() >= 0.0019

    # Thrown 20 mm, the bolt tip reaches y = -0.057 (overlapping the closed door's y in [-0.075, +0.075] by 18 mm)
    b_thrown = bolt_state(20.0)
    assert b_thrown.high[1] == pytest.approx(-0.057)
    assert b_thrown.high[1] - (-M.DOOR_LY / 2) == pytest.approx(0.018)
    # A closed door swinging outward hits the thrown bolt's back face (x = 0.359) at asin(1 / 150) = 0.382 deg <= 1.5 deg
    latched_deg = math.degrees(math.asin((M.HINGE_X - float(b_thrown.high[0])) / M.DOOR_LY))
    assert latched_deg == pytest.approx(0.382, abs=0.01)
    assert latched_deg <= M.DOOR_SHUT_DEG


# -- 4. scoring: goal, partial credit, and shortcuts ------------------------------------------------------------------


def test_goal_state_door_shut_and_bolt_thrown_20_mm_scores_one_and_succeeds(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array(M.LEFT_PARK), np.array(M.RIGHT_PARK)])
    s = make_objects(door_deg=-0.4, throw_mm=20.0)
    assert TASK.score(s) == pytest.approx(1.0)
    assert TASK.success(s)
    assert TASK.distance(s) == pytest.approx(0.4 / 35.0)


def test_bolt_thrown_10_mm_scores_half_and_does_not_succeed(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array(M.LEFT_PARK), np.array(M.RIGHT_PARK)])
    s = make_objects(door_deg=-0.4, throw_mm=10.0)
    assert TASK.score(s) == pytest.approx(0.5)
    assert not TASK.success(s)
    assert TASK.distance(s) == pytest.approx(0.4 / 35.0 + (15.0 - 10.0) / 15.0)


@pytest.mark.parametrize("throw_mm", [0.0, 4.5])
def test_shortcut_door_shut_but_bolt_not_thrown_scores_zero(monkeypatch, throw_mm):
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array(M.LEFT_PARK), np.array(M.RIGHT_PARK)])
    s = make_objects(door_deg=0.0, throw_mm=throw_mm)
    assert TASK.score(s) == 0.0 and not TASK.success(s)


def test_shortcut_bolt_thrown_20_mm_with_door_open_35_deg_scores_zero(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array(M.LEFT_PARK), np.array(M.RIGHT_PARK)])
    s = make_objects(door_deg=-35.0, throw_mm=20.0)
    assert TASK.score(s) == 0.0 and not TASK.success(s)
    assert TASK.distance(s) == pytest.approx(1.0)


@pytest.mark.parametrize("throw_mm", [5.0, 10.0, 15.0, 20.0, 25.0])
def test_shortcut_bolt_thrown_early_blocks_closing_door_past_3x_shut_threshold_and_scores_zero(monkeypatch, throw_mm):
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array(M.LEFT_PARK), np.array(M.RIGHT_PARK)])
    blocked_deg = blocked_door_angle_deg(throw_mm)
    assert blocked_deg >= 3.0 * M.DOOR_SHUT_DEG, (throw_mm, blocked_deg)
    s = make_objects(door_deg=-blocked_deg, throw_mm=throw_mm)
    assert TASK.score(s) == 0.0 and not TASK.success(s)


def test_shortcut_door_or_bolt_within_2_cm_of_a_tcp_scores_zero(monkeypatch):
    s = make_objects(door_deg=0.0, throw_mm=20.0)
    # Left TCP 2 cm in front of the closed door's front face (x = 0.360 - 0.020 = 0.340)
    near_door = np.array([0.340, 0.0, M.DOOR_ZERO_Z])
    monkeypatch.setattr(M, "tcp_points", lambda: [near_door, np.array(M.RIGHT_PARK)])
    assert M.box_distance(s["door"], M.DOOR_SIZE, near_door) == pytest.approx(0.020)
    assert TASK.score(s) == 0.0 and not TASK.success(s)

    # Right TCP 2 cm in -y from the thrown bolt's -y end face (y = -0.097 - 0.020 = -0.117)
    near_bolt = np.array([M.BOLT_X, -0.117, M.BOLT_Z])
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array(M.LEFT_PARK), near_bolt])
    assert M.box_distance(s["bolt"], M.BOLT_SIZE, near_bolt) == pytest.approx(0.020)
    assert TASK.score(s) == 0.0 and not TASK.success(s)


def test_moving_door_or_moving_bolt_prevents_success_and_scores_zero(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array(M.LEFT_PARK), np.array(M.RIGHT_PARK)])
    s_door_moving = make_objects(door_deg=0.0, throw_mm=20.0, door_vel=(0.05, 0.0, 0.0))
    assert TASK.score(s_door_moving) == 0.0 and not TASK.success(s_door_moving)
    s_bolt_moving = make_objects(door_deg=0.0, throw_mm=20.0, bolt_vel=(0.0, 0.05, 0.0))
    assert TASK.score(s_bolt_moving) == 0.0 and not TASK.success(s_bolt_moving)


# -- 5. layout: no start overlaps and left pusher stays outside open door until push ----------------------------------


def test_no_start_overlaps_between_open_door_frame_bolt_and_cage_body():
    d0 = door_state(M.DOOR_OPEN_DEG)
    b0 = bolt_state(0.0)
    fixed_boxes = [
        (b.name, np.asarray(b.pos, float), np.eye(3), np.asarray(b.size, float))
        for b in (M.CAGE_BODY, *M.FRAME)
    ]
    all_boxes = [
        ("door", d0.pos, _rot_matrix_from_quat_xyzw(d0.quat_xyzw), np.asarray(M.DOOR_SIZE, float)),
        ("bolt", b0.pos, np.eye(3), np.asarray(M.BOLT_SIZE, float)),
        *fixed_boxes,
    ]
    for name, c, r, s in all_boxes:
        lo, _ = _aabb_for_rotated_box(c, s, r)
        assert lo[2] >= TOP - 1e-9, (name, lo[2])

    # Pairwise oriented-box separation check (sampling corners and edge points of each box against the other)
    grid = np.linspace(-0.5, 0.5, 9)
    unit_pts = np.array([[x, y, z] for x in grid for y in grid for z in grid])
    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            n1, c1, r1, s1 = all_boxes[i]
            n2, c2, r2, s2 = all_boxes[j]
            if {n1, n2} <= {"cage_body", "hinge_post", "latch_post", "sill"}:
                # Fixed frame boxes touch the cage body at x = 0.370 and sill touches posts at y = +-0.077
                lo1, hi1 = _aabb_for_rotated_box(c1, s1, r1)
                lo2, hi2 = _aabb_for_rotated_box(c2, s2, r2)
                overlap = np.minimum(hi1, hi2) - np.maximum(lo1, lo2)
                assert not np.all(overlap > 1e-7), (n1, n2, overlap)
                continue
            pts1 = (unit_pts * s1) @ r1.T + c1
            pts2 = (unit_pts * s2) @ r2.T + c2
            gap = min(box_signed_distance(pts1, c2, r2, s2).min(), box_signed_distance(pts2, c1, r1, s1).min())
            assert gap >= 0.0009, (n1, n2, gap)


def test_left_pusher_path_stays_outside_the_open_door_until_the_push():
    d0 = door_state(M.DOOR_OPEN_DEG)
    rot_open = _rot_matrix_from_quat_xyzw(d0.quat_xyzw)
    _, steps = played("left")
    pre_push_steps = []
    for t, q, f, label in steps:
        if label == "push door shut":
            break
        pre_push_steps.append((t, q, f, label))
    assert len(pre_push_steps) > 0
    for t, q, f, label in pre_push_steps:
        dist, link = hulls_to_box(_hulls("left", q, f), d0.pos, rot_open, M.DOOR_SIZE)
        assert dist > 0.015, (t, label, link, dist)


# -- 6. top camera: first and last frames see the door and the bolt ---------------------------------------------------


def _hit(a, b, hull):
    d = b - a
    t0, t1 = 0.0, 1.0
    for n, off in zip(hull.planes[:, :3], hull.planes[:, 3], strict=True):
        num, den = off - n @ a, n @ d
        if abs(den) < 1e-12:
            if num < 0:
                return False
            continue
        t = num / den
        if den < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return t1 > 1e-6 and t0 < 1 - 1e-6


def _visible_fraction(lo, hi, hulls, faces=("top", "back")):
    cam = np.asarray(station.top_camera().pos, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    g = np.linspace(0.1, 0.9, 5)
    pts = []
    for a in g:
        for b in g:
            if "top" in faces:
                pts.append([lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1]), hi[2]])
            if "back" in faces:
                pts.append([lo[0], lo[1] + a * (hi[1] - lo[1]), lo[2] + b * (hi[2] - lo[2])])
    seen = sum(not any(_hit(cam, np.asarray(p, float), h) for _, h in hulls) for p in pts)
    return seen / len(pts)


def test_first_and_last_top_camera_frames_see_the_door_and_the_bolt():
    # First frame: both arms at READY with FINGER_OPEN, door at -35 deg, bolt at 0 mm
    hulls_first = _hulls("left", np.array(READY), FINGER_OPEN) + _hulls("right", np.array(READY), FINGER_OPEN)
    d_open = door_state(M.DOOR_OPEN_DEG)
    b_start = bolt_state(0.0)
    assert _visible_fraction(d_open.low, d_open.high, hulls_first) == 1.0
    assert _visible_fraction(b_start.low, b_start.high, hulls_first) == 1.0

    # Last frame: both arms parked at the end of the plan, door shut, bolt thrown 20 mm
    (_, left_steps), (_, right_steps) = played("left"), played("right")
    hulls_last = (
        _hulls("left", left_steps[-1][1], left_steps[-1][2])
        + _hulls("right", right_steps[-1][1], right_steps[-1][2])
    )
    d_shut = door_state(-0.4)
    b_thrown = bolt_state(20.0)
    assert _visible_fraction(d_shut.low, d_shut.high, hulls_last) == 1.0
    assert _visible_fraction(b_thrown.low, b_thrown.high, hulls_last) == 1.0


# -- 7. task.json and README checks -----------------------------------------------------------------------------------


def test_task_json_and_readme_have_every_required_field_and_limit():
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
        assert meta.get(key), key
    assert meta["title"] == "Hold a cage door shut and slide its bolt across"
    assert meta["category"] == "Impact on jobs"
    assert meta["instruction"] == "Shut the cage door and bolt it."
    assert len(meta["fidelity"]) <= 500, len(meta["fidelity"])
    assert meta["scores"] == "not yet run"
    readme = (HERE / "README.md").read_text()
    assert "## Third-party assets" in readme and "\nnone" in readme
    assert "robocurve_ext.py" in readme
    assert "## Verification" in readme
    assert "the scripted demo uses two arms (one holds the door, one slides the bolt)" in readme
