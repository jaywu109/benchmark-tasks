"""Offline checks for post-parcel-curbside-mailbox (NumPy and pytest only; no Isaac Sim).

    YAM_ARENA=~/yam-isaac-lab-arena pytest -q tasks/post-parcel-curbside-mailbox/test_task.py
"""

import json
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
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
TOP = TASK.table_top
HZ = 20.0
FAR_TCPS = [np.array(M.PARK_LEFT, dtype=float), np.array(M.PARK_RIGHT, dtype=float)]


def _rot_from_quat_xyzw(quat_xyzw):
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _state(pos, quat_xyzw, size, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> ObjectState:
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat_xyzw, dtype=float)
    rot = _rot_from_quat_xyzw(quat)
    ext = np.abs(rot) @ (np.asarray(size, dtype=float) / 2.0)
    return ObjectState(pos, quat, np.asarray(lin_vel, dtype=float), np.asarray(ang_vel, dtype=float), pos - ext, pos + ext)


def _door_pose(open_deg: float):
    joint = M.rx.Revolute(
        anchor=M.DOOR_ANCHOR,
        axis="Y",
        lower=M.DOOR_OPEN_INIT_DEG,
        upper=0.0,
        initial=-float(open_deg),
    )
    return M.rx._moved_pose((tuple(M.DOOR_ZERO_POS), (0.0, 0.0, 0.0, 1.0)), joint)


def _flag_pose(up_deg: float):
    joint = M.rx.Revolute(
        anchor=M.FLAG_ANCHOR,
        axis="Y",
        lower=0.0,
        upper=92.0,
        initial=float(up_deg),
    )
    return M.rx._moved_pose((tuple(M.FLAG_ZERO_POS), (0.0, 0.0, 0.0, 1.0)), joint)


def make_scene(
    parcel_pos=None,
    door_open_deg: float = 82.0,
    flag_deg: float = 0.0,
    vel=(0.0, 0.0, 0.0),
) -> dict[str, ObjectState]:
    """Construct ObjectState dictionary for parcel_pos, door_open_deg (82 open .. 0 closed), and flag_deg (0..90)."""
    if parcel_pos is None:
        parcel_pos = M.PARCEL_START_POS
    p_door, q_door = _door_pose(door_open_deg)
    p_flag, q_flag = _flag_pose(flag_deg)
    return {
        "mailbox": _state(M.MAILBOX_POS, (0.0, 0.0, 0.0, 1.0), M.MAILBOX_SIZE),
        "door": _state(p_door, q_door, M.DOOR_SIZE, lin_vel=vel),
        "flag": _state(p_flag, q_flag, M.FLAG_SIZE, lin_vel=vel),
        "parcel": _state(parcel_pos, (0.0, 0.0, 0.0, 1.0), M.PARCEL_SIZE, lin_vel=vel),
    }


INSIDE_PARCEL_POS = (0.435, M.Y_MAILBOX, TOP + M.FLOOR_H + 0.5 * M.PARCEL_SIZE[2])


@pytest.fixture(autouse=True)
def reset_episode_state():
    M.new_episode()


# -- 1. Task file contract and real-size constants ---------------------------------------------------------------------
def test_task_file_loads_and_uses_full_real_dimensions():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert M.SUCCESS_HOLD_S == 1.0 and M.CONTROL_HZ == 20.0
    assert M.INSTRUCTION == "Leave this parcel in the mailbox for the carrier to collect."
    assert [o.name for o in M.OBJECTS] == ["mailbox", "door", "flag", "parcel"]
    for obj in M.OBJECTS:
        assert all(type(v) is float for v in obj.pos), (obj.name, obj.pos)
        assert all(type(v) is float for v in obj.size), (obj.name, obj.size)
    # Gibraltar E1100B00 full 1:1 size: 509 mm deep x 175 mm wide x 225 mm tall
    assert M.MAILBOX_SIZE == (0.509, 0.175, 0.225)
    # USPS Small Flat Rate Box full 1:1 size: 219 x 41 x 137 mm, 0.25 kg
    assert M.PARCEL_SIZE == (0.219, 0.041, 0.137) and M.PARCEL_MASS == 0.250
    assert M.DOOR_CLOSED_MAX_DEG == 3.0
    assert M.FLAG_UP_TARGET_DEG == 90.0 and M.FLAG_UP_TOL_DEG == 6.0
    assert M.MAX_DOOR_PARCEL_FORCE_N == 15.0 and M.DOOR_CATCH_ENGAGE_DEG == 4.0
    assert not hasattr(M, "DOOR_BLOCKED_MIN_DEG")
    assert M.GRIPPER_CLEAR == 0.025
    assert "yam_arena.scene" not in sys.modules


def test_angle_and_inside_readouts():
    for d_deg in (82.0, 45.0, 10.0, 2.5, 0.0):
        for f_deg in (0.0, 30.0, 45.0, 86.0, 90.0):
            s = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=d_deg, flag_deg=f_deg)
            assert M.door_open_deg(s) == pytest.approx(d_deg, abs=1e-4)
            assert M.flag_angle_deg(s) == pytest.approx(f_deg, abs=1e-4)
            assert M.parcel_inside(s)
    # Start pose on table outside mailbox is not inside:
    s0 = make_scene()
    assert not M.parcel_inside(s0)
    # Protruding parcel (centre at x = 0.305 m -> low_x = 0.1955 m < X_SILL = 0.300 m) is not fully inside:
    s_protruding = make_scene(parcel_pos=(0.305, M.Y_MAILBOX, TOP + M.FLOOR_H + 0.5 * M.PARCEL_SIZE[2]))
    assert not M.parcel_inside(s_protruding)


# -- 2. Detent behaviour for door top catch and signal flag keeper -----------------------------------------------------
def test_door_and_flag_detents_engage_at_thresholds():
    # Spawn state: door unlatched (-82 deg target), flag down (0 deg target):
    M.advance(make_scene(door_open_deg=82.0, flag_deg=0.0))
    assert not M._S["door_latched"] and M._S["flag_target_deg"] == 0.0

    # Door carried to 6 deg (outside the 4 deg roller-catch window) stays unlatched:
    M.advance(make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=6.0, flag_deg=0.0))
    assert not M._S["door_latched"]

    # Door pushed within 4 deg of closed latches door_latched = True:
    M.advance(make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=2.0, flag_deg=0.0))
    assert M._S["door_latched"] and M._S["flag_target_deg"] == 0.0

    # Flag raised past 68 deg while parcel is inside and door is closed snaps flag_target_deg to 90.0 deg:
    M.advance(make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=1.0, flag_deg=72.0))
    assert M._S["door_latched"] and M._S["flag_target_deg"] == 90.0

    # Hysteresis: the catch keeps holding at 8 deg and lets go past 9 deg (engage + 5 deg):
    M.advance(make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=8.0, flag_deg=90.0))
    assert M._S["door_latched"]
    M.advance(make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=9.5, flag_deg=90.0))
    assert not M._S["door_latched"]


# -- 3. Scoring: 1.0 goal, no partial credit, 0.0 shortcuts ("try to beat it"), and latched faults --------------------
def test_goal_state_scores_one_and_succeeds(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    s = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=1.5, flag_deg=89.0)
    assert TASK.score(s) == pytest.approx(1.0)
    assert TASK.success(s)
    assert TASK.distance(s) == pytest.approx(0.0)


def test_parcel_inside_and_door_closed_with_flag_down_scores_zero(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    s = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=1.0, flag_deg=0.0)
    assert TASK.score(s) == pytest.approx(0.0)
    assert not TASK.success(s)
    assert 0.0 < TASK.distance(s) < 0.5


def test_try_to_beat_it_shortcuts_all_score_zero(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    # 1. Start state:
    s0 = make_scene()
    assert TASK.score(s0) == 0.0 and not TASK.success(s0)

    # 2. Parcel protruding across the sill (low_x = 0.2505 m < 0.300 m) even if door=0 and flag=90:
    M.new_episode()
    s_half_in = make_scene(
        parcel_pos=(0.360, M.Y_MAILBOX, TOP + M.FLOOR_H + 0.5 * M.PARCEL_SIZE[2]),
        door_open_deg=0.0,
        flag_deg=90.0,
    )
    assert TASK.score(s_half_in) == 0.0 and not TASK.success(s_half_in)

    # 3. Parcel inside and flag up, but door ajar at 6 deg (> 3 deg):
    M.new_episode()
    s_ajar = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=6.0, flag_deg=90.0)
    assert TASK.score(s_ajar) == 0.0 and not TASK.success(s_ajar)

    # 4. Parcel held in goal pose with gripper still within 25 mm (e.g. 12 mm from door):
    M.new_episode()
    s_goal = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=0.0, flag_deg=90.0)
    near_tcp = np.array([0.280, M.Y_MAILBOX, TOP + 0.120])
    monkeypatch.setattr(M, "tcp_points", lambda: [near_tcp, FAR_TCPS[1]])
    assert M.min_tcp_clearance_m(s_goal) < M.GRIPPER_CLEAR
    assert TASK.score(s_goal) == 0.0 and not TASK.success(s_goal)

    # 5. Offline with empty tcp_points() must return 0.0:
    monkeypatch.setattr(M, "tcp_points", lambda: [])
    assert TASK.score(s_goal) == 0.0 and not TASK.success(s_goal)

    # 6. Moving parts (not at rest) score 0.0:
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    s_moving = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=0.0, flag_deg=90.0, vel=(0.05, 0.0, 0.0))
    assert TASK.score(s_moving) == 0.0 and not TASK.success(s_moving)



def test_latched_fault_door_forced_against_parcel_permanently_zeros_score(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    protruding = (0.305, M.Y_MAILBOX, TOP + M.FLOOR_H + 0.5 * M.PARCEL_SIZE[2])
    # Contact force 10 N (<= 15 N) at door_open_deg = 25 deg does not fault:
    M.advance(make_scene(parcel_pos=protruding, door_open_deg=25.0, flag_deg=0.0), door_parcel_force_n=10.0)
    assert M._S["fault"] is None

    # Contact force 18 N (> 15 N) at door_open_deg = 25 deg latches door_forced_on_parcel:
    M.advance(make_scene(parcel_pos=protruding, door_open_deg=25.0, flag_deg=0.0), door_parcel_force_n=18.0)
    assert M._S["fault"] == "door_forced_on_parcel"

    s_goal = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=0.0, flag_deg=90.0)
    assert M.score(s_goal, FAR_TCPS, door_parcel_force_n=0.0) == 0.0
    assert not M.success(s_goal, FAR_TCPS, door_parcel_force_n=0.0)


def test_parcel_resting_against_closed_door_under_catch_force_does_not_fault(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    # Most the catch can press the door on a parcel at its top edge (0.200 m from the hinge): its spring torque at
    # the release angle (engage + 5 deg), plus the door's own weight leaning on it.
    catch_n = M.DOOR_CATCH_STIFFNESS * (M.DOOR_CATCH_ENGAGE_DEG + 5.0) / 0.200 + M.DOOR_MASS * 9.81
    assert catch_n < M.MAX_DOOR_PARCEL_FORCE_N / 5.0
    against_door = (0.300 + 0.5 * M.PARCEL_SIZE[0], M.Y_MAILBOX, TOP + M.FLOOR_H + 0.5 * M.PARCEL_SIZE[2])
    for d_deg in (0.0, 1.0, 3.0):
        M.advance(make_scene(parcel_pos=against_door, door_open_deg=d_deg, flag_deg=90.0), door_parcel_force_n=catch_n)
    assert M._S["fault"] is None
    assert M.score(make_scene(parcel_pos=against_door, door_open_deg=0.0, flag_deg=90.0), FAR_TCPS,
                   door_parcel_force_n=catch_n) == 1.0


def test_door_rammed_on_protruding_parcel_faults_at_any_door_angle(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    # Parcel sticking out ~25 mm; the door is rammed through its last degrees with > 15 N and shoves it past the sill.
    protruding = (0.300 + 0.5 * M.PARCEL_SIZE[0] - 0.025, M.Y_MAILBOX, TOP + M.FLOOR_H + 0.5 * M.PARCEL_SIZE[2])
    for d_deg in (8.0, 2.0, 0.0):
        M.new_episode()
        M.advance(make_scene(parcel_pos=protruding, door_open_deg=d_deg, flag_deg=90.0), door_parcel_force_n=15.5)
        assert M._S["fault"] == "door_forced_on_parcel", d_deg
        s_goal = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=0.0, flag_deg=90.0)
        assert M.score(s_goal, FAR_TCPS, door_parcel_force_n=0.0) == 0.0


def test_latched_fault_parcel_off_table_permanently_zeros_score(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    s_dropped = make_scene(parcel_pos=(0.250, +0.580, TOP - 0.050))
    M.advance(s_dropped)
    assert M._S["fault"] == "parcel_off_table"
    s_goal = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=0.0, flag_deg=90.0)
    assert TASK.score(s_goal) == 0.0


def test_contact_view_error_raises_instead_of_silently_returning_zero():
    class _BadView:
        filter_count = 1
        sensor_count = 1

        def get_contact_force_matrix(self, dt):
            raise RuntimeError("sim view gone")

    class _Sim:
        def get_physics_dt(self):
            return 0.005

    class _Base:
        sim = _Sim()

    M._CONTACT_VIEWS.clear()
    M._CONTACT_VIEWS.update(ready=True, door_parcel_view=_BadView())
    try:
        with pytest.raises(RuntimeError):
            M.read_door_parcel_contact_force_n(_Base())
    finally:
        M._CONTACT_VIEWS.clear()


# -- 4. Demo reachability, collision clearances, and top-camera visibility ---------------------------------------------
@pytest.fixture(scope="module")
def played():
    out = {}
    for side, waypoints in TASK.demo().items():
        arm = ArmPlan(waypoints, station.BASES[side])
        steps = []
        for k in range(round(arm.duration * HZ)):
            t = (k + 1) / HZ
            pos, _, finger, label = arm.target(t)
            action, _ = arm.action7(t)
            steps.append((t, action[:6].copy(), -float(action[6]) * 0.04695, pos, label))
        out[side] = (arm, steps)
    return out


def test_demo_is_reachable_with_large_joint_margins_at_20hz(played):
    assert set(played) == {"left", "right"}
    assert played["left"][0].duration == pytest.approx(played["right"][0].duration)
    assert played["left"][0].duration <= 25.0
    for side, (arm, steps) in played.items():
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (side, arm.worst_ik_mm, arm.worst_ik_deg)
        margin_deg = min(float(np.degrees(np.minimum(q - ik.LOWER, ik.UPPER - q)).min()) for _, q, *_ in steps)
        assert margin_deg >= 15.0, (side, margin_deg)


def _box_hull(c, r, h):
    v = np.array([c + r @ (h * np.array([a, b, d])) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    n = np.vstack([r.T, -r.T])
    return analysis.Hull(vertices=v, planes=np.c_[n, n @ c + np.concatenate([h, h])])


def test_demo_arms_clear_table_each_other_and_mailbox_body(played):
    (_, l_steps), (_, r_steps) = played["left"], played["right"]
    min_table_z = 1e9
    min_arm_arm = 1e9
    min_mailbox = 1e9

    boxes0 = M.scene_boxes_from_states(make_scene())
    mb_hulls = [(k, _box_hull(*v)) for k, v in boxes0.items() if k.startswith("mb_")]

    for (_, ql, fl, _, _), (_, qr, fr, _, _) in zip(l_steps[::5], r_steps[::5], strict=True):
        lh = analysis.world_hulls(
            analysis.fk_link_poses_world(
                {**{n: float(v) for n, v in zip(ARM_JOINTS, ql)}, "joint7": fl, "joint8": fl},
                station.BASES["left"],
            )
        )
        rh = analysis.world_hulls(
            analysis.fk_link_poses_world(
                {**{n: float(v) for n, v in zip(ARM_JOINTS, qr)}, "joint7": fr, "joint8": fr},
                station.BASES["right"],
            )
        )
        for name, h in lh + rh:
            if name != "base":
                min_table_z = min(min_table_z, float(h.vertices[:, 2].min() - TOP))
        d_aa, _ = analysis.hull_set_distance(lh, rh)
        min_arm_arm = min(min_arm_arm, d_aa)

        d_mb, _ = analysis.hull_set_distance(lh + rh, mb_hulls)
        min_mailbox = min(min_mailbox, d_mb)

    assert min_table_z >= 0.008
    assert min_arm_arm >= 0.080
    assert min_mailbox >= 0.004


def test_demo_ends_parked_at_least_25mm_clear_and_still_for_1s(played):
    for side in ("left", "right"):
        _, steps = played[side]
        assert steps[-1][4] in ("done", "l_done", "r_done")
        assert np.linalg.norm(steps[-1][3] - steps[-21][3]) < 1e-9  # stationary for final >= 1.0 s
    goal = make_scene(parcel_pos=INSIDE_PARCEL_POS, door_open_deg=0.0, flag_deg=90.0)
    end_tcps = [ik.tcp_pose(played[s][1][-1][1], station.BASES[s])[0] for s in ("left", "right")]
    assert M.min_tcp_clearance_m(goal, end_tcps) >= M.GRIPPER_CLEAR + 0.08


# -- 5. Metadata and README checks -------------------------------------------------------------------------------------
def test_task_json_and_readme_conform_to_guidelines():
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
    assert meta["category"] == "Societal relatability"
    assert meta["instruction"] == M.INSTRUCTION
    assert "Standard 7C" in meta["context"] and "quoted" in meta["context"]
    assert len(meta["fidelity"]) <= 500, len(meta["fidelity"])
    assert meta["scores"].startswith("Dream-1 run:")
    assert "media" not in meta
    readme = (HERE / "README.md").read_text()
    lines = [line.strip() for line in readme.splitlines()]
    assert "## Third-party assets" in lines and "none" in lines
    assert "E1100B00" in readme and "scripted drive" in readme
