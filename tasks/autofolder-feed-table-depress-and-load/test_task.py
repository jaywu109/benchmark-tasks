"""Offline checks for autofolder-feed-table-depress-and-load (NumPy and pytest only; no Isaac Sim).

    YAM_ARENA=~/yam-isaac-lab-arena pytest -q tasks/autofolder-feed-table-depress-and-load/test_task.py
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
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
TOP = TASK.table_top
HZ = 20.0
FAR_TCPS = [np.array(M.PARK_LEFT, dtype=float), np.array(M.PARK_RIGHT, dtype=float)]


def _state(pos, quat_xyzw, size, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> ObjectState:
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat_xyzw, dtype=float)
    rot = quat_xyzw_to_matrix(quat)
    ext = np.abs(rot) @ (np.asarray(size, dtype=float) / 2.0)
    return ObjectState(
        pos,
        quat,
        np.asarray(lin_vel, dtype=float),
        np.asarray(ang_vel, dtype=float),
        pos - ext,
        pos + ext,
    )


def _table_pose(table_deg: float, folder_shift=(0.0, 0.0, 0.0), folder_tilt_deg: float = 0.0):
    joint = M.rx.Revolute(
        anchor=M.FEED_TABLE_ANCHOR,
        axis="X",
        lower=M.FEED_TABLE_LOWER_DEG,
        upper=M.FEED_TABLE_UPPER_DEG,
        initial=float(table_deg),
    )
    p_local, q_local = M.rx._moved_pose((M.FEED_TABLE_ZERO_POS, (0.0, 0.0, 0.0, 1.0)), joint)
    r_fld = axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(float(folder_tilt_deg)))
    q_fld = tuple(float(v) for v in ik.matrix_to_quat_xyzw(r_fld))
    p_fld = tuple(float(c + d) for c, d in zip(M.FOLDER_SPAWN_POS, folder_shift))
    return M.rx._compose(
        (p_fld, q_fld),
        M.rx._compose(M.rx._inverse((M.FOLDER_SPAWN_POS, (0.0, 0.0, 0.0, 1.0))), (p_local, q_local)),
    )


def make_scene(
    *,
    loaded: bool = False,
    table_deg: float | None = None,
    stop_gap_m: float = 0.0,
    dx_m: float = 0.0,
    dz_deck_m: float = 0.0,
    packet_yaw_deg: float = 0.0,
    packet_offset=(0.0, 0.0, 0.0),
    folder_shift=(0.0, 0.0, 0.0),
    folder_tilt_deg: float = 0.0,
    packet_vel=(0.0, 0.0, 0.0),
) -> dict[str, ObjectState]:
    """Construct ObjectState dictionary for autofolder_housing, feed_table, and newsletter_packet."""
    if table_deg is None:
        table_deg = M.PACKET_CLAMPED_DEG if loaded else 0.0

    f_pos = tuple(float(c + d) for c, d in zip(M.FOLDER_SPAWN_POS, folder_shift))
    r_fld = axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(float(folder_tilt_deg)))
    q_fld = tuple(float(v) for v in ik.matrix_to_quat_xyzw(r_fld))

    p_tbl, q_tbl = _table_pose(table_deg, folder_shift=folder_shift, folder_tilt_deg=folder_tilt_deg)

    if loaded:
        p_pkt0, q_pkt = M.packet_pose_on_table(
            table_deg=table_deg,
            stop_gap_m=stop_gap_m,
            dx_m=dx_m,
            dz_deck_m=dz_deck_m,
            yaw_deg=packet_yaw_deg,
            folder_shift=folder_shift,
        )
    else:
        p_pkt0 = M.PACKET_SPAWN_POS
        r_pkt = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(float(packet_yaw_deg)))
        q_pkt = tuple(float(v) for v in ik.matrix_to_quat_xyzw(r_pkt))

    p_pkt = tuple(float(c + d) for c, d in zip(p_pkt0, packet_offset))
    return {
        "autofolder_housing": _state(f_pos, q_fld, M.FOLDER_SIZE),
        "feed_table": _state(p_tbl, q_tbl, M.FEED_TABLE_SIZE),
        "newsletter_packet": _state(p_pkt, q_pkt, M.PACKET_SIZE, lin_vel=packet_vel),
    }


@pytest.fixture(autouse=True)
def reset_episode_state():
    M.new_episode()


# -- 1. Task file contract, constants, and contact-only spring geometry ------------------------------------------------
def test_task_file_loads_and_declares_required_constants():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert M.SUCCESS_HOLD_S == 1.0 and M.CONTROL_HZ == 20.0
    assert M.INSTRUCTION == "Load this newsletter stack into the folder so it is ready to feed."
    assert [o.name for o in M.OBJECTS] == ["autofolder_housing", "feed_table", "newsletter_packet"]
    for obj in M.OBJECTS:
        assert all(type(v) is float for v in obj.pos), (obj.name, obj.pos)
        assert all(type(v) is float for v in obj.size), (obj.name, obj.size)
        assert type(obj.mass) is float and obj.mass > 0.0, (obj.name, obj.mass)

    # Candidate dimensions and masses (Martin Yale 1501X 14.1 kg net weight and half-letter 216 x 140 x 20 mm stack):
    assert M.FOLDER_SIZE == (0.252, 0.380, 0.176) and M.FOLDER_MASS == 14.1
    assert M.FEED_TABLE_SIZE == (0.210, 0.226, 0.028) and M.FEED_TABLE_MASS == 0.240
    assert M.PACKET_SIZE == (0.216, 0.140, 0.020) and M.PACKET_MASS == 0.540
    for comp, sz in (
        (M.AUTOFOLDER_COMPOUND, M.FOLDER_SIZE),
        (M.FEED_TABLE_COMPOUND, M.FEED_TABLE_SIZE),
        (M.PACKET_COMPOUND, M.PACKET_SIZE),
    ):
        lo = np.min([np.array(p.offset) - 0.5 * np.array(p.size) for p in comp.pieces], axis=0)
        hi = np.max([np.array(p.offset) + 0.5 * np.array(p.size) for p in comp.pieces], axis=0)
        assert np.allclose(0.5 * (lo + hi), 0.0, atol=1e-6)
        assert np.allclose(hi - lo, sz, atol=1e-6)
    assert M.GUIDE_SPACING_M - M.PACKET_LX == pytest.approx(0.007, abs=1e-6)  # 3.5 mm per side
    assert M.WHEEL_DIAMETER_M == pytest.approx(0.058, abs=1e-6)

    # Constant passive spring hinge on feed_table (parent="autofolder_housing"):
    assert M.FEED_TABLE_JOINT.parent == "autofolder_housing"
    assert M.FEED_TABLE_JOINT.axis == "X"
    assert M.FEED_TABLE_JOINT.upper == 0.0 and M.FEED_TABLE_JOINT.initial == 0.0
    assert M.FEED_TABLE_JOINT.lower == -22.0 and M.TABLE_STOP_DEG == -16.5
    assert M.FEED_TABLE_JOINT.stiffness == pytest.approx(0.055)
    assert M.FEED_TABLE_JOINT.target == pytest.approx(13.0)

    # Scored end-state and latched fault constants from job notes:
    assert M.STOP_GAP_MAX_M == 0.005
    assert M.PACKET_YAW_MAX_DEG == 5.0
    assert (M.CLAMP_ANGLE_MIN_DEG, M.CLAMP_ANGLE_MAX_DEG) == (-7.5, -3.5)
    assert M.GRIPPER_CLEAR == 0.030
    assert M.FOLDER_SHIFT_MAX_M == 0.015
    assert M.PACKET_DROP_MAX_M == 0.050
    assert M.TABLE_STOP_DEG == -16.5
    assert "yam_arena.scene" not in sys.modules


def test_contact_blocking_clamping_and_piece_clearances():
    s0 = make_scene(loaded=False)
    assert s0["autofolder_housing"].low[2] == pytest.approx(TOP, abs=1e-6)
    # At unloaded rest (0.0 deg), the nip gap under the 58 mm rubber feed wheel is 7.38..8.92 mm,
    # physically blocking the 20.0 mm newsletter_packet by > 11 mm unless feed_table is depressed:
    assert M.EMPTY_NIP_MIN_GAP_M == pytest.approx(0.00738, abs=1e-5)
    assert M.PACKET_LZ - M.EMPTY_NIP_MIN_GAP_M > 0.012

    # When released with the 20.0 mm packet on the feed table, contact with the overhead feed wheel occurs at
    # PACKET_CLAMPED_DEG = -5.50 deg (centre of [-7.5, -3.5] deg), with 0.0 mm wheel nip gap:
    s_clamped = make_scene(loaded=True, table_deg=-5.50, stop_gap_m=0.0)
    assert M.feed_table_angle_deg(s_clamped) == pytest.approx(-5.50, abs=1e-4)
    assert M.packet_wheel_gap_m(s_clamped) == pytest.approx(0.0, abs=1e-4)
    assert M.packet_leading_edge_gap_m(s_clamped) == pytest.approx(0.0, abs=1e-6)

    # Grasped 20.0 mm packet thickness and 18.0 mm paddle have > 73 mm margin inside the 93.9 mm YAM gripper opening:
    assert M.YAM_MAX_OPENING_M - M.PACKET_LZ >= 0.073
    assert M.YAM_MAX_OPENING_M - M.PADDLE_THICK_Z >= 0.075

    # Across the entire [-22.0, 0.0] deg rotation range, feed_table pieces maintain > 5.0 mm AABB clearance from
    # autofolder_housing pieces (zero internal binding):
    min_gap = 1e9
    for deg in np.linspace(-22.0, 0.0, 45):
        th = math.radians(float(deg))
        c, s = math.cos(th), math.sin(th)
        for _, tlo, thi, _ in M.FEED_TABLE_PIECE_BOUNDS:
            corners = []
            for x in (tlo[0], thi[0]):
                for y in (tlo[1], thi[1]):
                    for z in (tlo[2], thi[2]):
                        dy, dz = y - M.Y_PIVOT, z - M.Z_PIVOT_REL
                        corners.append((x, M.Y_PIVOT + dy * c - dz * s, M.Z_PIVOT_REL + dy * s + dz * c))
            corners = np.asarray(corners, dtype=float)
            rlo, rhi = corners.min(axis=0), corners.max(axis=0)
            for _, flo, fhi, _ in M.FOLDER_PIECE_BOUNDS:
                gap = np.maximum(np.asarray(flo) - rhi, rlo - np.asarray(fhi)).max()
                min_gap = min(min_gap, float(gap))
    assert min_gap >= 0.005


def test_table_angle_and_packet_readouts_across_states():
    for deg in (0.0, -3.5, -5.5, -7.5, -14.0, -16.5):
        s = make_scene(loaded=True, table_deg=deg, stop_gap_m=0.002, packet_yaw_deg=2.5)
        assert M.feed_table_angle_deg(s) == pytest.approx(deg, abs=1e-4)
        assert M.packet_leading_edge_gap_m(s) == pytest.approx(0.002, abs=1e-4)
        assert M.packet_yaw_deg(s) == pytest.approx(2.5, abs=0.1)
        assert M.packet_at_retarder_stop(s)
        assert M.packet_square_in_guides(s)
        assert M.feed_table_clamped(s) == (-7.5 <= deg <= -3.5 and M.packet_wheel_gap_m(s) <= M.WHEEL_NIP_GAP_MAX_M)


# -- 2. Scorer tests: 1.0 goal, 0.5 partial credit, 0.0 shortcuts, and latched faults ---------------------------------
def test_goal_state_scores_one_and_succeeds(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    s_goal = make_scene(loaded=True, table_deg=-5.5, stop_gap_m=0.001)
    assert TASK.score(s_goal) == pytest.approx(1.0)
    assert TASK.success(s_goal)
    assert TASK.distance(s_goal) == pytest.approx(0.0, abs=1e-6)


def test_partial_credit_cases_score_half(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    # Case A: feed_table depressed to -14.0 deg and packet slid within 15 mm of retarder stop before releasing table:
    s_depressed = make_scene(loaded=True, table_deg=-14.0, stop_gap_m=0.004)
    assert TASK.score(s_depressed) == pytest.approx(0.5)
    assert not TASK.success(s_depressed)

    # Case B: packet loaded at stop and clamped (-5.5 deg), but right gripper still within 30 mm:
    M.new_episode()
    s_goal = make_scene(loaded=True, table_deg=-5.5, stop_gap_m=0.001)
    near_tcp = [FAR_TCPS[0], np.asarray(s_goal["newsletter_packet"].pos, dtype=float)]
    assert M.score(s_goal, tcps=near_tcp) == pytest.approx(0.5)
    assert not M.success(s_goal, tcps=near_tcp)


def test_incomplete_or_shortcut_states_score_zero(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    # 1. Spawn state (table at 0 deg, packet on right-hand staging apron):
    assert TASK.score(make_scene(loaded=False)) == 0.0

    # 2. Feed table depressed and released without loading packet (returns to 0.0 deg outside [-7.5, -3.5] deg):
    M.new_episode()
    assert TASK.score(make_scene(loaded=False, table_deg=-14.0)) == 0.0
    assert TASK.score(make_scene(loaded=False, table_deg=0.0)) == 0.0

    # 3. Packet blocked short of retarder stop (25 mm > 5 mm stop gap):
    M.new_episode()
    s_short = make_scene(loaded=True, table_deg=-5.5, stop_gap_m=0.025)
    assert not M.packet_at_retarder_stop(s_short)
    assert TASK.score(s_short) == 0.0

    # 4. Packet skewed > 5 deg yaw in guides (8 deg yaw):
    M.new_episode()
    s_skewed = make_scene(loaded=True, table_deg=-5.5, stop_gap_m=0.001, packet_yaw_deg=8.0)
    assert not M.packet_square_in_guides(s_skewed)
    assert TASK.score(s_skewed) == 0.0


def test_latched_faults_permanently_zero_score(monkeypatch):
    monkeypatch.setattr(M, "tcp_points", lambda: FAR_TCPS)
    s_goal = make_scene(loaded=True, table_deg=-5.5, stop_gap_m=0.001)

    # 1. Folder moved > 15 mm:
    M.new_episode()
    s_shifted = make_scene(loaded=False, folder_shift=(0.018, 0.0, 0.0))
    assert TASK.score(s_shifted) == 0.0
    assert M._S["fault"] == "folder_moved"
    assert TASK.score(s_goal) == 0.0 and not TASK.success(s_goal)

    # 2. Packet dropped from > 50 mm:
    M.new_episode()
    s_dropped = make_scene(loaded=False, packet_offset=(0.0, 0.0, -0.060))
    assert TASK.score(s_dropped) == 0.0
    assert M._S["fault"] == "packet_dropped"
    assert TASK.score(s_goal) == 0.0 and not TASK.success(s_goal)

    # 3. Feed table pushed past its stop (< -16.5 deg):
    M.new_episode()
    s_overpressed = make_scene(loaded=False, table_deg=-18.0)
    assert TASK.score(s_overpressed) == 0.0
    assert M._S["fault"] == "table_pushed_past_stop"
    assert TASK.score(s_goal) == 0.0 and not TASK.success(s_goal)


def test_lifted_60mm_and_lowered_while_held_does_not_latch_drop():
    # Lifted 60 mm and lowered while held = no drop; releasing at goal scores 1.0:
    M.new_episode()
    s_lifted = make_scene(loaded=False, packet_offset=(0.0, 0.0, +0.060))
    held_lifted_tcps = [FAR_TCPS[0], np.asarray(s_lifted["newsletter_packet"].pos, dtype=float)]
    assert M.advance(s_lifted, tcps=held_lifted_tcps) is None
    assert M._S["fault"] is None

    s_lowered = make_scene(loaded=False, packet_offset=(0.0, 0.0, 0.0))
    held_lowered_tcps = [FAR_TCPS[0], np.asarray(s_lowered["newsletter_packet"].pos, dtype=float)]
    assert M.advance(s_lowered, tcps=held_lowered_tcps) is None
    assert M._S["fault"] is None

    s_goal = make_scene(loaded=True, table_deg=-5.5, stop_gap_m=0.001)
    held_goal_tcps = [FAR_TCPS[0], np.asarray(s_goal["newsletter_packet"].pos, dtype=float)]
    assert M.score(s_goal, tcps=held_goal_tcps) == pytest.approx(0.5)
    assert M.score(s_goal, tcps=FAR_TCPS) == pytest.approx(1.0)
    assert M._S["fault"] is None
    assert M.success(s_goal, tcps=FAR_TCPS)


# -- 3. Demo reachability, 20 Hz IK, collision clearances, and top-camera visibility -----------------------------------
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
            steps.append((t, action[:6].copy(), float(FINGER_OPEN) * float(action[6]), pos, label))
        out[side] = (arm, steps)
    return out


def test_demo_is_reachable_at_20hz_within_joint_limits_and_run_limit(played):
    assert set(played) == {"left", "right"}
    assert played["left"][0].duration == pytest.approx(played["right"][0].duration)
    limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0])
    assert played["left"][0].duration * HZ <= limit - 100
    for side, (arm, steps) in played.items():
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (side, arm.worst_ik_mm, arm.worst_ik_deg)
        joints = np.array([s[1] for s in steps])
        assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER)
        margin_deg = min(float(np.degrees(np.minimum(q - ik.LOWER, ik.UPPER - q)).min()) for _, q, *_ in steps)
        assert margin_deg >= 15.0, (side, margin_deg)


def test_demo_arms_clear_table_and_each_other(played):
    (_, l_steps), (_, r_steps) = played["left"], played["right"]
    min_table_z = 1e9
    min_arm_arm = 1e9

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

    assert min_table_z >= 0.015
    assert min_arm_arm >= 0.040


def test_demo_holds_table_depressed_while_sliding_packet_and_ends_parked(played):
    (_, l_steps), (_, r_steps) = played["left"], played["right"]
    seat_indices = [i for i, s in enumerate(r_steps) if s[4] in ("r_slide_into_guides", "r_seat_at_stop")]
    assert seat_indices
    for idx in seat_indices:
        assert l_steps[idx][4] == "l_hold_depressed"

    for side in ("left", "right"):
        _, steps = played[side]
        assert steps[-1][4] in ("l_done", "r_done", "done")
        assert np.linalg.norm(steps[-1][3] - steps[-30][3]) < 1e-9
    goal = make_scene(loaded=True, table_deg=-5.5, stop_gap_m=0.0)
    end_tcps = [ik.tcp_pose(played[s][1][-1][1], station.BASES[s])[0] for s in ("left", "right")]
    assert M.min_tcp_clearance_m(goal, end_tcps) >= M.GRIPPER_CLEAR + 0.05


def _ray_hits_hull(a, b, hull) -> bool:
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
    return t1 > 1e-6 and t0 < 1.0 - 1e-6


def _visible_fraction(state: ObjectState, hulls) -> float:
    cam = np.asarray(station.top_camera().pos, dtype=float)
    lo, hi = state.low, state.high
    g = np.linspace(0.1, 0.9, 5)
    pts = [[lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1]), hi[2]] for a in g for b in g]
    seen = sum(not any(_ray_hits_hull(cam, np.asarray(p, dtype=float), h) for _, h in hulls) for p in pts)
    return seen / len(pts)


def test_top_camera_sees_all_objects_in_first_and_last_frames(played):
    h_first, h_last = [], []
    for side in ("left", "right"):
        h_first += analysis.world_hulls(
            analysis.fk_link_poses_world(
                {**{n: float(v) for n, v in zip(ARM_JOINTS, READY)}, "joint7": FINGER_OPEN, "joint8": FINGER_OPEN},
                station.BASES[side],
            )
        )
        _, q_end, f_end, *_ = played[side][1][-1]
        h_last += analysis.world_hulls(
            analysis.fk_link_poses_world(
                {**{n: float(v) for n, v in zip(ARM_JOINTS, q_end)}, "joint7": f_end, "joint8": f_end},
                station.BASES[side],
            )
        )
    s_start = make_scene(loaded=False)
    s_end = make_scene(loaded=True, table_deg=-5.5, stop_gap_m=0.0)
    for name in ("autofolder_housing", "feed_table", "newsletter_packet"):
        assert _visible_fraction(s_start[name], h_first) >= 0.80, (name, "start")
        assert _visible_fraction(s_end[name], h_last) >= 0.80, (name, "end")


# -- 4. Metadata and README checks -------------------------------------------------------------------------------------
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
    assert meta["category"] == "Impact on jobs"
    assert meta["instruction"] == M.INSTRUCTION
    assert "43-9061" in meta["context"] and "2,487,670" in meta["context"]
    assert "43-6014" in meta["context"] and "1,722,060" in meta["context"]
    assert "https://www.bls.gov/oes/current/oes439061.htm" in meta["context"]
    assert "https://www.bls.gov/oes/current/oes436014.htm" in meta["context"]
    assert "https://www.martinyale.com/product/1501x-paper-folder/" in meta["context"]
    assert "https://www.martinyale.com/product/1501x-paper-folder/" in meta["setup"]
    assert '"' not in meta["context"]
    assert len(meta["fidelity"]) <= 500, len(meta["fidelity"])
    assert isinstance(meta["scores"], str) and len(meta["scores"]) > 0
    assert str(meta.get("media", "M-")).startswith("M-")
    readme = (HERE / "README.md").read_text()
    lines = [line.strip() for line in readme.splitlines()]
    assert "## Third-party assets" in lines and "none" in lines
    for tool_word in ("feeler gauge", "protractor", "ruler", "video"):
        assert tool_word in readme.lower(), tool_word
    assert "robocurve_ext.py" in readme and meta["instruction"] in readme
