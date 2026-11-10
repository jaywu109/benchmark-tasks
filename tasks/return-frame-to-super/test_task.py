"""Offline checks for return-frame-to-super (NumPy and pytest only; no Isaac Sim).

Run from this folder with the yam-isaac-lab-arena package importable (installed, as on the Brev VM, or a clone
named by YAM_ARENA):

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
"""

import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("YAM_ARENA"):
    sys.path.insert(0, os.path.expanduser(os.environ["YAM_ARENA"]))

import yam_arena  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.analysis import world_hulls  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import forward_kinematics, rpy_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
TOP = TASK.table_top
HZ = 20.0  # the rate the README's runs use
ARM_URDF = Path(yam_arena.__file__).resolve().parent / "assets" / "yam" / "yam_arm.urdf"  # links with collision meshes


# -- helpers -----------------------------------------------------------------------------------------------------------


def frame_state(x, y=0.0, z_top=None, tilt=0.0, yaw=0.0, vel=(0.0, 0.0, 0.0)):
    """ObjectState of a frame whose origin (top bar's top centre) is at (x, y, TOP + z_top), turned yaw about z then
    tilted about x (degrees)."""
    from yam_arena.urdf import axis_angle_matrix

    z_top = M.HANG_Z if z_top is None else z_top
    r = axis_angle_matrix(np.array([0.0, 0, 1]), math.radians(yaw)) @ axis_angle_matrix(np.array([1.0, 0, 0]),
                                                                                      math.radians(tilt))
    o = np.array([x, y, TOP + z_top])
    c = o + r @ M.FRAME_BBOX_C
    q = ik.matrix_to_quat_xyzw(r)
    half = np.abs(r) @ (np.asarray(M.FRAME_SIZE) / 2)
    return ObjectState(c, np.asarray(q, float), np.asarray(vel, float), np.zeros(3), c - half, c + half)


def goal_objects(**frame_kw):
    objs = {n: frame_state(x) for n, x in zip(M.KEPT_NAMES, M.KEPT_X, strict=True)}
    kw = {"x": M.GOAL_X, **frame_kw}
    objs["frame"] = frame_state(**kw)
    return objs


def _stl(path):
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:300]:
        return np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data), float)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84 : 84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.unique(rec["v"].reshape(-1, 3).astype(float), axis=0)


@lru_cache(maxsize=1)
def collision_points():
    """Every link's collision-mesh vertices in its own frame, from yam_arm.urdf."""
    root = ET.parse(ARM_URDF).getroot()
    out = {}
    for link in root.findall("link"):
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array(o.get("xyz").split(), float)
            rpy = np.array(o.get("rpy").split(), float)
            mesh = ARM_URDF.parent / col.find("geometry/mesh").get("filename")
            pts.append(_stl(mesh) @ rpy_matrix(rpy).T + xyz)
        if pts:
            out[link.get("name")] = np.vstack(pts)
    return out


def _poses(q6, finger, side):
    joints = {n: float(v) for n, v in zip(ARM_JOINTS, q6, strict=True)}
    joints["joint7"] = joints["joint8"] = float(finger)
    fk = forward_kinematics(ARM_URDF, joints)
    base = np.eye(4)
    base[:3, 3] = station.BASES[side]
    return {k: base @ v for k, v in fk.items() if k != "base"}


def links_in_world(q6, finger, side):
    poses = _poses(q6, finger, side)
    return {
        name: p @ poses[name][:3, :3].T + poses[name][:3, 3]
        for name, p in collision_points().items()
        if name in poses
    }


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


def world_pieces(compound, centre_world, rot=np.eye(3)):
    """(lo, hi) world AABBs of an axis-aligned compound's pieces (rot must be identity or a pure yaw of 0)."""
    out = []
    for p in compound.pieces:
        c = np.asarray(centre_world) + rot @ np.asarray(p.offset)
        h = np.abs(rot) @ (np.asarray(p.size) / 2)
        out.append((c - h, c + h))
    return out


def frame_pieces(x):
    return world_pieces(M._frame_compound(), (x, 0.0, TOP + M.HANG_Z + M.FRAME_BBOX_C[2]))


def super_pieces():
    return world_pieces(M._super_compound(), (M.SUPER_C[0], 0.0, TOP + M.SUPER_C[2]))


def perch_pieces():
    return world_pieces(M._perch_compound(), (M.PERCH_C[0], 0.0, TOP + M.PERCH_C[2]))


def box_gap(a, b):
    """Separation of two AABBs (m); negative = overlap depth along the least-overlapping axis."""
    (alo, ahi), (blo, bhi) = a, b
    sep = np.maximum(alo - bhi, blo - ahi)
    return float(sep.max()) if np.any(sep > 0) else float(sep.max())


def points_to_box(pts, lo, hi):
    """Distance from each point to an AABB (0 inside)."""
    return np.linalg.norm(pts - np.clip(pts, lo, hi), axis=1)


def step_at(side, label, last=True):
    _, steps = played(side)
    hits = [s for s in steps if s[3] == label]
    assert hits, label
    return hits[-1] if last else hits[0]


def finger_points(side, step):
    _, q, f, _ = step
    w = links_in_world(q, f, side)
    return np.vstack([w["tip_left"], w["tip_right"]])


def arm_points(side, step):
    _, q, f, _ = step
    return np.vstack(list(links_in_world(q, f, side).values()))


# -- 1. the file loads ------------------------------------------------------------------------------------------------


def test_the_task_file_loads():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert TASK.free_objects == ("frame", *M.KEPT_NAMES)
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))
    assert TASK.docs.startswith("On the station table")
    assert set(TASK.demo()) == {"left", "right"}
    assert "yam_arena.scene" not in sys.modules
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos), o.name


# -- 2. geometry -------------------------------------------------------------------------------------------------------


def test_real_sizes():
    assert 2 * M.LUG_Y == pytest.approx(0.483, abs=1e-3)  # 19 in over the lugs
    assert M.FRAME_H == pytest.approx(0.137, abs=1e-3)  # 5 3/8 in
    assert M.IN_X1 - M.IN_X0 == pytest.approx(0.375, abs=1e-3) and 2 * M.IN_Y == pytest.approx(0.467, abs=1e-3)
    assert M.BOX_H == pytest.approx(0.146, abs=1e-3) and M.BOX_H - M.REBATE_Z == pytest.approx(0.0095, abs=1e-4)
    assert M.SHOULDER_W == pytest.approx(0.035, abs=5e-4)


def test_hanging_frames_rest_on_the_rebates_clear_the_walls_and_each_other():
    sup = super_pieces()
    for x in (*M.KEPT_X, M.GOAL_X, M.SET_X):
        fr = frame_pieces(x)
        lugs = [fr[1], fr[2]]
        for lo, hi in lugs:
            assert lo[2] == pytest.approx(TOP + M.REBATE_Z, abs=1e-6)
            assert M.IN_Y < max(abs(lo[1]), abs(hi[1])) < M.LIP_Y  # each lug bears 8 mm on its rebate
        for k, piece in enumerate(fr):
            for s in sup:
                g = box_gap(piece, s)
                assert g >= -1e-6, (x, k, g)  # touching (lugs on the rebate floor) but never inside
        assert min(p[0][2] for p in fr) - TOP == pytest.approx(0.009, abs=1e-4)  # bottom bar 9 mm off the table
        end_bars = [p for p in fr if abs(p[0][1]) > M.END_IN_Y - 1e-6 and p[0][2] < TOP + M.HANG_Z - M.LUG_T - 1e-6]
        assert min(M.IN_Y - max(abs(lo[1]), abs(hi[1])) for lo, hi in end_bars) >= 0.009  # 9.5 mm to each end wall
    # the kept frames touch shoulder to shoulder; the last one touches the far wall
    for a, b in zip(M.KEPT_X, M.KEPT_X[1:]):
        assert a - b == pytest.approx(M.SHOULDER_W)
    assert M.KEPT_X[0] + M.SHOULDER_W / 2 == pytest.approx(M.IN_X1)


def test_the_perch_frame_hangs_on_the_perch_outside_the_near_wall():
    fr = frame_pieces(M.PERCH_FRAME_X)
    for piece in fr:
        for s in (*super_pieces(), *perch_pieces()):
            assert box_gap(piece, s) >= -1e-6
    assert max(p[1][0] for p in fr) < M.SUPER_X0  # all of it in front of the near wall


# -- 3. scorer ---------------------------------------------------------------------------------------------------------


def test_goal_scores_one_and_succeeds():
    objs = goal_objects()
    assert M.score(objs) == 1.0 and M.success(objs)
    assert M.distance(objs) == pytest.approx(0.0, abs=1e-9)


def test_frame_hung_but_left_with_a_gap_scores_half():
    objs = goal_objects(x=M.GOAL_X - 0.005)
    assert M.gap_mm(objs["frame"], objs["comb_7"]) == pytest.approx(5.0)
    assert M.score(objs) == 0.5 and not M.success(objs)
    assert M.score(goal_objects(x=M.GOAL_X - 0.0019)) == 1.0


def test_a_kept_frame_dragged_out_of_the_row_leaves_a_gap_and_scores_half():
    # comb_7 pulled 20 mm toward the near wall, the returned frame closed up against it: gap is now 6-7
    objs = goal_objects(x=M.GOAL_X - 0.020)
    objs["comb_7"] = frame_state(M.KEPT_X[6] - 0.020)
    assert M.gap_mm(objs["frame"], objs["comb_7"]) == pytest.approx(0.0, abs=1e-6)
    assert M.score(objs) == 0.5 and not M.success(objs)


@pytest.mark.parametrize("kw", [
    {"x": M.PERCH_FRAME_X},  # never moved: still on the perch
    {"z_top": M.HANG_Z + 0.004},  # held up
    {"z_top": M.HANG_Z - 0.010, "tilt": 8.0},  # one lug off its rebate
    {"yaw": 5.0},  # skewed
    {"y": 0.008},  # shifted along the rebates
    {"vel": (0.05, 0.0, 0.0)},  # still moving
])
def test_frame_not_hanging_square_and_still_scores_zero(kw):
    assert M.score(goal_objects(**kw)) == 0.0


def test_a_kept_frame_knocked_off_its_rebates_scores_zero():
    objs = goal_objects()
    objs["comb_3"] = frame_state(M.KEPT_X[2], z_top=M.HANG_Z - 0.02, tilt=12.0)
    assert M.score(objs) == 0.0


def test_a_kept_frame_knocked_off_and_hung_back_still_scores_zero_until_the_next_episode(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(M, "episode_t", lambda: clock["t"])
    monkeypatch.setitem(M._KNOCK, "t", None)
    monkeypatch.setitem(M._KNOCK, "knocked", False)
    assert M.score(goal_objects()) == 1.0
    clock["t"] = 5.0
    objs = goal_objects()
    objs["comb_3"] = frame_state(M.KEPT_X[2], z_top=M.HANG_Z - 0.006)  # dropped 6 mm off its rebates
    assert M.score(objs) == 0.0
    clock["t"] = 9.0
    assert M.score(goal_objects()) == 0.0 and not M.success(goal_objects())  # hung back: still latched
    clock["t"] = 0.05  # reset: the clock goes back
    assert M.score(goal_objects()) == 1.0


def test_a_tcp_within_3_cm_of_the_frame_scores_zero(monkeypatch):
    objs = goal_objects()
    o = M.origin(objs["frame"])
    monkeypatch.setattr(M, "tcp_points", lambda: [o + np.array([0.0, 0.08, 0.01]), np.array([0.2, -0.4, TOP + 0.12])])
    assert M.score(objs) == 0.0
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array([0.2, 0.4, TOP + 0.12]), np.array([0.2, -0.4, TOP + 0.12])])
    assert M.score(objs) == 1.0


def test_a_frame_turned_end_for_end_counts_the_same():
    assert M.score(goal_objects(yaw=180.0)) == 1.0


# -- 4. demo -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_every_arm_plan_is_reachable_within_the_joint_limits(side):
    arm, steps = played(side)
    joints = np.array([s[1] for s in steps])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (side, arm.worst_ik_mm, arm.worst_ik_deg)
    assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER), side
    assert arm.duration <= 30.0


def test_both_arms_share_one_clock_and_close_for_at_least_1_s():
    plan = TASK.demo()
    assert [w.duration for w in plan["left"]] == [w.duration for w in plan["right"]]
    close = [w for w in plan["right"] if w.label == "close"][0]
    assert close.duration >= 1.0 and close.finger == pytest.approx(-(M.TOPBAR_W / 2 - 0.008))


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_pads_straddle_the_top_bar_when_closing(side):
    pts = finger_points(side, step_at(side, "down to the top bar"))
    s = 1.0 if side == "left" else -1.0
    # fingertip points level with the top bar sit outside its 27 mm width (open jaws), none inside it
    lo = np.array([M.PERCH_FRAME_X - M.TOPBAR_W / 2, s * M.HAND_Y - 0.06, TOP + M.HANG_Z - M.TOPBAR_T])
    hi = np.array([M.PERCH_FRAME_X + M.TOPBAR_W / 2, s * M.HAND_Y + 0.06, TOP + M.HANG_Z])
    inside = np.all((pts > lo + 1e-4) & (pts < hi - 1e-4), axis=1)
    assert not inside.any(), side


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_closed_fingers_push_the_top_bar_near_face_past_the_touching_pose(side):
    pts = finger_points(side, step_at(side, "hold"))
    s = 1.0 if side == "left" else -1.0
    band = (np.abs(pts[:, 1] - s * M.HAND_Y) < 0.08) & (pts[:, 2] > TOP + M.HANG_Z - M.TOPBAR_T) & (
        pts[:, 2] < TOP + M.HANG_Z)
    assert band.any(), "no finger point level with the top bar"
    lead = float(pts[band, 0].max())
    face = M.GOAL_X - M.TOPBAR_W / 2
    # 0-2.5 mm past the face (left 2.0, right 0.4) plus the push's PUSH_OVER aim past contact
    assert M.PUSH_OVER <= lead - face <= M.PUSH_OVER + 0.0025, (side, lead - face)


@pytest.mark.parametrize("side", ["left", "right"])
def test_no_arm_link_enters_the_super_or_the_kept_frames(side):
    _, steps = played(side)
    boxes = super_pieces() + perch_pieces() + [p for x in M.KEPT_X for p in frame_pieces(x)]
    worst = np.inf
    for st in steps[::2]:
        pts = arm_points(side, st)
        for lo, hi in boxes:
            worst = min(worst, float(points_to_box(pts, lo, hi).min()))
    assert worst >= 0.003, (side, worst)


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
