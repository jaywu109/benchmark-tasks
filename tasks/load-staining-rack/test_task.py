"""Offline checks for load-staining-rack (NumPy and pytest only; no Isaac Sim).

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

for _candidate in (os.environ.get("YAM_ARENA"), os.environ.get("YAM_ARENA_DIR"), "~/yam-isaac-lab-arena"):
    if _candidate and Path(_candidate).expanduser().is_dir():
        sys.path.insert(0, str(Path(_candidate).expanduser()))
        break

import yam_arena  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.analysis import world_hulls  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, PAD_TIP_PAST_TCP, READY  # noqa: E402
from yam_arena.demo import ArmPlan, grip_rotation  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, forward_kinematics, quat_xyzw_to_matrix, rpy_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
TOP = TASK.table_top
HZ = 20.0
GRIPPER_MAX = 2 * abs(FINGER_OPEN)  # pads 93.9 mm apart fully open
MARGIN = 0.01
ARM_URDF = Path(yam_arena.__file__).resolve().parent / "assets" / "yam" / "yam_arm.urdf"
NAMES = M.SLIDE_NAMES
SIZE = np.asarray(M.SLIDE_SIZE, float)


# -- helpers -----------------------------------------------------------------------------------------------------------


def slide(x, y, bottom=M.SLAB_TOP, tilt_deg=0.0, tilt_axis=(0.0, 1.0, 0.0), yaw_deg=0.0, vel=(0.0, 0.0, 0.0)):
    """An ObjectState for a slide with its lowest point at ``bottom`` and tilted by ``tilt_deg``."""
    rz = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(yaw_deg))
    rt = axis_angle_matrix(np.asarray(tilt_axis, float), math.radians(tilt_deg))
    r = rz @ rt
    half_local = SIZE / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half_local
    rotated = corners @ r.T
    half_world = rotated.max(axis=0)
    pos = np.array([x, y, bottom + half_world[2]], float)
    return ObjectState(
        pos,
        ik.matrix_to_quat_xyzw(r),
        np.asarray(vel, float),
        np.zeros(3),
        pos - half_world,
        pos + half_world,
    )


def start():
    return {n: slide(x, M.DRYING_Y, M.SLAB_TOP) for n, x in zip(NAMES, M.SLOT_X, strict=True)}


def goal(**overrides):
    out = {n: slide(x, M.RACK_Y, M.SLAB_TOP) for n, x in zip(NAMES, M.SLOT_X, strict=True)}
    out.update(overrides)
    return out


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


def box_gap(points, lo, hi):
    """Smallest Euclidean distance (m) from mesh vertices to an axis-aligned box [lo, hi] (negative if inside)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    d = np.abs(points - centre) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return float((outside + inside).min())


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
    assert TASK.free_objects == NAMES and len(TASK.free_objects) == 5
    assert len(TASK.objects) == 5 + 7 + 7  # 5 slides + 7 drying-rack boxes + 7 staining-rack boxes
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))
    assert TASK.docs.startswith("On the station table")
    assert set(TASK.demo()) == {"left", "right"}


# -- 2. the demo is reachable and clears racks and neighbours ---------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_every_arm_plan_is_reachable_within_the_joint_limits_at_20_hz(side):
    arm, steps = played(side)
    joints = np.array([s[1] for s in steps])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (side, arm.worst_ik_mm, arm.worst_ik_deg)
    assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER), side
    assert max(w.pos[2] for w in TASK.demo()[side]) - TOP <= 0.14 + 1e-9


def test_both_arms_follow_one_clock_and_end_with_a_still_hold():
    plan = TASK.demo()
    left, right = plan["left"], plan["right"]
    assert [w.duration for w in left] == [w.duration for w in right]
    assert left[-1].pos == left[-2].pos == M.PARK_LEFT
    assert left[-1].finger == left[-2].finger == FINGER_OPEN
    assert left[-1].duration >= 2.0
    assert all(w.pos == M.PARK_RIGHT and w.finger == FINGER_OPEN for w in right)
    run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
    plan_steps = round(sum(w.duration for w in left) * HZ)
    expected_limit = int(math.ceil((plan_steps + 100) / 200.0) * 200)
    assert run_limit == expected_limit


def test_open_pads_at_grip_height_clear_groove_blocks_and_neighbouring_slides():
    """At the grip height on both the drying rack and the staining rack, the open finger pads (URDF collision meshes
    for tip_left and tip_right) clear all groove blocks and all neighbouring slides by at least 3 mm."""
    quat = ik.matrix_to_quat_xyzw(grip_rotation(M.GRIP_YAW))
    for side, rack_y, rack_boxes, tcp_z in (
        ("left", M.DRYING_Y, M.DRYING_RACK[1:], M.GRIP_Z),
        ("left", M.RACK_Y, M.STAINING_RACK[1:], M.PLACE_Z),
    ):
        base = np.asarray(station.BASES[side], float)
        slides = {i: slide(x, rack_y, M.SLAB_TOP) for i, x in enumerate(M.SLOT_X)}
        for k, x in enumerate(M.SLOT_X):
            target = np.array([x, rack_y, tcp_z], float)
            q, pe, re_ = ik.solve(target, quat, np.array(READY, float), base_pos=base, iters=400)
            assert pe < 5e-4 and re_ < 1e-3
            world = links_in_world(q, FINGER_OPEN, side)
            pads = np.vstack([world["tip_left"], world["tip_right"]])
            for b in rack_boxes:
                blo = np.asarray(b.pos) - np.asarray(b.size) / 2
                bhi = np.asarray(b.pos) + np.asarray(b.size) / 2
                gap = box_gap(pads, blo, bhi)
                assert gap >= 0.003, (rack_y, k, b.name, gap)
            for j, st in slides.items():
                if j != k:
                    gap = box_gap(pads, st.low, st.high)
                    assert gap >= 0.003, (rack_y, k, j, gap)


def test_carried_slide_clears_groove_block_tops_by_at_least_2_cm():
    slide_bottom_at_carry = M.CARRY_Z - M.GRIP_H
    drying_top = M.SLAB_TOP + M.DRYING_BLOCK_H
    staining_top = M.SLAB_TOP + M.RACK_BLOCK_H
    assert slide_bottom_at_carry - drying_top >= 0.020
    assert slide_bottom_at_carry - staining_top >= 0.020
    assert M.GRIP_H - PAD_TIP_PAST_TCP >= 0.005  # pad tips 7 mm above the slab top


def test_no_gripping_waypoint_commands_full_close_and_release_clears_slide_before_full_open():
    max_grip_finger = -(M.SLIDE_L / 2 - 0.010)
    plan = TASK.demo()
    for side in ("left", "right"):
        for w in plan[side]:
            assert w.finger <= max_grip_finger + 1e-9, (side, w.label, w.finger, max_grip_finger)
    gripping = [w for w in plan["left"] if w.label.startswith(("close", "lift", "carry", "lower"))]
    assert len(gripping) == 4 * len(M.SLOT_X)
    assert all(w.finger == pytest.approx(M.GRIP_FINGER) for w in gripping)
    releasing = [w for w in plan["left"] if w.label.startswith(("open on", "rise from"))]
    assert len(releasing) == 2 * len(M.SLOT_X)
    assert all(w.finger == pytest.approx(M.RELEASE_FINGER) for w in releasing)
    assert (M.RISE_Z - PAD_TIP_PAST_TCP) - (M.SLAB_TOP + M.SLIDE_H) >= 0.030 - 1e-9
    assert not hasattr(M, "PLACE_DX") and not hasattr(M, "PLACE_DY")


def test_staining_rack_ribs_have_chamfered_compound_mouths_and_4mm_groove_bodies():
    """Each staining rack rib has a compound with 4.0 mm groove walls in the lower 8 mm and chamfered mouths opening
    to 8.0 mm at the 12 mm top without protruding outside the rib's bounding box."""
    assert set(M.COMPOUNDS) == {b.name for b in M.STAINING_RACK[1:]}
    mouth_w = M.GROOVE_W + 2 * M.CHAMFER_DX
    assert 0.007 <= mouth_w <= 0.008 + 1e-9
    for i, b in enumerate(M.STAINING_RACK[1:]):
        comp = M.COMPOUNDS[b.name]
        want_pieces = 2 + int(i > 0) + int(i < 5)
        assert len(comp.pieces) == want_pieces
        # Piece 0 is the full-width lower 8 mm body; piece 1 is the inset 12 mm upper core.
        assert comp.pieces[0].size == pytest.approx((b.size[0], M.SLAB_W, M.RACK_BLOCK_H - M.CHAMFER_DZ))
        inset = (M.CHAMFER_DX if i > 0 else 0.0) + (M.CHAMFER_DX if i < 5 else 0.0)
        assert comp.pieces[1].size == pytest.approx((b.size[0] - inset, M.SLAB_W, M.RACK_BLOCK_H))
        # Every tilted chamfer strip stays strictly inside the rib's outer X-Z bounding box.
        for p in comp.pieces[2:]:
            r = quat_xyzw_to_matrix(np.asarray(p.rot_xyzw, float))
            half = np.asarray(p.size, float) / 2
            for sx in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    c = np.asarray(p.offset, float) + r @ np.array([sx * half[0], 0.0, sz * half[2]])
                    assert -b.size[0] / 2 - 1e-7 <= c[0] <= b.size[0] / 2 + 1e-7
                    assert -b.size[2] / 2 - 1e-7 <= c[2] <= b.size[2] / 2 + 1e-7


# -- 3. scoring: the goal and every partial-credit branch ------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_episode():
    """Every test starts a new episode, as reset() does on the GPU."""
    M.new_episode()
    yield
    M.new_episode()


def test_the_goal_state_succeeds_and_scores_one():
    g = goal()
    assert TASK.success(g) and TASK.score(g) == 1.0 and TASK.distance(g) == pytest.approx(0.0)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_each_partial_count_scores_0_2_per_slide(k):
    s = start()
    for i in range(k):
        n = NAMES[i]
        s[n] = slide(M.SLOT_X[i], M.RACK_Y, M.SLAB_TOP)
    assert TASK.score(s) == pytest.approx(round(0.2 * k, 10))
    assert not TASK.success(s)
    assert TASK.distance(s) == pytest.approx((5 - k) * abs(M.DRYING_Y - M.RACK_Y))


def test_moving_slides_do_not_succeed():
    g = goal(slide_3=slide(M.SLOT_X[2], M.RACK_Y, M.SLAB_TOP, vel=(0.0, 0.03, 0.0)))
    assert TASK.score(g) == pytest.approx(0.8) and not TASK.success(g)  # a slide still moving is not counted


# -- 4. construct validity: shortcuts score 0 -------------------------------------------------------------------------


def test_all_slides_left_in_the_drying_rack_score_zero():
    s = start()
    assert TASK.score(s) == 0.0 and not TASK.success(s)
    assert TASK.distance(s) == pytest.approx(5 * abs(M.DRYING_Y - M.RACK_Y))


def test_a_slide_leaning_across_two_slots_scores_zero_for_it():
    mid_x = (M.SLOT_X[0] + M.SLOT_X[1]) / 2
    s = start()
    s["slide_1"] = slide(mid_x, M.RACK_Y, M.SLAB_TOP, tilt_deg=35.0)
    assert TASK.score(s) == 0.0 and not TASK.success(s)
    g = goal(slide_1=slide(mid_x, M.RACK_Y, M.SLAB_TOP, tilt_deg=5.0))
    assert TASK.score(g) == 0.8 and not TASK.success(g)


def test_two_slides_in_one_slot_both_score_zero():
    g = goal(
        slide_1=slide(M.SLOT_X[0] - 0.0006, M.RACK_Y, M.SLAB_TOP),
        slide_2=slide(M.SLOT_X[0] + 0.0006, M.RACK_Y, M.SLAB_TOP),
    )
    assert M.slide_slot(g["slide_1"]) == 0 and M.slide_slot(g["slide_2"]) == 0
    assert TASK.score(g) == 0.6 and not TASK.success(g)
    two_only = start()
    two_only["slide_1"] = slide(M.SLOT_X[0] - 0.0006, M.RACK_Y, M.SLAB_TOP)
    two_only["slide_2"] = slide(M.SLOT_X[0] + 0.0006, M.RACK_Y, M.SLAB_TOP)
    assert TASK.score(two_only) == 0.0


def test_a_slide_that_lay_flat_and_was_stood_up_in_a_slot_scores_zero_until_new_episode():
    s = goal(slide_1=slide(M.SLOT_X[0], M.RACK_Y, M.SLAB_TOP, tilt_deg=90.0))
    assert TASK.score(s) == 0.8 and "slide_1 fell" in M.violations(s)
    restored = goal()
    assert TASK.score(restored) == 0.8 and not TASK.success(restored)
    M.new_episode()
    assert TASK.score(restored) == 1.0 and TASK.success(restored)


def test_a_slide_off_the_table_and_put_back_scores_zero_for_it():
    fallen = goal(slide_4=slide(0.30, -0.70, 0.10))
    assert TASK.score(fallen) == 0.8 and "slide_4 fell" in M.violations(fallen)
    restored = goal()
    assert TASK.score(restored) == 0.8 and not TASK.success(restored)
    M.new_episode()
    assert TASK.score(restored) == 1.0 and TASK.success(restored)



def test_a_slide_held_in_its_slot_by_a_gripper_scores_zero_for_it(monkeypatch):
    g = goal()
    held = np.asarray(g["slide_3"].pos, float) + (0.0, 0.0, 0.010)  # a TCP 1 cm above slide_3's centre
    monkeypatch.setattr(M, "tcp_points", lambda: [held, np.array(M.PARK_RIGHT)])
    # the neighbours at 30 mm pitch are 29.5 mm from that TCP, so slides 2, 3 and 4 do not count
    assert TASK.score(g) == pytest.approx(0.4) and not TASK.success(g)
    # pads pinching the overhanging end: the TCP is 36 mm from the centre but 8 mm above the slide's top edge
    pinch = np.asarray(g["slide_3"].pos, float) + (0.0, 0.030, M.SLIDE_H / 2 + 0.008)
    monkeypatch.setattr(M, "tcp_points", lambda: [pinch, np.array(M.PARK_RIGHT)])
    assert TASK.score(g) == pytest.approx(0.8) and not TASK.success(g)
    monkeypatch.setattr(M, "tcp_points", lambda: [np.array(M.PARK_LEFT), np.array(M.PARK_RIGHT)])
    assert TASK.score(g) == 1.0 and TASK.success(g)  # both arms parked as in the demo: every slide counts

# -- 5. layout: no overlaps, resting on slabs, gripper fit, camera views ---------------------------------------------


def test_no_start_overlaps_and_every_slide_rests_on_its_slab():
    boxes = [
        (o.name, np.asarray(o.pos) - np.asarray(o.size) / 2, np.asarray(o.pos) + np.asarray(o.size) / 2)
        for o in M.OBJECTS
    ]
    for i, (na, alo, ahi) in enumerate(boxes):
        for nb, blo, bhi in boxes[i + 1 :]:
            overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
            assert not np.all(overlap > 1e-9), (na, nb, overlap)
    for o, x in zip(M.SLIDES, M.SLOT_X, strict=True):
        lo = np.asarray(o.pos) - np.asarray(o.size) / 2
        assert lo[2] - M.SLAB_TOP == pytest.approx(M.REST_GAP)
        assert o.pos[0] == pytest.approx(x) and o.pos[1] == pytest.approx(M.DRYING_Y)
        assert o.size == M.SLIDE_SIZE and o.mass == pytest.approx(M.SLIDE_MASS)


def test_the_slide_length_plus_10_mm_fits_the_open_gripper():
    assert M.SLIDE_L + MARGIN <= GRIPPER_MAX


def _hit(a, b, hull):
    """Does the segment a -> b pass through a convex hull (outward planes n . x = d)?"""
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


def _visible_fraction(state, hulls, faces=("top", "back")):
    """Fraction of sample points on the state's faces with an unobstructed line to the top camera."""
    cam = np.asarray(station.top_camera().pos, float)
    lo, hi = np.asarray(state.low), np.asarray(state.high)
    g = np.linspace(0.1, 0.9, 5)
    pts = []
    for a in g:
        for b in g:
            if "top" in faces:
                pts.append([lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1]), hi[2]])
            if "back" in faces:
                pts.append([lo[0], lo[1] + a * (hi[1] - lo[1]), lo[2] + b * (hi[2] - lo[2])])
    seen = sum(not any(_hit(cam, np.asarray(p), h) for _, h in hulls) for p in pts)
    return seen / len(pts)


def _hulls(side, q, finger):
    return world_hulls(_poses(q, finger, side))


def test_the_top_camera_sees_every_slide_in_the_first_frame():
    hulls = _hulls("left", np.array(READY), FINGER_OPEN) + _hulls("right", np.array(READY), FINGER_OPEN)
    for n, s in start().items():
        assert _visible_fraction(s, hulls) == 1.0, n


def test_the_top_camera_sees_all_five_rack_slots_in_the_last_frame():
    (_, left), (_, right) = played("left"), played("right")
    hulls = _hulls("left", left[-1][1], left[-1][2]) + _hulls("right", right[-1][1], right[-1][2])
    for n, s in goal().items():
        assert _visible_fraction(s, hulls) == 1.0, n


# -- task.json --------------------------------------------------------------------------------------------------------


def test_task_json_has_every_field_and_names_no_one():
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
        "scores",
    ):
        assert meta.get(key), key
    assert meta["title"] == "Load five glass slides into a staining rack"
    assert meta["category"] == "Impact on jobs"
    assert meta["instruction"] == "Move the five slides from the drying rack into the staining rack, one slide per slot."
    assert len(meta["fidelity"]) <= 500, len(meta["fidelity"])
    assert meta["scores"] == "not yet run" or meta["scores"].startswith("Dream-1 run: "), meta["scores"]
    text = json.dumps(meta).lower() + (HERE / "README.md").read_text().lower() + TASK.docs.lower()
    for word in ("robocurve_sprint", "knowledge base", "kairos", "university", "claude", "argon"):
        assert word not in text, word
