"""Offline checks for pack-eggs-small-end-down (NumPy, SciPy and pytest only; no Isaac Sim).

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
from scipy.spatial import ConvexHull

for _candidate in (os.environ.get("YAM_ARENA"), os.environ.get("YAM_ARENA_DIR"), "~/yam-isaac-lab-arena"):
    if _candidate and Path(_candidate).expanduser().is_dir():
        sys.path.insert(0, str(Path(_candidate).expanduser()))
        break

import yam_arena  # noqa: E402
from yam_arena import YAM_URDF, ik, station  # noqa: E402
from yam_arena.analysis import hull_set_distance, hulls_to_box, world_hulls  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
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
EGG_SIZE = np.asarray(M.EGG_SIZE, float)


# -- helpers ----------------------------------------------------------------------------------------------------------


def egg_state(
    x: float,
    y: float,
    low_z: float = M.CUP_FLOOR_Z + 0.004,
    tilt_from_down_deg: float = 13.0,
    vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
) -> ObjectState:
    """An ObjectState for an egg whose small-end axis (local +z) is tilted by ``tilt_from_down_deg`` from world -z."""
    # Rotate local +z to world -z (180 deg about +Y), then tilt by tilt_from_down_deg about +Y:
    r_down = axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.pi - math.radians(tilt_from_down_deg))
    corners = (
        np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * (EGG_SIZE / 2.0)
    )
    rotated = corners @ r_down.T
    half_world = rotated.max(axis=0)
    pos = np.array([x, y, low_z + half_world[2]], float)
    return ObjectState(
        pos=pos,
        quat_xyzw=ik.matrix_to_quat_xyzw(r_down),
        lin_vel=np.asarray(vel, float),
        ang_vel=np.asarray(ang_vel, float),
        low=pos - half_world,
        high=pos + half_world,
    )


def start_scene() -> dict[str, ObjectState]:
    return {
        "egg_l": egg_state(M.CRADLE_X, M.LEFT_Y, low_z=M.RAIL_TOP_Z, tilt_from_down_deg=90.0),
        "egg_r": egg_state(M.CRADLE_X, M.RIGHT_Y, low_z=M.RAIL_TOP_Z, tilt_from_down_deg=90.0),
    }


def packed_scene(**overrides) -> dict[str, ObjectState]:
    scene = {
        "egg_l": egg_state(M.CUP_X, M.LEFT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=13.0),
        "egg_r": egg_state(M.CUP_X, M.RIGHT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=13.0),
    }
    scene.update(overrides)
    return scene


def _stl(path):
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:300]:
        return np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data), float)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84 : 84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.unique(rec["v"].reshape(-1, 3).astype(float), axis=0)


@lru_cache(maxsize=1)
def collision_points():
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
    fk = forward_kinematics(YAM_URDF, joints)
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
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    d = np.abs(points - centre) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return float((outside + inside).min())


@lru_cache(maxsize=None)
def played(side):
    plan = TASK.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    out = []
    for k in range(round(arm.duration * HZ)):
        t = (k + 1) / HZ
        action, label = arm.action7(t)
        out.append((t, action[:6].copy(), FINGER_OPEN * action[6], label))
    return arm, out


# -- 1. the file loads ------------------------------------------------------------------------------------------------


def test_task_file_loads():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert TASK.free_objects == ("egg_l", "egg_r")
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))
    assert TASK.docs.startswith("On the station table")
    assert set(TASK.demo()) == {"left", "right"}
    assert "yam_arena.scene" not in sys.modules


# -- 2. both arm plans reachable at 20 Hz within joint limits and TCP <= 14 cm ----------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_both_arm_plans_are_reachable_at_20_hz(side):
    arm, steps = played(side)
    joints = np.array([s[1] for s in steps])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (side, arm.worst_ik_mm, arm.worst_ik_deg)
    assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER), side
    assert max(w.pos[2] for w in TASK.demo()[side]) - TOP <= 0.14 + 1e-9


def test_both_arms_follow_one_clock_and_end_parked_with_hold():
    plan = TASK.demo()
    left, right = plan["left"], plan["right"]
    assert [w.duration for w in left] == [w.duration for w in right]
    assert left[-1].pos == left[-2].pos == M.PARK_LEFT
    assert right[-1].pos == right[-2].pos == M.PARK_RIGHT
    assert left[-1].finger == left[-2].finger == FINGER_OPEN
    assert right[-1].finger == right[-2].finger == FINGER_OPEN
    assert left[-1].duration >= 2.0 and right[-1].duration >= 2.0
    run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
    plan_steps = round(sum(w.duration for w in left) * HZ)
    expected_limit = int(math.ceil((plan_steps + 100) / 200.0) * 200)
    assert run_limit == expected_limit


# -- 3. two-arm hull distance >= 20 mm and housing-to-carton/egg clearance >= 10 mm -----------------------------------


def test_two_arm_hull_distance_at_least_20_mm_and_housing_clearance_at_least_10_mm():
    _, left_steps = played("left")
    _, right_steps = played("right")
    assert len(left_steps) == len(right_steps)

    carton_and_fixed_egg_boxes = [
        (b.name, np.asarray(b.pos, float), np.asarray(b.size, float))
        for b in (M.CUP_L, M.CUP_R, M.CARTON_PLATE, *M.FIXED_CUPS, *M.FIXED_EGGS)
    ]

    worst_arm_arm = np.inf
    worst_housing = np.inf
    for step_idx in range(0, len(left_steps), 4):
        t, ql, fl, label_l = left_steps[step_idx]
        _, qr, fr, _ = right_steps[step_idx]
        hl = world_hulls(_poses(ql, fl, "left"))
        hr = world_hulls(_poses(qr, fr, "right"))
        d_aa = hull_set_distance(hl, hr)[0]
        assert d_aa >= 0.020, f"arms {d_aa * 1000:.1f} mm apart at t={t:.2f}s ({label_l})"
        worst_arm_arm = min(worst_arm_arm, d_aa)

        for side, h_list in (("left", hl), ("right", hr)):
            housing_hulls = [(n, h) for n, h in h_list if not n.startswith("tip_")]
            for bname, bpos, bsize in carton_and_fixed_egg_boxes:
                d_box, lname = hulls_to_box(housing_hulls, bpos, np.eye(3), bsize)
                assert d_box >= 0.010, f"{side}:{lname} within {d_box * 1000:.1f} mm of {bname} at t={t:.2f}s"
                worst_housing = min(worst_housing, d_box)

    assert worst_arm_arm >= 0.020
    assert worst_housing >= 0.010


def test_bridging_plate_does_not_touch_carried_egg_path_and_pads_clear_cradles_and_cups():
    """The bridging plate (y in [-0.17, +0.17], x in [0.31, 0.37]) stays clear of both carried eggs' paths, and the
    open pads at grip/release clear the cradle rails and empty cup blocks by >= 3 mm."""
    plate_lo = np.asarray(M.CARTON_PLATE.pos, float) - np.asarray(M.CARTON_PLATE.size, float) / 2.0
    plate_hi = np.asarray(M.CARTON_PLATE.pos, float) + np.asarray(M.CARTON_PLATE.size, float) / 2.0
    assert plate_lo[1] == pytest.approx(-0.170) and plate_hi[1] == pytest.approx(+0.170)
    assert plate_lo[0] == pytest.approx(0.310) and plate_hi[0] == pytest.approx(0.370)

    # Along both arms' carry paths (y = +0.20 and y = -0.20), the 44 mm wide egg (r = 22 mm in y) stays >= 8 mm from y = +-0.17:
    for side in ("left", "right"):
        arm, steps = played(side)
        for t, _, _, label in steps:
            if "park" in label or "hold" == label.split()[-1]:
                continue
            pos, _, _, _ = arm.target(t)
            egg_y_lo = float(pos[1]) - M.EGG_R
            egg_y_hi = float(pos[1]) + M.EGG_R
            assert egg_y_hi <= plate_lo[1] - 0.005 or egg_y_lo >= plate_hi[1] + 0.005, (side, t, pos)

    # Open pads clear cradle rails and cup blocks by >= 3 mm:
    for side, y, rails, cup_spec in (
        ("left", M.LEFT_Y, M.CRADLE_L, M.CUP_L),
        ("right", M.RIGHT_Y, M.CRADLE_R, M.CUP_R),
    ):
        base = np.asarray(station.BASES[side], float)
        q_pick, pe, re_ = ik.solve(
            np.array([M.V2_PICK_X, y, M.V2_GRIP_Z], float),
            ik.matrix_to_quat_xyzw(M.V2_PICK_GRIP.m),
            np.array(READY, float),
            base_pos=base,
            iters=400,
        )
        assert pe < 5e-4 and re_ < 1e-3
        world_pick = links_in_world(q_pick, FINGER_OPEN, side)
        pads_pick = np.vstack([world_pick["tip_left"], world_pick["tip_right"]])
        for rail in rails:
            r_lo = np.asarray(rail.pos, float) - np.asarray(rail.size, float) / 2.0
            r_hi = np.asarray(rail.pos, float) + np.asarray(rail.size, float) / 2.0
            assert box_gap(pads_pick, r_lo, r_hi) >= 0.003, (side, rail.name)

        cup_lo = np.asarray(cup_spec.pos, float) - np.asarray(cup_spec.size, float) / 2.0
        cup_hi = np.asarray(cup_spec.pos, float) + np.asarray(cup_spec.size, float) / 2.0
        q_place, pe2, re2 = ik.solve(
            np.array([M.CUP_X, y, M.V2_PLACE_Z], float),
            ik.matrix_to_quat_xyzw(M.V2_PLACE_GRIP.m),
            q_pick,
            base_pos=base,
            iters=400,
        )
        assert pe2 < 5e-4 and re2 < 1e-3
        for f_cmd in (M.FINGER_RELEASE, FINGER_OPEN):
            world_place = links_in_world(q_place, f_cmd, side)
            pads_place = np.vstack([world_place["tip_left"], world_place["tip_right"]])
            assert box_gap(pads_place, cup_lo, cup_hi) >= 0.003, (side, f_cmd)


# -- 4. scoring tests (packed, partial, blunt end down, lying, 18 vs 22 deg, held, moving, outside cup) --------------


def test_packed_state_scores_1_and_succeeds():
    s = packed_scene()
    assert M.tilt_deg(s["egg_l"]) == pytest.approx(13.0, abs=1e-4)
    assert M.tilt_deg(s["egg_r"]) == pytest.approx(13.0, abs=1e-4)
    assert TASK.score(s) == pytest.approx(1.0)
    assert TASK.success(s)
    assert TASK.distance(s) == pytest.approx(0.0)


def test_one_packed_scores_0_5_and_does_not_succeed():
    s = packed_scene(egg_l=egg_state(M.CRADLE_X, M.LEFT_Y, low_z=M.RAIL_TOP_Z, tilt_from_down_deg=90.0))
    assert TASK.score(s) == pytest.approx(0.5)
    assert not TASK.success(s)
    assert TASK.distance(s) == pytest.approx(abs(M.CUP_X - M.CRADLE_X) + 0.001 * (90.0 - 20.0))

    # Each empty cup counts at most once (two eggs in cup_r -> 0.5, not 1.0):
    same_cup = packed_scene(egg_l=egg_state(M.CUP_X + 0.001, M.RIGHT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=13.0))
    assert TASK.score(same_cup) == pytest.approx(0.5)
    assert not TASK.success(same_cup)


def test_blunt_end_down_130_deg_scores_zero():
    s = {
        "egg_l": egg_state(M.CUP_X, M.LEFT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=130.0),
        "egg_r": egg_state(M.CUP_X, M.RIGHT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=130.0),
    }
    assert M.tilt_deg(s["egg_l"]) == pytest.approx(130.0, abs=1e-4)
    assert TASK.score(s) == 0.0 and not TASK.success(s)
    assert TASK.distance(s) == pytest.approx(2 * 0.001 * (130.0 - 20.0))


def test_lying_90_deg_in_cup_scores_zero():
    s = {
        "egg_l": egg_state(M.CUP_X, M.LEFT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=90.0),
        "egg_r": egg_state(M.CUP_X, M.RIGHT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=90.0),
    }
    assert M.tilt_deg(s["egg_l"]) == pytest.approx(90.0, abs=1e-4)
    assert TASK.score(s) == 0.0 and not TASK.success(s)


def test_tilt_threshold_18_deg_packed_vs_22_deg_zero():
    s_18 = {
        "egg_l": egg_state(M.CUP_X, M.LEFT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=18.0),
        "egg_r": egg_state(M.CUP_X, M.RIGHT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=18.0),
    }
    assert TASK.score(s_18) == 1.0 and TASK.success(s_18)

    s_22 = {
        "egg_l": egg_state(M.CUP_X, M.LEFT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=22.0),
        "egg_r": egg_state(M.CUP_X, M.RIGHT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=22.0),
    }
    assert TASK.score(s_22) == 0.0 and not TASK.success(s_22)
    assert TASK.distance(s_22) == pytest.approx(2 * 0.001 * 2.0)


def test_held_egg_with_tcp_2_cm_away_scores_zero(monkeypatch):
    s = packed_scene()
    near_l = np.array([M.CUP_X, M.LEFT_Y, float(s["egg_l"].high[2]) + 0.020], float)
    near_r = np.array([M.CUP_X, M.RIGHT_Y, float(s["egg_r"].high[2]) + 0.020], float)
    monkeypatch.setattr(M, "tcp_points", lambda: [near_l, near_r])
    assert TASK.score(s) == 0.0 and not TASK.success(s)

    monkeypatch.setattr(M, "tcp_points", lambda: [near_l, np.asarray(M.PARK_RIGHT, float)])
    assert TASK.score(s) == pytest.approx(0.5) and not TASK.success(s)

    monkeypatch.setattr(M, "tcp_points", lambda: [np.asarray(M.PARK_LEFT, float), np.asarray(M.PARK_RIGHT, float)])
    assert TASK.score(s) == 1.0 and TASK.success(s)


def test_moving_egg_scores_zero():
    s = {
        "egg_l": egg_state(M.CUP_X, M.LEFT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=13.0, vel=(0.03, 0.0, 0.0)),
        "egg_r": egg_state(M.CUP_X, M.RIGHT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=13.0, ang_vel=(0.0, 1.0, 0.0)),
    }
    assert TASK.score(s) == 0.0 and not TASK.success(s)


def test_outside_cup_xy_6_mm_scores_zero():
    s = {
        "egg_l": egg_state(M.CUP_X + 0.006, M.LEFT_Y, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=13.0),
        "egg_r": egg_state(M.CUP_X, M.RIGHT_Y - 0.006, low_z=M.CUP_FLOOR_Z + 0.004, tilt_from_down_deg=13.0),
    }
    assert TASK.score(s) == 0.0 and not TASK.success(s)
    assert TASK.distance(s) == pytest.approx(0.012)


# -- 5. meshes, start overlaps, gripper fit, and top-camera visibility -----------------------------------------------


def test_egg_and_carton_meshes_and_gripper_fit():
    assert M.EGG_D + MARGIN <= GRIPPER_MAX

    pts, faces = M.rx.read_obj(M.EGG_MESH_PATH)
    arr = np.asarray(pts, float)
    assert len(faces) > 0
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    assert (hi - lo) == pytest.approx((0.044, 0.044, 0.057), abs=1e-5)
    r_xy = np.hypot(arr[:, 0], arr[:, 1])
    assert r_xy[arr[:, 2] > 0.015].max() < r_xy[arr[:, 2] < -0.015].max() - 0.002
    hull = ConvexHull(arr)
    signed = arr @ hull.equations[:, :3].T + hull.equations[:, 3]
    assert float(np.abs(signed.max(axis=1)).max()) < 1e-5

    c_pts, c_faces = M.rx.read_obj(M.CARTON_MESH_PATH)
    c_arr = np.asarray(c_pts, float)
    assert len(c_faces) > 0
    c_lo, c_hi = c_arr.min(axis=0), c_arr.max(axis=0)
    assert (c_hi - c_lo) == pytest.approx((0.060, 0.060, 0.035), abs=1e-5)


def test_no_start_overlaps_for_eggs_on_cradles_and_fixed_eggs_in_cups():
    egg_pts_local = np.asarray(M.rx.read_obj(M.EGG_MESH_PATH)[0], float)

    # Expand compound pieces for carton_plate so each collider box is checked directly:
    colliders: list[tuple[str, np.ndarray, np.ndarray]] = []
    for o in M.OBJECTS:
        entry = M.rx.JOINTS.get(o.name)
        if isinstance(entry, M.rx.Compound):
            for idx, p in enumerate(entry.pieces):
                c = np.asarray(o.pos, float) + np.asarray(p.offset, float)
                half = np.asarray(p.size, float) / 2.0
                colliders.append((f"{o.name}_p{idx}", c - half, c + half))
        elif o.name.startswith(("egg_", "fixed_egg_")):
            rot = quat_xyzw_to_matrix(np.asarray(entry.rot_xyzw, float))
            pts_w = np.asarray(o.pos, float) + egg_pts_local @ rot.T
            colliders.append((o.name, pts_w.min(axis=0), pts_w.max(axis=0)))
        else:
            lo = np.asarray(o.pos, float) - np.asarray(o.size, float) / 2.0
            hi = np.asarray(o.pos, float) + np.asarray(o.size, float) / 2.0
            colliders.append((o.name, lo, hi))

    for name, lo, _ in colliders:
        assert lo[2] >= TOP - 1e-9, (name, lo[2])

    fixed_egg_cup_pairs = {(f"fixed_egg_{i + 1}", f"cup_{i + 1}") for i in range(len(M.FIXED_CUP_YS))}
    fixed_egg_cup_pairs |= {(b, a) for a, b in fixed_egg_cup_pairs}

    for i, (n1, lo1, hi1) in enumerate(colliders):
        for n2, lo2, hi2 in colliders[i + 1 :]:
            if (n1, n2) in fixed_egg_cup_pairs:
                continue
            overlap = np.minimum(hi1, hi2) - np.maximum(lo1, lo2)
            assert not np.all(overlap > 1e-7), (n1, n2, overlap)

    # Verify each fixed egg sits inside its cup's conical cavity with >= 0.4 mm clearance from the floor and cone walls:
    rot_s = quat_xyzw_to_matrix(np.asarray(M.ROT_SMALL_END_DOWN, float))
    for egg_box, cup_y in zip(M.FIXED_EGGS, M.FIXED_CUP_YS, strict=True):
        pts_s = np.asarray(egg_box.pos, float) + egg_pts_local @ rot_s.T
        assert float(pts_s[:, 2].min()) - M.CUP_FLOOR_Z == pytest.approx(0.002, abs=1e-5)
        for x, y, z in pts_s:
            if z <= M.CUP_TOP_Z + 1e-9:
                h = z - M.CUP_FLOOR_Z
                r_cup = 0.012 + (0.023 - 0.012) * (h / M.CUP_DEPTH)
                r_egg = math.hypot(x - M.CUP_X, y - cup_y)
                assert r_cup - r_egg >= 0.0004, (egg_box.name, h, r_egg, r_cup)


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


def _visible_fraction(lo, hi, hulls):
    cam = np.asarray(station.top_camera().pos, float)
    g = np.linspace(0.1, 0.9, 5)
    pts = [[lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1]), hi[2]] for a in g for b in g]
    seen = sum(not any(_hit(cam, np.asarray(p), h) for _, h in hulls) for p in pts)
    return seen / len(pts)


def test_first_and_last_top_camera_frames_see_both_eggs_and_both_empty_cups():
    hulls_ready = world_hulls(_poses(READY, FINGER_OPEN, "left")) + world_hulls(_poses(READY, FINGER_OPEN, "right"))
    s0 = start_scene()
    for name in ("egg_l", "egg_r"):
        assert _visible_fraction(s0[name].low, s0[name].high, hulls_ready) == 1.0, name
    for cup in (M.CUP_L, M.CUP_R):
        lo = np.asarray(cup.pos, float) - np.asarray(cup.size, float) / 2.0
        hi = np.asarray(cup.pos, float) + np.asarray(cup.size, float) / 2.0
        assert _visible_fraction(lo, hi, hulls_ready) == 1.0, cup.name

    (_, left), (_, right) = played("left"), played("right")
    hulls_end = world_hulls(_poses(left[-1][1], left[-1][2], "left")) + world_hulls(
        _poses(right[-1][1], right[-1][2], "right")
    )
    s_end = packed_scene()
    for name in ("egg_l", "egg_r"):
        assert _visible_fraction(s_end[name].low, s_end[name].high, hulls_end) == 1.0, name


# -- 6. task.json and README fields -----------------------------------------------------------------------------------


def test_task_json_and_readme_have_every_field_and_name_no_one():
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
    assert meta["title"] == "Pack the last two eggs small end down"
    assert meta["category"] == "Societal relatability"
    assert meta["instruction"] == "Put the two loose eggs into the empty carton cups, small end down."
    assert len(meta["fidelity"]) <= 500, len(meta["fidelity"])
    assert meta["scores"] == "not yet run" or meta["scores"].startswith("Dream-1 run: "), meta["scores"]
    readme = (HERE / "README.md").read_text()
    for section in ("## Run and score a trial", "## Scoring", "## Design numbers", "## Verification", "## Third-party assets"):
        assert section in readme, section
    text = json.dumps(meta).lower() + readme.lower() + TASK.docs.lower()
    for word in ("robocurve_sprint", "knowledge base", "kairos", "university", "claude", "argon"):
        assert word not in text, word
