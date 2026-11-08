"""Offline checks for laminate-drawer-front (NumPy and pytest only; no Isaac Sim).

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

for _candidate in (os.environ.get("YAM_ARENA"), os.environ.get("YAM_ARENA_DIR"), "~/yam-isaac-lab-arena"):
    if _candidate and Path(_candidate).expanduser().is_dir():
        sys.path.insert(0, str(Path(_candidate).expanduser()))
        break

from yam_arena import YAM_URDF, station  # noqa: E402
from yam_arena.analysis import box_signed_distance, world_hulls  # noqa: E402
from yam_arena.constants import FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.ik import matrix_to_quat_xyzw  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, forward_kinematics  # noqa: E402

HZ = 20.0
HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
T = TASK.module


@pytest.fixture(autouse=True)
def fresh():
    T.new_episode()
    yield
    T.new_episode()


@pytest.fixture(scope="module")
def played():
    arm = ArmPlan(TASK.demo()["right"], station.BASES["right"])
    steps = []
    for k in range(round(arm.duration * HZ) + 5):
        t = (k + 1) / HZ
        _, _, finger, label = arm.target(t)
        action, _ = arm.action7(t)
        steps.append((t, action[:6].copy(), finger, label))
    return arm, steps


def hulls(q, finger, side="right"):
    d = {f"joint{i + 1}": float(v) for i, v in enumerate(q)}
    d.update(joint7=finger, joint8=finger)
    poses = forward_kinematics(YAM_URDF, d)
    base = np.eye(4)
    base[:3, 3] = station.BASES[side]
    return world_hulls({k: base @ v for k, v in poses.items() if k != "base"})


def box_clearance(hs, spec):
    c, size = np.asarray(spec.pos, float), np.asarray(spec.size, float)
    corners = c + size / 2 * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    best = np.inf
    for _, h in hs:
        best = min(best, box_signed_distance(h.vertices, c, np.eye(3), size).min())
        best = min(best, (corners @ h.planes[:, :3].T - h.planes[:, 3]).max(axis=1).min())
    return float(best)


def state(pos, tilt_deg=0.0, size=T.PANEL_SIZE, vel=(0.0, 0.0, 0.0)):
    r = axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(tilt_deg))
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.asarray(size) / 2
    w = np.asarray(pos, float) + corners @ r.T
    return ObjectState(np.asarray(pos, float), np.asarray(matrix_to_quat_xyzw(r)), np.asarray(vel, float),
                       np.zeros(3), w.min(0), w.max(0))


FLAT_Z = T.BOARD_TOP + T.PANEL_SIZE[2] / 2


def scene(down=("m", "n", "f"), dx=0.0, dy=0.0):
    """Panels in ``down`` flat on the blank, the rest on their dowels; the sheet shifted by (dx, dy)."""
    out = {}
    for k, x in T.PANEL_X.items():
        out[f"panel_{k}"] = state((x + dx, dy, FLAT_Z if k in down else T.PANEL_Z))
    for k, d in T.DOWELS.items():
        out[d.name] = state(d.pos, size=d.size)
    return out


class FakeEmbodiment:
    """Just enough of the embodiment for the scorer: the episode clock, arms parked far from the sheet."""

    def __init__(self):
        self._t, self._held, self._base, self.resets = 0.0, 0, object(), 0

    def reset(self, *a, **k):
        self.resets += 1
        self._t = 0.0

    def call(self, fn, objects):
        self = self  # noqa: PLW0127 - the scorer finds the embodiment as a frame local named "self"
        return fn(objects)


# -- 1. load and layout --------------------------------------------------------------------------------------------


def test_loads():
    assert [o.name for o in T.OBJECTS] == ["blank", "panel_m", "panel_n", "panel_f", "dowel_m", "dowel_n", "dowel_f"]
    assert TASK.score_over == "end"


def test_spawn_stack_touches_without_overlap():
    for k in T.PANEL_X:
        d, p = T.DOWELS[k], T.PANELS[k]
        assert abs(d.pos[2] - d.size[2] / 2 - T.BOARD_TOP) < 1e-9  # dowel on the blank
        assert abs(p.pos[2] - p.size[2] / 2 - (d.pos[2] + T.DOWEL_R)) < 1e-9  # panel on its dowel
        assert d.pos[0] == p.pos[0]  # dowel under the panel's centre
        assert d.pos[1] + d.size[1] / 2 >= p.pos[1] + p.size[1] / 2 - 1e-9  # it runs under the whole panel
        assert d.pos[1] - d.size[1] / 2 < T.BOARD.pos[1] - T.BOARD.size[1] / 2 - 0.04  # and sticks out 4+ cm
    xs = sorted(T.PANEL_X.values())
    assert all(abs(b - a - T.PANEL_SIZE[0] - 0.001) < 1e-9 for a, b in zip(xs, xs[1:]))  # 1 mm gaps
    sheet_x = (xs[0] - T.PANEL_SIZE[0] / 2, xs[-1] + T.PANEL_SIZE[0] / 2)
    blank_x = (T.BOARD_XY[0] - T.BOARD_SIZE[0] / 2, T.BOARD_XY[0] + T.BOARD_SIZE[0] / 2)
    assert blank_x[0] - sheet_x[0] == pytest.approx(0.008) and sheet_x[1] - blank_x[1] == pytest.approx(0.008)
    assert (T.PANEL_SIZE[1] - T.BOARD_SIZE[1]) / 2 == pytest.approx(0.010)
    assert T.REG_X == pytest.approx(0.008 - 0.002) and T.REG_Y == pytest.approx(0.010 - 0.002)


def test_dowel_mesh_matches_its_box():
    from robocurve_ext import read_obj

    pts, _ = read_obj(str(HERE / "dowel.obj"))
    assert np.allclose(np.ptp(np.asarray(pts), axis=0), T.DOWELS["m"].size, atol=1e-4)


# -- 2. the demo ------------------------------------------------------------------------------------------------------


def test_reachable_at_20hz(played):
    arm, steps = played
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (arm.worst_ik_mm, arm.worst_ik_deg)
    run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0])
    assert len(steps) < run_limit


def test_pads_clear_blank_sheet_and_other_dowels(played):
    _, steps = played
    worst = {}
    for t, q, finger, label in steps:
        k = label.split()[0]
        hs = hulls(q, finger)
        held = T.DOWELS.get(k)
        pulled = set(T.ORDER[: T.ORDER.index(k)]) if k in T.ORDER else set(T.ORDER)
        specs = [T.BOARD, *T.PANELS.values()] + [d for j, d in T.DOWELS.items() if j not in pulled and d is not held]
        for spec in specs:
            c = 1000 * box_clearance(hs, spec)
            if c < worst.get(spec.name, (np.inf,))[0]:
                worst[spec.name] = (round(c, 1), t, label)
    assert all(v[0] >= 3.0 for v in worst.values()), worst


def test_hand_stays_above_the_table(played):
    _, steps = played
    low = min(min(h.vertices[:, 2].min() for _, h in hulls(q, f)) for _, q, f, _ in steps)
    assert low > T.TOP + 0.002, low - T.TOP


def test_park_is_3_cm_clear_of_every_panel():
    tcp = np.asarray(T.PARK)
    for k in T.PANEL_X:
        assert T.box_distance(scene()[f"panel_{k}"], T.PANEL_SIZE, tcp) >= 0.03


# -- 3. bonding and scoring ---------------------------------------------------------------------------------------


def test_glue_condition():
    assert T.glue_ready(state((0.3, 0.0, FLAT_Z)))
    assert not T.glue_ready(state((0.3, 0.0, FLAT_Z + 0.002)))
    tilted = state((0.3, 0.0, FLAT_Z), 5.0)
    lift = T.BOARD_TOP - float(tilted.low[2])
    assert not T.glue_ready(state((0.3, 0.0, FLAT_Z + lift), 5.0))  # a corner on the blank at 5 deg: no bond


def test_start_scores_zero():
    assert T.score(scene(down=())) == 0.0 and not T.success(scene(down=()))


def test_all_bonded_registered_released_is_full_score():
    s = scene()
    assert T.score(s) == 1.0 and T.success(s)
    assert T.distance(s) == 0.0


def test_two_bonded_is_two_thirds():
    s = scene(down=("m", "n"))
    assert T.score(s) == pytest.approx(2 / 3) and not T.success(s)
    assert T.distance(s) == pytest.approx(2 * T.DOWEL_R)


def test_a_bond_lasts_after_the_panel_is_lifted():
    T.score(scene(down=("m",)))
    assert T.score(scene(down=())) == pytest.approx(1 / 3)  # bonded panels are held by their joints in the sim


def test_off_register_scores_zero():
    for dx, dy in ((0.0065, 0.0), (0.0, -0.0085)):
        T.new_episode()
        s = scene(dx=dx, dy=dy)
        assert T.score(s) == 0.0 and not T.success(s), (dx, dy)
        assert T.distance(s) > 0.0
    T.new_episode()
    assert T.score(scene(dx=0.0055, dy=-0.0075)) == 1.0


def test_a_panel_near_a_tcp_does_not_count(monkeypatch):
    s = scene()
    near = np.asarray(s["panel_n"].pos) + [0.0, 0.0, 0.02]
    far = np.asarray(T.PARK)
    monkeypatch.setattr(T, "tcp_points", lambda: [far, near])
    assert T.score(s) == pytest.approx(2 / 3) and not T.success(s)


def test_bonds_update_once_per_step_and_reset_unbonds(monkeypatch):
    monkeypatch.setattr(T, "tcp_points", lambda: [])  # the fake has no arms
    emb = FakeEmbodiment()
    emb._t = 0.05
    up = scene(down=())
    assert emb.call(T.score, up) == 0.0
    emb._t = 0.10
    assert emb.call(T.score, scene()) == 1.0  # all bond this step
    assert emb.call(T.success, up)  # same step: no new update, the record stands
    emb.reset()  # the wrapped reset clears the record first
    assert emb.resets == 1 and T._BOND["bond"] == {}
    emb._t = 0.05
    assert emb.call(T.score, up) == 0.0
    emb._t = 0.10
    emb.call(T.score, scene(down=("m",)))
    emb._t = 0.05  # the clock went back without a wrapped reset: a new episode
    assert emb.call(T.score, up) == 0.0


# -- 4. camera and text --------------------------------------------------------------------------------------------


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


def test_the_top_camera_sees_the_sheet_and_the_dowel_ends_in_the_first_frame():
    cam = np.asarray(station.top_camera().pos, float)
    hs = hulls(np.array(READY), FINGER_OPEN, "left") + hulls(np.array(READY), FINGER_OPEN, "right")
    pts = [(x, y, T.PANEL_Z + 0.002) for x in T.PANEL_X.values() for y in (-0.12, 0.0, 0.12)]
    pts += [(d.pos[0], -0.16, d.pos[2] + T.DOWEL_R) for d in T.DOWELS.values()]
    for p in pts:
        assert not any(_hit(cam, np.asarray(p, float), h) for _, h in hs), p


def test_task_json_has_every_field_and_names_no_one():
    meta = json.loads((HERE / "task.json").read_text())
    for key in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit", "fidelity",
                "scores"):
        assert meta.get(key), key
    assert meta["category"] == "Impact on jobs"
    assert meta["instruction"] == ("Pull out the three spacer dowels so the laminate sheet bonds flat to the drawer "
                                   "front, still overhanging every edge.")
    assert len(meta["fidelity"]) <= 500, len(meta["fidelity"])
    assert meta["scores"] == "not yet run" or meta["scores"].startswith("Dream-1 run: "), meta["scores"]
    text = json.dumps(meta).lower() + (HERE / "README.md").read_text().lower() + TASK.docs.lower()
    for word in ("robocurve_sprint", "knowledge base", "kairos", "university", "claude", "argon", "spike"):
        assert word not in text, word
