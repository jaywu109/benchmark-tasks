"""Offline checks for pollinate-cucumber-flowers (NumPy and pytest only; no Isaac Sim).

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
"""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("YAM_ARENA"):
    sys.path.insert(0, os.path.expanduser(os.environ["YAM_ARENA"]))

from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
HZ = 20.0


def brush_at(tip):
    """The brush lying along +x with its tip at ``tip``."""
    c = np.asarray(tip, float) - np.array([M.TIP, 0.0, 0.0])
    half = np.array([M.BRUSH_L / 2, 0.004, 0.004])
    return ObjectState(c, np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3), c - half, c + half)


def on(name, dx=0.0, dy=0.0, dz=0.0):
    return {"brush": brush_at(M.CENTRES[name] + np.array([dx, dy, dz]))}


def away():
    return {"brush": brush_at(M.BRUSH_C + np.array([M.TIP, 0.0, 0.0]))}


def dwell(name, steps, dx=0.0, dy=0.0):
    for _ in range(steps):
        s = M.score(on(name, dx, dy))
    return s


def stroke(name, dx=0.0, dz=0.0, half=0.003, steps=7):
    """The tip drawn across the face from ``half`` right of the centre to ``half`` left of it, then lifted away."""
    for dy in np.linspace(-half, half, steps):
        s = M.score(on(name, dx, dy, dz))
    M.score(away())
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
    assert list(TASK.free_objects) == ["brush"]
    assert TASK.docs.startswith("On the station table")
    assert set(TASK.demo()) == {"left", "right"}


def test_flowers_alternate_so_a_plain_sweep_meets_a_female_first():
    names = [n for n, *_ in sorted(M.FLOWERS, key=lambda f: f[1])]
    assert names == ["female_1", "male_1", "female_2", "male_2"]
    assert all(M.X_FRONT[True] < M.X_FRONT[False] for _ in [0])  # males on shorter stalks (UC leaflet 21242)


def test_the_brush_fits_the_open_jaws():
    assert 0.007 < 2 * abs(FINGER_OPEN)


def test_nothing_touched_scores_zero():
    assert M.score(away()) == 0.0 and not M._S["loaded"]


def test_a_female_brushed_before_any_male_does_not_count():
    assert stroke("female_1") == 0.0
    assert stroke("male_1") == 0.0 and M._S["loaded"]
    assert stroke("female_1") == 0.5


def test_load_then_both_females_scores_one():
    stroke("male_2")
    assert stroke("female_2") == 0.5
    assert stroke("female_1") == 1.0 and M.success(away())


def test_a_still_touch_is_not_a_stroke():
    dwell("male_1", 20)
    assert not M._S["loaded"]
    M.score(away())
    stroke("male_1")
    assert dwell("female_1", 20) == 0.0
    M.score(away())
    assert stroke("female_1", half=0.002) == 0.0  # 4 mm across: short of 5 mm
    assert stroke("female_1") == 0.5


def test_the_stroke_starts_over_when_the_tip_leaves_the_face():
    stroke("male_1")
    for dy in (-0.003, -0.001, 0.0):
        M.score(on("female_1", dy=dy))
    M.score(away())
    for dy in (0.001, 0.003):
        assert M.score(on("female_1", dy=dy)) == 0.0


def test_a_tip_more_than_4_mm_off_does_not_count():
    stroke("male_1")
    assert stroke("female_1", dx=0.001, dz=0.0042) == 0.0
    assert stroke("female_1", dx=0.001, half=0.0035) == 0.5
    M.new_episode()
    assert stroke("male_1", dz=0.0045) == 0.0 and not M._S["loaded"]


def test_a_tip_hovering_in_front_of_the_flower_does_not_count():
    assert stroke("male_1", dx=-0.002) == 0.0 and not M._S["loaded"]
    stroke("male_1", dx=-0.0008)
    assert M._S["loaded"]
    assert stroke("female_1", dx=-0.002) == 0.0
    assert stroke("female_1", dx=-0.0008) == 0.5


def test_a_right_to_left_sweep_earns_half():
    for n in ("female_1", "male_1", "female_2", "male_2"):
        stroke(n)
    assert M.score(away()) == 0.5


def test_a_new_episode_clears_the_pollen_and_the_credit():
    stroke("male_1")
    stroke("female_1")
    M.new_episode()
    assert M.score(away()) == 0.0 and not M._S["loaded"]


def test_distance_leads_to_a_male_then_to_the_females():
    assert M.distance(on("male_1")) == pytest.approx(0.0, abs=1e-9)
    assert M.distance(on("female_1")) > 0.05
    stroke("male_1")
    assert M.distance(on("female_1")) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_plan_is_reachable_within_the_joint_limits_and_the_run_limit(side):
    arm, joints = played(side)
    joints = np.array(joints)
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (arm.worst_ik_mm, arm.worst_ik_deg)
    assert np.all(joints >= ik.LOWER) and np.all(joints <= ik.UPPER)
    limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0])
    assert arm.duration * HZ <= limit - 100


def test_the_demo_loads_on_a_male_before_any_female():
    labels = [w.label for w in TASK.demo()["right"] if w.label.endswith("tip on")]
    assert labels[0].startswith("male") and sum(lab.startswith("female") for lab in labels) == 2


def test_task_json_and_readme_have_every_required_field():
    meta = json.loads((HERE / "task.json").read_text())
    for key in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                "fidelity", "reset_notes", "scores"):
        assert meta.get(key), key
    assert len(meta["fidelity"]) <= 500
    readme = (HERE / "README.md").read_text()
    lines = [line.strip() for line in readme.splitlines()]
    assert "## Third-party assets" in lines and "none" in lines
    assert "robocurve_ext.py" in readme and meta["instruction"] in readme
