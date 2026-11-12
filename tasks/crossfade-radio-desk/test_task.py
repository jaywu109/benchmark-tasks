"""Offline checks for crossfade-radio-desk (NumPy and pytest only; no Isaac Sim).

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("YAM_ARENA"):
    sys.path.insert(0, os.path.expanduser(os.environ["YAM_ARENA"]))

from yam_arena.task_file import ObjectState, load  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
HZ = 20.0


def cap(k, travel, vel=0.0):
    c = np.array([M.X0 + travel, M.CH_Y[k], M.CAP_Z])
    h = np.array(M.CAP_SIZE) / 2
    return ObjectState(c, np.array([0.0, 0.0, 0.0, 1.0]), np.array([vel, 0.0, 0.0]), np.zeros(3), c - h, c + h)


def objs(on_air, cued, vel=0.0):
    o = {f"fader{k + 1}": cap(k, 0.0) for k in range(M.N_CH)}
    o["fader1"], o[f"fader{M.N_CH}"] = cap(M.ON_AIR, on_air, vel), cap(M.CUED, cued, vel)
    return o


def run(pairs):
    s = None
    for a, b in pairs:
        s = M.score(objs(a, b))
    return held(a, b)


def held(a, b, vel=0.0):
    """Leave the faders there for 1 s; the last score."""
    return [M.score(objs(a, b, vel=vel)) for _ in range(M.HOLD_STEPS + 1)][-1]


@pytest.fixture(autouse=True)
def fresh():
    M.new_episode()


def test_the_task_file_loads():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert list(TASK.free_objects) == [f"fader{k}" for k in range(1, 9)] and M.CUED == 7
    assert TASK.docs.startswith("On the station table")
    for o in M.OBJECTS:
        assert all(type(v) is float for v in o.pos), o.name


def test_the_start_is_not_a_success_and_not_a_fault():
    assert held(M.MARK, 0.0) == 0.0 and M._S["fault"] is None


def test_a_true_crossfade_scores_one():
    f = np.linspace(0, 1, 40)
    assert run(zip(M.MARK * (1 - f), M.MARK * f)) == 1.0


def test_down_first_then_up_is_dead_air():
    pairs = [(a, 0.0) for a in np.linspace(M.MARK, 0, 20)] + [(0.0, b) for b in np.linspace(0, M.MARK, 20)]
    assert run(pairs) == 0.0 and M._S["fault"] == "dead air"


def test_up_first_then_down_is_both_up():
    pairs = [(M.MARK, b) for b in np.linspace(0, M.MARK, 20)] + [(a, M.MARK) for a in np.linspace(M.MARK, 0, 20)]
    assert run(pairs) == 0.0 and M._S["fault"] == "both up"


def test_a_fault_is_kept_for_the_episode_and_cleared_by_a_new_one():
    run([(0.01, 0.01)])
    assert held(0.0, M.MARK) == 0.0
    M.new_episode()
    assert held(0.0, M.MARK) == 1.0


def test_the_end_tolerances():
    for a, b, want in ((0.0029, M.MARK + 0.0039, 1.0), (0.0029, M.MARK - 0.0039, 1.0), (0.004, M.MARK, 0.0),
                       (0.0, M.MARK + 0.005, 0.0), (0.0, M.MARK - 0.005, 0.0)):
        M.new_episode()
        assert held(a, b) == want, (a, b)
    M.new_episode()
    assert held(0.0, M.MARK, vel=0.02) == 0.0  # still moving


def test_success_needs_one_second():
    got = [M.score(objs(0.0, M.MARK)) for _ in range(M.HOLD_STEPS + 1)]
    assert got[:-1] == [0.0] * M.HOLD_STEPS and got[-1] == 1.0


def test_the_demo_hand_targets_never_trip_a_rule():
    for wl, wr in zip(TASK.demo()["left"][4:12], TASK.demo()["right"][4:12]):
        a, b = max(0.0, wl.pos[0] - M.X0), wr.pos[0] - M.X0
        assert not (a < M.QUIET and b < M.QUIET) and not (a > M.LOUD and b > M.LOUD), (a, b)
    assert M.QUIET < M.MARK / 2 < M.LOUD  # the halfway point of a linear crossfade sits between the two lines
