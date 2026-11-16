"""Offline geometry, mechanism, reach and scoring checks (numpy + pytest; no Isaac)."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("YAM_ARENA"):
    sys.path.insert(0, os.path.expanduser(os.environ["YAM_ARENA"]))

from yam_arena import ik, station  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

TASK = load(Path(__file__).with_name("task.py"))
M = TASK.module
rx = M.rx


def _state(pos, quat=(0.0, 0.0, 0.0, 1.0), still=True):
    p = np.asarray(pos, float)
    return ObjectState(p, np.asarray(quat, float), np.zeros(3) if still else np.array([0.1, 0, 0]), np.zeros(3), p - 0.01, p + 0.01)


def st(name, origin, quat=(0.0, 0.0, 0.0, 1.0), still=True):
    """State of part ``name`` with its body origin at ``origin`` (yam_arena reports the geometry-bounds centre as pos)."""
    return _state(M.centroid_of(name, origin, quat), quat, still)


def scene(theta=0.0, **over):
    """All six parts at their nominal poses with the lever at ``theta`` and the loop taut."""
    cfg = M.configuration(theta)
    objs = {o.name: _state(o.pos) for o in TASK.objects}
    for n, (p, q) in cfg.items():
        objs[n] = st(n, p, q)
    objs.update(over)
    return objs


def grade(objects):
    M.new_episode()
    return M.score(objects)


def spawn_poses():
    """Spawn pose of every jointed part as the extension composes them (zero pose, own move, parent's move)."""
    joints = {"strap_end": M.STRAP, "lever": M.HINGE, "loop": M.PIN}
    zero = {o.name: (tuple(o.pos), (0.0, 0.0, 0.0, 1.0)) for o in TASK.objects}
    spawn = dict(zero)
    for n in ("strap_end", "lever", "loop"):
        j = joints[n]
        own = rx._moved_pose(zero[n], j)
        par = j.parent
        spawn[n] = rx._compose(rx._compose(spawn[par], rx._inverse(zero[par])), own)
    return spawn


# -- constants and sizes ---------------------------------------------------------------------------------------------
def test_load_constants_and_sizes():
    assert TASK.scene == "station" and TASK.score_over == "end"
    assert M.INSTRUCTION == "Secure the fire extinguisher in its vehicle bracket."
    assert M.EXT_SIZE == (0.175, 0.410, 0.167) and M.EXT_MASS == 3.155
    assert M.PLATE_SIZE == (0.210, 0.430, 0.018) and M.BRACKET_MASS == 0.340
    assert math.isclose(M.BRACKET_SIZE[0], 0.146) and math.isclose(M.BRACKET_SIZE[1], 0.277)
    assert M.BRACKET_SIZE[2] <= 0.127  # the real bracket is 127 mm tall
    assert M.OPEN_DEG == 100.0 and M.CLOSED_DEG == 10.0 and M.PARTIAL_DEG == 55.0
    assert M.EXT_SHIFT_M == 0.015 and M.BRACKET_SHIFT_M == 0.010 and M.LOOP_OFF_M == 0.012
    assert [o.name for o in TASK.objects] == ["plate", "bracket", "strap_end", "lever", "loop", "extinguisher"]
    assert all(type(v) is float for o in TASK.objects for v in o.pos)
    assert set(TASK.demo()) == {"right"}
    assert M.HINGE.stiffness == 0 and M.PIN.stiffness == 0  # no spring on either hinge: the only spring is the strap's


def test_mesh_matches_the_extinguisher_envelope():
    pts, _ = rx.read_obj(str(Path(__file__).with_name("extinguisher.obj")))
    a = np.asarray(pts)
    np.testing.assert_allclose(a.max(0) - a.min(0), M.EXT_SIZE, atol=1e-6)
    np.testing.assert_allclose(a.max(0) + a.min(0), 0.0, atol=1e-6)


# -- the over-centre catch: strap tension through the loop, no scripted latch ------------------------------------------
def test_catch_is_a_real_over_centre_point():
    th = np.arange(0.0, 100.5, 0.5)
    u = np.array([M.mech_px(t) for t in th]) - M.PX0
    k = int(np.argmax(u))
    assert abs(th[k] - M.CATCH_DEG) <= 1.0
    assert 10.0 < th[k] < 45.0  # catch lies between the "closed" band and the half-way mark
    assert np.all(np.diff(u[: k + 1]) > 0) and np.all(np.diff(u[k:]) < 0)  # strap tension closes / opens each side
    assert 0.002 <= u[k] - u[0] <= 0.005  # the strap stretches 2-5 mm at the catch
    assert M.STRAP.lower <= u.min() and u.max() <= M.STRAP.upper
    torque = lambda a, b: M.STRAP_PULL_N * (np.interp(b, th, u) - np.interp(a, th, u)) / math.radians(b - a)  # noqa: E731
    assert torque(0.0, 3.0) > 0.2  # N m holding the lever on its closed stop
    assert torque(40.0, 43.0) < 0 and torque(95.0, 98.0) < 0  # opening torque on the raised side
    push = max(abs(torque(a, a + 3.0)) for a in np.arange(35.0, 97.0, 3.0))
    assert push / M.PUSH_R < 8.0  # N at the push point to cross over from the raised side: well within a finger


def test_spawn_is_the_taut_raised_configuration():
    sp, cfg = spawn_poses(), M.configuration(M.OPEN_DEG)
    for n in ("strap_end", "lever", "loop"):
        np.testing.assert_allclose(sp[n][0], cfg[n][0], atol=1e-9)
        a, b = np.asarray(sp[n][1]), np.asarray(cfg[n][1])
        assert min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-9
    o = scene(M.OPEN_DEG)
    assert math.isclose(M.lever_angle_deg(o["lever"]), M.OPEN_DEG, abs_tol=1e-6)
    # the loop's hole centre sits on the peg at every lever angle (the loop is taut)
    for t in (0, 20, 32, 60, 100):
        o = scene(t)
        d = M.origin_of("loop", o["loop"]) + quat_xyzw_to_matrix(np.asarray(o["loop"].quat_xyzw)) @ np.asarray(M.RING_PEG_LOCAL) - M.hook_world(o)
        assert abs(d[0]) < 1e-9 and abs(d[2]) < 1e-9
        assert M.loop_on_hook(o) and M.latch_intact(o)


# -- parts do not overlap each other at any lever angle ---------------------------------------------------------------
def _contains(points, centre, rot, piece):
    c = np.asarray(centre, float) + rot @ np.asarray(piece.offset)
    pr = quat_xyzw_to_matrix(np.asarray(piece.rot_xyzw))
    local = (points - c) @ (rot @ pr)
    return np.all(np.abs(local) <= np.asarray(piece.size) / 2 + 1e-9, axis=1)


def _part(name, objects):
    comp = {"bracket": M._bracket_compound(), "strap_end": M._carriage_compound(), "lever": M._lever_compound(),
            "loop": M._ring_compound(), "plate": M._plate_compound()}[name]
    o = objects[name]
    pos = M.origin_of(name, o) if name in M.PART_C else np.asarray(o.pos, float)
    return pos, quat_xyzw_to_matrix(np.asarray(o.quat_xyzw)), tuple(p for p in comp.pieces if isinstance(p, rx.Piece))


def _in_peg(points, objects, shrink=1e-6):
    """Points inside the bracket's round hook peg (3 mm radius, 15 mm long along y, centred on the hook)."""
    d = np.asarray(points) - M.hook_world(objects)
    return (np.hypot(d[:, 0], d[:, 2]) < 0.003 - shrink) & (np.abs(d[:, 1]) < 0.0075)


def _volume_points(objects, name, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    pos, rot, pieces = _part(name, objects)
    out = []
    for p in pieces:
        pr = quat_xyzw_to_matrix(np.asarray(p.rot_xyzw))
        loc = (rng.random((n, 3)) - 0.5) * np.asarray(p.size)
        out.append(pos + (rot @ (pr @ loc.T + np.asarray(p.offset)[:, None])).T)
    return np.vstack(out)


def _ext_points(n=60000, seed=1):
    """Random points inside the extinguisher's solids (body, shoulder cone, neck, head block), world frame."""
    rng = np.random.default_rng(seed)
    s = rng.random(n) * M.EXT_SIZE[1]
    ang, rad = rng.random(n) * 2 * math.pi, np.sqrt(rng.random(n))
    r = np.where(s < 0.2, M.R_BODY, np.where(s < 0.23, M.R_BODY - (s - 0.2) / 0.03 * (M.R_BODY - M.R_NECK), M.R_NECK)) * rad
    x, z = M.XA + r * np.cos(ang), M.ZA + r * np.sin(ang)
    head = s >= 0.29
    x[head] = M.XA + (rng.random(head.sum()) - 0.5) * 0.175
    z[head] = M.ZA - 0.03 + rng.random(head.sum()) * 0.147  # head block: 30 mm below the axis to 117 above
    return np.stack([x, M.Y0 + s, M.TOP + z + 2 * M.LIFT], 1)


@pytest.mark.parametrize("theta", [0.0, 20.0, 32.0, 50.0, 75.0, 100.0])
def test_no_part_overlaps_another_it_is_not_jointed_to(theta):
    o = scene(theta)
    ext = _ext_points()
    for name in ("bracket", "strap_end", "lever", "loop"):
        pos, rot, pieces = _part(name, o)
        for p in pieces:
            assert not _contains(ext, pos, rot, p).any(), f"{name} cuts into the extinguisher at {theta} deg"
    pairs = [("lever", "bracket"), ("loop", "strap_end"), ("loop", "bracket"), ("lever", "plate"), ("strap_end", "plate")]
    for a, b in pairs:
        pts = _volume_points(o, a)
        pos, rot, pieces = _part(b, o)
        for p in pieces:
            assert not _contains(pts, pos, rot, p).any(), f"{a} overlaps {b} at {theta} deg"
        if b == "bracket":
            assert not _in_peg(pts, o).any(), f"{a} cuts the hook peg at {theta} deg"
    # the rigid strap ends clear the fixed half over the whole stroke (8 mm or more)
    pts = _volume_points(o, "strap_end")
    pos, rot, pieces = _part("bracket", o)
    arm = pieces[5]
    grown = rx.Piece(size=tuple(np.asarray(arm.size) + 0.008), offset=arm.offset, rot_xyzw=arm.rot_xyzw)
    assert not _contains(pts, pos, rot, grown).any()


def test_extinguisher_rests_on_the_bracket_and_the_hook_peg_passes_through_the_loop_hole():
    ext, br = TASK.objects[5], TASK.objects[1]
    floor_top = br.pos[2] - M.BRACKET_SIZE[2] / 2 + (M.PLATE_T + M.FLOOR_T - M.PLATE_T)
    body_bottom = ext.pos[2] - M.EXT_SIZE[2] / 2
    assert 0.0 < body_bottom - floor_top <= 0.0015
    o = scene(M.OPEN_DEG)
    ang = np.radians(np.arange(0.0, 360.0, 30.0))
    peg = np.array([[0.0029 * math.cos(a_), dy, 0.0029 * math.sin(a_)] for a_ in ang for dy in (-0.007, 0.0, 0.007)]) + M.hook_world(o)
    pos, rot, pieces = _part("loop", o)
    assert not any(_contains(peg, pos, rot, p).any() for p in pieces)  # the peg's surface sits in the hole, not in the ring bars
    assert M.LOOP_OFF_M > 0.008  # the 8 mm nominal loop/hook clearance still counts as "on the hook"


# -- reach ---------------------------------------------------------------------------------------------------------------
def test_reach_at_20_hz():
    plan = ArmPlan(TASK.demo()["right"], station.BASES["right"])
    qs = []
    for k in range(round(plan.duration * 20)):
        q, _ = plan.action7((k + 1) / 20)
        qs.append(q[:6])
    q = np.asarray(qs)
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1
    assert np.all(q >= ik.LOWER) and np.all(q <= ik.UPPER)
    assert plan.duration * 20 + 100 <= 600  # inside the run limit with the usual margin


def test_demo_sweeps_the_lever_face_and_ends_clear():
    wp = TASK.demo()["right"]
    c100, c0 = M._contact(100.0, 0.0), M._contact(0.0, 0.0)
    assert math.isclose(wp[1].pos[0], c100[0]) and wp[1].pos[2] > M.TOP + 0.15
    assert c0[2] - M.TOP - M.LIFT - (M.ZA + M.PZ + 0.005) - M.PAD_TIP == pytest.approx(0.0, abs=1e-9)  # pad tips on the lever top
    park = np.asarray(wp[-1].pos)
    o = scene(0.0)
    assert M.hands_clear(o, [park])
    assert not M.hands_clear(o, [np.asarray(c0)])
    assert all(w.finger == 0.0 for w in wp)  # closed pads only: nothing is gripped


# -- scoring ---------------------------------------------------------------------------------------------------------------
def test_scoring_tiers():
    assert grade(scene(0.0)) == 1.0 and M.success(scene(0.0))
    assert grade(scene(9.0)) == 1.0
    assert grade(scene(11.0)) == 0.5  # past 45 deg of travel but not yet within 10 deg of the stop
    assert grade(scene(M.OPEN_DEG)) == 0.0  # untouched
    assert grade(scene(54.0)) == 0.5
    assert grade(scene(56.0)) == 0.0


def test_latched_partial_and_no_credit_for_a_lever_that_popped_back(monkeypatch):
    class Episode:
        _held = None
        _base = object()
        _t = 0.05

    ep = Episode()
    monkeypatch.setattr(M, "_embodiment", lambda: ep)
    monkeypatch.setattr(M, "tcp_points", lambda: [])
    M.new_episode()
    assert M.score(scene(100.0)) == 0.0
    ep._t = 3.0
    assert M.score(scene(50.0)) == 0.5
    ep._t = 3.5
    assert M.score(scene(100.0)) == 0.5  # it sprang back open but had passed 45 deg: partial stays, never 1.0
    ep._t = 9.0
    assert not M.success(scene(100.0))


def test_closed_lever_must_stay_down_for_one_second_after_the_hands_leave(monkeypatch):
    class Episode:
        _held = None
        _base = object()
        _t = 0.05

    ep = Episode()
    monkeypatch.setattr(M, "_embodiment", lambda: ep)
    monkeypatch.setattr(M, "tcp_points", lambda: [])
    M.new_episode()
    assert M.score(scene(100.0)) == 0.0
    M.new_episode()
    ep._t = 0.05
    M.score(scene(100.0))
    ep._t = 2.0
    assert M.score(scene(0.0)) == 0.5  # first closed step: timer starts
    ep._t = 2.6
    assert M.score(scene(30.0)) == 0.5  # lever drifts up past 10 deg: timer resets
    ep._t = 3.0
    assert M.score(scene(0.0)) == 0.5
    ep._t = 3.9
    assert M.score(scene(0.0)) == 0.5
    ep._t = 4.1
    assert M.score(scene(0.0)) == 1.0


def test_lever_is_judged_still_by_its_angle_over_the_hold(monkeypatch):
    class Episode:
        _held = None
        _base = object()
        _t = 0.05

    ep = Episode()
    monkeypatch.setattr(M, "_embodiment", lambda: ep)
    monkeypatch.setattr(M, "tcp_points", lambda: [])
    M.new_episode()
    M.score(scene(100.0))
    jitter = lambda th: scene(th, lever=st("lever", *M.configuration(th)["lever"], still=False))  # noqa: E731
    for i, t in enumerate(np.arange(2.0, 3.25, 0.05)):  # on its stop, velocity jitter, angle steady: full credit
        ep._t = float(t)
        val = M.score(jitter(0.3 if i % 2 else 0.4))
    assert val == 1.0
    M.new_episode()
    ep._t = 0.05
    M.score(scene(100.0))
    for i, t in enumerate(np.arange(2.0, 3.25, 0.05)):  # rocking 0.6 deg within the 10 deg band: not at rest
        ep._t = float(t)
        val = M.score(scene(0.2 if i % 2 else 0.8))
    assert val == 0.5


def test_hands_still_on_the_lever_cannot_score_full(monkeypatch):
    c = np.asarray(M._contact(0.0, 0.0))
    monkeypatch.setattr(M, "tcp_points", lambda: [c])
    assert grade(scene(0.0)) == 0.5  # held down by the hand: only the partial
    assert not M.success(scene(0.0))
    monkeypatch.setattr(M, "tcp_points", lambda: [np.asarray(scene(0.0)["extinguisher"].pos) + np.array([0, 0, 0.1])])
    assert grade(scene(0.0)) == 0.5  # a hand resting on the extinguisher is not released either


def test_faults_score_zero():
    assert grade(scene(0.0)) == 1.0
    # loop slid off the hook
    cfg = M.configuration(0.0)["loop"]
    off = scene(0.0, loop=st("loop", np.asarray(cfg[0]) + np.array([0.0, 0.0, 0.03]), cfg[1]))
    assert not M.loop_on_hook(off) and grade(off) == 0.0
    off = scene(0.0, loop=st("loop", np.asarray(cfg[0]) + np.array([0.0, 0.02, 0.0]), cfg[1]))
    assert grade(off) == 0.0  # slid along the peg and off
    # extinguisher lifted, shifted or tipped out of the cradle
    e = np.asarray(TASK.objects[5].pos)
    assert grade(scene(0.0, extinguisher=_state(e + np.array([0, 0, 0.02])))) == 0.0
    assert grade(scene(0.0, extinguisher=_state(e + np.array([0.03, 0, 0])))) == 0.0
    assert grade(scene(0.0, extinguisher=_state(e + np.array([0, 0.02, 0])))) == 0.0
    h = math.radians(30) / 2
    assert grade(scene(0.0, extinguisher=_state(e, (math.sin(h), 0, 0, math.cos(h))))) == 0.0
    assert grade(scene(0.0, extinguisher=_state([0.5, 0.2, M.TOP + 0.05]))) == 0.0  # on the floor
    # bracket shifted > 10 mm
    b = np.asarray(TASK.objects[1].pos)
    assert grade(scene(0.0, bracket=_state(b + np.array([0.012, 0, 0])))) == 0.0
    assert grade(scene(0.0, bracket=_state(b + np.array([0.006, 0, 0])))) == 1.0
    # a joint torn apart
    c = M.configuration(0.0)["lever"]
    assert grade(scene(0.0, lever=st("lever", np.asarray(c[0]) + np.array([0.02, 0, 0]), c[1]))) == 0.0


def test_faults_stay_latched_and_a_fresh_episode_starts_clean(monkeypatch):
    class Episode:
        _held = None
        _base = object()
        _t = 0.05

    ep = Episode()
    monkeypatch.setattr(M, "_embodiment", lambda: ep)
    monkeypatch.setattr(M, "tcp_points", lambda: [])
    M.new_episode()
    assert M.score(scene(100.0)) == 0.0
    cfg = M.configuration(100.0)["loop"]
    ep._t = 1.0
    assert M.score(scene(100.0, loop=st("loop", np.asarray(cfg[0]) + np.array([0, 0, 0.05]), cfg[1]))) == 0.0
    for t in (2.0, 4.0, 6.0):  # loop back on the hook and the lever closed afterwards: still 0
        ep._t = t
        assert M.score(scene(0.0)) == 0.0
    ep._t = 0.05  # reset: clock goes back
    assert M.score(scene(100.0)) == 0.0 and not M._STATE["fault"]


def test_start_pose_checks(monkeypatch):
    class Episode:
        _held = None
        _base = object()
        _t = 0.05

    monkeypatch.setattr(M, "_embodiment", lambda: Episode())
    monkeypatch.setattr(M, "tcp_points", lambda: [])
    M.new_episode()
    assert M.score(scene(0.0)) == 0.0  # lever already closed at the first step: never starts, never scores
    assert M._STATE["fault"]
    M.new_episode()
    M.score(scene(M.OPEN_DEG))
    assert M._STATE["started"] and not M._STATE["fault"]


def test_the_start_state_scores_zero_and_distance_is_the_lever_angle():
    o = scene(M.OPEN_DEG)
    assert grade(o) == 0.0 and not M.success(o)
    assert math.isclose(M.distance(o), math.radians(M.OPEN_DEG), abs_tol=1e-6)
    assert M.distance(scene(0.0)) < 1e-6


def test_moving_extinguisher_is_not_at_rest():
    assert grade(scene(0.0, extinguisher=_state(TASK.objects[5].pos, still=False))) == 0.5


def test_reported_pos_is_the_geometry_bounds_centre_not_the_origin():
    # yam_arena reports the centre of a part's local geometry bounds; the scorer must undo that offset
    assert np.abs(M.PART_C["lever"]).max() > 0.005 and np.abs(M.PART_C["strap_end"]).max() > 0.02
    o = scene(M.OPEN_DEG)
    cfg = M.configuration(M.OPEN_DEG)
    for n in ("strap_end", "lever", "loop"):
        np.testing.assert_allclose(M.origin_of(n, o[n]), cfg[n][0], atol=1e-12)
    assert not np.allclose(o["lever"].pos, cfg["lever"][0], atol=1e-3)
    assert M.loop_on_hook(o) and M.latch_intact(o) and M.extinguisher_seated(o) and M.bracket_in_place(o)
    # the same scene read as if pos were the origin would be wrong: the offsets are what makes the checks pass
    wrong = dict(o, lever=_state(cfg["lever"][0], cfg["lever"][1]))
    assert not M.latch_intact(wrong)


def test_extinguisher_centre_of_mass_is_over_the_cradle_not_the_head():
    s_com = M.EXT_COM[1] + M.EXT_C[1]  # metres from the bottom end
    assert math.isclose(s_com, 0.160) and 0.0 < s_com < M.BRACKET_S[1] - 0.05  # well inside the bracket's 277 mm
    assert math.isclose(M.EXT_COM[2], -M.EXT_DZ) and M.EXT_COM[0] == 0.0  # on the axis
    # a uniform-density head block alone would put the centre of mass at the bracket's end (what tipped the first runs)
    vol_head, vol_rest = 0.175 * 0.120 * 0.147, math.pi * 0.05 ** 2 * 0.2 + math.pi * 0.035 ** 2 * 0.06 + 0.0002
    s_uniform = (vol_head * 0.35 + vol_rest * 0.11) / (vol_head + vol_rest)
    assert s_uniform > 0.24
    assert rx.CenterOfMass(M.EXT_COM) in M.rx.JOINTS["extinguisher"]
