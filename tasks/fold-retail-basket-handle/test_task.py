"""Offline geometry, reach and scoring checks for the retail basket."""
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, os.environ.get('YAM_ARENA','/home/sparklab/yam-isaac-lab-arena'))
from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

HERE = Path(__file__).resolve().parent
T = load(HERE/'task.py')
M = T.module


def state(pos, size, deg=0):
    q = np.array([0.,math.sin(math.radians(deg)/2),0.,math.cos(math.radians(deg)/2)])
    p = np.array(pos,float); h=np.array(size,float)/2
    return ObjectState(p,q,np.zeros(3),np.zeros(3),p-h,p+h)


def scene(deg=0, body_shift=(0,0,0), body_tilt=0, detached=0):
    bpos=np.array(M.BODY_CENTER)+np.array(body_shift)
    qb=np.array([0,math.sin(math.radians(body_tilt)/2),0,math.cos(math.radians(body_tilt)/2)])
    r= M.axis_angle_matrix(np.array([0.,1.,0.]),math.radians(body_tilt+deg+M.RAISED_STOP_DEG))
    pivot=bpos+M.axis_angle_matrix(np.array([0.,1.,0.]),math.radians(body_tilt))@np.array(M.HINGE_REL_BODY)
    hpos=pivot+r@np.array([0,0,.060])+np.array([detached,0,0])
    b=state(bpos,M.BODY_SIZE,body_tilt)
    h=state(hpos,M.HANDLE_SIZE,body_tilt+deg+M.RAISED_STOP_DEG)
    return {'basket':b,'handle':h}

FAR=[np.array([.0,.35,M.TOP+.35]),np.array([.0,-.35,M.TOP+.35])]

@pytest.fixture(autouse=True)
def reset(): M.new_episode()


def test_spec_and_mesh():
    j=json.loads((HERE/'task.json').read_text())
    assert T.scene=='station' and T.score_over=='end' and T.success_hold_s==1
    assert j['instruction']==M.INSTRUCTION=="Fold the shopping basket's handle flat for storage."
    assert M.BODY_SIZE==(.300,.430,.225) and M.BODY_MASS+M.HANDLE_MASS==pytest.approx(.620)
    assert (M.PARTIAL_TRAVEL_DEG,M.FULL_STOP_TOL_DEG,M.TIP_FAULT_DEG,M.RELEASE_CLEAR_M)==(60,5,30,.030)
    assert M.HINGE.stiffness==0 and M.HINGE.friction==.015 and M.HINGE.lower==-5 and M.HINGE.upper==90 and M.HINGE.initial==-5
    assert len(M.rx.read_obj_parts(str(HERE/'basket.obj')))==5
    assert all(all(isinstance(v,float) for v in o.pos) for o in M.OBJECTS)
    assert len(j['title'])<=120 and len(j['context'])<=1000 and isinstance(j['scores'],str) and 'media' not in j


def test_reach():
    assert set(T.demo())=={'left','right'}
    plans={s:ArmPlan(w,station.BASES[s]) for s,w in T.demo().items()}
    assert plans['left'].duration==plans['right'].duration
    for side,p in plans.items():
        qs=np.array([p.action7((k+1)/20)[0][:6] for k in range(round(p.duration*20))])
        margin=np.min(np.minimum(qs-ik.LOWER,ik.UPPER-qs))
        assert p.worst_ik_mm<.5 and p.worst_ik_deg<.1 and margin>.07,(side,p.worst_ik_mm,p.worst_ik_deg,margin)


def test_start_and_angles():
    for deg in (0,60,85,90):
        o=scene(deg)
        assert M.handle_angle_deg(o)==pytest.approx(deg,abs=1e-5)
        assert M.hinge_error_m(o)<1e-8
    o=scene()
    assert o['basket'].low[2]==pytest.approx(M.TOP)
    assert M.TABLE_MIN_X<o['basket'].low[0]<o['basket'].high[0]<M.TABLE_MAX_X
    assert M.TABLE_MIN_Y<o['basket'].low[1]<o['basket'].high[1]<M.TABLE_MAX_Y


def test_tiers_and_release():
    assert M.score(scene(59),FAR)==0
    assert M.score(scene(60),FAR)==.5
    o=scene(90)
    for _ in range(21): M.advance(o)
    assert M.success(o,FAR) and M.score(o,FAR)==1
    assert M.score(o,[M.bow_top(o),FAR[1]])==.5
    M.new_episode(); assert M.score(o,FAR)==.5  # one-second stillness is required


def test_faults_latch():
    for faulty in (scene(90,body_shift=(1,0,0)),scene(90,body_tilt=31),scene(90,detached=.03)):
        M.new_episode();M.advance(faulty)
        assert M._S['fault'] is not None
        for _ in range(21): M.advance(scene(90))
        assert M.score(scene(90),FAR)==0


def test_held_basket_does_not_score():
    for shift in ((0,0,.05),(0,0,.004)):
        M.new_episode(); o=scene(90,body_shift=shift)
        for _ in range(25): M.advance(o)
        assert not M.success(o,FAR) and M.score(o,FAR)==0
    # lifted 5 cm once, then set back down: latched 0
    M.new_episode(); M.advance(scene(90,body_shift=(0,0,.05)))
    for _ in range(25): M.advance(scene(90))
    assert M.score(scene(90),FAR)==0
    # on the table, handle folded, hands clear
    M.new_episode(); o=scene(90)
    for _ in range(25): M.advance(o)
    assert M.success(o,FAR) and M.score(o,FAR)==1.0
    # a gripper touching the basket rim blocks the full score
    rim=np.array([M.BODY_CENTER[0]-.1,0,M.TOP+.225])
    assert M.score(o,[rim,FAR[1]])<1.0
