import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('new_grasp_screen',Path(__file__).with_name('screen_numeric.py'))
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)


def fixture(lift=0.06):
    pose=np.zeros((20,7));pose[:,6]=1
    pose[:,2]=np.r_[np.zeros(8),np.linspace(0,lift,8),np.full(4,lift)]
    grip=np.r_[np.full(8,-0.785),np.zeros(12)]
    thresholds={'terminal_hold_saved_frames':4,'initial_gripper_held_min':-0.55,
                'terminal_gripper_open_max':-0.7,'grasp_min_lift_m_per_arm':0.05,
                'terminal_stability_max_m_per_arm':0.01}
    return pose,grip,grip.copy(),thresholds


def test_contract_is_read_from_current_episode_not_old_task():
    pose,g,ng,t=fixture()
    metrics,reasons=screen.task_metrics(pose,g,ng,t)
    assert reasons==[] and metrics['grasp_index']==8
    t['grasp_min_lift_m_per_arm']=0.1
    assert 'lift_below_this_task_contract' in screen.task_metrics(pose,g,ng,t)[1]


def test_open_terminal_and_last_next_are_not_accepted():
    pose,g,ng,t=fixture();g[-2:]=-0.785;ng[-1]=-0.785
    reasons=screen.task_metrics(pose,g,ng,t)[1]
    assert 'terminal_not_held' in reasons and 'last_next_not_held' in reasons


def test_quaternion_sign_equivalence():
    a=np.array([[0.,0.,0.,0.,0.,0.,1.]])
    b=a.copy();b[:,3:]*=-1
    assert screen.compare_numeric(a,b,(3,))
    b[0,0]=.1
    assert not screen.compare_numeric(a,b,(3,))


def test_nonfinite_and_shape_mismatch_rejected():
    assert not screen.compare_numeric([np.nan],[0.])
    assert not screen.compare_numeric([1.],[1.,2.])
