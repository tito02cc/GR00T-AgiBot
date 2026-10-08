import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('task_grasp_runner', HERE/'run_inference.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
PROFILE = json.loads((HERE/'inference.json').read_text())


def row(i, z, grip=0.):
    return {'command_id': f'test-h{i}', 'status': 'COMPLETED', 'completion': {'ok': True,
        'live_pose_at_100ms': [.5, -.2, z, 0, 0, 0, 1], 'execution_finished_monotonic_ns': i*100_000_000,
        'gripper_status': {'feedback_encoding': 'native_radians', 'last_observation': {'raw_position': grip}}}}


def status(z, grip=0.):
    return {'ready': True, 'queue_depth': 0, 'live_pose': [.5, -.2, z, 0, 0, 0, 1],
            'right_gripper': {'last_observation': {'raw_position': grip}}}


def test_uses_pinned_successful_transport():
    # Other tests may preload the mutable root module into sys.modules. The
    # actual runner starts in a fresh interpreter, so test that entrypoint.
    code = f"""
import importlib.util
import json
from pathlib import Path
spec = importlib.util.spec_from_file_location('grasp_runtime_check', {str(HERE / 'run_inference.py')!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
print(json.dumps([str(Path(runner.shared.__file__).resolve()),
                  str(Path(runner.execute_action_chunk.__code__.co_filename).resolve())]))
"""
    result = subprocess.run([sys.executable, '-c', code], cwd=module.ROOT,
                            capture_output=True, text=True, check=True)
    paths = json.loads(result.stdout)
    assert all(Path(path).is_relative_to(module.RUNTIME) for path in paths)


def test_only_measured_held_lift_and_stability_finish():
    p=module.GraspProgress(PROFILE['completion'])
    p.observe([row(1,.8)])
    assert not p.status(status(.91))['feedback_sequence_complete']
    p.observe([row(i,.91) for i in range(2,7)])
    assert p.status(status(.91))['feedback_sequence_complete']
    assert not p.status(status(.91,-.785))['feedback_sequence_complete']


def test_unstable_terminal_does_not_finish():
    p=module.GraspProgress(PROFILE['completion']);p.observe([row(1,.8)])
    p.observe([row(i,.84+i*.03) for i in range(2,7)])
    assert not p.status(status(1.02))['feedback_sequence_complete']


def test_reopen_clears_anchor_and_no_latch_modifies_action():
    p=module.GraspProgress(PROFILE['completion']);p.observe([row(1,.8),row(2,.9,-.785)])
    assert p.anchor is None and not p.window
    action={'right_eef':np.tile([.5,-.2,.8,1,0,0,0,1,0],(1,16,1)),
            'right_gripper':np.linspace(-.785,0,16).reshape(1,16,1)}
    result=module.policy_targets(action)
    np.testing.assert_allclose(result[:,7],action['right_gripper'][0,:,0],atol=1e-7)
    np.testing.assert_allclose(result[:,:3],action['right_eef'][0,:,:3],atol=1e-7)


def test_bad_receipt_and_stale_feedback_cannot_finish():
    p=module.GraspProgress(PROFILE['completion']);p.observe([row(1,.8)])
    with pytest.raises(RuntimeError):p.observe([row(1,.8)])
    p.observe([row(i,.91) for i in range(2,7)])
    s=status(.91);s['live_pose_stale']=True
    assert not p.status(s)['feedback_sequence_complete']


def test_no_fixed_step_cap_and_no_rtc():
    assert PROFILE['max_cycles']==PROFILE['session_limit_s']==0
    assert PROFILE['horizon']==16 and PROFILE['control_hz']==50
    assert not PROFILE['rtc_enabled'] and PROFILE['trajectory_filter_enabled']


def test_closed_start_is_rejected_without_opening():
    s=status(.8);s['right_gripper']['feedback_encoding']='native_radians'
    with pytest.raises(RuntimeError,match='open-gripper'):module.pre_activation(s,PROFILE)


def test_out_of_range_gripper_is_not_hidden_by_decoder():
    a={'right_eef':np.tile([.5,-.2,.8,1,0,0,0,1,0],(1,16,1)),
       'right_gripper':np.ones((1,16,1))}
    with pytest.raises(RuntimeError,match='range'):module.policy_targets(a)


def test_hover_smoothing_preserves_model_chunk_endpoints_and_gripper():
    original=np.tile([.725,-.185,.895,0.,0.,0.,1.,-.785],(16,1)).astype(float)
    original[:,0]+=np.linspace(0,.008,16)+np.array([0,.001,-.001,.001,-.001,.001,-.001,.001,
        -.001,.001,-.001,.001,-.001,.001,-.001,0])
    current=np.array([.725,-.185,.895,0,0,0,1],float)
    smoothed,detail=module.smooth_open_gripper_hover(original,current,-.785,
        PROFILE['open_gripper_hover_smoothing'])
    assert detail['applied']
    assert detail['max_xyz_adjustment_m']<=.0015
    np.testing.assert_array_equal(smoothed[[0,-1]],original[[0,-1]])
    np.testing.assert_array_equal(smoothed[:,7],original[:,7])
    assert np.linalg.norm(np.diff(smoothed[:,:3],axis=0),axis=1).sum()<np.linalg.norm(
        np.diff(original[:,:3],axis=0),axis=1).sum()
    closing=original.copy();closing[-1,7]=-.5
    untouched,why=module.smooth_open_gripper_hover(closing,current,-.785,
        PROFILE['open_gripper_hover_smoothing'])
    assert not why['applied']
    np.testing.assert_array_equal(untouched,closing)


def test_hover_smoothing_does_not_change_fast_model_motion():
    original=np.tile([.5,-.2,.8,0,0,0,1,-.785],(16,1)).astype(float)
    original[:,0]+=np.linspace(0,.2,16)
    result,detail=module.smooth_open_gripper_hover(original,original[0,:7],-.785,
        PROFILE['open_gripper_hover_smoothing'])
    assert not detail['applied']
    assert detail['reason']=='waypoint_step_above_hover_limit'
    np.testing.assert_array_equal(result,original)


def test_hover_smoothing_records_precise_skip_reason_without_modifying_targets():
    original=np.tile([.5,-.2,.8,0,0,0,1,-.785],(16,1)).astype(float)
    original[:,0]+=np.linspace(0,.03,16)
    result,detail=module.smooth_open_gripper_hover(original,original[0,:7],-.785,
        PROFILE['open_gripper_hover_smoothing'])
    assert detail['reason']=='net_motion_above_hover_limit'
    assert detail['net_displacement_m']>PROFILE['open_gripper_hover_smoothing']['max_net_m']
    np.testing.assert_array_equal(result,original)
    original[-1,7]=-.5
    result,detail=module.smooth_open_gripper_hover(original,original[0,:7],-.785,
        PROFILE['open_gripper_hover_smoothing'])
    assert detail['reason']=='gripper_not_open_throughout_chunk'
    np.testing.assert_array_equal(result,original)


@pytest.mark.parametrize('fail_first', [False, True])
def test_complete_h16_loop_and_no_retry_after_uncertain_execution(monkeypatch, fail_first):
    events=[]; submitted=[]; status_requests=[]
    action={'right_eef':np.tile([.5,-.2,.8,1,0,0,0,1,0],(1,16,1)),
            'right_gripper':np.linspace(-.785,0,16).reshape(1,16,1)}
    class Client:
        def __init__(self,*a,**k): pass
        def __enter__(self): return self
        def __exit__(self,*a): events.append('closed')
        def get_info(self): return {'control_api_exposed':False}
        def get_snapshot(self): return SimpleNamespace(metadata={'right_eef_xyz_quaternion_xyzw':[.5,-.2,.8,0,0,0,1],
            'right_gripper':{'training_position':-.785}},head_color_rgb=np.zeros((480,640,3),dtype=np.uint8),
            hand_right_rgb=np.zeros((480,640,3),dtype=np.uint8))
        def get_action(self,obs): events.append('inference');return action,{}
        def request(self,payload):
            if payload['op']=='activate':events.append('activate')
            if payload['op']=='status':status_requests.append(payload.copy())
            return dict(status(.8),ok=True)
    for name in ['G2LiveObservationClient','PolicyClient']:monkeypatch.setattr(module,name,Client)
    monkeypatch.setattr(module.shared,'PlacementBridgeSession',Client)
    monkeypatch.setattr(module.shared,'inspect_standby_bridge',lambda *a:None)
    monkeypatch.setattr(module.shared,'prepare_model',lambda *a:events.append('warmup'))
    monkeypatch.setattr(module.shared,'configure_transport_optimization',lambda *a:{})
    monkeypatch.setattr(module.shared,'validate_activation_state',lambda *a:None)
    monkeypatch.setattr(module.shared,'validate_snapshot',lambda *a:{})
    monkeypatch.setattr(module,'validate_contract',lambda *a,**k:None)
    monkeypatch.setattr(module,'pre_activation',lambda *a:{})
    monkeypatch.setattr(module,'calibrate_bridge_clock',lambda *a:(0,1000))
    def execute(client,targets,prefix,offset,**kwargs):
        submitted.append(targets.copy())
        assert targets.shape==(16,8) and kwargs['native_chunk_submission'] and kwargs['native_ack_pacing']
        if fail_first:raise TimeoutError('unknown physical execution; do not retry')
        cycle=len(submitted)-1;z=.8 if cycle==0 else .91
        kwargs['execution_rows'].extend(row(cycle*16+i+1,z) for i in range(16))
        kwargs['completion_status_out'].update(dict(status(z),ok=True))
    monkeypatch.setattr(module,'execute_action_chunk',execute)
    report={'cycles':[]}
    if fail_first:
        with pytest.raises(TimeoutError):module.run(PROFILE,report,lambda:None)
        assert len(submitted)==1
    else:
        module.run(PROFILE,report,lambda:None)
        assert len(submitted)==2 and report['status']=='FEEDBACK_GRASP_LIFT_STABLE'
        assert all(len(r['executions'])==16 for r in report['cycles'])
        assert all(r['snapshot_request_decode_s']>=0 and r['action_status_request_s']>=0
                   for r in report['cycles'])
        assert all(r['action_status_payload_bytes_estimate']>0 for r in report['cycles'])
        assert any(request.get('after_command_id')==report['cycles'][0]['executions'][-1]['command_id']
                   for request in status_requests)
    assert events.index('warmup')<events.index('activate')<events.index('inference')
    np.testing.assert_allclose(submitted[0][:,7],action['right_gripper'][0,:,0],atol=1e-7)
