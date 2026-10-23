"""Validate GRIT against the pinned author's observations, warmup and action path.

uv run validate_inference.py --source GRIT_teleop_deploy --weights policy.onnx --output parity.json
"""
import argparse
import ast
from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
from adapter import GRIT
from motion_challenge.sdk import StatePacket, ReferenceFrame


def definitions(path, scope, names=None, methods=None, constants=False):
    nodes=[]
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names):
            if methods is not None and isinstance(n,ast.ClassDef):
                n.body=[m for m in n.body if isinstance(m,ast.Assign) or (isinstance(m,ast.FunctionDef) and m.name in methods)]
            nodes.append(n)
        elif constants and isinstance(n,ast.Assign):nodes.append(n)
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+nodes,type_ignores=[]),str(path),'exec'),scope)


def main():
    p=argparse.ArgumentParser()
    for name in ('source','weights','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parent
    pins=json.loads((root/'native-sources.json').read_text())
    for name,expected in pins['files'].items():assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==expected,name
    runtime=json.loads((root/'runtime.json').read_text());runtime['weights']=str(args.weights.resolve());policy=GRIT(root,runtime)
    src=args.source/'sim2real/src';sys.path.insert(0,str(src))
    from common.joint_mapper import JointMapper
    scope={'np':np,'R':R,'ABC':ABC,'abstractmethod':abstractmethod}
    definitions(src/'runtime/observation.py',scope)
    definitions(src/'runtime/grit_observation.py',scope,constants=True)
    definitions(src/'runtime/policy.py',scope,['Policy'],{'_reset_obs_modules','update_obs'})
    scope['ReferenceTrackingPolicy']=scope['Policy'];scope['ONNXRuntimeModel']=object
    definitions(src/'runtime/grit_policy.py',scope,['GritONNXModel','GritPolicy','_resolve_by_name'],{'__call__','compute_action'},constants=True)
    definitions(src/'deploy.py',scope,['Controller'],{'_apply_action'})
    ctrl_cfg=yaml.safe_load((args.source/'sim2real/config/g1/controller.yaml').read_text())
    track_cfg=yaml.safe_load((args.source/'sim2real/config/g1/tracking.yaml').read_text())
    assert policy.names==track_cfg['action_joint_names']==scope['GRIT_ACTION_JOINT_NAMES']
    np.testing.assert_allclose(policy.default,scope['_resolve_keyframe'](scope['grit_default_joint_pos_by_name'](),policy.names),rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(policy.scale,track_cfg['action_scale'],rtol=1e-7,atol=1e-7)
    mapper=JointMapper(ctrl_cfg['policy_joint_names'],policy.names)
    np.testing.assert_allclose(policy.kp,mapper.map_action_from_to(ctrl_cfg['kps']),rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(policy.kd,mapper.map_action_from_to(ctrl_cfg['kds']),rtol=1e-7,atol=1e-7)
    offsets=scope['grit_context_offsets']();assert list(offsets)==runtime['reference_steps']
    np.testing.assert_allclose(np.array(runtime['reference_offsets_s']),offsets/50,rtol=0,atol=1e-12)
    errors=dict(reference=0.,history=0.,network=0.,action=0.,motor_target=0.,reset_history=0.,synthetic_action_target=0.)
    def err(key,a,b):errors[key]=max(errors[key],float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    rng=np.random.default_rng(291023);names=tuple(np.array(policy.names)[rng.permutation(29)]);raw_max=0.
    for reset in range(4):
        ctrl=scope['Controller'].__new__(scope['Controller'])
        ctrl.config=NS(policy_joint_names=list(names));ctrl.dof_size=29
        ctrl.default_qpos=np.zeros(29,np.float32)
        ctrl.cmd_q=np.zeros(29,np.float32);ctrl.cmd_qd=np.zeros(29,np.float32);ctrl.cmd_kp=np.zeros(29,np.float32);ctrl.cmd_kd=np.zeros(29,np.float32)
        ctrl.kps=JointMapper(policy.names,list(names)).map_action_from_to(policy.kp);ctrl.kds=JointMapper(policy.names,list(names)).map_action_from_to(policy.kd)
        native=scope['GritPolicy'].__new__(scope['GritPolicy']);native.controller=ctrl;native.config=NS(reference_fps=50.)
        native.reference_joint_names=list(policy.names);native.last_action=np.zeros(29,np.float32);native.mapper_action=JointMapper(policy.names,list(names))
        native.grit_action_scale=np.array(track_cfg['action_scale'],np.float32);native.grit_default_pos_action=scope['_resolve_keyframe'](scope['grit_default_joint_pos_by_name'](),policy.names);native.grit_encoder_offset=np.zeros(29,np.float32)
        model=scope['GritONNXModel'].__new__(scope['GritONNXModel']);model.ort_session=policy.session;model.out_keys=['actions'];model.policy_input_dim=1620;native.module=model
        native.obs_modules=[scope['GritReferenceContext'](native),scope['GritProprioHistory'](native)];native.num_obs=1620;native.policy_input=None
        for step in range(40):
            actual=R.random(random_state=rng);q=rng.normal(0,.35,29);qd=rng.normal(0,.5,29);omega=rng.normal(0,.3,3)
            state=StatePacket(step*.02,names,tuple(q),tuple(qd),(0,0,.8),tuple(actual.as_quat()),(0,0,0),tuple(actual.apply(omega)))
            frame_count=2 if step>=35 else 14
            refs=[ReferenceFrame(step*.02+i*.02,step*.02+i*.02,tuple(rng.normal(0,.35,29)),tuple(rng.normal(0,.3,3)),tuple(R.random(random_state=rng).as_quat()),tuple(rng.normal(0,.5,29)),tuple(rng.normal(0,.5,3)),tuple(rng.normal(0,.3,3))) for i in range(frame_count)]
            selected=tuple(refs[min(i,frame_count-1)] for i in offsets)
            order=[names.index(n) for n in policy.names]
            refrot=R.from_quat([f.root_orientation_xyzw for f in refs])
            native.ref_idx=0;native.ref_len=len(refs);native.ref_root_quat=np.array([f.root_orientation_xyzw for f in refs])[:,[3,0,1,2]]
            native.grit_ref_features=scope['build_grit_reference_features'](
                np.array([f.joint_position_rad for f in refs])[:,order],np.array([f.root_position_world_m for f in refs]),native.ref_root_quat,
                joint_vel=np.array([f.joint_velocity_rad_s for f in refs])[:,order],root_lin_vel_b=refrot.inv().apply([f.root_linear_velocity_world_m_s for f in refs]),root_ang_vel_b=refrot.inv().apply([f.root_angular_velocity_world_rad_s for f in refs]))
            ctrl.qj=q;ctrl.dqj=qd;ctrl.quat=actual.as_quat()[[3,0,1,2]];ctrl.gyro=omega
            if step==0:
                native._reset_obs_modules();policy.reset(state,selected,reset)
                err('reset_history',np.concatenate([h.reshape(-1) for h in policy.history]),native.obs_modules[1].compute())
            native.update_obs();inputs=policy.observations(state,selected)
            err('reference',inputs['reference_context'].reshape(-1),native.policy_input['policy'][0,:630]);err('history',inputs['proprio_history'],native.policy_input['policy'][:,630:])
            expected=model(native.policy_input)['actions'];raw_max=max(raw_max,float(np.max(np.abs(expected))))
            err('network',policy.infer({'reference_context':native.policy_input['policy'][:,:630].reshape(1,9,70),'proprio_history':native.policy_input['policy'][:,630:]})['actions'],expected)
            ctrl._apply_action(native.compute_action())
            fn=policy.observations;policy.observations=lambda *_:inputs;out=policy.act(state,selected);policy.observations=fn
            err('action',policy.last,native.last_action);err('motor_target',out.joint_position_target_rad,ctrl.cmd_q)
        # Explicitly exercise raw outputs beyond a transport-side clipping threshold.
        synthetic=np.linspace(-20,20,29,dtype=np.float32)[None]
        native.module=lambda _: {'actions':synthetic};ctrl._apply_action(native.compute_action())
        fn=policy.infer;policy.infer=lambda _: {'actions':synthetic};out=policy.act(state,selected);policy.infer=fn
        err('synthetic_action_target',out.joint_position_target_rad,ctrl.cmd_q)
    limits=dict(reference=2e-5,history=2e-4,network=0.,action=2e-4,motor_target=2e-4,reset_history=2e-5,synthetic_action_target=2e-5)
    report={'passed':all(errors[k]<=limits[k] for k in errors),'cases':160,'resets':4,'max_abs_error':errors,'limits':limits,'max_abs_native_action':raw_max,'configured_action_clip':policy.cfg.get('action_clip'),'checkpoint_sha256':runtime['weights_sha256'],'source_commit':pins['source_commit'],'scope':'Original GRIT reference features, term-major history, reset warmup, ONNX call and final controller target; transport/import glue omitted. Synthetic raw action range tests conversion separately.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()
