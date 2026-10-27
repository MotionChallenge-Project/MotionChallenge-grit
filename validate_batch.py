"""Compare batch observations and controls to the verified author-native adapter."""
import argparse,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,default=Path.home()/'.cache/MotionChallenge/lafan')
p.add_argument('--weights',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.core=root/'MotionChallenge-Core';a.runtime=root/'runtime.json'
sys.path.insert(0,str(a.core/'src'))
import numpy as np
from dataclasses import fields
from motion_challenge.sdk import StateBatch, ReferenceBatch
from motion_challenge.runtime.observations import controller_state
from motion_challenge.simulation.physics import World
from motion_challenge.data.dataset import load_csv_motion
from motion_challenge.validation import sha256_file
runtime=json.loads(a.runtime.read_text());runtime['adapter_root']=str(root)
if a.weights:runtime['weights']=str(a.weights.resolve())
sys.path.insert(0,str(root))
from batch_adapter import create
from restricted import create as single
batch=create(root,runtime);world=World(a.core/'robots/g1-29dof-cpu-v0.json')
manifest=json.loads((a.core/'suites/lafan/dataset.json').read_text());motion=load_csv_motion(a.data/manifest['files'][0]['path'],manifest['joint_names'],manifest['source_fps']);world.reset(motion)
refs=[tuple(world.reference(t+x) for x in runtime['reference_offsets_s']) for t in (.0,.03,.2,.91,1.2)]
slots=[9,2,12,0,6,5,20];singles={s:single(root,runtime) for s in slots};rng=np.random.default_rng(17)
max_observation=max_action=0.;cases=resets=0
def pack(states, windows):
 state = StateBatch(joint_names=states[0].joint_names, **{
  f.name: np.array([getattr(s, f.name) for s in states])
  for f in fields(StateBatch) if f.name != 'joint_names'})
 body_fields = {'body_position_world_m': 'position_world_m',
  'body_orientation_xyzw': 'orientation_xyzw',
  'body_linear_velocity_world_m_s': 'linear_velocity_world_m_s',
  'body_angular_velocity_world_rad_s': 'angular_velocity_world_rad_s'}
 values = {'body_names': tuple(b.name for b in windows[0][0].bodies)}
 for f in fields(ReferenceBatch):
  if f.name == 'body_names': continue
  values[f.name] = np.array([
   [[getattr(b, body_fields[f.name]) for b in r.bodies] for r in refs]
   if f.name in body_fields else [getattr(r, f.name) for r in refs]
   for refs in windows])
 return state, ReferenceBatch(**values)

for step in range(100):
 ids=slots if step==0 else [slots[i] for i in rng.permutation(7)[:(1,3,7)[step%3]]]
 states=[];windows=[]
 for row,slot in enumerate(ids):
  world.data.qpos[world.qadr]=rng.uniform(-.8,.8,29);world.data.qvel[:]=rng.normal(size=world.model.nv)
  import mujoco
  mujoco.mj_forward(world.model,world.data)
  states.append(controller_state(world.state(),'ideal-proprioceptive-no-position-v1'));windows.append(refs[step%len(refs)])
 sa,ra=pack(states,windows)
 if step==0 or step%7==0:
  batch.reset_batch(tuple(ids),sa,ra,(0,)*len(ids));resets+=len(ids)
  for slot,state,ref in zip(ids,states,windows):singles[slot].reset(state,ref,0)
 # Capture actual observations without double-advancing either history.
 feeds=[]
 original=batch.observations
 def capture(*args):
  value=original(*args);feeds.append(value);return value
 batch.observations=capture
 actual=batch.act_batch(tuple(ids),sa,ra);batch.observations=original
 for row,(slot,state,ref) in enumerate(zip(ids,states,windows)):
  policy=singles[slot];observed=[];old=policy.inner.observations
  def save(*args):
   value=old(*args);observed.append(value);return value
  policy.inner.observations=save;expected=policy.act(state,ref);policy.inner.observations=old
  for key in observed[0]:max_observation=max(max_observation,float(np.max(np.abs(observed[0][key]-feeds[0][key][row:row+1]))))
  max_action=max(max_action,float(np.max(np.abs(np.array(expected.joint_position_target_rad)-actual.joint_position_target_rad[row]))));cases+=1
record=dict(cases=cases,reset_slots=resets,max_observation_difference=max_observation,max_target_difference_rad=max_action,weights_sha256=runtime['weights_sha256'],batch_adapter_sha256=sha256_file(root/'batch_adapter.py'))
a.output.write_text(json.dumps(record,indent=2));print(json.dumps(record))
assert max_observation<=1e-5 and max_action<=1e-5
