"""GRIT's author-published nine-frame / 990D observation and action contract."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController


class GRIT(ONNXController):
    action_output='actions'
    raw_action_history=True
    def frame(self,state,reference):
        q,qd,omega,gravity,rotation=self.proprio(state)
        relative=(rotation.inv()*R.from_quat(reference[0].root_orientation_xyzw)).as_matrix()[:,:2].reshape(6)
        return [omega,gravity,relative,q-self.default,qd,self.last]

    def reset(self,state,reference,seed):
        super().reset(state,reference,seed)
        # Author Policy._reset_observations performs max_step+1 warmup updates.
        self.history=[np.tile(np.asarray(v,np.float32),(10,1)) for v in self.frame(state,reference)]

    def observations(self,state,reference):
        for i,value in enumerate(self.frame(state,reference)):
            self.history[i]=np.concatenate([self.history[i][1:],np.asarray(value,np.float32)[None]],axis=0)
        rotations=R.from_quat([f.root_orientation_xyzw for f in reference])
        q=np.array([f.joint_position_rad for f in reference])[:,self.order]
        qd=np.array([f.joint_velocity_rad_s for f in reference])[:,self.order]
        linear=rotations.inv().apply([f.root_linear_velocity_world_m_s for f in reference])
        angular=rotations.inv().apply([f.root_angular_velocity_world_rad_s for f in reference])
        context=np.concatenate([q,qd,rotations.as_matrix()[:,:,:2].reshape(9,6),linear,angular],axis=-1)
        return {'reference_context':context[None],'proprio_history':np.concatenate([h.reshape(-1) for h in self.history])[None]}


def create(root,runtime):return GRIT(root,runtime)
