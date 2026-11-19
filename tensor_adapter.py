"""GRIT's nine-frame reference graph and ten-frame observation history."""
from tensor_common import TensorPD,normalize,conjugate,rotate,multiply,matrix,torch


class TensorGRIT(TensorPD):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.history=[torch.zeros((self.slots,10,n),device=self.device) for n in (3,3,6,29,29,29)]

    def observations(self,state,reference):
        q,qd,omega,gravity,rotation=self.proprio(state)
        rotations=normalize(reference.root_orientation_xyzw.double())
        relative=matrix(multiply(conjugate(rotation),rotations[:,0]))[...,:,:2].flatten(1)
        frame=(omega,gravity,relative,q-self.default,qd,self.last)
        for history,value in zip(self.history,frame):
            value=value.float()
            history.copy_(torch.where(self.fresh[:,None,None],value[:,None],history))
            history[:,:-1]=history[:,1:].clone();history[:,-1]=value
        self.fresh.zero_()
        context=torch.cat((reference.joint_position_rad[:,:,self.order].double(),
            reference.joint_velocity_rad_s[:,:,self.order].double(),matrix(rotations)[...,:,:2].flatten(-2),
            rotate(conjugate(rotations),reference.root_linear_velocity_world_m_s.double()),
            rotate(conjugate(rotations),reference.root_angular_velocity_world_rad_s.double())),dim=-1).float()
        return {'reference_context':context,'proprio_history':torch.cat([h.flatten(1) for h in self.history],dim=-1)}

ADAPTER_CLASS='TensorGRIT'
NATIVE_CLASS='GRIT'
