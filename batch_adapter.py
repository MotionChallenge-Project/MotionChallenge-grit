"""Author GRiT observations, with independent histories and one batched ONNX call."""
import json

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from motion_challenge.adapters.onnx import vector
from motion_challenge.sdk import ControlBatch
from motion_challenge.validation import sha256_file


class GRiTBatch:
    def __init__(self, root, runtime):
        path = root/runtime['inference_config']
        if sha256_file(path) != runtime['inference_config_sha256']:
            raise ValueError('config checksum mismatch')
        self.config = json.loads(path.read_text())
        path = root/runtime['weights']
        if sha256_file(path) != runtime['weights_sha256']:
            raise ValueError('checkpoint checksum mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = runtime.get('inference_threads', 8)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.names = self.config['policy_joint_names']
        self.default = vector(self.config['default_joint_pos'], self.names)
        self.scale = vector(self.config['action_scale'], self.names, 1.)
        self.kp = vector(self.config['joint_kp'], self.names)
        self.kd = vector(self.config['joint_kd'], self.names)
        self.history, self.last = {}, {}

    def frame(self, slots, states, references):
        rotation = Rotation.from_quat(states.root_orientation_xyzw)
        inverse = rotation.inv()
        relative = (inverse*Rotation.from_quat(references.root_orientation_xyzw[:,0])).as_matrix()[:,:,:2]
        return [inverse.apply(states.root_angular_velocity_world_rad_s),
                inverse.apply((0.,0.,-1.)), relative.reshape(len(slots),6),
                states.joint_position_rad[:,self.order]-self.default,
                states.joint_velocity_rad_s[:,self.order], np.stack([self.last[s] for s in slots])]

    def reset_batch(self, slots, states, references, seeds):
        self.order = [states.joint_names.index(name) for name in self.names]
        self.inverse = [self.names.index(name) for name in states.joint_names]
        for slot in slots:
            self.last[slot] = np.zeros(len(self.names),np.float32)
        frames = self.frame(slots, states, references)
        for row, slot in enumerate(slots):
            # Matches the author's fully filled ten-frame reset history.
            self.history[slot] = [np.tile(np.asarray(frame[row],np.float32),(10,1)) for frame in frames]

    def observations(self, slots, states, references):
        frames = self.frame(slots, states, references)
        histories = []
        for row, slot in enumerate(slots):
            history = self.history[slot]
            for index, frame in enumerate(frames):
                history[index][:-1] = history[index][1:]
                history[index][-1] = frame[row]
            histories.append(np.concatenate([h.reshape(-1) for h in history]))
        n,k,_ = references.root_orientation_xyzw.shape
        rotation = Rotation.from_quat(references.root_orientation_xyzw.reshape(n*k,4))
        linear = rotation.inv().apply(references.root_linear_velocity_world_m_s.reshape(n*k,3)).reshape(n,k,3)
        angular = rotation.inv().apply(references.root_angular_velocity_world_rad_s.reshape(n*k,3)).reshape(n,k,3)
        context = np.concatenate([references.joint_position_rad[:,:,self.order],
            references.joint_velocity_rad_s[:,:,self.order], rotation.as_matrix()[:,:,:2].reshape(n,k,6),
            linear, angular],axis=-1)
        return dict(reference_context=context.astype(np.float32), proprio_history=np.stack(histories))

    def act_batch(self, slots, states, references):
        feed = self.observations(slots, states, references)
        raw = self.session.run(['actions'], feed)[0].reshape(len(slots),len(self.names))
        if not np.isfinite(raw).all():
            raise ValueError('nonfinite action')
        clip = self.config.get('action_clip')
        action = np.clip(raw,-clip,clip) if clip is not None else raw
        for row, slot in enumerate(slots):
            self.last[slot] = raw[row].copy()  # Author history stores raw actions.
        target = (self.default+self.scale*action)[:,self.inverse]
        zero = np.zeros_like(target)
        return ControlBatch(target, zero, np.broadcast_to(self.kp[self.inverse],target.shape),
                            np.broadcast_to(self.kd[self.inverse],target.shape), zero)

    def close(self):
        self.session = None


def create(root, runtime):
    return GRiTBatch(root, runtime)
