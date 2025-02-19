import torch
import numpy as np

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import Articulation


@configclass
class CustomRLEnvCfg(DirectRLEnvCfg):
    usd_path: str = None


def quaternion_multiply(q1, q2):
    """Multiplies two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quaternion_conjugate(q):
    """Returns the conjugate of a quaternion q."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def translate_point(point, translation, quaternion):
    """Translates a point using position and quaternion."""
    # Convert point to a quaternion (w=0)
    point_quat = (0, *point)

    # Rotate the point using the quaternion
    q_conj = quaternion_conjugate(quaternion)
    rotated_point = quaternion_multiply(
        quaternion_multiply(quaternion, point_quat), q_conj
    )
    # Extract the rotated point (x, y, z)
    rotated_point = np.array(rotated_point[1:])  # Skip the w component

    # Add the translation
    translated_point = rotated_point + np.array(translation)
    return translated_point


class CustomRLEnv(DirectRLEnv):
    cfg: CustomRLEnvCfg

    def __init__(self, cfg: CustomRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.default_ori_dict = None
        self.body_part_to_capsules_dict = None
        self.body_part = None

    def get_obs(self):
        self._compute_intermediate_values()
        return self._get_observations()

    def _compute_intermediate_values(self):
        raise NotImplementedError("This method should be implemented by a subclass")

    # bs (body state) is state of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
    def obs_query_to_scene_input(self, obs_query: np.ndarray, bs_query: np.ndarray):
        frames = []

        default_pos = bs_query[0, 0, :2]  # torso position at start
        bs_query = bs_query.copy()
        bs_query[:, :, :2] -= default_pos

        for bs in bs_query:
            capsules = []
            for i in range(len(self.body_part)):
                pos = bs[i][:3]
                quat = bs[i][3:7]

                capsule_names = self.body_part_to_capsules_dict[self.body_part[i]]
                for capsule_name in capsule_names:
                    rad, start, end = self.default_ori_dict[capsule_name]
                    start = translate_point(np.array(start), pos, quat)
                    end = translate_point(np.array(end), pos, quat)
                    capsules.append((start, end, rad, (1.0, 1.0, 0.5)))

            frames.append((capsules, [], []))
        return frames

    def get_mirrored_state(self, state):
        raise NotImplementedError("This method should be implemented by a subclass")

    def get_mirrored_action(self, action):
        raise NotImplementedError("This method should be implemented by a subclass")

    def get_mirrored_state_action_query(self, state_action_query: np.ndarray):
        mirrored_sa_query = np.zeros_like(state_action_query)
        for i in range(len(state_action_query)):
            state = state_action_query[i][: self.cfg.observation_space]
            action = state_action_query[i][self.cfg.observation_space :]
            mirrored_state = self.get_mirrored_state(state.copy())
            mirrored_action = self.get_mirrored_action(action.copy())
            mirrored_sa_query[i, : self.cfg.observation_space] = mirrored_state
            mirrored_sa_query[i, self.cfg.observation_space :] = mirrored_action
        return mirrored_sa_query
