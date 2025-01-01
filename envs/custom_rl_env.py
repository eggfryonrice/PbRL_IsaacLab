import torch
import numpy as np

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import Articulation


@configclass
class CustomRLEnvCfg(DirectRLEnvCfg):
    usd_path: str = None


class CustomRLEnv(DirectRLEnv):
    cfg: CustomRLEnvCfg

    def __init__(self, cfg: CustomRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def get_obs(self):
        self._compute_intermediate_values()
        return self._get_observations()

    def _compute_intermediate_values(self):
        raise NotImplementedError("This method should be implemented by a subclass")

    def obs_query_to_scene_input(self, obs_query: np.ndarray, bs_query: np.ndarray):
        raise NotImplementedError("This method should be implemented by a subclass")

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
