import torch
import numpy as np

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.utils import configclass


@configclass
class CustomRLEnvCfg(DirectRLEnvCfg):
    pass


class CustomRLEnv(DirectRLEnv):
    cfg: CustomRLEnvCfg

    def __init__(self, cfg: CustomRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def get_obs(self):
        self._compute_intermediate_values()
        return self._get_observations()

    def _compute_intermediate_values(self):
        raise NotImplementedError("This method should be implemented by a subclass")

    def obs_query_to_scene_input(self, obs_query: np.ndarray):
        raise NotImplementedError("This method should be implemented by a subclass")
