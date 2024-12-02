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

    def reset_idx(self, idx: torch.Tensor | None = None):
        if idx is None:
            idx = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        self._reset_idx(idx)

        return self._get_observations(), self.extras

    def obs_query_to_scene_input(self, obs_query: np.ndarray):
        raise NotImplementedError("This method should be implemented by a subclass")
