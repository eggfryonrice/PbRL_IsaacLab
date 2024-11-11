# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .ant_env import AntEnv, AntEnvCfg

##
# Register Gym environments.
##


def ant_env_create(seed=None, device="cuda:0", render_mode=None, **kwargs):
    cfg = AntEnvCfg()
    cfg.seed = seed
    cfg.sim.device = device
    return AntEnv(cfg=cfg, render_mode=render_mode, **kwargs)


gym.register(
    id="ant",
    entry_point=ant_env_create,
    disable_env_checker=True,
)
