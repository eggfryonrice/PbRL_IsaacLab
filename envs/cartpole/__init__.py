# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym
from .cartpole_env import CartpoleEnv, CartpoleEnvCfg


def cartpole_env_create(seed=None, device="cuda:0", render_mode=None, **kwargs):
    cfg = CartpoleEnvCfg()
    cfg.seed = seed
    cfg.sim.device = device
    return CartpoleEnv(cfg=cfg, render_mode=render_mode, **kwargs)


gym.register(
    id="cartpole",
    entry_point=cartpole_env_create,
    disable_env_checker=True,
)
