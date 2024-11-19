# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .walker2d import WALKER2D_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from gymnasium.spaces import Box

from .locomotion_env import LocomotionEnv


@configclass
class Walker2dEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 25.0 + 1e-6
    decimation = 2
    action_space = Box(low=-1.0, high=1.0, shape=(6,))
    observation_space = 24
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.0125, render_interval=decimation)

    # set friction coefficients
    sim.physics_material.static_friction = 0.7
    sim.physics_material.dynamic_friction = 0.7

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.7,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10, env_spacing=8.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = WALKER2D_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    joint_gears: list = [
        100.0000,  # thigh_right
        100.0000,  # thigh_left
        50.0000,  # leg_right
        50.0000,  # leg_left
        20.0000,  # foot_right
        20.0000,  # foot_left
    ]

    vel_scale: float = 1.0
    ang_vel_scale: float = 0.1
    dof_vel_scale: float = 0.1

    stand_height: float = 1.2
    move_speed: float = 1.0

    reset_noise: float = 5e-3


class Walker2dEnv(LocomotionEnv):
    cfg: Walker2dEnvCfg

    def __init__(self, cfg: Walker2dEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
