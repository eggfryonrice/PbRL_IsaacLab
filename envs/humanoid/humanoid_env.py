# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab_assets import HUMANOID_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

import numpy as np

from .locomotion_env import LocomotionEnv
from ..custom_rl_env import CustomRLEnvCfg


@configclass
class HumanoidEnvCfg(CustomRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_space = 21
    observation_space = 57
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    usd_path: str = robot.spawn.usd_path
    joint_gears: list = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ]

    # # joint information
    # 'lower_waist:0', 'lower_waist:1'
    # 'right_upper_arm:0', 'right_upper_arm:2'
    # 'left_upper_arm:0', 'left_upper_arm:2'
    # 'pelvis'
    # 'right_lower_arm'
    # 'left_lower_arm'
    # 'right_thigh:0', 'right_thigh:1', 'right_thigh:2'
    # 'left_thigh:0', 'left_thigh:1', 'left_thigh:2'
    # 'right_shin'
    # 'left_shin'
    # 'right_foot:0', 'right_foot:1'
    # 'left_foot:0', 'left_foot:1'

    # joint pos limit
    # lower limit
    # -0.7854, -1.3090, -1.5708, -1.5708, -1.5708, -1.5708, -0.6109, -1.5708,
    # -1.5708, -0.7854, -2.0944, -1.0472, -0.7854, -2.0944, -1.0472, -2.7925,
    # -2.7925, -0.8727, -0.8727, -0.8727, -0.8727
    # upper limit
    # 0.7854, 0.5236, 1.2217, 1.2217, 1.2217, 1.2217, 0.6109, 0.8727,
    # 0.8727, 0.2618, 0.7854, 0.6109, 0.2618, 0.7854, 0.6109, 0.0349,
    # 0.0349, 0.8727, 0.8727, 0.8727, 0.8727

    # # body part information
    # 'torso', 'head', 'lower_waist', 'right_upper_arm', 'left_upper_arm', 'pelvis',
    # 'right_lower_arm', 'left_lower_arm', 'right_thigh', 'left_thigh', 'right_hand',
    # 'left_hand', 'right_shin', 'left_shin', 'right_foot', 'left_foot'

    PD_Kp = 0.5
    PD_Kd = 0.05

    termination_height: float = 0.6

    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25

    stand_height: float = 1.2
    move_speed: float = 1.0
    hand_height: float = 0.9


class HumanoidEnv(LocomotionEnv):
    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.default_ori_dict = {
            "torso": (0.07, (0.0, -0.07, 0.0), (0.0, 0.07, 0.0)),
            "upper_waist": (0.06, (-0.01, -0.06, -0.12), (-0.01, 0.06, -0.12)),
            "head": (0.09, (0, 0, 0), (0, 0, 0)),
            "lower_waist": (0.06, (0.0, -0.06, 0.0), (0.0, 0.06, 0.0)),
            "butt": (0.09, (-0.02, -0.05, 0.0), (-0.02, 0.05, 0.0)),
            "right_thigh": (0.06, (0.0, 0.0, 0.0), (0.0, 0.0, -0.34)),
            "right_shin": (0.05, (0.0, 0.0, 0.0), (0.0, 0.0, -0.3)),
            "right_right_foot": (0.027, (-0.07, -0.02, 0.0), (0.14, -0.04, 0.0)),
            "left_right_foot": (0.027, (-0.07, 0.0, 0.0), (0.14, 0.02, 0.0)),
            "left_thigh": (0.06, (0.0, 0.0, 0.0), (0.0, 0.0, -0.34)),
            "left_shin": (0.05, (0.0, 0.0, 0.0), (0.0, 0.0, -0.3)),
            "right_left_foot": (0.027, (-0.07, 0.0, 0.0), (0.14, -0.02, 0.0)),
            "left_left_foot": (0.027, (-0.07, 0.02, 0.0), (0.14, 0.04, 0.0)),
            "right_upper_arm": (0.04, (0.0, 0.0, 0.0), (0.16, -0.16, -0.16)),
            "right_lower_arm": (0.031, (0.0, 0.0, 0.0), (0.17, 0.17, 0.17)),
            "right_hand": (0.04, (0, 0, 0), (0, 0, 0)),
            "left_upper_arm": (0.04, (0.0, 0.0, 0.0), (0.16, 0.16, -0.16)),
            "left_lower_arm": (0.031, (0.0, 0.0, 0.0), (0.17, -0.17, 0.17)),
            "left_hand": (0.04, (0, 0, 0), (0, 0, 0)),
        }

        self.body_part_to_capsules_dict = {
            "torso": ("torso", "upper_waist"),
            "head": ("head",),
            "lower_waist": ("lower_waist",),
            "right_upper_arm": ("right_upper_arm",),
            "left_upper_arm": ("left_upper_arm",),
            "pelvis": ("butt",),
            "right_lower_arm": ("right_lower_arm",),
            "left_lower_arm": ("left_lower_arm",),
            "right_thigh": ("right_thigh",),
            "left_thigh": ("left_thigh",),
            "right_hand": ("right_hand",),
            "left_hand": ("left_hand",),
            "right_shin": ("right_shin",),
            "left_shin": ("left_shin",),
            "right_foot": ("right_right_foot", "left_right_foot"),
            "left_foot": ("right_left_foot", "left_left_foot"),
        }

        self.body_part = [
            "torso",
            "head",
            "lower_waist",
            "right_upper_arm",
            "left_upper_arm",
            "pelvis",
            "right_lower_arm",
            "left_lower_arm",
            "right_thigh",
            "left_thigh",
            "right_hand",
            "left_hand",
            "right_shin",
            "left_shin",
            "right_foot",
            "left_foot",
        ]

    def get_mirrored_state(self, state):
        mirrored_state = state.copy()
        # dof_pos
        mirrored_state[[2, 3, 7, 9, 10, 11, 15, 17, 18]] = state[
            [4, 5, 8, 12, 13, 14, 16, 19, 20]
        ]
        mirrored_state[[4, 5, 8, 12, 13, 14, 16, 19, 20]] = state[
            [2, 3, 7, 9, 10, 11, 15, 17, 18]
        ]

        mirrored_state[0] = -mirrored_state[0]  # lower waist x
        mirrored_state[6] = -mirrored_state[6]  # pelvis
        mirrored_state[[18, 20]] = -mirrored_state[
            [18, 20]
        ]  # I don't know the reason, but foot:1 joint is not symmetric, but rather aligned to same orientation in global

        # dof_vel
        mirrored_state[[23, 24, 28, 30, 31, 32, 36, 38, 39]] = state[
            [25, 26, 29, 33, 34, 35, 37, 40, 41]
        ]
        mirrored_state[[25, 26, 29, 33, 34, 35, 37, 40, 41]] = state[
            [23, 24, 28, 30, 31, 32, 36, 38, 39]
        ]

        mirrored_state[21] = -mirrored_state[21]  # lower waist x
        mirrored_state[27] = -mirrored_state[27]  # pelvis
        mirrored_state[[39, 41]] = -mirrored_state[[39, 41]]

        # vel_loc, angvel_loc, yaw, roll
        mirrored_state[[44, 46, 48, 49, 50]] = -mirrored_state[[44, 46, 48, 49, 50]]

        # hand and foot z position
        mirrored_state[[53, 54, 55, 56]] = state[[54, 53, 56, 55]]

        return mirrored_state

    def get_mirrored_action(self, action):
        mirrored_action = action.copy()

        # joint force
        mirrored_action[[2, 3, 7, 9, 10, 11, 15, 17, 18]] = action[
            [4, 5, 8, 12, 13, 14, 16, 19, 20]
        ]
        mirrored_action[[4, 5, 8, 12, 13, 14, 16, 19, 20]] = action[
            [2, 3, 7, 9, 10, 11, 15, 17, 18]
        ]

        mirrored_action[0] = -mirrored_action[0]  # lower waist x
        mirrored_action[6] = -mirrored_action[6]  # pelvis
        mirrored_action[[18, 20]] = -mirrored_action[[18, 20]]

        return mirrored_action
