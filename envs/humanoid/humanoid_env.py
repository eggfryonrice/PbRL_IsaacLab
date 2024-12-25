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
    action_space = 42
    observation_space = 54
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

    PD_Kp = 0.5
    PD_Kd = 0.05

    termination_height: float = 0.6

    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25

    stand_height: float = 1.2
    move_speed: float = 1.0


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

    def obs_query_to_scene_input(self, obs_query: np.ndarray, bs_query: np.ndarray):
        frames = []

        default_pos = bs_query[0, 0, :2]  # torso position at start
        bs_query = bs_query.copy()
        bs_query[:, :, :2] -= default_pos

        for bs in bs_query:
            capsules = []
            for i in range(16):
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
