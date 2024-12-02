# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .walker2d import WALKER2D_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from gymnasium.spaces import Box

import numpy as np

from .locomotion_env import LocomotionEnv
from ..custom_rl_env import CustomRLEnvCfg


@configclass
class Walker2dEnvCfg(CustomRLEnvCfg):
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
        num_envs=10, env_spacing=32.0, replicate_physics=True
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

    def obs_query_to_scene_input(self, obs_query: np.ndarray):
        frames = []

        torso_x = 0
        for i in range(len(obs_query)):
            obs = obs_query[i]
            torso_xx, torso_xz = obs[0], obs[7]
            thigh_right_xx, thigh_right_xz = obs[1], obs[8]
            thigh_left_xx, thigh_left_xz = obs[2], obs[9]
            leg_right_xx, leg_right_xz = obs[3], obs[10]
            leg_left_xx, leg_left_xz = obs[4], obs[11]
            foot_right_xx, foot_right_xz = obs[5], obs[12]
            foot_left_xx, foot_left_xz = obs[6], obs[13]
            torso_z = obs[14]
            torso_xvel = obs[15] / self.cfg.vel_scale

            torso_length = 0.6
            thigh_length = 0.45
            leg_length = 0.5
            foot_length = 0.2

            torso_center = np.array([torso_x, 0, torso_z])
            torso_top = torso_center + torso_length / 2 * np.array(
                [torso_xz, 0, torso_xx]
            )
            torso_bottom = torso_center + torso_length / 2 * np.array(
                [-torso_xz, 0, -torso_xx]
            )
            torso_bottom_right = torso_bottom + np.array([0, -0.1, 0])
            torso_bottom_left = torso_bottom + np.array([0, 0.1, 0])

            thigh_right = torso_bottom_right + thigh_length * np.array(
                [-thigh_right_xz, 0, -thigh_right_xx]
            )
            leg_right = thigh_right + leg_length * np.array(
                [-leg_right_xz, 0, -leg_right_xx]
            )
            foot_right = leg_right + foot_length * np.array(
                [foot_right_xx, 0, -foot_right_xz]
            )

            thigh_left = torso_bottom_left + thigh_length * np.array(
                [-thigh_left_xz, 0, -thigh_left_xx]
            )
            leg_left = thigh_left + leg_length * np.array(
                [-leg_left_xz, 0, -leg_left_xx]
            )
            foot_left = leg_left + foot_length * np.array(
                [foot_left_xx, 0, -foot_left_xz]
            )

            color = (1.0, 1.0, 0.5)
            left_color = (0.5, 0.5, 1.0)
            jointPositions = [
                (torso_top, color),
                (torso_bottom, color),
                (torso_bottom_right, color),
                (torso_bottom_left, left_color),
                (thigh_right, color),
                (leg_right, color),
                (foot_right, color),
                (thigh_left, left_color),
                (leg_left, left_color),
                (foot_left, left_color),
            ]

            def quat_z_axis_to_v(v):
                # Normalize the target vector
                v = v / np.linalg.norm(v)
                x, y, z = v

                # Calculate the angle between (0, 0, 1) and v
                theta = np.arccos(z)

                # Calculate the axis of rotation (cross product with (0, 0, 1))
                axis = np.array([-y, x, 0])
                axis_norm = np.linalg.norm(axis)

                # If the axis is zero, the vectors are already aligned
                if axis_norm < 1e-6:
                    return np.array([1, 0, 0, 0])

                # Normalize the axis
                axis = axis / axis_norm

                # Calculate the quaternion
                w = np.cos(theta / 2)
                x, y, z = axis * np.sin(theta / 2)

                return np.array([w, x, y, z])

            links = [
                (
                    torso_top,
                    torso_length,
                    quat_z_axis_to_v(torso_bottom - torso_top),
                    color,
                ),
                (
                    torso_bottom_right,
                    thigh_length,
                    quat_z_axis_to_v(thigh_right - torso_bottom_right),
                    color,
                ),
                (
                    thigh_right,
                    leg_length,
                    quat_z_axis_to_v(leg_right - thigh_right),
                    color,
                ),
                (
                    leg_right,
                    foot_length,
                    quat_z_axis_to_v(foot_right - leg_right),
                    color,
                ),
                (
                    torso_bottom_left,
                    thigh_length,
                    quat_z_axis_to_v(thigh_left - torso_bottom_left),
                    left_color,
                ),
                (
                    thigh_left,
                    leg_length,
                    quat_z_axis_to_v(leg_left - thigh_left),
                    left_color,
                ),
                (
                    leg_left,
                    foot_length,
                    quat_z_axis_to_v(foot_left - leg_left),
                    left_color,
                ),
            ]

            frames.append((jointPositions, links))

            torso_x += torso_xvel * self.step_dt

        return frames
