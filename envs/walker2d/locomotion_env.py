# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from ..custom_rl_env import CustomRLEnv, CustomRLEnvCfg


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class LocomotionEnv(CustomRLEnv):
    cfg: CustomRLEnvCfg

    def __init__(self, cfg: CustomRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_gears = torch.tensor(
            self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device
        )

        self._joint_dof_idx, joint_names = self.robot.find_joints(".*")
        excluded_joints = {
            "torso_rot_constraint",
            "torso_z_constraint",
            "torso_x_constraint",
        }
        self._joint_dof_idx = [
            idx
            for idx, name in zip(self._joint_dof_idx, joint_names)
            if name not in excluded_joints
        ]

        self._body_dof_idx, body_names = self.robot.find_bodies(".*")
        excluded_bodies = {"slider1", "slider2", "anchor"}
        self._body_dof_idx = [
            idx
            for idx, name in zip(self._body_dof_idx, body_names)
            if name not in excluded_bodies
        ]

        self.render_mode = render_mode

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position = self.robot.data.root_pos_w
        self.velocity, self.ang_velocity = (
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_w,
        )
        self.body_rotation = self.robot.data.body_quat_w
        self.dof_pos, self.dof_vel = (
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
        )

        self.orientation_xx, self.orientation_xz = compute_intermediate_values(
            self.body_rotation[:, self._body_dof_idx]
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.orientation_xx,
                self.orientation_xz,
                self.torso_position[:, 2].view(-1, 1),
                self.velocity[:, [0, 2]] * self.cfg.vel_scale,
                self.ang_velocity[:, [1]],
                self.dof_vel[:, self._joint_dof_idx] * self.cfg.dof_vel_scale,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.torso_position[:, 2],
            self.cfg.stand_height,
            self.orientation_xx[:, 0],
            self.velocity[:, 0],
            self.cfg.move_speed,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # die due to torso height less than termination_height
        died1 = self.torso_position[:, 2] < self.cfg.termination_height
        # die due to torso angle lies down too much, so that xz value higher than termination_xz
        died2 = torch.logical_or(
            self.orientation_xz[:, 0] > self.cfg.termination_xz,
            self.orientation_xz[:, 0] < -self.cfg.termination_xz,
        )
        died = torch.logical_or(died1, died2)
        return died, time_out
        # return False, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._compute_intermediate_values()


@torch.jit.script
def tolerance_gaussian(
    x: torch.Tensor,
    lower: float,
    upper: float,
    margin: float = 0.0,
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    in_bounds = (lower <= x) & (x <= upper)

    if margin == 0.0:
        return torch.where(
            in_bounds,
            torch.tensor(1.0, dtype=x.dtype, device=x.device),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
        )

    scale = margin / torch.sqrt(
        -2 * torch.log(torch.tensor(value_at_margin, dtype=x.dtype, device=x.device))
    )

    d = torch.where(x < lower, lower - x, x - upper) / scale
    value = torch.where(
        in_bounds,
        torch.tensor(1.0, dtype=x.dtype, device=x.device),
        torch.exp(-0.5 * d**2),
    )
    return value


@torch.jit.script
def tolerance_linear(
    x: torch.Tensor,
    lower: float,
    upper: float,
    margin: float = 0.0,
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    in_bounds = (lower <= x) & (x <= upper)
    if margin == 0.0:
        return torch.where(
            in_bounds,
            torch.tensor(1.0, dtype=x.dtype),
            torch.tensor(0.0, dtype=x.dtype),
        )

    d = torch.where(x < lower, lower - x, x - upper) / margin
    slope = 1 - value_at_margin
    value = torch.where(
        in_bounds,
        torch.tensor(1.0, dtype=x.dtype),
        torch.clamp(1.0 - slope * d, min=0.0),
    )
    return value


@torch.jit.script
def compute_rewards(
    torso_height: torch.Tensor,
    stand_height: float,
    torso_xx: torch.Tensor,
    torso_speed: torch.Tensor,
    move_speed: float,
):
    # stand reward
    standing = tolerance_gaussian(
        torso_height, stand_height, float("inf"), stand_height / 2, 0.1
    )
    upright = (1 + torso_xx) / 2
    stand_reward = (3 * standing + upright) / 4

    # move reward
    move_reward = tolerance_linear(
        torso_speed, move_speed, float("inf"), move_speed / 2, 0.5
    )
    return stand_reward * (5 * move_reward + 1) / 6


@torch.jit.script
def compute_intermediate_values(
    body_rotation: torch.Tensor,
):
    w, x, y, z = (
        body_rotation[..., 0],
        body_rotation[..., 1],
        body_rotation[..., 2],
        body_rotation[..., 3],
    )

    xx = 1 - 2 * (y**2 + z**2)
    xz = 2 * (x * z + y * w)

    return xx, xz
