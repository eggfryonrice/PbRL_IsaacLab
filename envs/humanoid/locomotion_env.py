# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.utils.torch.rotations import (
    compute_heading_and_up,
    compute_rot,
    quat_conjugate,
)

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

        self.motor_effort_ratio = torch.ones_like(
            self.joint_gears, device=self.sim.device
        )
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        self.potentials = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.sim.device
        )
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor(
            [1000, 0, 0], dtype=torch.float32, device=self.sim.device
        ).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor(
            [1, 0, 0, 0], device=self.sim.device, dtype=torch.float32
        )
        self.up_vec = torch.tensor(
            [0, 0, 1], dtype=torch.float32, device=self.sim.device
        ).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor(
            [1, 0, 0], dtype=torch.float32, device=self.sim.device
        ).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat(
            (self.num_envs, 1)
        )
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self._joint_pos_lowerbound = self.robot.data.soft_joint_pos_limits[
            0, :, 0
        ].unsqueeze(0)
        self._joint_pos_upperbound = self.robot.data.soft_joint_pos_limits[
            0, :, 1
        ].unsqueeze(0)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        action_position = self.actions[:, : len(self._joint_dof_idx)]
        margin = 0.0
        # lower bound are all negative, upper bound are all positive, default is 0
        # target is 0 when action is 0, target is lowerbound - margin when action is -1,
        # upperbound + margin when action is 1 and between [-1,0], [0,1] is line
        target_position = torch.where(
            action_position > 0,
            action_position * (self._joint_pos_upperbound + margin),
            -action_position * (self._joint_pos_lowerbound - margin),
        )
        target_velocity = (
            self.actions[:, len(self._joint_dof_idx) :] / self.cfg.dof_vel_scale
        )
        forces = self.joint_gears * self.cfg.PD_Kp * (
            target_position - self.robot.data.joint_pos
        ) + self.cfg.PD_Kd * (target_velocity - self.robot.data.joint_vel)
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = (
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
        )
        self.velocity, self.ang_velocity = (
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_w,
        )
        self.dof_pos, self.dof_vel = (
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
        )

        # z position of hand and foot
        self.acro_position_z = self.robot.data.body_pos_w[:, [10, 11, 14, 15], 2]

        self.body_state = self.robot.data.body_state_w
        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
            self.torso_position_z_scaled,
            self.acro_position_z_scaled,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
            self.acro_position_z,
            self.cfg.termination_height,
            self.cfg.stand_height,
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.torso_position_z_scaled.view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.acro_position_z_scaled,
            ),
            dim=-1,
        )
        observations = {"policy": obs, "body_state": self.body_state}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.torso_position[:, 2],
            self.cfg.stand_height,
            self.up_proj,
            self.vel_loc[:, 0],
            self.cfg.move_speed,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_position[:, 2] < self.cfg.termination_height
        # died = False
        return died, time_out

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

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

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
    up_proj: torch.Tensor,
    torso_speed: torch.Tensor,
    move_speed: float,
):
    # stand reward
    standing = tolerance_gaussian(
        torso_height, stand_height, float("inf"), stand_height / 2, 0.1
    )
    upright = (1 + up_proj) / 2
    stand_reward = (3 * standing + upright) / 4

    # move reward
    move_reward = tolerance_linear(
        torso_speed, move_speed, float("inf"), move_speed / 2, 0.5
    )
    return stand_reward * (5 * move_reward + 1) / 6


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
    acro_position_z,
    termination_height: float,
    stand_height: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(
        dof_pos, dof_lower_limits, dof_upper_limits
    )

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_position_z = torso_position[:, 2]
    torso_position_z_scaled = -1 + 2 * (torso_position_z - termination_height) / (
        stand_height - termination_height
    )

    acro_position_z_scaled = -1 + 2 * acro_position_z / stand_height

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
        torso_position_z_scaled,
        acro_position_z_scaled,
    )
