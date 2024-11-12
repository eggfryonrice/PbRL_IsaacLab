"""Configuration for the Walker2d robot."""

from __future__ import annotations

import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

current_dir = os.path.dirname(os.path.abspath(__file__))
usd_file_path = os.path.join(current_dir, "walker2d.usd")

WALKER2D_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_file_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.3),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "thigh_right": 0.0,
                "thigh_left": 0.0,
                "leg_right": 0.0,
                "leg_left": 0.0,
                "foot_right": 0.0,
                "foot_left": 0.0,
                "torso_rot_constraint": 0.0,
                "torso_z_constraint": 0.0,
            },
            damping={
                "thigh_right": 0.1,
                "thigh_left": 0.1,
                "leg_right": 0.1,
                "leg_left": 0.1,
                "foot_right": 0.1,
                "foot_left": 0.1,
                "torso_rot_constraint": 0.0,
                "torso_z_constraint": 0.0,
            },
        ),
    },
)
"""Configuration for the Mujoco walker2d robot."""
