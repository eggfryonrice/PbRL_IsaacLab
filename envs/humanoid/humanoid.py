"""Configuration for the Mujoco Humanoid robot."""

from __future__ import annotations

import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

current_dir = os.path.dirname(os.path.abspath(__file__))
usd_file_path = os.path.join(current_dir, "humanoid.usda")

HUMANOID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_file_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.34),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_waist.*": 20.0,
                ".*_upper_arm.*": 10.0,
                "pelvis": 10.0,
                ".*_lower_arm": 2.0,
                ".*_thigh:0": 10.0,
                ".*_thigh:1": 20.0,
                ".*_thigh:2": 10.0,
                ".*_shin": 5.0,
                ".*_foot.*": 2.0,
            },
            damping={
                ".*_waist.*": 5.0,
                ".*_upper_arm.*": 5.0,
                "pelvis": 5.0,
                ".*_lower_arm": 1.0,
                ".*_thigh:0": 5.0,
                ".*_thigh:1": 5.0,
                ".*_thigh:2": 5.0,
                ".*_shin": 0.1,
                ".*_foot.*": 1.0,
            },
        ),
    },
)
"""Configuration for the Mujoco Humanoid robot."""
