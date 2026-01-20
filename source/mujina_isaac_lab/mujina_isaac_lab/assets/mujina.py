# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for mujina robots.

The following configuration parameters are available:

* :obj:`MUJINA_CFG`: The mujina robot

"""

# from isaaclab.sensors.camera.camera_cfg import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators.
##

ROBSTRIDE02_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_collar_joint", ".*_hip_joint"],
    effort_limit=12.0,
    velocity_limit=37.699,  # [rad/s] = 360 [rpm]
    stiffness={".*_collar_joint": 30.0, ".*_hip_joint": 30.0},
    damping={".*_collar_joint": 1.0, ".*_hip_joint": 1.0},
    armature=0.002,
    min_delay=1,  # 0.005*1 = 0.005 [s]
    max_delay=4,  # 0.005*4 = 0.020 [s]
)
ROBSTRIDE02_X2_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_knee_joint"],
    effort_limit=24.0,
    velocity_limit=18.85,  # [rad/s] = 180 [rpm]
    stiffness={".*_knee_joint": 30.0},
    damping={".*_knee_joint": 1.0},
    armature=0.002,
    min_delay=1,  # 0.005*1 = 0.005 [s]
    max_delay=4,  # 0.005*4 = 0.020 [s]
)
"""Configuration for mujina Delayed PDActuator model."""

##
# Configuration - Articulation.
##
import os

MUJINA_JOINT_NAMES = [
    "RL_collar_joint", "RL_hip_joint", "RL_knee_joint",
    "RR_collar_joint", "RR_hip_joint", "RR_knee_joint",
    "FL_collar_joint", "FL_hip_joint", "FL_knee_joint",
    "FR_collar_joint", "FR_hip_joint", "FR_knee_joint",
]

MUJINA_ASSETS_BASEPATH = os.path.dirname(__file__) 
MUJINA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MUJINA_ASSETS_BASEPATH}/data/usd/mujina.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),  # x,y,z [m]
        joint_pos={  # = target angles [rad] when action = 0.0
            '[F,R]R_collar_joint': -0.05,
            '[F,R]L_collar_joint': 0.05,
            'F[R,L]_hip_joint': 0.8,
            'R[R,L]_hip_joint': 1.0,
            '.*knee_joint': -1.4,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.8,
    actuators={
        "legs": ROBSTRIDE02_CFG,
        "knees": ROBSTRIDE02_X2_CFG
    },
)
"""Configuration of MUJINA robot using simple actuator config.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""