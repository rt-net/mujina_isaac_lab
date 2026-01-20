
from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from mujina_isaac_lab.tasks.locomotion.velocity import mdp
from mujina_isaac_lab.assets.mujina import MUJINA_CFG, MUJINA_JOINT_NAMES
from isaaclab.utils import modifiers

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##
@configclass
class MujinaSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MUJINA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    def __post_init__(self):

        # terrain parameter settings
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 10
        self.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.02, 0.10)
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.02, 0.10)
        self.terrain.terrain_generator.sub_terrains["steps"] = mdp.terrains.MeshConsecutiveStepsTerrainCfg(
            step_height_range=(0.02, 0.17),
            step_width_range=(0.35, 0.45),
            step_margin_range=(0.35, 0.45),
            border_width=0.5,
            platform_width=1.0,
        )

        # set the terrain proportions
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion       = 0.1
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion   = 0.1
        self.terrain.terrain_generator.sub_terrains["random_rough"].proportion         = 0.2
        self.terrain.terrain_generator.sub_terrains["boxes"].proportion                = 0.2
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion     = 0.1
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.1
        self.terrain.terrain_generator.sub_terrains["steps"].proportion                = 0.0


@configclass
class MujinaObservationsCfg:

    @configclass
    class CommonCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

            # set the parameters to match the hardware
            self.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=MUJINA_JOINT_NAMES, preserve_order=True)
            }
            self.joint_vel.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=MUJINA_JOINT_NAMES, preserve_order=True)
            }

            # scale observations
            self.joint_pos.scale    = 1.0
            self.joint_vel.scale    = 0.05
            self.base_ang_vel.scale = 0.25

            # clip observations
            self.base_ang_vel.clip      = (-100.0, 100.0)
            self.velocity_commands.clip = (-100.0, 100.0)
            self.joint_pos.clip         = (-100.0, 100.0)
            self.joint_vel.clip         = (-100.0, 100.0)
            self.actions.clip           = ( -10.0,  10.0)

    @configclass
    class PolicyCfg(CommonCfg):
        height_scan = None
        base_lin_vel = None

        def __post_init__(self):
            super().__post_init__()

            # add noise to the observations
            self.enable_corruption = True
            self.base_ang_vel.noise      = Unoise(n_min=-0.25, n_max=0.25)
            self.joint_pos.noise         = Unoise(n_min=-0.01, n_max=0.01)
            self.joint_vel.noise         = Unoise(n_min=-1.0, n_max=1.0)
            self.projected_gravity.noise = Unoise(n_min=-0.15, n_max=0.15)

    @configclass
    class CriticCfg(CommonCfg):
        ## add more privileged observations
        foot_contact_forces = ObsTerm(
            func=mdp.foot_contact_forces,
            scale=0.01,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        foot_contact_force_dirs = ObsTerm(
            func=mdp.foot_contact_force_dirs,
            scale=1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        
        def __post_init__(self):
            super().__post_init__()

            # disable noise for critic observations
            self.enable_corruption = False
            
            self.base_lin_vel.scale = 2.0
            self.base_lin_vel.clip      = (-100.0, 100.0)

            # change offset for base height scan
            self.height_scan.params["offset"] = 0.3

    policy = PolicyCfg()
    critic = CriticCfg()

@configclass
class MujinaActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=MUJINA_JOINT_NAMES, scale=0.25, use_default_offset=True)
    joint_pos.preserve_order = True
    joint_pos.clip = {".*": (-10.0, 10.0) }

@configclass
class MujinaCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


##
# MDP settings
##

@configclass
class MujinaRewardsCfg:

    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.00,
        params={
            "command_name": "base_velocity",
            "cmd_lin_vel_threshold": 0.1,
            "cmd_ang_vel_threshold": 0.1,
        }
    )
    gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=0.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.1,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")), 
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        }
    )
    foot_rhythm = RewTerm(
        func=spot_mdp.air_time_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "mode_time": 0.25,
            "velocity_threshold": 0.1,
        }
    )
    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    air_time_variance = RewTerm(
        func=spot_mdp.air_time_variance_penalty,
        weight=-0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    knee_torque_std = RewTerm(
        func=mdp.JointTorquesStd,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"),
            "alpha": 0.975,
        },
    )
    collar_torque_std = RewTerm(
        func=mdp.JointTorquesStd,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_collar_joint"),
            "alpha": 0.975,
        },
    )
    hip_torque_std = RewTerm(
        func=mdp.JointTorquesStd,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "alpha": 0.975,
        },
    )
    
    def __post_init__(self):

        # weights
        self.track_lin_vel_xy_exp.weight = 1.5
        self.track_ang_vel_z_exp.weight  = 1.0
        self.lin_vel_z_l2.weight         = -1.0
        self.ang_vel_xy_l2.weight        = -0.05
        self.dof_torques_l2.weight       = -1.0e-5
        self.dof_acc_l2.weight           = -2.5e-7
        self.action_rate_l2.weight       = -0.1
        self.feet_air_time.weight        = 0.125
        self.undesired_contacts.weight   = -1.0
        self.flat_orientation_l2.weight  = -1.0
        self.dof_pos_limits.weight       = -5.0
        self.stand_still.weight          = -0.5
        self.gait.weight                 = 0.1
        self.foot_rhythm.weight          = 0.1
        self.foot_slip.weight            = -0.3
        self.air_time_variance.weight    = -0.1
        self.knee_torque_std.weight      = -0.01 
        self.collar_torque_std.weight    = -0.01
        self.hip_torque_std.weight       = -0.01

@configclass
class MujinaTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Base",".*_scapula", ".*_thigh"]), "threshold": 1.0},
    )

@configclass
class MujinaEventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-1.0, 1.0)}},
    )


@configclass
class MujinaCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)



@configclass
class MujinaRoughEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MujinaSceneCfg = MujinaSceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: MujinaObservationsCfg = MujinaObservationsCfg()
    actions: MujinaActionsCfg = MujinaActionsCfg()
    commands: MujinaCommandsCfg = MujinaCommandsCfg()

    # MDP settings
    rewards: MujinaRewardsCfg = MujinaRewardsCfg()
    terminations: MujinaTerminationsCfg = MujinaTerminationsCfg()
    events: MujinaEventCfg = MujinaEventCfg()
    curriculum: MujinaCurriculumCfg = MujinaCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False



@configclass
class MujinaRoughEnvCfg_PLAY(MujinaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
