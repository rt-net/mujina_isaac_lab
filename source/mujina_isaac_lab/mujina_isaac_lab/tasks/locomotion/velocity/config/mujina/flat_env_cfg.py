from isaaclab.utils import configclass

from .rough_env_cfg import MujinaRoughEnvCfg


@configclass
class MujinaFlatEnvCfg(MujinaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # command scale
        self.commands.base_velocity.heading_command = False

        # event params
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.0)

        # self.commands.base_velocity.resampling_time_range = (0.5, 3.0)

        # reward scales
        # self.rewards.base_height_l2.weight = -0.01
        # self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.stand_still.weight = -1.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class MujinaFlatEnvCfg_PLAY(MujinaFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
