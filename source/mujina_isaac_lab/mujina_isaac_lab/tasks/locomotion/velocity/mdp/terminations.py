from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def arrive_goal(env: ManagerBasedRLEnv, goal_x: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the asset arrives at the goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_origins_x = env.scene.env_origins[:, 0]
    return (asset.data.root_link_pos_w[:, 0] - env_origins_x) > goal_x