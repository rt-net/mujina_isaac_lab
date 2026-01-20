from __future__ import annotations

from dataclasses import MISSING
import numpy as np
import trimesh

from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass
from isaaclab.terrains import SubTerrainBaseCfg


def consecutive_steps_terrain(
    difficulty: float, cfg: MeshConsecutiveStepsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    Generate a terrain with many consecutive box rail steps as extrusions.
    
    Starting from the central platform (of size cfg.platform_width), a sequence
    of rail borders is generated. For each step, a random rail thickness is chosen
    from cfg.rail_thickness_range. The rail is constructed as a border between an
    inner box (previous step) and an outer box (expanded by 2 * thickness).
    This is repeated as long as the outer box fits within the overall terrain (cfg.size).
    
    Args:
        difficulty: The difficulty of the terrain. A value between 0 and 1.
                    It is used to compute the rail height.
        cfg: The terrain configuration, expected to include:
             - size: Tuple (width, length) in meters for the overall terrain.
             - platform_width: Width of the central platform (in m).
             - rail_height_range: Tuple (min, max) for rail height (in m).
             - rail_thickness_range: Tuple (min, max) thickness (in m) for each rail.
    
    Returns:
        A tuple containing:
          - A list of trimesh.Trimesh objects representing the rail steps.
          - A numpy array for the terrain origin (in m).
    """

    # Compute the rail height
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    step_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], step_height * 0.5)
    
    meshes_list = []

    # Initial inner area is the central platform
    current_inner_width = cfg.platform_width
    current_inner_length = cfg.platform_width

    while True:
        # choose a random step width
        step_width = np.random.uniform(cfg.step_width_range[0], cfg.step_width_range[1])
        # 
        outer_width = current_inner_width + 2.0 * step_width
        outer_length = current_inner_length + 2.0 * step_width

        # 地形全体のサイズを超えた場合は終了
        if outer_width > cfg.size[0] - cfg.border_width or outer_length > cfg.size[1] - cfg.border_width:
            break

        # 現在のリング領域のレールを生成
        inner_size = (current_inner_width, current_inner_length)
        outer_size = (outer_width, outer_length)
        meshes_list += make_border(outer_size, inner_size, step_height, step_center)

        # 次回の内側領域を，今回作成した外側サイズ+marginに更新
        step_margin = np.random.uniform(cfg.step_margin_range[0], cfg.step_margin_range[1])
        current_inner_width = outer_width + 2.0 * step_margin
        current_inner_length = outer_length + 2.0 * step_margin

    # 地形全体の下部（地面）を生成
    terrain_height = 1.0
    ground_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground_mesh = trimesh.creation.box(
        (cfg.size[0], cfg.size[1], terrain_height),
        trimesh.transformations.translation_matrix(ground_pos)
    )
    meshes_list.append(ground_mesh)

    # 地形の原点は地面の高さ 0 を基準とする（必要に応じて調整）
    origin = np.array([ground_pos[0], ground_pos[1], 0.0])
    return meshes_list, origin




@configclass
class MeshConsecutiveStepsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a consecutive steps mesh terrain."""

    function = consecutive_steps_terrain


    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width_range: tuple[float, float] = MISSING
    """The width of the steps (in m)."""
    step_margin_range: tuple[float, float] = MISSING
    """The margin between steps (in m)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
