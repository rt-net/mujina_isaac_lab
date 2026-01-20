# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys
import os

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  # set camera view

MUJINA_ASSETS_BASEPATH = os.path.dirname(__file__) 

# Add Franka
asset_path = MUJINA_ASSETS_BASEPATH + "assets/data/usd/mujina.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/mujina/Base")  # add robot to stage
robot = Articulation(prim_paths_expr="/mujina/Base", name="mujina")  # create an articulation object

print(str(robot.dof_names))
robot.initialize()
print(str(robot.dof_names))

# set the initial poses of the arm and the car so they don't collide BEFORE the simulation starts
robot.set_world_poses(positions=np.array([[0.0, 0.0, 1.0]]) / get_stage_units())

# initialize the world
my_world.reset()

for i in range(4):
    print("running cycle: ", i)
    if i == 1 or i == 3:
        print("moving")
        # move the arm
        robot.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])
    if i == 2:
        print("stopping")
        # reset the arm
        robot.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    for j in range(100):
        # step the simulation, both rendering and physics
        my_world.step(render=True)
        # print the joint positions of the car at every physics step

simulation_app.close()