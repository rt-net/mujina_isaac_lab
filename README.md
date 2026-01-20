# Mujina Isaac Lab Project

## Overview

This project is based on the Isaac Lab project template.

## Requirements
- Isaac Sim 5.1.0
- Isaac Lab 2.3.0

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/mujina_isaac_lab

- Verify that the extension is correctly installed by:

    - Running a train task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/rsl_rl/train.py --task=<TASK_NAME>
        ```

## Acknowledgements
- This package is forked from [CoRE-MA-KING/mevius_isaac_lab](https://github.com/CoRE-MA-KING/mevius_isaac_lab)
- [mujina.usd](./source/mujina_isaac_lab/mujina_isaac_lab/assets/data/usd/mujina.usd) is based on CAD models from [Mujina_Hardware](https://github.com/noriakinakagawa/Mujina_Hardware).
  - We have obtained permission from Noriaki Nakagawa to use the CAD models.
