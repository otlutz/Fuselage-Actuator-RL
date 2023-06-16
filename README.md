# AssemblyGym

This package contains environments for developing reinforcement learning for manufacturing assembly tasks.
So far, a number of subvariants have been implemented for the "FuselageActuators" environment. In the future, a multi-step assembly task will be added.

## Requirements

The "FuselageActuators" environments rely on Ansys simulations. The implementation has been tested on Ansys Mechanical 2022 R1 running on Windows 10 Pro. A connection to a license server may be required if no local license is available.

Necessary Python dependencies can be installed from the requirements.txt file. 

## Usage

See included Jupyter notebook for examples of running Ansys simulations from Python and running reinforcement learning experiments.

## What's included

Under the FuselageActuators environment folder, there are the following subfolders:

- **AnsysFiles**: Contains solution input files that were exported from Ansys. They are split into training, testing, and benchmarking examples. Each input file was generated from a respective Design Point in Ansys Workbench, where the initial shape was generated in an intermediate simulation step.
- **Shapes**: To speed up the environment reset procedure, the initial shapes for every design point have been saved into numpy arrays inside this folder. This avoids the need for re-calculating the starting shape in Ansys with zero forces applied
- **Recordings**: When enabled, interactions with the environment are recorded in csv files and stored in this folder. 

## Run training script
The ppo_FuselageActuators_v12 training script can be executed by the following command from inside a python shell

    python .\AssemblyGym\ppo_FuselageActuators_v12.py

Options can be passed in for parsing. E.g.,

    python .\AssemblyGym\ppo_FuselageActuators_v12.py --learning_rate 0.002

sets the initial learning rate to 0.002. Refer to the top of the script for available options.

The script incorporates logging to Tensorboard (pip install tensorboard) and wandb (pip install wandb). For wandb, the user's own wandb name should be entered for \<user\> below

    python .\AssemblyGym\ppo_FuselageActuators_v12.py --wandb-entity <user>

