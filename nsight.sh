#!/bin/bash

# Source the Conda setup script. Modify this line according to your Conda installation path.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate opt
python generate.py
