#!/bin/bash

# IMPORTANT: This .job must be read with an argument ($1) that is equal to the parent directory of the respective simulation.

# This is a master script that will concat all the wrfout files (raw), extract variables (L1), interpolate variables (L2), and do cross-sectional analysis (L3).

# Concat
sbatch post_proc.job $1
# Extract
sbatch run_extract.job $1
# Interpolate
sbatch run_interp.job $1
# Cross-section
sbatch run_cross_section.job $1
