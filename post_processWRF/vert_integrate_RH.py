"""
=======================================================================================
Script: vert_integrate_RH.py
Author: Hrag Najarian
Date: July 2, 2025
Description:
    This script performs vertical integration of specified atmospheric variables 
    between two pressure levels (e.g., 1000 hPa to 500 hPa) using data from WRF output.

    The code:
    - Loads the relevant interpolated variable files from the `/L2` directory
    - Replaces missing/fill values with NaNs to avoid contamination of the results
    - Vertically integrates the variable over a specified pressure range using the 
      trapezoidal method in pressure coordinates
    - Saves the output as a new NetCDF file in the `/L4` directory with the prefix "VI_"

    This is especially useful for calculating mass-integrated quantities such as:
    - Vertically integrated moisture
    - Integrated kinetic energy
    - Vertical mass flux, etc.

Inputs:
    - `parent_dir` (command-line argument): Full path to the directory containing 
      the `L2` subdirectory with WRF-interpolated NetCDF files.
    - `L2_vars`: List of variable names (as strings) to vertically integrate (e.g., 'W').
    - `p_bot` and `p_top`: Pressure boundaries for integration in hPa (e.g., 1000 and 500).
    - Assumes a raw WRF dataset (`/raw/d02` or `/raw/d02_sunrise`) exists for assigning 
      necessary coordinates.

Requirements:
    - Python 3.x
    - Packages: xarray, numpy, netCDF4, wrf-python, os, sys, time

Outputs:
    - One NetCDF file per variable with the integrated result stored in the `/L4` folder 
      under the name `VI_<variable>.nc`.

Notes:
    - Integration assumes pressure is sorted from high (bottom) to low (top).
    - Fill values from WRF are replaced with NaNs to ensure accuracy.

Example Usage:
    python vertical_integration.py /path/to/parent_dir
=======================================================================================
"""

import netCDF4 as nc
import numpy as np
import xarray as xr
import sys
import os
import time
from wrf import default_fill

#######################################################################################
#######################################################################################

start1_time = time.perf_counter()

# variables needed to compute RH
L4_vars = ['QV','ws']
## What pressure levels are you integrating between?
    # Keep in hPa, and the '*100' in the function will converts to pascal
    # p_bot must be greater than p_top
p_bot=[1000]    #, [1000,1000,500]]
p_top=[100]     #, [700,100,200]]

## Assign parent_dir that is where your raw, L1, L2, etc. directories live.
parent_dir = sys.argv[1]
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00/SN_CTRL'

## This is the string that's added depending on the experiment 
    # (i.e., '_sunrise', '_swap', '_adjLH', 
    # or '' if ctrl)
exp_string = ''

## Assign bottom_top coordinates to make computations simpler using xarray
interp_P_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))  # Should be in hPa
coords = dict(bottom_top=(('bottom_top'),interp_P_levels))
step2_time = time.perf_counter()
print('Created coordinate dictionaries \N{check mark}', step2_time-start1_time, 'seconds')

## Create full paths of the variables to vertically integrate
L4_dir = parent_dir + '/L4'
prefix = f'd02_VI_{exp_string}'
L4_var_files = {
    f"{prefix}{var}_{pb}-{pt}"
     for var in L4_vars
     for pb, pt in zip(p_bot,p_top)}
L4_paths = [os.path.join(L4_dir, f) for f in os.listdir(L4_dir) if f in L4_var_files]

start2_time = time.perf_counter()

# ## Open data set, index appropriate variable, assign coords, and replace fill values with nans
ds_QV = xr.open_dataset(L4_paths[0], chunks='auto')#.isel(Time=[0])   # Debugging purposes
ds_ws = xr.open_dataset(L4_paths[1], chunks='auto')
da_QV = ds_QV[L4_vars[0]]
da_ws = ds_ws[L4_vars[1]]
    # Important step over regions of terrain
da_QV = da_QV.where(da_QV != default_fill(np.float32))
da_ws = da_ws.where(da_ws != default_fill(np.float32))
step2_time = time.perf_counter()
print('Open Data Set \N{check mark}', step2_time-start2_time, 'seconds')

## Vertically Integrate
da_VI_RH = da_QV/da_ws
step1_time = time.perf_counter()
print('Calculate RH \N{check mark}', step1_time-step2_time, 'seconds')

## Change Variable Name and assign attributes
da_VI_RH.name = 'VI_RH'
da_VI_RH = da_VI_RH.assign_attrs(
Pressure_bounds=f'{p_bot[0]}-{p_top[0]} hPa',
Units='%'
)

## Save File
out_path = f"{parent_dir}/L4/{prefix}RH_{p_bot[0]}-{p_top[0]}"
da_VI_RH.to_netcdf(out_path, mode='w', format='NETCDF4')
step2_time = time.perf_counter()
print(f'Calculated vertically integrated RH over {p_bot[0]}-{p_top[0]} hPa and saved \N{check mark}', step2_time-step1_time, 'seconds')