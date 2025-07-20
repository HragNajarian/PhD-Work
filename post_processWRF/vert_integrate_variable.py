"""
=======================================================================================
Script: vert_integrate_variable.py
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

## What variables would you like to integrate?
L2_vars = ['QV', 'W']
## What pressure levels are you integrating between?
    # p_bot must be greater than p_top
p_bot=[[1000,1000,1000], [1000,1000,500]]
p_top=[[700,500,100], [700,100,200]]
## Assign parent_dir that is where your raw, L1, L2, etc. directories live.
parent_dir = sys.argv[1]
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00'

#######################################################################################
#######################################################################################

def vertical_integration(da, p_bot, p_top, g=9.81):
    """
    Vertically integrates a variable in pressure coordinates.

    Assumes `da` is ordered from top (low pressure) to bottom (high pressure),
    and that 'bottom_top' is the vertical dimension with pressure as the coordinate.

    Parameters:
    - da: xarray.DataArray with dimension 'bottom_top' representing pressure levels.
    - p_bot, p_top: int representing the pressure 
    - g: Gravitational acceleration (default 9.81 m/s²)

    Returns:
    - da_integrated: Vertically integrated DataArray (e.g., in J/m² or kg/m²)
    """

    # Calculate the pressure difference
    dp = da.bottom_top.diff('bottom_top').sel(bottom_top=slice(p_bot, p_top))

    # Calculate mean value between levels
    da_roll = da.rolling(bottom_top=2).mean().sel(bottom_top=dp.bottom_top.values)

    # Broadcast dp to match da_roll
    dp_broadcasted = dp.broadcast_like(da_roll)

    # Perform integration (sum over vertical)
    da_integrated = (da_roll * dp_broadcasted).sum(dim='bottom_top') / g

    # Convert to float32
    da_integrated = da_integrated.astype(np.float32)

    return da_integrated

start1_time = time.perf_counter()


## First open the raw dataset in order to create the coordinates
raw_subdir = 'd02_sunrise' if 'CRFoff' in parent_dir else 'd02'
ds_raw = xr.open_dataset(os.path.join(parent_dir, 'raw', raw_subdir), chunks='auto')#.isel(Time=[0])
step1_time = time.perf_counter()
print('Dataset loaded \N{check mark}', step1_time-start1_time, 'seconds')


## Assign bottom_top coordinates to make computations simpler using xarray
interp_P_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))  # Should be in hPa
coords = dict(bottom_top=(('bottom_top'),interp_P_levels))
step2_time = time.perf_counter()
print('Created coordinate dictionaries \N{check mark}', step2_time-step1_time, 'seconds')


## Create full paths of the variables to vertically integrate
L2_dir = parent_dir + '/L2'
prefix = 'd02_sunrise_interp_' if 'CRFoff' in parent_dir else 'd02_interp_'
L2_var_files = {f"{prefix}{var}" for var in L2_vars}
L2_paths = [os.path.join(L2_dir, f) for f in os.listdir(L2_dir) if f in L2_var_files]


## Loop through the variable paths
for i, path in enumerate(L2_paths):
    start2_time = time.perf_counter()

    ## Open data set, index appropriate variable, assign coords, and replace fill values with nans
    ds = xr.open_dataset(path, chunks='auto')#.isel(Time=[0])
    da = ds[L2_vars[i]].assign_coords(coords)
        # Important step over regions of terrain
    da = da.where(da != default_fill(np.float32))
    step2_time = time.perf_counter()
    print('Open Data Set \N{check mark}', step2_time-step1_time, 'seconds')


    ## Loop over the number of pressure layers you want to vertically integrate
    for j in range(len(p_bot[i])):
        p1 = p_bot[i][j]
        p2 = p_top[i][j]

        ## Vertically Integrate
        da_VI = vertical_integration(da, p1, p2, g=9.81)
        step1_time = time.perf_counter()
        print('Vertically Integrated \N{check mark}', step1_time-step2_time, 'seconds')

        ## Change Variable Name and assign attributes
        da_VI.name = L2_vars[i]
        da_VI = da_VI.assign_attrs(
        Pressure_bounds=f'{p1}-{p2} hPa',
        Units=da.attrs['units']
        )

        ## Save File
        filename_prefix = 'd02_sunrise' if 'CRFoff' in parent_dir else 'd02'
        out_path = f"{parent_dir}/L4/{filename_prefix}_VI_{L2_vars[i]}_{p1}-{p2}"
        da_VI.to_netcdf(out_path, mode='w', format='NETCDF4')
        step2_time = time.perf_counter()
        print(f'{L2_vars[i]} integrated over {p1}-{p2} hPa saved \N{check mark}', step2_time-step1_time, 'seconds')

    print(f'Total time for {L2_vars[i]} \N{check mark}', step2_time-start2_time, 'seconds')
