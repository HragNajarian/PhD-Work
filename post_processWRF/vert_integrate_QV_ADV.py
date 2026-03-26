"""
=======================================================================================
Script: vert_integrate_QV_advection.py
Author: Hrag Najarian
Date: March 26, 2026
Description:
    This script computes vertically integrated horizontal moisture advection
    between two pressure levels (e.g., 1000 hPa to 100 hPa) using WRF output.

    The code:
    - Loads interpolated WRF variables (QV, U, V) from the `/L2` directory
    - Replaces missing/fill values with NaNs to avoid contamination of results
    - Computes horizontal moisture advection: -(u dq/dx + v dq/dy), where
      q is specific humidity (QV)
    - Vertically integrates the advection over a specified pressure range using a
      trapezoidal method in pressure coordinates via a reusable function
    - Saves the output as a NetCDF file in the `/L4` directory

    This is useful for diagnosing:
    - Column-integrated moisture tendency due to horizontal advection

Inputs:
    - `parent_dir` (command-line argument): Full path to the directory containing 
      the `L2` subdirectory with WRF-interpolated NetCDF files.
    - `L2_vars`: List of variables required for computation (QV, U, V).
    - `p_bot` and `p_top`: Pressure boundaries for integration in hPa.
    - Assumes interpolated pressure levels are stored along the `bottom_top` dimension.

Requirements:
    - Python 3.x
    - Packages: xarray, numpy, netCDF4, wrf-python, os, sys, time

Outputs:
    - NetCDF file containing vertically integrated moisture advection stored in `/L4`
      with naming convention: `QV_ADV_<p_bot>-<p_top>`

Notes:
    - Assumes winds are already destaggered
    - Assumes constant horizontal grid spacing (dx = dy = 3000 m)
    - Pressure coordinate is assumed to be in hPa and converted internally to Pa
    - Vertical integration uses a trapezoidal method with pressure thickness (dp/g)
    - Fill values from WRF are replaced with NaNs to ensure numerical accuracy

Example Usage:
    python vert_integrate_QV_advection.py /path/to/parent_dir
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
    dp = da.bottom_top.diff('bottom_top').sel(bottom_top=slice(p_bot, p_top))*100

    # Calculate mean value between levels
    da_roll = np.abs(da.rolling(bottom_top=2).mean().sel(bottom_top=dp.bottom_top.values))  # if issues, it's probably due to np.abs converting it to an numpy array

    # Broadcast dp to match da_roll
    dp_broadcasted = dp.broadcast_like(da_roll)

    # Perform integration (sum over vertical)
    da_integrated = (da_roll * dp_broadcasted).sum(dim='bottom_top') / g

    # Convert to float32
    da_integrated = da_integrated.astype(np.float32)

    return da_integrated


start1_time = time.perf_counter()

# variables needed to compute RH
L2_vars = ['QV','U','V']
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
interp_P_levels = np.concatenate((np.arange(1000,950,-10),
                                  np.arange(950,350,-30),
                                  np.arange(350,0,-50)))  # Should be in hPa
coords = dict(bottom_top=(('bottom_top'),interp_P_levels))
step2_time = time.perf_counter()
print('Created coordinate dictionaries \N{check mark}', step2_time-start1_time, 'seconds')

## Create full paths of the variables to vertically integrate
L2_dir = parent_dir + '/L2'
prefix = f'd02{exp_string}_interp_'
L2_var_files = {f"{prefix}{var}" for var in L2_vars}
L2_paths = [os.path.join(L2_dir, f) for f in os.listdir(L2_dir) if f in L2_var_files]

start2_time = time.perf_counter()

# ## Open data set, index appropriate variable, assign coords, and replace fill values with nans
ds_QV = xr.open_dataset(L2_paths[0], chunks='auto')
ds_U  = xr.open_dataset(L2_paths[1], chunks='auto')
ds_V  = xr.open_dataset(L2_paths[2], chunks='auto')

da_QV = ds_QV['QV']
da_U  = ds_U['U']
da_V  = ds_V['V']
    # Important step over regions of terrain
da_QV = da_QV.where(da_QV != default_fill(np.float32))
da_U  = da_U.where(da_U  != default_fill(np.float32))
da_V  = da_V.where(da_V  != default_fill(np.float32))

step2_time = time.perf_counter()
print('Open Data Set \N{check mark}', step2_time-start2_time, 'seconds')

######################################################################################
## Compute horizontal moisture advection: -(u dq/dx + v dq/dy)
#######################################################################################

dx = 3000.0  # meters
dy = 3000.0  # meters

# Compute gradients
dqdx = da_QV.differentiate('west_east') / dx
dqdy = da_QV.differentiate('south_north') / dy

# Moisture advection
adv_q = -(da_U * dqdx + da_V * dqdy)

step1_time = time.perf_counter()
print('Calculated horizontal moisture advection \N{check mark}', step1_time-step2_time, 'seconds')

#######################################################################################
## Vertical Integration (pressure coordinates)
#######################################################################################

# Vertical integral
da_VI_QV_ADV = vertical_integration(adv_q, p_bot[0], p_top[0], g=9.81)

step2_time = time.perf_counter()
print('Vertical integration complete \N{check mark}', step2_time-step1_time, 'seconds')

## Change Variable Name and assign attributes
da_VI_QV_ADV.name = 'VI_QV_ADV'
da_VI_QV_ADV = da_VI_QV_ADV.assign_attrs(
    description='Vertically integrated horizontal QV advection',
    Pressure_bounds=f'{p_bot[0]}-{p_top[0]} hPa',
    Units='kg kg^-1 m s^-1'
)

## Save File
filename_prefix = f'd02{exp_string}'
out_path = f"{parent_dir}/L4/{filename_prefix}_VI_QV_ADV_{p_bot[0]}-{p_top[0]}"
da_VI_QV_ADV.to_netcdf(out_path, mode='w', format='NETCDF4')

step2_time = time.perf_counter()
print(f'Calculated vertically integrated moisture advection over {p_bot[0]}-{p_top[0]} hPa and saved \N{check mark}', step2_time-step1_time, 'seconds')