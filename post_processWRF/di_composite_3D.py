'''
Script: di_composite_3D.py
Author: Hrag Najarian
Date: July 25, 2025
'''

import netCDF4 as nc
import numpy as np
import xarray as xr
from flox.xarray import xarray_reduce
import sys
import os
import time
import glob
from wrf import default_fill


#######################################################################################
#######################################################################################

## What 3-D variables would you like to diurnally coposite?
L2_vars = ['Temp']
## Slice lat and lon bounds set for diurnal calculations
lat_bound = [-10, 10]
lon_bound = [80, 135]
interp_P_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))  # Should be in hPa
rolls = 1   # Smoother

## Assign parent_dir that is where your raw, L1, L2, etc. directories live.
parent_dir = sys.argv[1]
time_bound = [np.datetime64('2015-11-23T01'), np.datetime64('2015-12-02T00')] if '2015-11-22-12' in parent_dir else [np.datetime64('2015-12-10T01'), np.datetime64('2015-12-20T00')]
## This is the string that's added depending on the experiment 
    # (i.e., '_sunrise', '_swap', '_adjLH', 
    # or '' if ctrl)
exp_string = '_swap'

#######################################################################################
#######################################################################################


# Purpose: To grab the indicies that correspond to the times, latitudes, and longitudes of the WRF dataset file.

# Input:
	# file == path to the .nc file
	# times == np.datetime64 array [Start,End]
	# lats == np.array [south,north]
	# lons == np.array [west,east]

# Output:
	# time_ind, lat_ind, lon_ind == corresponds to the indicies of the times, lats, and lons provided within that file

# Example:
	# file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/raw/d01'
	# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-11-23T12')]
	# lats = [-7.5, 7.5]
	# lons = [90, 110]
	# isel_ind(file, times, lats, lons)

def isel_ind(file,times,lats,lons):
	# Declare the variables
	time_ind = np.zeros(2, dtype=int)
	lat_ind = np.zeros(2, dtype=int)
	lon_ind = np.zeros(2, dtype=int)
	# Open the file
	ds = xr.open_dataset(file)
	# Times
	time_vals = ds.XTIME.compute().values
	time_mask = (time_vals >= times[0]) & (time_vals <= times[1])
	time_indices = np.where(time_mask)[0]
	time_ind = [time_indices[0], time_indices[-1] + 1]  # +1 for Python slicing to include last index
	# Latitudes
	lat_vals = ds.XLAT[0,:,0].compute().values
	lat_mask = (lat_vals >= lats[0]) & (lat_vals <= lats[1])
	lat_indices = np.where(lat_mask)[0]
	lat_ind = [lat_indices[0], lat_indices[-1] + 1]  # +1 for Python slicing to include last index
	# Longitude
	lon_vals = ds.XLONG[0,0,:].compute().values
	lon_mask = (lon_vals >= lons[0]) & (lon_vals <= lons[1])
	lon_indices = np.where(lon_mask)[0]
	lon_ind = [lon_indices[0], lon_indices[-1] + 1]  # +1 for Python slicing to include last index


	return time_ind, lat_ind, lon_ind

# Purpose: Opens a dataset with restrictive bounds to make opening large files less intensive

# Input:
	# file == path to the .nc file
	# time_ind == indicies (inclusive) of the dates you want to look at
	# lat_ind == indicies (inclusive) of the latitudes you want to look at
	# lon_ind == indicies (inclusive) of the longitudes you want to look at

# Output:
	# ds == dataset that corresponds to the times, lats, and lons provided.

# Example:
	# To get the indicies, I suggest using the function isel_ind() I have coded up in tandem with this function (see above).
	# file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/raw/d01'
	# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-11-23T12')]
	# lats = [-7.5, 7.5]
	# lons = [90, 110]
	# time_ind, lat_ind, lon_ind = isel_ind(file, times, lats, lons)
	# ds = open_ds(file,time_ind,lat_ind,lon_ind)

def open_ds(file, time_ind, lat_ind, lon_ind):

	if time_ind[-1] == -1:
		ds = xr.open_dataset(file, chunks='auto').isel(
		Time=slice(None),
		# south_north=slice(lat_ind[0],lat_ind[1]),
		# west_east=slice(lon_ind[0],lon_ind[1])
		south_north=slice(None),
		west_east=slice(None)
	)
	else:
		ds = xr.open_dataset(file, chunks='auto').isel(
			Time=slice(time_ind[0],time_ind[1]),
			south_north=slice(lat_ind[0],lat_ind[1]),
			west_east=slice(lon_ind[0],lon_ind[1])
		)
	
	return ds


def build_path(parent_dir: str, sub_dir: str, name: str) -> str:
    return f"{parent_dir}/{sub_dir}/{name}"

def load_variable(file_dict: dict, coords: dict) -> xr.DataArray:
    step_start = time.perf_counter()
    
    ds = open_ds(file_dict['path'], time_ind, lat_ind, lon_ind)
        
    da = ds[file_dict['varname']].compute()
    da = da.assign_coords(coords)
    # Replace fill value with nan
    da = da.where(da != default_fill(np.float32))
    print(f"{file_dict['description']} loaded ✓", time.perf_counter() - step_start, "seconds")
    return da

# Function that can removes the bottom_top dimension for 2-D datasets
def without_keys(d, keys):
	return {x: d[x] for x in d if x not in keys}


# Purpose: Create a LocalTime coordinate within your DataArray.

def assign_LT_coord(da, dim_num):
	hour_offset = (da.XLONG.values[:,0,:]/15).round(decimals=0)

	# Local Time is a function of only Time and Longitude
	if dim_num==2:
		local_time = np.empty([len(da.Time),len(da.west_east)], dtype=object)
		for i in range(local_time.shape[0]):
			for j in range(local_time.shape[1]):
				local_time[i,j] = da.Time.values[i] + np.timedelta64(int(hour_offset[0,j]),'h')
		da = da.assign_coords(LocalTime=(('Time','west_east'),local_time))
	
	# Local Time is a function of Time, Longitude, and Latitude
	else:
		local_time = np.empty([len(da.Time),len(da.south_north),len(da.west_east)], dtype='datetime64[ns]')
		for i in range(local_time.shape[0]):
			for j in range(local_time.shape[2]):
				local_time[i,:,j] = da.Time.values[i] + np.timedelta64(int(hour_offset[0,j]),'h')
		da = da.assign_coords(LocalTime=(('Time','south_north','west_east'),local_time))
	return da


# Purpose: Diurnal Composite

def di_process_da(da):

	## Slice in time
	da = da.sel(Time=slice(*time_bound))

	## Rolling mean
	if rolls > 1:
		da = da.rolling({'south_north': rolls, 'west_east': rolls}, min_periods=1, center=True).mean()

	## Add local time coordinate and group by hour
	da = assign_LT_coord(da, dim_num=3)
	
	## Group by local time and the 'dim' parameter ensures 
		# only the Time dimension is averaged into the 'hour' dimension
	da = xarray_reduce(
		da, 'LocalTime.hour', func='nanmean', dim='Time',
		expected_groups=np.arange(24), isbin=[False], fill_value=np.nan
	).transpose('hour', 'south_north', 'west_east')

	return da.values


# Function that can removes the bottom_top dimension for 2-D datasets
def without_keys(d, keys):
	return {x: d[x] for x in d if x not in keys}


#######################################################################################
#######################################################################################

start1_time = time.perf_counter()

## Assign bottom_top coordinates to make computations simpler using xarray
raw_file = f'd02{exp_string}'
time_ind, lat_ind, lon_ind = isel_ind(build_path(parent_dir, "raw", raw_file), time_bound, lat_bound, lon_bound)
ds_raw = open_ds(build_path(parent_dir, "raw", raw_file), time_ind, lat_ind, lon_ind)
step1_time = time.perf_counter()
print('Dataset loaded \N{check mark}', step1_time-start1_time, 'seconds')


## Coordinate dictionaries
step1_time = time.perf_counter()

coords = dict(
	XLAT=(('Time','south_north','west_east'),ds_raw.XLAT.values),
	XLONG=(('Time','south_north','west_east'),ds_raw.XLONG.values),
	bottom_top=(('bottom_top'),interp_P_levels),
	Time=('Time',ds_raw.XTIME.values),
	south_north=(('south_north'),ds_raw.XLAT[0,:,0].values),
	west_east=(('west_east'),ds_raw.XLONG[0,0,:].values)
	)
di_coords = dict(
	hour=('hour',np.arange(0,24)),
	bottom_top=(('bottom_top'),interp_P_levels),
	south_north=(('south_north'),ds_raw.XLAT[0,:,0].values),
	west_east=(('west_east'),ds_raw.XLONG[0,0,:].values)
	)

step2_time = time.perf_counter()
print('Created coordinate dictionaries \N{check mark}', step2_time-step1_time, 'seconds')


## Create full paths of the variables to vertically integrate
L2_dir = parent_dir + '/L2'

prefix = f'd02{exp_string}_interp_'
L2_var_files = {f"{prefix}{var}" for var in L2_vars}
L2_paths = [os.path.join(L2_dir, f) for f in os.listdir(L2_dir) if f in L2_var_files]


## Loop through the variable paths
for i, path in enumerate(L2_paths):

	start1_time = time.perf_counter()

	## Open data set, index appropriate variable, assign coords, and replace fill values with nans
	ds = open_ds(path, time_ind, lat_ind, lon_ind)
	da = ds[L2_vars[i]].assign_coords(coords)
		# Important step over regions of terrain
	da = da.where(da != default_fill(np.float32))
	step2_time = time.perf_counter()
	print('Open Data Set \N{check mark}', step2_time-start1_time, 'seconds')

	## Create the 4-D array that will be populated diurnally
	da_di = np.full((24, len(interp_P_levels), *da.shape[2:]), np.nan, dtype=np.float32)
	
	
	## Loop over all the pressure layers 
	for j, level in enumerate(interp_P_levels):

		## Diurnally Composite
		da_di_z = di_process_da(da[:,j])

		## Append Diurnal composite into array
		da_di[:,j] = da_di_z

	## Assign coordinates before saving as an .nc file
	da_di = xr.DataArray(da_di, name=L2_vars[i], coords=di_coords, dims=('hour', 'bottom_top', 'south_north', 'west_east'))

	## Assign attributes
	da_di = da_di.assign_attrs(
		Units=da.attrs['units'],
		rolling=str(rolls),
		sim='NCRF' if 'CRFoff' in path else 'CTRL',
		Longitude_Bounds=f'{lon_bound[0]} to {lon_bound[1]}',
		Latitude_Bounds=f'{lat_bound[0]} to {lat_bound[1]}',
		Time_Bounds=f'{time_bound[0]} to {time_bound[1]}')

	## Save File
	out_path = f'{parent_dir}/L4/{L2_vars[i]}_di_ctrl' if exp_string=='' else f'{parent_dir}/L4/{L2_vars[i]}_di{exp_string}'
	da_di.to_netcdf(path=out_path, mode='w', format='NETCDF4', compute=True) #  ,unlimited_dims='Time'
	step2_time = time.perf_counter()
	print(f'{L2_vars[i]} saved \N{check mark}', step2_time-start1_time, 'seconds')