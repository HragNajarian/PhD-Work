
from wrf import getvar, ALL_TIMES
import matplotlib as mpl
import cartopy.crs as ccrs
from flox.xarray import xarray_reduce
from flox import Aggregation
# import glob
# import dask
# import cartopy.feature as cfeature
# import metpy.calc as mpcalc
# from metpy.cbook import get_test_data
# from metpy.interpolate import cross_section
# from cartopy.io.img_tiles import Stamen
# from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from matplotlib import cm
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import os
# from braceexpand import braceexpand
# import pandas as pd
import time
# import netCDF4 as nc
from math import cos, asin, sqrt, pi, atan, degrees
from scipy.stats import bootstrap
# from scipy.optimize import curve_fit


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
	time_ind[0] = np.absolute(ds.XTIME.compute().values - (times[0])).argmin()
	time_ind[1] = np.absolute(ds.XTIME.compute().values - (times[1])).argmin()+1	# + 1 is because of the way argmin works
	# Latitudes
	lat_ind[0] = np.absolute(ds.XLAT[0,:,0].compute().values-(lats[0])).argmin()
	lat_ind[1] = np.absolute(ds.XLAT[0,:,0].compute().values-(lats[1])).argmin()+1
	# Longitude
	lon_ind[0] = np.absolute(ds.XLONG[0,0,:].compute().values-(lons[0])).argmin()
	lon_ind[1] = np.absolute(ds.XLONG[0,0,:].compute().values-(lons[1])).argmin()+1

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

def open_ds(file,time_ind,lat_ind,lon_ind):
	# if cross==True :	# If it is a cross-section ds
	# 	ds = xr.open_dataset(file).isel(
	# 		Time=slice(time_ind[0],time_ind[1]),
	# 	)
	# else:
	ds = xr.open_dataset(file, chunks='auto').isel(
		Time=slice(time_ind[0],time_ind[1]),
		south_north=slice(lat_ind[0],lat_ind[1]),
		west_east=slice(lon_ind[0],lon_ind[1])
	)
	return ds


# Purpose: Create a LocalTime coordinate within your DataArray.

# Input:
    # da = xr.DataArray;  Must be in shape: time x south_north x west_east
		# Make sure west_east/XLONG values are 0->360, not -180->+180
	# dim_num = 2 or 3;  This indicates to the function if you want Local Time
		# within the dataarray to be only a function of time and longitude, or
		# time, lognitude, and latitude. This is a preference and if you don't need
		# it as a function of latitude, it will save lots of time going with dim_num = 2.
# Output:
    # da: This will be the DataArray with the newly assigned coordinate
# Process:
    # First create a matrix of hours to be offset relative to UTC.
    # Create an empty array that has dimensions Time and Longitude.
    # Loop through each timestep and longitude to determine the local time.
    # Assign the new Local Time coordinate to the da and return it.


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


# Function that can removes the bottom_top dimension for 2-D datasets
def without_keys(d, keys):
	return {x: d[x] for x in d if x not in keys}


start_time = time.perf_counter()

parent_dir_CNTL = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00'
# parent_dir_CNTL = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26/2015-11-25-03--11-26-12'
file_d01_raw = parent_dir_CNTL + '/raw/d01'
file_d02_raw = parent_dir_CNTL + '/raw/d02'
# 2-D data
file_d02_RR = parent_dir_CNTL + '/L1/d02_RR'				# [mm/dt]


######################################################################################
################ Declare the bounds you want to specifically look at #################
#### All the data 
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-02T12')]
lats = [-20, 20]
lons = [80, 135]

#### Some of the data
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-03T00')]
times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-12-02T00')]        # NCRF Sunrise
# times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-11-25T02')]        # NCRF Sunrise Debugging
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-11-23T12')]
# times = [np.datetime64('2015-11-25T00'), np.datetime64('2015-11-26T12')]
# lats = [-7.5, 7.5]
# lons = [90, 115]
######################################################################################
# Setup the indicies that will be used throughout
time_ind_d01, lat_ind_d01, lon_ind_d01 = isel_ind(file_d01_raw, times, lats, lons)
time_ind_d02, lat_ind_d02, lon_ind_d02 = isel_ind(file_d02_raw, times, lats, lons)

# Raw datasets
ds_d01 = open_ds(file_d01_raw,time_ind_d01,lat_ind_d01,lon_ind_d01)
ds_d02 = open_ds(file_d02_raw,time_ind_d02,lat_ind_d02,lon_ind_d02)
step1_time = time.perf_counter()
print('Dataset loaded \N{check mark}', step1_time-start_time, 'seconds')


# Coordinate dictionaries:
step2_time = time.perf_counter()
interp_P_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))

d01_coords = dict(
    XLAT=(('Time','south_north','west_east'),ds_d01.XLAT.values),
    XLONG=(('Time','south_north','west_east'),ds_d01.XLONG.values),
    bottom_top=(('bottom_top'),interp_P_levels),
    Time=('Time',ds_d01.XTIME.values),
    south_north=(('south_north'),ds_d01.XLAT[0,:,0].values),
    west_east=(('west_east'),ds_d01.XLONG[0,0,:].values)
    )

d02_coords = dict(
    XLAT=(('Time','south_north','west_east'),ds_d02.XLAT.values),
    XLONG=(('Time','south_north','west_east'),ds_d02.XLONG.values),
    bottom_top=(('bottom_top'),interp_P_levels),
    Time=('Time',ds_d02.XTIME.values),
    south_north=(('south_north'),ds_d02.XLAT[0,:,0].values),
    west_east=(('west_east'),ds_d02.XLONG[0,0,:].values)
    )

step1_time = time.perf_counter()
print('Created coordinate dictionaries \N{check mark}', step1_time-step2_time, 'seconds')

###########################################################
################# Load in the variables ###################
###########################################################


############ Rain Rate     [mm/hr] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_RR,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_RR = ds['RR'].compute()
da_d02_RR = da_d02_RR.assign_coords(without_keys(d02_coords,'bottom_top'))
step1_time = time.perf_counter()
print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ Detection of land & water  ############
step2_time = time.perf_counter()
# d02
da_d02_LANDMASK = ds_d02['LANDMASK'].sel(Time=slice(1)).compute().squeeze()   # Land = 1, Water = 0
step1_time = time.perf_counter()
print('Landmask loaded \N{check mark}', step1_time-step2_time, 'seconds')


######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

start_time = time.perf_counter()

parent_dir_NCRF = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'

# Raw data
file_d02_sunrise = parent_dir_NCRF + '/raw/d02_sunrise'
file_d02_sunset = parent_dir_NCRF + '/raw/d02_sunset'
# 2-D data
file_d02_sunrise_RR = parent_dir_NCRF + '/L1/d02_sunrise_RR'				# [mm/dt]
file_d02_sunset_RR = parent_dir_NCRF + '/L1/d02_sunset_RR'				# [mm/dt]


######################################################################################
################ Declare the bounds you want to specifically look at #################
#### All the data 
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-02T12')]
lats = [-20, 20]
lons = [80, 135]

#### Some of the data
# times = [np.datetime64('2015-11-23T13'), np.datetime64('2015-12-02T12')]      # NCRF Sunset
times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-12-02T00')]        # NCRF Sunrise
# times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-11-25T02')]        # NCRF Sunrise Debugging
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-11-23T12')]
# times = [np.datetime64('2015-11-25T00'), np.datetime64('2015-11-26T12')]
# lats = [-7.5, 7.5]
# lons = [90, 115]
######################################################################################
# Setup the indicies that will be used throughout
time_ind_d02, lat_ind_d02, lon_ind_d02 = isel_ind(file_d02_sunrise, times, lats, lons)

# Raw datasets
ds_d02 = open_ds(file_d02_sunrise,time_ind_d02,lat_ind_d02,lon_ind_d02)
step1_time = time.perf_counter()
print('Dataset loaded \N{check mark}', step1_time-start_time, 'seconds')

# Coordinate dictionaries:
step2_time = time.perf_counter()
interp_P_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))

d02_coords = dict(
    XLAT=(('Time','south_north','west_east'),ds_d02.XLAT.values),
    XLONG=(('Time','south_north','west_east'),ds_d02.XLONG.values),
    bottom_top=(('bottom_top'),interp_P_levels),
    Time=('Time',ds_d02.XTIME.values),
    south_north=(('south_north'),ds_d02.XLAT[0,:,0].values),
    west_east=(('west_east'),ds_d02.XLONG[0,0,:].values)
    )

step1_time = time.perf_counter()
print('Created coordinate dictionaries \N{check mark}', step1_time-step2_time, 'seconds')


###########################################################
################# Load in the variables ###################
###########################################################


############ Rain Rate     [mm/hr] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_RR,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_RR = ds['RR'].compute()
da_d02_sunrise_RR = da_d02_sunrise_RR.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')


# time_bound = ['2015-11-23T02','2015-12-01T15']	# Does not include Dec 2, ends at Dec 1, 23:00 local time
time_bound = ['2015-11-23T02','2015-12-02T00']

## Select a slice of the times, and create a local time coordinate
RR_sunrise = da_d02_sunrise_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy()
RR_sunrise = assign_LT_coord(RR_sunrise, dim_num=3)	# Create a local time coordinte
RR_cntl = da_d02_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy()
RR_cntl = assign_LT_coord(RR_cntl, dim_num=3)		# Create a local time coordinte




# ## Number of itterations
# N = 500
# N_amplitude_RR_di_diff = np.zeros((N, RR_sunrise.shape[1], RR_sunrise.shape[2]))
# sig_bars = np.empty((2, RR_sunrise.shape[1], RR_sunrise.shape[2]))	# Significant bounds [low, high]

# for n in range(N):
# 	## Create a random set of indices with repeats for the time dimension
# 	rand_time_inds = np.random.randint(RR_sunrise.shape[0], size=RR_sunrise.shape[0])

# 	## Select random time indices, then diurnally composite
# 	RR_di_sunrise = xarray_reduce(RR_sunrise[rand_time_inds,...], 'LocalTime.hour', func='nanmean', dim='Time', expected_groups=(np.arange(0,24)), isbin=[False], fill_value=np.nan).transpose('hour','south_north','west_east')
# 	RR_di_cntl = xarray_reduce(RR_cntl[rand_time_inds,...], 'LocalTime.hour', func='nanmean', dim='Time', expected_groups=(np.arange(0,24)), isbin=[False], fill_value=np.nan).transpose('hour','south_north','west_east')

# 	## Find the amplitudes and their difference
# 	amplitude_RR_di_cntl = RR_di_cntl.max(dim='hour', keep_attrs=True, skipna=None) - RR_di_cntl.min(dim='hour', keep_attrs=True, skipna=None)
# 	amplitude_RR_di_sunrise = RR_di_sunrise.max(dim='hour', keep_attrs=True, skipna=None) - RR_di_sunrise.min(dim='hour', keep_attrs=True, skipna=None)
# 	amplitude_RR_di_diff = amplitude_RR_di_sunrise - amplitude_RR_di_cntl

# 	N_amplitude_RR_di_diff[n,...] = amplitude_RR_di_diff
# 	print(f'Iteration: {n}/{N}, Number of nan\'s: {np.sum(np.isnan(amplitude_RR_di_diff)).values}')

# ## 95% condifence intervals
# N_amplitude_RR_di_diff = np.sort(N_amplitude_RR_di_diff, axis=0)
# sig_bars[0,...] = N_amplitude_RR_di_diff[int(np.floor(.025*(N-1)))]
# sig_bars[1,...] = N_amplitude_RR_di_diff[int(np.ceil(.975*(N-1)))]

# ## Save as xarray.dataarray -> netcdf
# da_sig_bars = xr.DataArray(
# 	data=sig_bars,
# 	dims=['interval','south_north','west_east'],
# 	coords=dict(
# 		interval=np.array([0.025,0.975]),
# 		south_north=amplitude_RR_di_diff.coords['south_north'].values,
# 		west_east=amplitude_RR_di_diff.coords['west_east'].values
# 		),
# 	attrs=dict(
#         description="95% Confidence Intervals of the Spatial Diurnal Rain Rate Range Difference (NCRF-CNTL)",
#         units="mm/hr"
# 		),
# 	name='RR_range_diff_sig_bars'
# )

# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'
# file_name = parent_dir + '/L4/RR_di_range_diff_sig'
# da_sig_bars.to_netcdf(path=file_name, mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)


## Diurnal Composite of Domain Averaged Rain Rate

# Define the statistic to calculate (e.g., mean)
def mean_statistic(data):
    return np.nanmean(data, axis=0)

def bootstrap_di_composite(rain_rates, local_time, land_mask_3d, downselect_ratio, N):

	sig_bars = np.empty((2, 24))	# Significant bounds
	nbin = 0						# keeps track of the index of the bin through the loop

	for i in range(24):
		
		# Extract data from a specfic local time
		values = rain_rates[(local_time==i) & (land_mask_3d==True)]

		# Downselect
		values = np.random.choice(a=values, size=int(len(values)*downselect_ratio), replace=True)
		
		# Perform the bootstrap
		result = bootstrap(
			(values,),  # The data needs to be a tuple
			statistic=mean_statistic,
			confidence_level=0.95,	# 95% significance
			n_resamples=N,  # Number of bootstrap samples
			method='percentile',  # Use percentile method for CI
			# random_state=42  # For reproducibility
		)

		sig_bars[0,nbin] = result.confidence_interval.low		# Lower 2.5% significance
		sig_bars[1,nbin] = result.confidence_interval.high	# Upper 97.5% significance

		# print(nbin)
		nbin+=1
	return sig_bars

## Control
# Assign data
dataarray = RR_cntl.copy()
rain_rates = dataarray.values						# Rain rate values
land_mask_3d = np.repeat(np.expand_dims(da_d02_LANDMASK.values, axis=0), len(dataarray.Time), axis=0)
local_time = dataarray.LocalTime.dt.hour.values		# Only look at hours
N = 500				# Number of itterations
downselect_ratio = 1	# Ratio of total data selected when sig testing
# Net
sig_bars_cntl_net = bootstrap_di_composite(rain_rates, local_time, (land_mask_3d==0)|(land_mask_3d==1), downselect_ratio, N)
# Land
sig_bars_cntl_land = bootstrap_di_composite(rain_rates, local_time, (land_mask_3d==1), downselect_ratio, N)
# Ocean
sig_bars_cntl_ocean = bootstrap_di_composite(rain_rates, local_time, land_mask_3d==0, downselect_ratio, N)

## Sunrise
# Assign data
dataarray = RR_sunrise.copy()
rain_rates = dataarray.values						# Rain rate values

# Net
sig_bars_sunrise_net = bootstrap_di_composite(rain_rates, local_time, (land_mask_3d==0)|(land_mask_3d==1), downselect_ratio, N)
# Land
sig_bars_sunrise_land = bootstrap_di_composite(rain_rates, local_time, (land_mask_3d==1), downselect_ratio, N)
# Ocean
sig_bars_sunrise_ocean = bootstrap_di_composite(rain_rates, local_time, land_mask_3d==0, downselect_ratio, N)

## Save the significant bounds

# Control
file_name = parent_dir_CNTL + '/L4/RR_di_sig_bars_cntl'
da_sig_bars_cntl = xr.DataArray(
	data=np.stack((sig_bars_cntl_net,sig_bars_cntl_land,sig_bars_cntl_ocean),axis=2),
	dims=['interval','hours','domain'],
	coords=dict(
		interval = ('interval', np.array([0.025,0.975])),
		hours = ('hours', np.arange(0,24)),
		domain = ('domain', np.array(['net','land','ocean']))
	),
	attrs=dict(
        description="95% Confidence Intervals of the Diurnal Composite of Rain Rate over varying domains",
        units="mm/hr",
		N_itterations=f"{N}",
	),
	name='RR_domain_avg_sig_bars'
)
# Save file
da_sig_bars_cntl.to_netcdf(path=file_name, mode='w', format='NETCDF4', compute=True)

# NCRF
file_name = parent_dir_NCRF + '/L4/RR_di_sig_bars_sunrise'
da_sig_bars_sunrise = xr.DataArray(
	data=np.stack((sig_bars_sunrise_net,sig_bars_sunrise_land,sig_bars_sunrise_ocean),axis=2),
	dims=['interval','hours','domain'],
	coords=dict(
		interval = ('interval', np.array([0.025,0.975])),
		hours = ('hours', np.arange(0,24)),
		domain = ('domain', np.array(['net','land','ocean']))
	),
	attrs=dict(
        description="95% Confidence Intervals of the Diurnal Composite of Rain Rate over varying domains",
        units="mm/hr",
		N_itterations=f"{N}",
	),
	name='RR_domain_avg_sig_bars'
)
# Save file
da_sig_bars_sunrise.to_netcdf(path=file_name, mode='w', format='NETCDF4', compute=True)