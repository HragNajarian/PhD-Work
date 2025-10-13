from wrf import getvar, ALL_TIMES, default_fill
import glob
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import os
import time
from scipy.stats import bootstrap
import sys

## Function List:

def bin_by_rh(da_RH, da_path, RH_threshold, varname):

	# Load in 3-D Radiative Data
	da = load_variable(da_path, d02_coords, False)

	# Declare variables being returned
	binned_array = np.empty(len(RH_threshold)-1, dtype=float)
	# percentile_array = np.empty((len(RH_threshold)-1, 2), dtype=float)
	sig_bars = np.empty((len(RH_threshold)-1, 2), dtype=float)
	
	# Loop through the RH bins
	for i in range(len(binned_array)):
		# Create RH mask to only look between two RH values
		rh_mask = (da_RH >= RH_threshold[i]) & (da_RH < RH_threshold[i+1])

		# Bin and replace masked values with nans
		binned = xr.where(rh_mask, da, np.nan).values.flatten()

		# Remove masked values
		binned = binned[~np.isnan(binned)]

		# Bootstrap
		sig_bars[i,:] = bootstrap_values(binned, significance_percentile=significance_percentile, downselect_ratio=downselect_ratio, N=N)

		# Populate average binned into array
		binned_array[i] = np.nanmean(binned)

	da_binned = xr.DataArray(
		data=binned_array,
		name=varname,
		coords=dict(RH=RH_threshold[:-1]), 
		dims='RH')
	
	da_sig_bars = xr.DataArray(
		data=sig_bars,
		name='sig_bars',
		coords=dict(RH=RH_threshold[:-1], sig_bars=['0.05', '0.95']),
		dims=('RH', 'sig_bars'))
	
	print(f'Successfully calculated {varname} binned.')

	return da_binned, da_sig_bars


## Bootstrap
# Define the statistic to calculate (e.g., mean)
def mean_statistic(data):
	return np.nanmean(data, axis=0)

def bootstrap_values(values, significance_percentile:float, downselect_ratio:float, N:int):

	# Declare return variable 
	sig_bars = np.empty((2,1))	# Significant bounds [low, high]

	# Downselect
	values = np.random.choice(a=values, size=int(len(values)*downselect_ratio), replace=True)
	
	# Perform the bootstrap
	result = bootstrap(
		(values,),  # The data needs to be a tuple
		statistic=mean_statistic,
		confidence_level=significance_percentile,	# % significance
		n_resamples=N,  # Number of bootstrap samples
		method='percentile',  # Use percentile method for CI
		# random_state=42  # For reproducibility
	)

	# Populate Sig Bars
	sig_bars[0] = result.confidence_interval.low		# Lower % significance
	sig_bars[1] = result.confidence_interval.high		# Upper % significance

	return sig_bars.flatten()


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


def load_variable(file_dict: dict, coords: dict, is_diff: bool) -> xr.DataArray:
	step_start = time.perf_counter()
	# If its a difference of already sliced datasets, don't slice further
	if is_diff:
		# ds = open_ds(file_dict['path'], [0,-1], lat_ind_d02, lon_ind_d02)
		ds = open_ds(file_dict['path'], [0,-1], [0,-1], [0,-1])
	else:
		ds = open_ds(file_dict['path'], time_ind_d02, lat_ind_d02, lon_ind_d02)
		
	da = ds[file_dict['varname']].compute()
	da = da.assign_coords(coords)
	# Replace fill value with nan
	da = da.where(da != default_fill(np.float32))
	print(f"{file_dict['description']} loaded ✓", time.perf_counter() - step_start, "seconds")
	return da


# Function that can removes the bottom_top dimension for 2-D datasets
def without_keys(d, keys):
	return {x: d[x] for x in d if x not in keys}


#########################################################################################################################################################
## User Inputs #### User Inputs #### User Inputs #### User Inputs #### User Inputs #### User Inputs #### User Inputs #### User Inputs #### User Inputs ##
#########################################################################################################################################################

## Select Parent Directory
parent_dir = sys.argv[1]
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00'

## Select thresholds
# times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-11-23T02')] # Debug
times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-12-02T00')]
lats = [-10, 10]
lons = [80, 135]

## Select RH Bins
RH_min_max_dt = np.array([0,100.01,2.5])
RH_threshold = np.arange(RH_min_max_dt[0],RH_min_max_dt[1]+RH_min_max_dt[2],RH_min_max_dt[2])

## Bootstrapping parameters
significance_percentile=0.95
downselect_ratio=.1
N=50

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################


## Create Dictionary with file names, units, description.
	# Interpolated 3-D data 
l2_files = {
	'd02_interp_LWAll':     ('K/s',     'LWAll',     'Interpolated longwave heating (all-sky)'),
	'd02_interp_LWClear':   ('K/s',     'LWClear',   'Interpolated longwave heating (clear-sky)'),
	'd02_interp_SWAll':     ('K/s',     'SWAll',     'Interpolated shortwave heating (all-sky)'),
	'd02_interp_SWClear':   ('K/s',     'SWClear',   'Interpolated shortwave heating (clear-sky)'),
	'd02_interp_RH':        ('%',       'RH',        'Relative humidity'),
}
# Build structured dictionary
files = {
	'L2': {
		var: {
			"path": build_path(parent_dir, 'L2', var),
			"unit": unit,
			"varname": varname,
			"description": desc
		}
		for var, (unit, varname, desc) in l2_files.items()
	}
}


time_ind_d02, lat_ind_d02, lon_ind_d02 = isel_ind(build_path(parent_dir, "raw", "d02"), times, lats, lons)


## Load in Coordinates
file_path_coords = build_path(parent_dir,'raw','d02_coords')
ds_coords = xr.open_dataset(file_path_coords, chunks='auto').isel(south_north=slice(None), west_east=slice(None))
d02_coords = {
	var: (ds_coords[var].dims, ds_coords[var].values)
	for var in ds_coords.coords}


## Load in 3-D RH Data
da_d02_RH = load_variable(files['L2']['d02_interp_RH'], d02_coords, False)
counts = np.histogram(da_d02_RH.values.flatten(), bins=RH_threshold[:-1])[0]
da_counts = xr.DataArray(
	data=np.append(counts,0),	# Match the len of dim 'RH'
	name='RH_count',
	coords=dict(RH=RH_threshold[:-1]), 
	dims='RH')

## Start Binning
LWAll_binned, LWAll_sig_bars = bin_by_rh(da_d02_RH, files['L2']['d02_interp_LWAll'], RH_threshold, varname='LWAll')
LWClear_binned, LWClear_sig_bars = bin_by_rh(da_d02_RH, files['L2']['d02_interp_LWClear'], RH_threshold, varname='LWClear')
SWAll_binned, SWAll_sig_bars = bin_by_rh(da_d02_RH, files['L2']['d02_interp_SWAll'], RH_threshold, varname='SWAll')
SWClear_binned, SWClear_sig_bars = bin_by_rh(da_d02_RH, files['L2']['d02_interp_SWClear'], RH_threshold, varname='SWClear')
# TotalAll_binned, TotalAll_percentiles = bin_by_rh(da_d02_RH, da_d02_TotalAll, RH_threshold, varname='TotalAll')
# TotalClear_binned, TotalClear_percentiles = bin_by_rh(da_d02_RH, da_d02_TotalClear, RH_threshold, varname='TotalClear')


# Combine all output into a dataset
ds = xr.Dataset({
	"RH_count":da_counts,
	"LWAll": LWAll_binned,
	"LWAll_sig_bars": LWAll_sig_bars,
	"LWClear": LWClear_binned,
	"LWClear_sig_bars": LWClear_sig_bars,
	"SWAll": SWAll_binned,
	"SWAll_sig_bars": SWAll_sig_bars,
	"SWClear": SWClear_binned,
	"SWClear_sig_bars": SWClear_sig_bars,
	# "TotalClear": TotalClear_binned,
	# "TotalClear_percentile": TotalClear_percentiles,
	# "TotalAll": TotalAll_binned,
	# "TotalAll_percentile": TotalAll_percentiles,
})

## Save File
out_path = f'{parent_dir}/L4/rh_binned_radiative_heating_sunrise' if 'CRFoff' in parent_dir else f'{parent_dir}/L4/rh_binned_radiative_heating_ctrl'
ds.to_netcdf(path=out_path, mode='w', format='NETCDF4', compute=True) #  ,unlimited_dims='Time'