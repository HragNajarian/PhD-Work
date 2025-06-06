#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Run this command on the command line to create a .py script instead of .ipynb
	# jupyter nbconvert cross_section_variable.ipynb --to python


# In[1]:


# Purpose: Read variables and create cross sections of the variable. 

# Input:
    # This script is made out of many functions, but its strucutre is simple. It goes variable by variable, reading
		# in the .nc file, calculating a cross-section, and then saving that cross-section.
	# Important lessons developing this code is:
		# You will run out of memory if you do not delete variables as you move through the script.
		# Memory is a big issue and so you also have to go variable by variable rather than loading everything in first, and then
			# calculating your cross-sections.

##############################################################################

import numpy as np
import xarray as xr
import wrf
from math import cos, asin, sqrt, pi, atan
import time
import os
import sys
import glob

##############################################################################


# ### Pre-requisite Functions 

# In[2]:


# Assumes cartesian coordinate system
def calculate_angle_between_points(p1, p2):
    # Calculate differences
    dy = p2[0] - p1[0]  # Lats
    dx = p2[1] - p1[1]  # Lons
    # Find the angle (radians)
    theta = atan(dy/dx)
    # Theta will be:
        # Negative if NW or SE direction
        # Positive if NE or SW direction
    
    return theta

# start_coord		= [-1.8,103.8]
# end_coord 		= [-5.8,99.8]
# degrees(calculate_angle_between_points(start_coord, end_coord))


# In[3]:


# Purpose: To grab the indicies that correspond to the times, latitudes, and longitudes of the WRF dataset file.

# Input:
	# file == path to the .nc file
	# times == np.datetime64 array [start_coord,end_coord]
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


# In[4]:


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
	ds = xr.open_dataset(file).isel(
		Time=slice(time_ind[0],time_ind[1]),
		south_north=slice(lat_ind[0],lat_ind[1]),
		west_east=slice(lon_ind[0],lon_ind[1])
	)
	return ds


# In[5]:


# This function finds the distance [km] between two coordinates in lat & lon
def dist(lat1, lon1, lat2, lon2):
    r = 6371 # km
    p = pi / 180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 2 * r * asin(sqrt(a))


# In[6]:


# Purpose: Rotate a vectorized variable like a wind vector in x/lon and y/lat space!
	# See Lecture02_Vector.pdf in METR 5113 Advanced Atmospheric Dynamics folder

# Input:
    # da_x  = 		xr.data_array		 	# Should be the X component of the vector
    # da_y  = 		xr.data_array			# Should be the Y component of the vector
		# Can be one time step or pressure level or even a 4-D variable [Time x Height x Lat x Lon]
    # theta =		rotation in radians		
		# Must be -pi/2 < theta < pi/2, you don't need to rotate any more than the 180° provided
		# Positive theta indiates anticlockwise turn of coordinates
		# Negative theta indiates clockwise turn of coordinates
# Output:
    # da_x_rot:		da_x but rotated
	# da_y_rot:		da_y but rotated
# Process:
    # Check if theta is positive or negative
		# Based on the value, the projections using theta will change
	# Project the current x and y coordinate to the new rotated x and y coordinates for each component.
	# Once rotated, return the rotated x and y components
# Example:
	# da_x = da_d01_U
	# da_y = da_d01_V
	# theta = pi/4	# 45° rotation anti-clockwise
	# da_d01_U_rotated, da_d01_V_rotated = rotate_vec(da_x, da_y, theta)

def rotate_vec(da_x, da_y, theta):
	# anti-clockwise rotation
	if (theta > 0) & (theta<pi/2):
		da_x_rot = da_x*cos(theta) + da_y*cos((pi/2)-theta)
		da_y_rot = -da_x*cos((pi/2)-theta) + da_y*cos(theta)
	# clockwise rotation
	if (theta < 0) & (theta>(-pi/2)):
		da_x_rot = da_x*cos(-theta) - da_y*cos((pi/2)+theta)
		da_y_rot = da_x*cos((pi/2)+theta) + da_y*cos(-theta)

	return da_x_rot, da_y_rot


# In[7]:


def make_da_cross(da, da_cross_temp, var_name, distance, width, all_line_coords):
	
	if 'bottom_top' in da.dims:		# If the da is 3-D
		da_cross = xr.DataArray(
		data=da_cross_temp,
		name=var_name,
		dims=('Time','bottom_top','Distance','Spread'),
		coords={'Time':da['XTIME'].values,	# da is the da that was used to create da_cross_temp
			'bottom_top':da['bottom_top'].values,
			'Distance':distance,
			'Spread':np.arange(width/2,-width/2,-dx),
			'Lat': (('Distance','Spread'), all_line_coords[0,:,:]),
			'Lon': (('Distance','Spread'), all_line_coords[1,:,:])}
		)
	else:  							# If the da is 2-D
		da_cross = xr.DataArray(
		data=da_cross_temp,
		name=var_name,
		dims=('Time','Distance','Spread'),
		coords={'Time':da['XTIME'].values,	# da is the da that was used to create da_cross_temp
			'Distance':distance,
			'Spread':np.arange(width/2,-width/2,-dx),
			'Lat': (('Distance','Spread'), all_line_coords[0,:,:]),
			'Lon': (('Distance','Spread'), all_line_coords[1,:,:])}
		)
	return da_cross


# In[8]:


# Purpose: Create an array with multiple cross-sectional data from WRFoutput.
# Input:
    # da = 			xr.data_array		 	works with both 2-D and 3-D variables!
    # start_coord = [latitude, longitude] 
    # end_coord = 	[latitude, longitude]
	# width = 		spread of cross-section in degrees i.e., 0.75° = 0.75
	# dx = 			distance between each cross-sectional line i.e., 0.05° = 0.05
# Output:
    # da_cross: 	matrix in time, height, distance, and # of lines
	# 					or time, distance, and # of lines if using a 2-D variable
# Process:
	# Make sure you 'da' have assigned coordinates corresponding to south_north and west_east.
    # We first create a main line between start_coord -> end_coord. This line will be the center
		# line for all other lines to sit next to.
	# Depending on the angle of the line (more latitudinal change or longitudinal change), this
		# function will account for that and make the longest side the length of the cross-sectional line
	# We then create an empty np.array, calculate the change in spread needed, then start_coord filling in the data.

def cross_section_multi(da, start_coord, end_coord, width, dx):

	# We want to first create a line between start_coord and end_coords
		# Gather the indicies of the closest gridboxes of start_coord and end_coords.
	start_ilat = int((abs((da.XLAT[0,:,0]) - (start_coord[0]))).argmin())
	end_ilat = int((abs((da.XLAT[0,:,0]) - (end_coord[0]))).argmin())
	start_ilon = int((abs((da.XLONG[0,0,:]) - (start_coord[1]))).argmin())
	end_ilon = int((abs((da.XLONG[0,0,:]) - (end_coord[1]))).argmin())
	# This statement ensures that the length of the line will be the size of the LONGEST side (either lat or lon)
	if abs(start_ilat-end_ilat)>=abs(start_ilon-end_ilon):
		line_coords = np.zeros([2,abs(start_ilat-end_ilat)])
		# Populate latitudes
		line_coords[0,:] = np.linspace(da.XLAT[0,start_ilat,0], da.XLAT[0,end_ilat,0], abs(start_ilat-end_ilat))
		# Populate longitudes
		line_coords[1,:] = np.linspace(da.XLONG[0,0,start_ilon], da.XLONG[0,0,end_ilon], abs(start_ilat-end_ilat))
	else:
		line_coords = np.zeros([2,abs(start_ilon-end_ilon)])
		# Populate latitudes
		line_coords[0,:] = np.linspace(da.XLAT[0,start_ilat,0], da.XLAT[0,end_ilat,0], abs(start_ilon-end_ilon))
		# Populate longitudes
		line_coords[1,:] = np.linspace(da.XLONG[0,0,start_ilon], da.XLONG[0,0,end_ilon], abs(start_ilon-end_ilon))

	##########################################################################################################################
	# Now that we have the coordinates between the start_coord and end_coords, we need to replicate it for all the lines
	num_lines = int(width/dx)
	spread = np.arange(width/2,-width/2,-dx)
	if 'bottom_top' in da.dims:	# If the dataset is 3-D
		#						TIME	x	HEIGHT	x	   DISTANCE	   x   #ofLINES
		da_cross = np.zeros([da.shape[0],da.shape[1],line_coords.shape[1],num_lines])
	else:
		#						TIME	x	   DISTANCE	   x   #ofLINES
		da_cross = np.zeros([da.shape[0],line_coords.shape[1],num_lines])

	# Create all_line_coords that holds all the coordinates for every line produced
	all_line_coords = np.zeros([line_coords.shape[0],line_coords.shape[1],spread.shape[0]])

	# Looping over all the lines
	for i in range(len(spread)):

		if (end_coord[0] > start_coord[0]):
			all_line_coords[0,:,i] = line_coords[0,:]+spread[i]
		else:
			all_line_coords[0,:,i] = line_coords[0,:]-spread[i]
		if (end_coord[1] > start_coord[1]):
			all_line_coords[1,:,i] = line_coords[1,:]-spread[i]
		else:
			all_line_coords[1,:,i] = line_coords[1,:]+spread[i]

		# Now that we have our lines, we can interpolate the dataset with the offset for each line applied
		da_interp = da.interp(south_north=all_line_coords[0,:,i], west_east=all_line_coords[1,:,i], method="linear")

		# Populate the new data array with data from the cross section
			# Loop through the length of the line, find the match, and then populate it.
		if 'bottom_top' in da.dims:	# If the dataset is 3-D
			for j in range(da_cross.shape[2]):
				data = da_interp.sel(
					south_north = da_interp.south_north[j],
					west_east = da_interp.west_east[j])
				da_cross[:,:,j,i] = data
		else:
			for j in range(da_cross.shape[1]):
				data = da_interp.sel(
					south_north = da_interp.south_north[j],
					west_east = da_interp.west_east[j])
				da_cross[:,j,i] = data

	return da_cross, all_line_coords


# In[9]:


# Function that can removes the bottom_top dimension for 2-D datasets
def without_keys(d, keys):
	return {x: d[x] for x in d if x not in keys}


# ## Main Code
# #### This code is designed to read in the variable, create a cross-section of that variable, save the cross-section as an .nc file, then delete the variable. Originally, reading all the variables first and then saving files was attempted, but the system runs out of memory quite quickly.

# In[46]:


##cd into the appropriate directory (L3) and then assign a parent directory
parent_dir = sys.argv[1]
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00/CRFoff'

times = [np.datetime64('2015-12-09T12'), np.datetime64('2015-12-20T12')]
# times = [np.datetime64('2015-12-10T01'), np.datetime64('2015-12-10T03')]

## Assign the correct location directory depending on where you're trying to cross-section
local_dirs = ['/L3/Sumatra_mid_central','/L3/Sumatra_northwest','/L3/Borneo_northwest']

## Define cross-section settings
region_settings = {
    'Sumatra_mid_central': {
        'start_coord': [-2.2, 103.2],
        'end_coord': [-6.7, 98.7],
        'width': 1.5,
        'dx': 0.025
    },
    'Sumatra_northwest': {
        'start_coord': [5.2, 96.4],
        'end_coord': [1.2, 92.4],
        'width': 1.5,
        'dx': 0.025
    },
    'Borneo_northwest': {
        'start_coord': [1.2, 112.8],
        'end_coord': [5.9, 108.1],
        'width': 1.5,
        'dx': 0.025
    }
}

#### NOT WORKING, KEEP '' OR '_sunrise' for now
## starter string to diseminate between experiments
    # icloud=0 at sunrise, starter_str='_sunrise'
    # icloud=1, starter_str=''
    # icloud=depends, starter_str='_oceanoff'
starter_str = '_sunrise'

# Declare variables to interpolate (they must exist in 'l1_files' or 'l2_files')
    # variables_to_process = [
    #     # L1 varnames
    #     'RR', 'U10', 'V10', 'PSFC', 'T2', 'HFX', 'QFX', 'LH', 'CAPE', 'CIN',
    #     'LWDNB', 'LWUPB', 'LWDNBC', 'LWUPBC', 'SWDNB', 'SWUPB', 'SWDNBC', 'SWUPBC',
    #     'LWDNT', 'LWUPT', 'LWDNTC', 'LWUPTC', 'SWDNT', 'SWUPT', 'SWDNTC', 'SWUPTC',
    #     # L2 varnames
    #     'U', 'V', 'W', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'H_DIABATIC',
    #     'Theta', 'LWAll', 'LWClear', 'SWAll', 'SWClear'
    # ]

variables_to_process = [
    # L1 varnames
    'RR', 'U10', 'V10', 'PSFC', 'T2', 'HFX', 'QFX', 'LH', 'CAPE', 'CIN',
    'LWDNB', 'LWUPB', 'LWDNBC', 'LWUPBC', 'SWDNB', 'SWUPB', 'SWDNBC', 'SWUPBC',
    'LWDNT', 'LWUPT', 'LWDNTC', 'LWUPTC', 'SWDNT', 'SWUPT', 'SWDNTC', 'SWUPTC',
    # L2 varnames
    'U', 'V', 'LWAll', 'LWClear', 'SWAll', 'SWClear'
]
# variables_to_process = ['U','V']

###############################################################################################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ USER INPUTS ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
###############################################################################################################

## Dictionary containing all variables supported:
    # 2-D data
l1_files = {
    f'd02{starter_str}_RR':       ('mm/dt',   'RR',      'Rain rate'),
    f'd02{starter_str}_U10':      ('m/s',     'U10',     '10-meter U-wind'),
    f'd02{starter_str}_V10':      ('m/s',     'V10',     '10-meter V-wind'),
    f'd02{starter_str}_PSFC':     ('hPa',     'PSFC',    'Surface pressure'),
    f'd02{starter_str}_T2':       ('K',       'T2',      '2-meter temperature'),
    f'd02{starter_str}_HFX':      ('W/m^2',   'HFX',     'Surface sensible heat flux'),
    f'd02{starter_str}_QFX':      ('W/m^2',   'QFX',     'Surface moisture flux'),
    f'd02{starter_str}_LH':       ('W/m^2',   'LH',      'Surface latent heat flux'),
    f'd02{starter_str}_CAPE':     ('J/kg',    'CAPE',    'Convective available potential energy'),
    f'd02{starter_str}_CIN':      ('J/kg',    'CIN',     'Convective Inhibition'),
    # Rad vars
    f'd02{starter_str}_LWDNB':    ('W/m^2',   'LWDNB',   'Downward longwave radiation at SFC'),
    f'd02{starter_str}_LWUPB':    ('W/m^2',   'LWUPB',   'Upward longwave radiation at SFC'),
    f'd02{starter_str}_LWDNBC':   ('W/m^2',   'LWDNBC',  'Clear-sky downward longwave radiation at SFC'),
    f'd02{starter_str}_LWUPBC':   ('W/m^2',   'LWUPBC',  'Clear-sky upward longwave radiation at SFC'),
    f'd02{starter_str}_SWDNB':    ('W/m^2',   'SWDNB',   'Downward shortwave radiation at SFC'),
    f'd02{starter_str}_SWUPB':    ('W/m^2',   'SWUPB',   'Upward shortwave radiation at SFC'),
    f'd02{starter_str}_SWDNBC':   ('W/m^2',   'SWDNBC',  'Clear-sky downward SW radiation at SFC'),
    f'd02{starter_str}_SWUPBC':   ('W/m^2',   'SWUPBC',  'Clear-sky upward SW radiation at SFC'),
    f'd02{starter_str}_LWDNT':    ('W/m^2',   'LWDNT',   'Downward longwave radiation at TOA'),
    f'd02{starter_str}_LWUPT':    ('W/m^2',   'LWUPT',   'Upward longwave radiation at TOA'),
    f'd02{starter_str}_LWDNTC':   ('W/m^2',   'LWDNTC',  'Clear-sky downward longwave radiation at TOA'),
    f'd02{starter_str}_LWUPTC':   ('W/m^2',   'LWUPTC',  'Clear-sky upward longwave radiation at TOA'),
    f'd02{starter_str}_SWDNT':    ('W/m^2',   'SWDNT',   'Downward shortwave radiation at TOA'),
    f'd02{starter_str}_SWUPT':    ('W/m^2',   'SWUPT',   'Upward shortwave radiation at TOA'),
    f'd02{starter_str}_SWDNTC':   ('W/m^2',   'SWDNTC',  'Clear-sky downward SW radiation at TOA'),
    f'd02{starter_str}_SWUPTC':   ('W/m^2',   'SWUPTC',  'Clear-sky upward SW radiation at TOA'),
}
    # Interpolated 3-D data
l2_files = {
    f'd02{starter_str}_interp_U':         ('m/s',     'U',         'Interpolated U-wind'),
    f'd02{starter_str}_interp_V':         ('m/s',     'V',         'Interpolated V-wind'),
    f'd02{starter_str}_interp_W':         ('m/s',     'W',         'Interpolated W-wind'),
    f'd02{starter_str}_interp_QV':        ('kg/kg',   'QV',        'Interpolated specific humidity'),
    f'd02{starter_str}_interp_QC':        ('kg/kg',   'QC',        'Interpolated cloud water'),
    f'd02{starter_str}_interp_QR':        ('kg/kg',   'QR',        'Interpolated rain water'),
    f'd02{starter_str}_interp_QI':        ('kg/kg',   'QI',        'Interpolated ice mixing ratio'),
    f'd02{starter_str}_interp_QS':        ('kg/kg',   'QS',        'Interpolated snow mixing ratio'),
    f'd02{starter_str}_interp_QG':        ('kg/kg',   'QG',        'Interpolated Graupel mixing ratio'),
    f'd02{starter_str}_interp_CLDFRA':    ('0-1',     'CLDFRA',    'Interpolated cloud fraction'),
    f'd02{starter_str}_interp_H_DIABATIC':('K/s',     'H_DIABATIC','Interpolated diabatic heating'),
    f'd02{starter_str}_interp_Theta':     ('K',       'Theta',     'Interpolated potential temperature'),
    # Rad vars
    f'd02{starter_str}_interp_LWAll':     ('K/s',     'LWAll',     'Interpolated longwave heating (all-sky)'),
    f'd02{starter_str}_interp_LWClear':   ('K/s',     'LWClear',   'Interpolated longwave heating (clear-sky)'),
    f'd02{starter_str}_interp_SWAll':     ('K/s',     'SWAll',     'Interpolated shortwave heating (all-sky)'),
    f'd02{starter_str}_interp_SWClear':   ('K/s',     'SWClear',   'Interpolated shortwave heating (clear-sky)'),
}


## Build structured dictionary
def build_path(parent_dir: str, sub_dir: str, name: str) -> str:
    return f"{parent_dir}/{sub_dir}/{name}"

files = {
    'L1': {
        var: {
            "path": build_path(parent_dir, 'L1', var),
            "unit": unit,
            "varname": varname,
            "description": desc
        }
        # Only include variables from variables_to_process
        for var, (unit, varname, desc) in l1_files.items()
        if varname in variables_to_process
    },
    'L2': {
        var: {
            "path": build_path(parent_dir, 'L2', var),
            "unit": unit,
            "varname": varname,
            "description": desc
        }
        for var, (unit, varname, desc) in l2_files.items()
        if varname in variables_to_process
    }
}


# In[ ]:


## Helper to load and cross-section a variable
def process_variable(var_name, file_path, coords, save_name):
    step_start = time.perf_counter()

    ## Open, slice, and extract variable from dataset
    ds = open_ds(file_path, time_inds, lat_inds, lon_inds)
    da = ds[var_name].compute()

    ## If 3-D, keep bottom_top coordinate, else remove it when assigning coords
    if 'bottom_top' in da.dims:
        da = da.assign_coords(coords).where(lambda x: x != fill_value_f8)
    else:
        da = da.assign_coords(without_keys(coords,'bottom_top')).where(lambda x: x != fill_value_f8)

    ## 
    da_cross_temp, all_line_coords = cross_section_multi(da, start_coord, end_coord, width, dx)
    da_out = make_da_cross(da, da_cross_temp, var_name, distance, width, all_line_coords)
    da_out.to_netcdf(f'./{save_name}')

    print(f'{var_name} saved ✔ ', round(time.perf_counter() - step_start, 2), ' seconds')

    return


## Calculate Normal Wind
def process_normalwind():
    step_start = time.perf_counter()

    ds_U = open_ds(files['L2'][f'd02{starter_str}_interp_U']['path'], time_inds, lat_inds, lon_inds)
    da_U = ds_U['U'].compute().assign_coords(coords).where(lambda x: x != fill_value_f8)

    ds_V = open_ds(files['L2'][f'd02{starter_str}_interp_V']['path'], time_inds, lat_inds, lon_inds)
    da_V = ds_V['V'].compute().assign_coords(coords).where(lambda x: x != fill_value_f8)

    da_NormalWind, _ = rotate_vec(da_U, da_V, theta)
    da_cross_temp, all_line_coords = cross_section_multi(da_NormalWind, start_coord, end_coord, width, dx)
    da_NormalWind = make_da_cross(da_U, da_cross_temp, 'NormalWind', distance, width, all_line_coords)
    da_NormalWind.to_netcdf(f'./d02{starter_str}_cross_NormalWind')

    print(f'Normal Wind saved ✔ ', round(time.perf_counter() - step_start, 2), ' seconds')

    return


## Calculate Normal Wind at the Surface
def process_surfacenormalwind():
    step_start = time.perf_counter()

    ds_U10 = open_ds(files['L1'][f'd02{starter_str}_U10']['path'], time_inds, lat_inds, lon_inds)
    da_U10 = ds_U10['U10'].compute().assign_coords(without_keys(coords,'bottom_top')).where(lambda x: x != fill_value_f8)

    ds_V10 = open_ds(files['L1'][f'd02{starter_str}_V10']['path'], time_inds, lat_inds, lon_inds)
    da_V10 = ds_V10['V10'].compute().assign_coords(without_keys(coords,'bottom_top')).where(lambda x: x != fill_value_f8)

    da_SurfaceNormalWind, _ = rotate_vec(da_U10, da_V10, theta)
    da_cross_temp, all_line_coords = cross_section_multi(da_SurfaceNormalWind, start_coord, end_coord, width, dx)
    da_SurfaceNormalWind = make_da_cross(da_U10, da_cross_temp, 'NormalWind', distance, width, all_line_coords)
    da_SurfaceNormalWind.to_netcdf(f'./d02{starter_str}_cross_SurfaceNormalWind')

    print(f'Normal Wind saved ✔ ', round(time.perf_counter() - step_start, 2), ' seconds')

    return


## Calculate 3-D Radiative Variables
def process_CRF():
    step_start = time.perf_counter()

    ## Longwave
    ds_LWAll = open_ds(files['L2'][f'd02{starter_str}_interp_LWAll']['path'], time_inds, lat_inds, lon_inds)
    da_LWAll = ds_LWAll['LWAll'].compute().assign_coords(coords).where(lambda x: x != fill_value_f8)
    da_cross_temp, all_line_coords = cross_section_multi(da_LWAll, start_coord, end_coord, width, dx)
    da_cross_LWAll = make_da_cross(da_LWAll, da_cross_temp, 'LWAll', distance, width, all_line_coords)
    da_cross_LWAll.to_netcdf(f'./d02{starter_str}_cross_LWAll')

    ds_LWClear = open_ds(files['L2'][f'd02{starter_str}_interp_LWClear']['path'], time_inds, lat_inds, lon_inds)
    da_LWClear = ds_LWClear['LWClear'].compute().assign_coords(coords).where(lambda x: x != fill_value_f8)
    da_cross_temp, all_line_coords = cross_section_multi(da_LWClear, start_coord, end_coord, width, dx)
    da_cross_LWClear = make_da_cross(da_LWClear, da_cross_temp, 'LWClear', distance, width, all_line_coords)
    da_cross_LWClear.to_netcdf(f'./d02{starter_str}_cross_LWClear')

    ## Shortwave
    ds_SWAll = open_ds(files['L2'][f'd02{starter_str}_interp_SWAll']['path'], time_inds, lat_inds, lon_inds)
    da_SWAll = ds_SWAll['SWAll'].compute().assign_coords(coords).where(lambda x: x != fill_value_f8)
    da_cross_temp, all_line_coords = cross_section_multi(da_SWAll, start_coord, end_coord, width, dx)
    da_cross_SWAll = make_da_cross(da_SWAll, da_cross_temp, 'SWAll', distance, width, all_line_coords)
    da_cross_SWAll.to_netcdf(f'./d02{starter_str}_cross_SWAll')

    ds_SWClear = open_ds(files['L2'][f'd02{starter_str}_interp_SWClear']['path'], time_inds, lat_inds, lon_inds)
    da_SWClear = ds_SWClear['SWClear'].compute().assign_coords(coords).where(lambda x: x != fill_value_f8)
    da_cross_temp, all_line_coords = cross_section_multi(da_SWClear, start_coord, end_coord, width, dx)
    da_cross_SWClear = make_da_cross(da_SWClear, da_cross_temp, 'SWClear', distance, width, all_line_coords)
    da_cross_SWClear.to_netcdf(f'./d02{starter_str}_cross_SWClear')

    ## CRF Calculations:
    da_cross_LWCRF = da_cross_LWAll - da_cross_LWClear
    da_cross_LWCRF = xr.DataArray(data=da_cross_LWCRF, name='LWCRF')
    da_cross_LWCRF.to_netcdf(f'./d02{starter_str}_cross_LWCRF')

    da_cross_SWCRF = da_cross_SWAll - da_cross_SWClear
    da_cross_SWCRF = xr.DataArray(data=da_cross_SWCRF, name='SWCRF')
    da_cross_SWCRF.to_netcdf(f'./d02{starter_str}_cross_SWCRF')

    da_cross_TotalCRF = da_cross_LWCRF - da_cross_SWCRF
    da_cross_TotalCRF = xr.DataArray(data=da_cross_TotalCRF, name='TotalCRF')
    da_cross_TotalCRF.to_netcdf(f'./d02{starter_str}_cross_TotalCRF')

    print(f'3-D Radiative saved ✔ ', round(time.perf_counter() - step_start, 2), ' seconds')

    return


## Loop through the regions you want to cross section
for loc_dir in local_dirs:

    ## Change directory
    os.chdir(parent_dir+loc_dir)
    print(f"You're in directory {parent_dir+loc_dir}")
    region_key = loc_dir.split('/')[-1]
    
    ## Assign parameters based on region
    params = region_settings[region_key]
    start_coord, end_coord = params['start_coord'], params['end_coord']
    width, dx = params['width'], params['dx']
    theta = calculate_angle_between_points(start_coord, end_coord)

    ## Prepare slicing indicies
    lats = [min(start_coord[0], end_coord[0]) - width, max(start_coord[0], end_coord[0]) + width]
    lons = [min(start_coord[1], end_coord[1]) - width, max(start_coord[1], end_coord[1]) + width]
    time_inds, lat_inds, lon_inds = isel_ind(build_path(parent_dir, 'raw', f'd02{starter_str}'), times, lats, lons)
    
    ## Open main dataset and set coordinates
    t0 = time.perf_counter()
    ds_main = open_ds(build_path(parent_dir, 'raw', f'd02{starter_str}'), time_inds, lat_inds, lon_inds)
    print('Main dataset loaded ✔', round(time.perf_counter() - t0, 2), 'seconds')

    ## Determine the pressure levels needed
    interp_P_levels_1D = np.concatenate((np.arange(1000, 950, -10), np.arange(950, 350, -30), np.arange(350, 0, -50)))
    interp_P_levels = np.swapaxes(np.resize(interp_P_levels_1D, (ds_main.XLAT.shape[2], ds_main.XLAT.shape[1], len(interp_P_levels_1D))), 0, 2)

    ## Setup Coordinates
    coords = dict(
        XLAT=(('Time','south_north','west_east'),ds_main.XLAT.values),
        XLONG=(('Time','south_north','west_east'),ds_main.XLONG.values),
        bottom_top=(('bottom_top'),interp_P_levels_1D),
        XTIME=('Time',ds_main.XTIME.values),
        south_north=(('south_north'),ds_main.XLAT[0,:,0].values),
        west_east=(('west_east'),ds_main.XLONG[0,0,:].values)
        )

    ## Coast detection
    da_LANDMASK = ds_main['LANDMASK'].compute().assign_coords(without_keys(coords,'bottom_top'))
    da_cross_LANDMASK, all_line_coords = cross_section_multi(da_LANDMASK, start_coord, end_coord, width, dx)
    distance = np.linspace(0, dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]), da_cross_LANDMASK.shape[1])
    mid_idx = da_cross_LANDMASK.shape[2] // 2
    coast_idx = np.where(da_cross_LANDMASK[0, :, mid_idx] != da_cross_LANDMASK[0, 0, mid_idx])[0][0]
    distance -= distance[coast_idx]

    ## Determine fill_value
    fill_value_f8 = wrf.default_fill(np.float32)


    ## Loop through all variables
    for dir, file in files.items():
        for filename, inner_dict in file.items():

            # Declare the variable
            varname = inner_dict['varname']

            ## Process Normal Wind
            if varname in ('U','V'):
                # Special case where two variables need to be loaded in to calculate normal wind
                if f'd02{starter_str}_cross_NormalWind' not in glob.glob('*'):
                    # If statement makes sure it doesn't calculate it multiple times
                    print(f'Variable name: {varname}')
                    process_normalwind()
                else:
                    print(f'd02{starter_str}_cross_NormalWind already saved ✔ ')

            ## Process Surface Normal Wind
            elif varname in ('U10','V10'):
                # Special case where two variables need to be loaded in to calculate normal wind
                if f'd02{starter_str}_cross_SurfaceNormalWind' not in glob.glob('*'):
                    # If statement makes sure it doesn't calculate it multiple times
                    print(f'Variable name: {varname}')
                    process_surfacenormalwind()
                else:
                    print(f'd02{starter_str}_cross_SurfaceNormalWind already saved ✔ ')

            ## Process 3-D rad vars + CRF calculations
            elif varname in ('LWAll','LWClear','SWAll','SWClear'):
                # Special case to calculate cloud-radaitive forcing
                if f'd02{starter_str}_cross_TotalCRF' not in glob.glob('*'):
                    # If statement makes sure it doesn't calculate it multiple times
                    print(f'Variable name: {varname}')
                    process_CRF()
                else:
                    print(f'd02{starter_str}_cross_TotalCRF already saved ✔ ')

            ## All other variables
            else:
                if f'd02{starter_str}_cross_{varname}' not in glob.glob('*'):
                    # If statement makes sure it doesn't calculate it multiple times
                    print(f'Variable name: {varname}')
                    process_variable(varname, inner_dict['path'], coords, f'd02{starter_str}_cross_{varname}')
                else:
                    print(f'd02{starter_str}_cross_{varname} already saved ✔ ')


# In[ ]:


#### OLD inefficient version

#     ## Declare Files

#     # Raw dataset
#     file_d02_raw = parent_dir + '/raw/d02'

#     # 2-D data
#     file_d02_RR = parent_dir + '/L1/d02_RR'						# [mm/dt]
#     file_d02_U10 = parent_dir + '/L1/d02_U10'					# [m/s]
#     file_d02_V10 = parent_dir + '/L1/d02_V10'					# [m/s]
#     file_d02_PSFC = parent_dir + '/L1/d02_PSFC'					# [hPa]
#     file_d02_T2 = parent_dir + '/L1/d02_T2'						# [K]
#     file_d02_HFX = parent_dir + '/L1/d02_HFX'					# [W/m^2]
#     file_d02_QFX = parent_dir + '/L1/d02_QFX'					# [W/m^2]
#     file_d02_LH = parent_dir + '/L1/d02_LH'					    # [kg/(m^2s^1)]
#     file_d02_CAPE = parent_dir + '/L1/d02_CAPE'	                # [J/kg]
#     file_d02_CIN = parent_dir + '/L1/d02_CIN'	                # [J/kg]

#     # All-sky
#     file_d02_LWUPT = parent_dir + '/L1/d02_LWUPT'				# [W/m^2]
#     file_d02_LWDNT = parent_dir + '/L1/d02_LWDNT'				# [W/m^2]
#     file_d02_LWUPB = parent_dir + '/L1/d02_LWUPB'				# [W/m^2]
#     file_d02_LWDNB = parent_dir + '/L1/d02_LWDNB'				# [W/m^2]
#     file_d02_SWUPT = parent_dir + '/L1/d02_SWUPT'				# [W/m^2]
#     file_d02_SWDNT = parent_dir + '/L1/d02_SWDNT'				# [W/m^2]
#     file_d02_SWUPB = parent_dir + '/L1/d02_SWUPB'				# [W/m^2]
#     file_d02_SWDNB = parent_dir + '/L1/d02_SWDNB'				# [W/m^2]
#     # Clear-sky
#     file_d02_LWUPTC = parent_dir + '/L1/d02_LWUPTC'				# [W/m^2]
#     file_d02_LWDNTC = parent_dir + '/L1/d02_LWDNTC'				# [W/m^2]
#     file_d02_LWUPBC = parent_dir + '/L1/d02_LWUPBC'				# [W/m^2]
#     file_d02_LWDNBC = parent_dir + '/L1/d02_LWDNBC'				# [W/m^2]
#     file_d02_SWUPTC = parent_dir + '/L1/d02_SWUPTC'				# [W/m^2]
#     file_d02_SWDNTC = parent_dir + '/L1/d02_SWDNTC'				# [W/m^2]
#     file_d02_SWUPBC = parent_dir + '/L1/d02_SWUPBC'				# [W/m^2]
#     file_d02_SWDNBC = parent_dir + '/L1/d02_SWDNBC'				# [W/m^2]

#     # Interpolated data 
#     file_d02_U = parent_dir + '/L2/d02_interp_U'	        # [m/s]
#     file_d02_V = parent_dir + '/L2/d02_interp_V'	        # [m/s]
#     file_d02_W = parent_dir + '/L2/d02_interp_W'	        # [m/s]
#     file_d02_QV = parent_dir + '/L2/d02_interp_QV'	        # [kg/kg]
#     file_d02_QC = parent_dir + '/L2/d02_interp_QC'	        # [kg/kg]
#     file_d02_QR = parent_dir + '/L2/d02_interp_QR'	        # [kg/kg]
#     file_d02_QI = parent_dir + '/L2/d02_interp_QI'	        # [kg/kg]
#     file_d02_QS = parent_dir + '/L2/d02_interp_QS'	        # [kg/kg]
#     file_d02_QG = parent_dir + '/L2/d02_interp_QG'	        # [kg/kg]
#     file_d02_CLDFRA = parent_dir + '/L2/d02_interp_CLDFRA'	# 
#     file_d02_H_DIABATIC = parent_dir + '/L2/d02_interp_H_DIABATIC'	        # [K/s]
#     file_d02_LWAll = parent_dir + '/L2/d02_interp_LWAll'	# [K/s]
#     file_d02_LWClear = parent_dir + '/L2/d02_interp_LWClear'# [K/s]
#     file_d02_SWAll = parent_dir + '/L2/d02_interp_SWAll'	# [K/s]
#     file_d02_SWClear = parent_dir + '/L2/d02_interp_SWClear'# [K/s]
#     file_d02_Theta = parent_dir + '/L2/d02_interp_Theta'	# [K]

#     ######################################################################################



#     # Raw datasets
#     start_time = time.perf_counter()

#     ds_main = open_ds(file_d02_raw,time_inds,lat_inds,lon_inds)
#     step1_time = time.perf_counter()
#     print('Dataset loaded \N{check mark}', step1_time-start_time, 'seconds')

#     # Coordinate dictionaries:
#     step2_time = time.perf_counter()
#     interp_P_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))

#     interp_P_levels_d02 = np.resize(interp_P_levels,(ds_main.XLAT.shape[2],ds_main.XLAT.shape[1],len(interp_P_levels)))
#     interp_P_levels_d02 = np.swapaxes(interp_P_levels_d02, 0, 2)
#     coords = dict(
#         XLAT=(('Time','south_north','west_east'),ds_main.XLAT.values),
#         XLONG=(('Time','south_north','west_east'),ds_main.XLONG.values),
#         bottom_top=(('bottom_top'),interp_P_levels),
#         XTIME=('Time',ds_main.XTIME.values),
#         south_north=(('south_north'),ds_main.XLAT[0,:,0].values),
#         west_east=(('west_east'),ds_main.XLONG[0,0,:].values)
#         )

#     step1_time = time.perf_counter()
#     print('Created coordinate dictionaries \N{check mark}', step1_time-step2_time, 'seconds')

#     step2_time = time.perf_counter()
#     #### Figure out where the coast is
#     ## Load Landmask
#     da_d02_LANDMASK = ds_main['LANDMASK'].compute()
#     da_d02_LANDMASK = da_d02_LANDMASK.assign_coords(without_keys(coords,'bottom_top'))

#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LANDMASK, start_coord, end_coord, width, dx)
#     # Create distance coordinate
#     distance_d02 = np.linspace(0,dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]),da_cross_temp.shape[1])
#     mid_cross_ind = int(da_cross_temp.shape[2]/2)	# Find middle cross-section index
#     if da_cross_temp[0,0,mid_cross_ind]==0:     # Figure out if the start_coord is over land or ocean
#         coast_ind = np.where(da_cross_temp[0,:,mid_cross_ind]==1)[0][0]	# First 1 (ocean->land)
#     else:
#         coast_ind = np.where(da_cross_temp[0,:,mid_cross_ind]==0)[0][0]	# First 0 (land->ocean)
#     distance_d02 = distance_d02 - distance_d02[coast_ind]   # Negative values is land
#     step1_time = time.perf_counter()
#     print('Calculated distance measurements \N{check mark}', step1_time-step2_time, 'seconds')

#     # Do this to first figure out what the missing data value is
#     fill_value_f8 = wrf.default_fill(np.float32)      # This is the fill_value meaning missing_data

#     ######################################################################################################################################
#     ################# Load in the variables ##############################################################################################
#     ######################################################################################################################################


#     ##################### 3-D variables ##################################################################################################

#     ############ Interpolated zonal winds   [m/s] #############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_U,time_inds,lat_inds,lon_inds)
#     da_d02_U = ds['U'].compute()
#     da_d02_U = da_d02_U.assign_coords(coords)
#     da_d02_U = da_d02_U.where(da_d02_U!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated zonal winds loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Interpolated Meridional winds   [m/s] ############
#     step2_time = time.perf_counter()

#     # d02
#     ds = open_ds(file_d02_V,time_inds,lat_inds,lon_inds)
#     da_d02_V = ds['V'].compute()
#     da_d02_V = da_d02_V.assign_coords(coords)
#     da_d02_V = da_d02_V.where(da_d02_V!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated meridional winds loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################################ Calculate Normal Wind ################################

#     # d02
#     da_u, da_v = rotate_vec(da_d02_U, da_d02_V, theta)
#     ################ Normal Wind - Cross-section Analysis ################
#     da_cross_temp, all_line_coords = cross_section_multi(da_u, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_NormalWind_cross = make_da_cross(da_d02_U, da_cross_temp, 'NormalWind', distance_d02, width, all_line_coords)
#     da_d02_NormalWind_cross.to_netcdf('./d02_cross_NormalWind')
#     # Delete variables after to aliviate memory strain
#     del da_d02_NormalWind_cross, da_d02_U, da_d02_V

#     ############ Interpolated Vertical winds   [m/s] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_W,time_inds,lat_inds,lon_inds)
#     da_d02_W = ds['W'].compute()
#     da_d02_W = da_d02_W.assign_coords(coords)
#     da_d02_W = da_d02_W.where(da_d02_W!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated vertical winds loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############################ Vertical Wind - Cross-section Analysis ############################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_W, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_W_cross = make_da_cross(da_d02_W, da_cross_temp, 'W', distance_d02, width, all_line_coords)
#     da_d02_W_cross.to_netcdf('./d02_cross_W')
#     # Delete variables after to aliviate memory strain
#     del da_d02_W_cross, da_d02_W

#     ############ Interpolated water vapor mixing ratio  [kg/kg] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_QV,time_inds,lat_inds,lon_inds)
#     da_d02_QV = ds['QV'].compute()
#     da_d02_QV = da_d02_QV.assign_coords(coords)
#     da_d02_QV = da_d02_QV.where(da_d02_QV!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated water vapor mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############################ Water Vapor - Cross-section Analysis ############################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_QV, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_QV_cross = make_da_cross(da_d02_QV, da_cross_temp, 'QV', distance_d02, width, all_line_coords)
#     da_d02_QV_cross.to_netcdf('./d02_cross_QV')
#     # Delete variables after to aliviate memory strain
#     del da_d02_QV_cross, da_d02_QV

#     ############ Interpolated cloud water mixing ratio  [kg/kg] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_QC,time_inds,lat_inds,lon_inds)
#     da_d02_QC = ds['QC'].compute()
#     da_d02_QC = da_d02_QC.assign_coords(coords)
#     da_d02_QC = da_d02_QC.where(da_d02_QC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############################ Cloud water - Cross-section Analysis ############################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_QC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_QC_cross = make_da_cross(da_d02_QC, da_cross_temp, 'QC', distance_d02, width, all_line_coords)
#     da_d02_QC_cross.to_netcdf('./d02_cross_QC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_QC_cross, da_d02_QC

#     ############ Interpolated rain water mixing ratio  [kg/kg] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_QR,time_inds,lat_inds,lon_inds)
#     da_d02_QR = ds['QR'].compute()
#     da_d02_QR = da_d02_QR.assign_coords(coords)
#     da_d02_QR = da_d02_QR.where(da_d02_QR!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############################ Rain water - Cross-section Analysis ############################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_QR, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_QR_cross = make_da_cross(da_d02_QR, da_cross_temp, 'QR', distance_d02, width, all_line_coords)
#     da_d02_QR_cross.to_netcdf('./d02_cross_QR')
#     # Delete variables after to aliviate memory strain
#     del da_d02_QR_cross, da_d02_QR

#     ############ Interpolated ice mixing ratio  [kg/kg] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_QI,time_inds,lat_inds,lon_inds)
#     da_d02_QI = ds['QI'].compute()
#     da_d02_QI = da_d02_QI.assign_coords(coords)
#     da_d02_QI = da_d02_QI.where(da_d02_QI!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############################ Ice mixing ratio - Cross-section Analysis ############################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_QI, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_QI_cross = make_da_cross(da_d02_QI, da_cross_temp, 'QI', distance_d02, width, all_line_coords)
#     da_d02_QI_cross.to_netcdf('./d02_cross_QI')
#     # Delete variables after to aliviate memory strain
#     del da_d02_QI_cross, da_d02_QI

#     ############ Interpolated snow mixing ratio  [kg/kg] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_QS,time_inds,lat_inds,lon_inds)
#     da_d02_QS = ds['QS'].compute()
#     da_d02_QS = da_d02_QS.assign_coords(coords)
#     da_d02_QS = da_d02_QS.where(da_d02_QS!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############################ Snow mixing ratio - Cross-section Analysis ############################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_QS, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_QS_cross = make_da_cross(da_d02_QS, da_cross_temp, 'QS', distance_d02, width, all_line_coords)
#     da_d02_QS_cross.to_netcdf('./d02_cross_QS')
#     # Delete variables after to aliviate memory strain
#     del da_d02_QS_cross, da_d02_QS

#     ############ Interpolated graupel mixing ratio  [kg/kg] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_QG,time_inds,lat_inds,lon_inds)
#     da_d02_QG = ds['QG'].compute()
#     da_d02_QG = da_d02_QG.assign_coords(coords)
#     da_d02_QG = da_d02_QG.where(da_d02_QG!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############################ Graupel mixing ratio - Cross-section Analysis ############################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_QG, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_QG_cross = make_da_cross(da_d02_QG, da_cross_temp, 'QG', distance_d02, width, all_line_coords)
#     da_d02_QG_cross.to_netcdf('./d02_cross_QG')
#     # Delete variables after to aliviate memory strain
#     del da_d02_QG_cross, da_d02_QG

#     ############ Cloud Fraction [0-1] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_CLDFRA,time_inds,lat_inds,lon_inds)
#     da_d02_CLDFRA = ds['CLDFRA'].compute()
#     da_d02_CLDFRA = da_d02_CLDFRA.assign_coords(coords)
#     da_d02_CLDFRA = da_d02_CLDFRA.where(da_d02_CLDFRA!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Cloud fraction loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Cloud Fraction - Cross-section Analysis ################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_CLDFRA, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_CLDFRA_cross = make_da_cross(da_d02_CLDFRA, da_cross_temp, 'CLDFRA', distance_d02, width, all_line_coords)
#     da_d02_CLDFRA_cross.to_netcdf('./d02_cross_CLDFRA')
#     # Delete variables after to aliviate memory strain
#     del da_d02_CLDFRA_cross, da_d02_CLDFRA

#     ############ Latent Heating [K/s] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_H_DIABATIC,time_inds,lat_inds,lon_inds)
#     da_d02_H_DIABATIC = ds['H_DIABATIC'].compute()
#     da_d02_H_DIABATIC = da_d02_H_DIABATIC.assign_coords(coords)
#     da_d02_H_DIABATIC = da_d02_H_DIABATIC.where(da_d02_H_DIABATIC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Latent heating loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Latent Heat - Cross-section Analysis ################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_H_DIABATIC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_H_DIABATIC_cross = make_da_cross(da_d02_H_DIABATIC, da_cross_temp, 'H_DIABATIC', distance_d02, width, all_line_coords)
#     da_d02_H_DIABATIC_cross.to_netcdf('./d02_cross_H_DIABATIC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_H_DIABATIC_cross, da_d02_H_DIABATIC

#     ############ Potential Temperature [K] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_Theta,time_inds,lat_inds,lon_inds)
#     da_d02_Theta = ds['Theta'].compute()
#     da_d02_Theta = da_d02_Theta.assign_coords(coords)
#     da_d02_Theta = da_d02_Theta.where(da_d02_Theta!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Clear-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Potential Temperature/Theta - Cross-section Analysis ################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_Theta, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_Theta_cross = make_da_cross(da_d02_Theta, da_cross_temp, 'Theta', distance_d02, width, all_line_coords)
#     da_d02_Theta_cross.to_netcdf('./d02_cross_Theta')
#     # Delete variables after to aliviate memory strain
#     del da_d02_Theta_cross, da_d02_Theta

#     ############ Longwave Radiative Heating All-Sky [K/s] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWAll,time_inds,lat_inds,lon_inds)
#     da_d02_LWAll = ds['LWAll'].compute()
#     da_d02_LWAll = da_d02_LWAll.assign_coords(coords)
#     da_d02_LWAll = da_d02_LWAll.where(da_d02_LWAll!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave All-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Longwave All - Cross-section Analysis ################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWAll, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWAll_cross = make_da_cross(da_d02_LWAll, da_cross_temp, 'LWAll', distance_d02, width, all_line_coords)
#     da_d02_LWAll_cross.to_netcdf('./d02_cross_LWAll')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWAll

#     ############ Longwave Radiative Heating Clear-Sky [K/s] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWClear,time_inds,lat_inds,lon_inds)
#     da_d02_LWClear = ds['LWClear'].compute()
#     da_d02_LWClear = da_d02_LWClear.assign_coords(coords)
#     da_d02_LWClear = da_d02_LWClear.where(da_d02_LWClear!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Clear-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Longwave Clear - Cross-section Analysis ################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWClear, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWClear_cross = make_da_cross(da_d02_LWClear, da_cross_temp, 'LWClear', distance_d02, width, all_line_coords)
#     da_d02_LWClear_cross.to_netcdf('./d02_cross_LWClear')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWClear

#     ################ Longwave CRF - Cross-section Analysis ################
#     # Calculate the CRF
#     da_d02_LWCRF_cross = da_d02_LWAll_cross - da_d02_LWClear_cross
#     da_d02_LWCRF_cross = xr.DataArray(data=da_d02_LWCRF_cross, name='LWCRF')
#     da_d02_LWCRF_cross.to_netcdf('./d02_cross_LWCRF')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWAll_cross, da_d02_LWClear_cross

#     ############ Shortwave Radiative Heating All-Sky [K/s] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWAll,time_inds,lat_inds,lon_inds)
#     da_d02_SWAll = ds['SWAll'].compute()
#     da_d02_SWAll = da_d02_SWAll.assign_coords(coords)
#     da_d02_SWAll = da_d02_SWAll.where(da_d02_SWAll!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave All-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Shortwave All - Cross-section Analysis ################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWAll, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWAll_cross = make_da_cross(da_d02_SWAll, da_cross_temp, 'SWAll', distance_d02, width, all_line_coords)
#     da_d02_SWAll_cross.to_netcdf('./d02_cross_SWAll')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWAll

#     ############ Shortwave Radiative Heating Clear-Sky [K/s] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWClear,time_inds,lat_inds,lon_inds)
#     da_d02_SWClear = ds['SWClear'].compute()
#     da_d02_SWClear = da_d02_SWClear.assign_coords(coords)
#     da_d02_SWClear = da_d02_SWClear.where(da_d02_SWClear!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Clear-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Shortwave Clear - Cross-section Analysis ################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWClear, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWClear_cross = make_da_cross(da_d02_SWClear, da_cross_temp, 'SWClear', distance_d02, width, all_line_coords)
#     da_d02_SWClear_cross.to_netcdf('./d02_cross_SWClear')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWClear

#     ################ Shortwave CRF - Cross-section Analysis ################
#     # Calculate the CRF
#     da_d02_SWCRF_cross = da_d02_SWAll_cross - da_d02_SWClear_cross
#     da_d02_SWCRF_cross = xr.DataArray(data=da_d02_SWCRF_cross, name='SWCRF')
#     da_d02_SWCRF_cross.to_netcdf('./d02_cross_SWCRF')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWClear_cross, da_d02_SWAll_cross

#     ################ Total CRF - Cross-section Analysis ################
#     # Calculate the Total CRF
#     da_d02_TotalCRF_cross = da_d02_SWCRF_cross + da_d02_LWCRF_cross
#     da_d02_TotalCRF_cross = xr.DataArray(data=da_d02_TotalCRF_cross, name='TotalCRF')
#     da_d02_TotalCRF_cross.to_netcdf('./d02_cross_TotalCRF')
#     # Delete variables after to aliviate memory strain
#     del da_d02_TotalCRF_cross, da_d02_SWCRF_cross, da_d02_LWCRF_cross

#     #################################################################################################################################################
#     ##################### 2-D variables #############################################################################################################
#     #################################################################################################################################################

#     ############ Rain Rate     [mm/hr] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_RR,time_inds,lat_inds,lon_inds)
#     da_d02_RR = ds['RR'].compute()
#     da_d02_RR = da_d02_RR.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_RR = da_d02_RR.where(da_d02_RR!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Rain Rate - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_RR, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_RR_cross = make_da_cross(da_d02_RR, da_cross_temp, 'RR', distance_d02, width, all_line_coords)
#     da_d02_RR_cross.to_netcdf('./d02_cross_RR')
#     # Delete variables after to aliviate memory strain
#     del da_d02_RR, da_d02_RR_cross

#     ############ Surface U Wind     [m/s] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_U10,time_inds,lat_inds,lon_inds)
#     da_d02_U10 = ds['U10'].compute()
#     da_d02_U10 = da_d02_U10.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_U10 = da_d02_U10.where(da_d02_U10!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('U10 rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Surface U Wind - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_U10, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_U10_cross = make_da_cross(da_d02_U10, da_cross_temp, 'U10', distance_d02, width, all_line_coords)
#     da_d02_U10_cross.to_netcdf('./d02_cross_U10')
#     # Delete variables after to aliviate memory strain
#     del da_d02_U10_cross

#     ############ Surface V Wind     [m/s] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_V10,time_inds,lat_inds,lon_inds)
#     da_d02_V10 = ds['V10'].compute()
#     da_d02_V10 = da_d02_V10.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_V10 = da_d02_V10.where(da_d02_V10!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('V10 rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Surface V Wind - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_V10, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_V10_cross = make_da_cross(da_d02_V10, da_cross_temp, 'V10', distance_d02, width, all_line_coords)
#     da_d02_V10_cross.to_netcdf('./d02_cross_V10')
#     # Delete variables after to aliviate memory strain
#     del da_d02_V10_cross

#     ################################ Calculate Surface Normal Wind ################################
#     # d02
#     da_u10, da_v10 = rotate_vec(da_d02_U10, da_d02_V10, theta)
#     ################ Normal Wind - Cross-section Analysis ################
#     da_cross_temp, all_line_coords = cross_section_multi(da_u10, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_NormalWind_cross = make_da_cross(da_d02_U10, da_cross_temp, 'NormalWind', distance_d02, width, all_line_coords)
#     da_d02_NormalWind_cross.to_netcdf('./d02_cross_SurfaceNormalWind')
#     # Delete variables after to aliviate memory strain
#     del da_d02_NormalWind_cross, da_d02_U10, da_d02_V10

#     ############ Upward Heat Flux at Surface     [W/m^2] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_HFX,time_inds,lat_inds,lon_inds)
#     da_d02_HFX = ds['HFX'].compute()
#     da_d02_HFX = da_d02_HFX.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_HFX = da_d02_HFX.where(da_d02_HFX!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Upward Heat Flux at Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Upward Heat Flux at Surface - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_HFX, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_HFX_cross = make_da_cross(da_d02_HFX, da_cross_temp, 'HFX', distance_d02, width, all_line_coords)
#     da_d02_HFX_cross.to_netcdf('./d02_cross_HFX')
#     # Delete variables after to aliviate memory strain
#     del da_d02_HFX, da_d02_HFX_cross


#     ############ Upward Moisture Flux at Surface     [kg/(m^2s^1)] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_QFX,time_inds,lat_inds,lon_inds)
#     da_d02_QFX = ds['QFX'].compute()
#     da_d02_QFX = da_d02_QFX.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_QFX = da_d02_QFX.where(da_d02_QFX!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Upward Moisture Flux at Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Upward Moisture Flux at Surface - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_QFX, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_QFX_cross = make_da_cross(da_d02_QFX, da_cross_temp, 'QFX', distance_d02, width, all_line_coords)
#     da_d02_QFX_cross.to_netcdf('./d02_cross_QFX')
#     # Delete variables after to aliviate memory strain
#     del da_d02_QFX, da_d02_QFX_cross


#     ############ Latent Heat Flux at Surface		[W/m^2] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LH,time_inds,lat_inds,lon_inds)
#     da_d02_LH = ds['LH'].compute()
#     da_d02_LH = da_d02_LH.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LH = da_d02_LH.where(da_d02_LH!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Latent Heat Flux at Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Latent Heat Flux at Surface - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LH, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LH_cross = make_da_cross(da_d02_LH, da_cross_temp, 'LH', distance_d02, width, all_line_coords)
#     da_d02_LH_cross.to_netcdf('./d02_cross_LH')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LH, da_d02_LH_cross


#     ############ Temperature at 2m     [K] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_T2,time_inds,lat_inds,lon_inds)
#     da_d02_T2 = ds['T2'].compute()
#     da_d02_T2 = da_d02_T2.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_T2 = da_d02_T2.where(da_d02_T2!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Temperature at 2m loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Temperature at 2m - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_T2, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_T2_cross = make_da_cross(da_d02_T2, da_cross_temp, 'T2', distance_d02, width, all_line_coords)
#     da_d02_T2_cross.to_netcdf('./d02_cross_T2')
#     # Delete variables after to aliviate memory strain
#     del da_d02_T2, da_d02_T2_cross


#     ############ Surface Pressure     [hPa] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_PSFC,time_inds,lat_inds,lon_inds)
#     da_d02_PSFC = ds['PSFC'].compute()
#     da_d02_PSFC = da_d02_PSFC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_PSFC = da_d02_PSFC.where(da_d02_PSFC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Surface Pressure loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ Surface Pressure - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_PSFC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_PSFC_cross = make_da_cross(da_d02_PSFC, da_cross_temp, 'PSFC', distance_d02, width, all_line_coords)
#     da_d02_PSFC_cross.to_netcdf('./d02_cross_PSFC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_PSFC, da_d02_PSFC_cross


#     ############ CAPE     [hPa] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_CAPE,time_inds,lat_inds,lon_inds)
#     da_d02_CAPE = ds['CAPE'].compute()
#     da_d02_CAPE = da_d02_CAPE.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_CAPE = da_d02_CAPE.where(da_d02_CAPE!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('CAPE loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ CAPE - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_CAPE, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_CAPE_cross = make_da_cross(da_d02_CAPE, da_cross_temp, 'CAPE', distance_d02, width, all_line_coords)
#     da_d02_CAPE_cross.to_netcdf('./d02_cross_CAPE')
#     # Delete variables after to aliviate memory strain
#     del da_d02_CAPE, da_d02_CAPE_cross


#     ############ CIN     [hPa] ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_CIN,time_inds,lat_inds,lon_inds)
#     da_d02_CIN = ds['CIN'].compute()
#     da_d02_CIN = da_d02_CIN.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_CIN = da_d02_CIN.where(da_d02_CIN!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('CIN loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ################ CIN - Cross-section Analysis #################
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_CIN, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_CIN_cross = make_da_cross(da_d02_CIN, da_cross_temp, 'CIN', distance_d02, width, all_line_coords)
#     da_d02_CIN_cross.to_netcdf('./d02_cross_CIN')
#     # Delete variables after to aliviate memory strain
#     del da_d02_CIN, da_d02_CIN_cross

#     #####################################################################################################################################################
#     ################################ All-sky ############################################################################################################
#     #####################################################################################################################################################

#     ############ Load Longwave Upwelling at TOA ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWUPT,time_inds,lat_inds,lon_inds)
#     da_d02_LWUPT = ds['LWUPT'].compute()
#     da_d02_LWUPT = da_d02_LWUPT.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LWUPT = da_d02_LWUPT.where(da_d02_LWUPT!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Upwelling at TOA loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Longwave Upwelling at TOA ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWUPT, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWUPT_cross = make_da_cross(da_d02_LWUPT, da_cross_temp, 'LWUPT', distance_d02, width, all_line_coords)
#     da_d02_LWUPT_cross.to_netcdf('./d02_cross_LWUPT')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWUPT, da_d02_LWUPT_cross

#     ############ Load Longwave Downwelling at TOA ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWDNT,time_inds,lat_inds,lon_inds)
#     da_d02_LWDNT = ds['LWDNT'].compute()
#     da_d02_LWDNT = da_d02_LWDNT.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LWDNT = da_d02_LWDNT.where(da_d02_LWDNT!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Downwelling at TOA loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Longwave Downwelling at TOA ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWDNT, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWDNT_cross = make_da_cross(da_d02_LWDNT, da_cross_temp, 'LWDNT', distance_d02, width, all_line_coords)
#     da_d02_LWDNT_cross.to_netcdf('./d02_cross_LWDNT')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWDNT, da_d02_LWDNT_cross

#     ############ Load Longwave Upwelling at SFC ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWUPB,time_inds,lat_inds,lon_inds)
#     da_d02_LWUPB = ds['LWUPB'].compute()
#     da_d02_LWUPB = da_d02_LWUPB.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LWUPB = da_d02_LWUPB.where(da_d02_LWUPB!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Upwelling at SFC loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Longwave Upwelling at SFC ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWUPB, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWUPB_cross = make_da_cross(da_d02_LWUPB, da_cross_temp, 'LWUPB', distance_d02, width, all_line_coords)
#     da_d02_LWUPB_cross.to_netcdf('./d02_cross_LWUPB')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWUPB, da_d02_LWUPB_cross

#     ############ Load Longwave Downwelling at SFC ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWDNB,time_inds,lat_inds,lon_inds)
#     da_d02_LWDNB = ds['LWDNB'].compute()
#     da_d02_LWDNB = da_d02_LWDNB.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LWDNB = da_d02_LWDNB.where(da_d02_LWDNB!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Downwelling at SFC loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Longwave Downwelling at SFC ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWDNB, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWDNB_cross = make_da_cross(da_d02_LWDNB, da_cross_temp, 'LWDNB', distance_d02, width, all_line_coords)
#     da_d02_LWDNB_cross.to_netcdf('./d02_cross_LWDNB')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWDNB, da_d02_LWDNB_cross

#     ############ Load Shortwave Upwelling at TOA ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWUPT,time_inds,lat_inds,lon_inds)
#     da_d02_SWUPT = ds['SWUPT'].compute()
#     da_d02_SWUPT = da_d02_SWUPT.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_SWUPT = da_d02_SWUPT.where(da_d02_SWUPT!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Upwelling at TOA loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Shortwave Upwelling at TOA ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWUPT, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWUPT_cross = make_da_cross(da_d02_SWUPT, da_cross_temp, 'SWUPT', distance_d02, width, all_line_coords)
#     da_d02_SWUPT_cross.to_netcdf('./d02_cross_SWUPT')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWUPT, da_d02_SWUPT_cross

#     ############ Load Shortwave Downwelling at TOA ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWDNT,time_inds,lat_inds,lon_inds)
#     da_d02_SWDNT = ds['SWDNT'].compute()
#     da_d02_SWDNT = da_d02_SWDNT.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_SWDNT = da_d02_SWDNT.where(da_d02_SWDNT!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Downwelling at TOA loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Shortwave Downwelling at TOA ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWDNT, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWDNT_cross = make_da_cross(da_d02_SWDNT, da_cross_temp, 'SWDNT', distance_d02, width, all_line_coords)
#     da_d02_SWDNT_cross.to_netcdf('./d02_cross_SWDNT')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWDNT, da_d02_SWDNT_cross

#     ############ Load Shortwave Upwelling at SFC ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWUPB,time_inds,lat_inds,lon_inds)
#     da_d02_SWUPB = ds['SWUPB'].compute()
#     da_d02_SWUPB = da_d02_SWUPB.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_SWUPB = da_d02_SWUPB.where(da_d02_SWUPB!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Upwelling at SFC loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Shortwave Upwelling at SFC ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWUPB, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWUPB_cross = make_da_cross(da_d02_SWUPB, da_cross_temp, 'SWUPB', distance_d02, width, all_line_coords)
#     da_d02_SWUPB_cross.to_netcdf('./d02_cross_SWUPB')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWUPB, da_d02_SWUPB_cross

#     ############ Load Shortwave Downwelling at SFC ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWDNB,time_inds,lat_inds,lon_inds)
#     da_d02_SWDNB = ds['SWDNB'].compute()
#     da_d02_SWDNB = da_d02_SWDNB.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_SWDNB = da_d02_SWDNB.where(da_d02_SWDNB!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Downwelling at SFC loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Shortwave Downwelling at SFC ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWDNB, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWDNB_cross = make_da_cross(da_d02_SWDNB, da_cross_temp, 'SWDNB', distance_d02, width, all_line_coords)
#     da_d02_SWDNB_cross.to_netcdf('./d02_cross_SWDNB')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWDNB, da_d02_SWDNB_cross

#     #######################################################################################################################################################
#     ################################ Clear-sky ############################################################################################################
#     #######################################################################################################################################################

#     ############ Load Longwave Upwelling at TOA ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWUPTC,time_inds,lat_inds,lon_inds)
#     da_d02_LWUPTC = ds['LWUPTC'].compute()
#     da_d02_LWUPTC = da_d02_LWUPTC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LWUPTC = da_d02_LWUPTC.where(da_d02_LWUPTC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Upwelling at TOA loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Longwave Upwelling at TOA ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWUPTC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWUPTC_cross = make_da_cross(da_d02_LWUPTC, da_cross_temp, 'LWUPTC', distance_d02, width, all_line_coords)
#     da_d02_LWUPTC_cross.to_netcdf('./d02_cross_LWUPTC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWUPTC, da_d02_LWUPTC_cross

#     ############ Load Longwave Downwelling at TOA ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWDNTC,time_inds,lat_inds,lon_inds)
#     da_d02_LWDNTC = ds['LWDNTC'].compute()
#     da_d02_LWDNTC = da_d02_LWDNTC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LWDNTC = da_d02_LWDNTC.where(da_d02_LWDNTC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Downwelling at TOA loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Longwave Downwelling at TOA ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWDNTC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWDNTC_cross = make_da_cross(da_d02_LWDNTC, da_cross_temp, 'LWDNTC', distance_d02, width, all_line_coords)
#     da_d02_LWDNTC_cross.to_netcdf('./d02_cross_LWDNTC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWDNTC, da_d02_LWDNTC_cross

#     ############ Load Longwave Upwelling at SFC ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWUPBC,time_inds,lat_inds,lon_inds)
#     da_d02_LWUPBC = ds['LWUPBC'].compute()
#     da_d02_LWUPBC = da_d02_LWUPBC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LWUPBC = da_d02_LWUPBC.where(da_d02_LWUPBC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Upwelling at SFC loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Longwave Upwelling at SFC ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWUPBC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWUPBC_cross = make_da_cross(da_d02_LWUPBC, da_cross_temp, 'LWUPBC', distance_d02, width, all_line_coords)
#     da_d02_LWUPBC_cross.to_netcdf('./d02_cross_LWUPBC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWUPBC, da_d02_LWUPBC_cross

#     ############ Load Longwave Downwelling at SFC ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_LWDNBC,time_inds,lat_inds,lon_inds)
#     da_d02_LWDNBC = ds['LWDNBC'].compute()
#     da_d02_LWDNBC = da_d02_LWDNBC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_LWDNBC = da_d02_LWDNBC.where(da_d02_LWDNBC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Longwave Downwelling at SFC loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Longwave Downwelling at SFC ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWDNBC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_LWDNBC_cross = make_da_cross(da_d02_LWDNBC, da_cross_temp, 'LWDNBC', distance_d02, width, all_line_coords)
#     da_d02_LWDNBC_cross.to_netcdf('./d02_cross_LWDNBC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_LWDNBC, da_d02_LWDNBC_cross

#     ############ Load Shortwave Upwelling at TOA ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWUPTC,time_inds,lat_inds,lon_inds)
#     da_d02_SWUPTC = ds['SWUPTC'].compute()
#     da_d02_SWUPTC = da_d02_SWUPTC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_SWUPTC = da_d02_SWUPTC.where(da_d02_SWUPTC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Upwelling at TOA loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Shortwave Upwelling at TOA ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWUPTC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWUPTC_cross = make_da_cross(da_d02_SWUPTC, da_cross_temp, 'SWUPTC', distance_d02, width, all_line_coords)
#     da_d02_SWUPTC_cross.to_netcdf('./d02_cross_SWUPTC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWUPTC, da_d02_SWUPTC_cross

#     ############ Load Shortwave Downwelling at TOA ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWDNTC,time_inds,lat_inds,lon_inds)
#     da_d02_SWDNTC = ds['SWDNTC'].compute()
#     da_d02_SWDNTC = da_d02_SWDNTC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_SWDNTC = da_d02_SWDNTC.where(da_d02_SWDNTC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Downwelling at TOA loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Shortwave Downwelling at TOA ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWDNTC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWDNTC_cross = make_da_cross(da_d02_SWDNTC, da_cross_temp, 'SWDNTC', distance_d02, width, all_line_coords)
#     da_d02_SWDNTC_cross.to_netcdf('./d02_cross_SWDNTC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWDNTC, da_d02_SWDNTC_cross

#     ############ Load Shortwave Upwelling at SFC ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWUPBC,time_inds,lat_inds,lon_inds)
#     da_d02_SWUPBC = ds['SWUPBC'].compute()
#     da_d02_SWUPBC = da_d02_SWUPBC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_SWUPBC = da_d02_SWUPBC.where(da_d02_SWUPBC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Upwelling at SFC loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Shortwave Upwelling at SFC ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWUPBC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWUPBC_cross = make_da_cross(da_d02_SWUPBC, da_cross_temp, 'SWUPBC', distance_d02, width, all_line_coords)
#     da_d02_SWUPBC_cross.to_netcdf('./d02_cross_SWUPBC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWUPBC, da_d02_SWUPBC_cross

#     ############ Load Shortwave Downwelling at SFC ############
#     step2_time = time.perf_counter()
#     # d02
#     ds = open_ds(file_d02_SWDNBC,time_inds,lat_inds,lon_inds)
#     da_d02_SWDNBC = ds['SWDNBC'].compute()
#     da_d02_SWDNBC = da_d02_SWDNBC.assign_coords(without_keys(coords,'bottom_top'))
#     da_d02_SWDNBC = da_d02_SWDNBC.where(da_d02_SWDNBC!=fill_value_f8)    # Change fill_value points to nans

#     step1_time = time.perf_counter()
#     print('Shortwave Downwelling at SFC loaded \N{check mark}', step1_time-step2_time, 'seconds')

#     ############ Cross-sectional analysis of Shortwave Downwelling at SFC ############
#     # d02
#     da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWDNBC, start_coord, end_coord, width, dx)
#     # Create da with coordinates
#     da_d02_SWDNBC_cross = make_da_cross(da_d02_SWDNBC, da_cross_temp, 'SWDNBC', distance_d02, width, all_line_coords)
#     da_d02_SWDNBC_cross.to_netcdf('./d02_cross_SWDNBC')
#     # Delete variables after to aliviate memory strain
#     del da_d02_SWDNBC, da_d02_SWDNBC_cross

