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
# This code is designed to read in the variable, create a cross-section of that variable, save the cross-section as an .nc file, then delete the variable. Originally, reading all the variables first and then saving files was attempted, but the system runs out of memory quite quickly.

# ### WRF

# In[17]:


##cd into the appropriate directory (L3) and then assign a parent directory
parent_dir = sys.argv[1]
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/SN_CTRL'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00/CRFoff'

# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-03T00')]
# times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-11-23T03')]  # Testing
times = [np.datetime64('2015-12-09T12'), np.datetime64('2015-12-20T12')]
# times = [np.datetime64('2015-12-10T01'), np.datetime64('2015-12-10T03')]  # Testing

## Assign the correct location directory depending on where you're trying to cross-section
# local_dirs = ['/L3/Sumatra_mid_central','/L3/Sumatra_northwest','/L3/Borneo_northwest',   '/L3/Sumatra_mid_central_long_wide', '/L3/Borneo_northwest_long_wide']
local_dirs = ['/L3/Sumatra_mid_central_long_wide', '/L3/Borneo_northwest_long_wide']

## Define cross-section settings
region_settings = {
    'Sumatra_mid_central': {
        'start_coord': [-2.2, 103.2],
        'end_coord': [-6.7, 98.7],
        'width': 1.5,
        'dx': 0.025
    },
    'Sumatra_mid_central_long_wide': {
        'start_coord': [0.1, 105.5],
        'end_coord': [-4.5, 100.9],
        'width': 3,
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
    },
    'Borneo_northwest_long_wide': {
        'start_coord': [4.8, 110.8],
        'end_coord': [-2.3, 117.9],
        'width': 3,
        'dx': 0.025
    }
}

#### NOT WORKING, KEEP '' OR '_sunrise' for now
## starter string to diseminate between experiments
    # icloud=0 at sunrise, starter_str='_sunrise'
    # icloud=1, starter_str=''
    # icloud=depends, starter_str='_oceanoff'
starter_str = ''

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

# variables_to_process = [
#     # L1 varnames
#     'RR', 'U10', 'V10', 'PSFC', 'T2', 'HFX', 'QFX', 'LH', 'CAPE', 'CIN',
#     'LWDNB', 'LWUPB', 'LWDNBC', 'LWUPBC', 'SWDNB', 'SWUPB', 'SWDNBC', 'SWUPBC',
#     'LWDNT', 'LWUPT', 'LWDNTC', 'LWUPTC', 'SWDNT', 'SWUPT', 'SWDNTC', 'SWUPTC',
#     # L2 varnames
#     'U', 'V', 'LWAll', 'LWClear', 'SWAll', 'SWClear'
#     # L4 varnames
#     'VI_QV', 'VI_RH', 'VI_ws'
# ]

variables_to_process = ['RR', 'U10', 'V10', 'U', 'V', 'VI_QV', 'VI_RH', 'VI_ws']


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
    # Vertically Integrated data
l4_files = {
    f'd02{starter_str}_VI_QV_1000-100':         ('kg kg-1',     'VI_QV',       'Vertically Integrated (1000-100hPa) Water Vapor Mixing Ratio'),
    f'd02{starter_str}_VI_RH_1000-100':         ('%',           'VI_RH',       'Vertically Integrated (1000-100hPa) Relative Humidity'),
    f'd02{starter_str}_VI_ws_1000-100':         ('kg kg-1',     'VI_ws',       'Vertically Integrated (1000-100hPa) Saturation Water Vapor Mixing Ratio'),
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
    },
    'L4': {
        var: {
            "path": build_path(parent_dir, 'L4', var),
            "unit": unit,
            "varname": varname,
            "description": desc
        }
        for var, (unit, varname, desc) in l4_files.items()
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

