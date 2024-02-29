#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

######## EXAMPLE ########
# There is an example at the very end.
##############################################################################

import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import xarray as xr
from math import cos, asin, sqrt, pi
import time
import os
import sys

##############################################################################


# ### Pre-requisite Functions 

# In[2]:


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


# In[3]:


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


# In[4]:


# This function finds the distance [km] between two coordinates in lat & lon
def dist(lat1, lon1, lat2, lon2):
    r = 6371 # km
    p = pi / 180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 2 * r * asin(sqrt(a))


# In[5]:


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


# In[6]:


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


# In[21]:


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
	# We then create an empty np.array, calculate the change in spread needed, then start filling in the data.

def cross_section_multi(da, start_coord, end_coord, width, dx):

	# We want to first create a line between start and end_coords
		# Gather the indicies of the closest gridboxes of start and end_coords.
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
	# Now that we have the coordinates between the start and end_coords, we need to replicate it for all the lines
	num_lines = int(width/dx)
	spread = np.arange(width/2,-width/2,-dx)
	if 'bottom_top' in da.dims:	# If the dataset is 3-D
		#						TIME	x	HEIGHT	x	   DISTANCE	   x   #ofLINES
		da_cross = np.zeros([da.shape[0],da.shape[1],line_coords.shape[1],num_lines])
	else:
		#						TIME	x	   DISTANCE	   x   #ofLINES
		da_cross = np.zeros([da.shape[0],line_coords.shape[1],num_lines])
		print(line_coords.shape[1])

	# Create all_line_coords that holds all the coordinates for every line produced
	all_line_coords = np.zeros([line_coords.shape[0],line_coords.shape[1],spread.shape[0]])

	# Looping over all the lines
	for i in range(len(spread)):
		# Now that we have our lines, we can interpolate the dataset with the offset for each line applied
		da_interp = da.interp(south_north=line_coords[0,:]+spread[i], west_east=line_coords[1,:]-spread[i], method="linear")

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

		all_line_coords[0,:,i] = line_coords[0,:]+spread[i]
		all_line_coords[1,:,i] = line_coords[1,:]-spread[i]

	return da_cross, all_line_coords


# ## Main Code
# #### This code is designed to read in the variable, create a cross-section of that variable, save the cross-section as an .nc file, then delete the variable. Originally, reading all the variables first and then saving files was attempted, but the system runs out of memory quite quickly.

# In[8]:


# cd into the appropriate directory (L3) and then assign a parent directory
parent_dir = sys.argv[1]
# # CRFOff experiments
# os.chdir(parent_dir+'/L3/Sumatra_mid_central')
# Control
os.chdir(parent_dir+'/L3')

# # Control
# os.chdir('/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/L3')
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00'

file_d01_raw = parent_dir + '/raw/d01'
file_d02_raw = parent_dir + '/raw/d02'

# 2-D data
file_d01_RR = parent_dir + '/L1/d01_RR'						# [mm/dt]
file_d02_RR = parent_dir + '/L1/d02_RR'						# [mm/dt]
file_d01_HFX = parent_dir + '/L1/d01_HFX'					# [W/m^2]
file_d02_HFX = parent_dir + '/L1/d02_HFX'					# [W/m^2]
# All-sky
file_d01_LWUPT = parent_dir + '/L1/d01_LWUPT'				# [W/m^2]
file_d02_LWUPT = parent_dir + '/L1/d02_LWUPT'				# [W/m^2]
file_d01_LWDNT = parent_dir + '/L1/d01_LWDNT'				# [W/m^2]
file_d02_LWDNT = parent_dir + '/L1/d02_LWDNT'				# [W/m^2]
file_d01_LWUPB = parent_dir + '/L1/d01_LWUPB'				# [W/m^2]
file_d02_LWUPB = parent_dir + '/L1/d02_LWUPB'				# [W/m^2]
file_d01_LWDNB = parent_dir + '/L1/d01_LWDNB'				# [W/m^2]
file_d02_LWDNB = parent_dir + '/L1/d02_LWDNB'				# [W/m^2]
file_d01_SWUPT = parent_dir + '/L1/d01_SWUPT'				# [W/m^2]
file_d02_SWUPT = parent_dir + '/L1/d02_SWUPT'				# [W/m^2]
file_d01_SWDNT = parent_dir + '/L1/d01_SWDNT'				# [W/m^2]
file_d02_SWDNT = parent_dir + '/L1/d02_SWDNT'				# [W/m^2]
file_d01_SWUPB = parent_dir + '/L1/d01_SWUPB'				# [W/m^2]
file_d02_SWUPB = parent_dir + '/L1/d02_SWUPB'				# [W/m^2]
file_d01_SWDNB = parent_dir + '/L1/d01_SWDNB'				# [W/m^2]
file_d02_SWDNB = parent_dir + '/L1/d02_SWDNB'				# [W/m^2]
# Clear-sky
file_d01_LWUPTC = parent_dir + '/L1/d01_LWUPTC'				# [W/m^2]
file_d02_LWUPTC = parent_dir + '/L1/d02_LWUPTC'				# [W/m^2]
file_d01_LWDNTC = parent_dir + '/L1/d01_LWDNTC'				# [W/m^2]
file_d02_LWDNTC = parent_dir + '/L1/d02_LWDNTC'				# [W/m^2]
file_d01_LWUPBC = parent_dir + '/L1/d01_LWUPBC'				# [W/m^2]
file_d02_LWUPBC = parent_dir + '/L1/d02_LWUPBC'				# [W/m^2]
file_d01_LWDNBC = parent_dir + '/L1/d01_LWDNBC'				# [W/m^2]
file_d02_LWDNBC = parent_dir + '/L1/d02_LWDNBC'				# [W/m^2]
file_d01_SWUPTC = parent_dir + '/L1/d01_SWUPTC'				# [W/m^2]
file_d02_SWUPTC = parent_dir + '/L1/d02_SWUPTC'				# [W/m^2]
file_d01_SWDNTC = parent_dir + '/L1/d01_SWDNTC'				# [W/m^2]
file_d02_SWDNTC = parent_dir + '/L1/d02_SWDNTC'				# [W/m^2]
file_d01_SWUPBC = parent_dir + '/L1/d01_SWUPBC'				# [W/m^2]
file_d02_SWUPBC = parent_dir + '/L1/d02_SWUPBC'				# [W/m^2]
file_d01_SWDNBC = parent_dir + '/L1/d01_SWDNBC'				# [W/m^2]
file_d02_SWDNBC = parent_dir + '/L1/d02_SWDNBC'				# [W/m^2]

# Interpolated data 
file_d01_U = parent_dir + '/L2/d01_interp_U'	        # [m/s]
file_d02_U = parent_dir + '/L2/d02_interp_U'	        # [m/s]
file_d01_V = parent_dir + '/L2/d01_interp_V'	        # [m/s]
file_d02_V = parent_dir + '/L2/d02_interp_V'	        # [m/s]
file_d01_W = parent_dir + '/L2/d01_interp_W'	        # [m/s]
file_d02_W = parent_dir + '/L2/d02_interp_W'	        # [m/s]
file_d01_QV = parent_dir + '/L2/d01_interp_QV'	        # [kg/kg]
file_d02_QV = parent_dir + '/L2/d02_interp_QV'	        # [kg/kg]
file_d01_QC = parent_dir + '/L2/d01_interp_QC'	        # [kg/kg]
file_d02_QC = parent_dir + '/L2/d02_interp_QC'	        # [kg/kg]
file_d01_QR = parent_dir + '/L2/d01_interp_QR'	        # [kg/kg]
file_d02_QR = parent_dir + '/L2/d02_interp_QR'	        # [kg/kg]
file_d01_QI = parent_dir + '/L2/d01_interp_QI'	        # [kg/kg]
file_d02_QI = parent_dir + '/L2/d02_interp_QI'	        # [kg/kg]
file_d01_QS = parent_dir + '/L2/d01_interp_QS'	        # [kg/kg]
file_d02_QS = parent_dir + '/L2/d02_interp_QS'	        # [kg/kg]
file_d01_QG = parent_dir + '/L2/d01_interp_QG'	        # [kg/kg]
file_d02_QG = parent_dir + '/L2/d02_interp_QG'	        # [kg/kg]
file_d01_CLDFRA = parent_dir + '/L2/d01_interp_CLDFRA'	# 
file_d02_CLDFRA = parent_dir + '/L2/d02_interp_CLDFRA'	# 
file_d01_LH = parent_dir + '/L2/d01_interp_LH'	        # [K/s]
file_d02_LH = parent_dir + '/L2/d02_interp_LH'	        # [K/s]
file_d01_LWAll = parent_dir + '/L2/d01_interp_LWAll'	# [K/s]
file_d02_LWAll = parent_dir + '/L2/d02_interp_LWAll'	# [K/s]
file_d01_LWClear = parent_dir + '/L2/d01_interp_LWClear'# [K/s]
file_d02_LWClear = parent_dir + '/L2/d02_interp_LWClear'# [K/s]
file_d01_SWAll = parent_dir + '/L2/d01_interp_SWAll'	# [K/s]
file_d02_SWAll = parent_dir + '/L2/d02_interp_SWAll'	# [K/s]
file_d01_SWClear = parent_dir + '/L2/d01_interp_SWClear'# [K/s]
file_d02_SWClear = parent_dir + '/L2/d02_interp_SWClear'# [K/s]
file_d01_Theta = parent_dir + '/L2/d01_interp_Theta'	# [K]
file_d02_Theta = parent_dir + '/L2/d02_interp_Theta'	# [K]

######################################################################################
################ Declare the bounds you want to specifically look at #################
######################################################################################
#### All the data 
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-02T12')]
# lats = [-20, 20]
# lons = [80, 135]

#### Some of the data
times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-03T00')]
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-11-23T12')]
lats = [-7.5, 7.5]
lons = [90, 110]
######################################################################################
# Based on the bounds you set above, setup the indicies that will be used throughout
time_ind_d01, lat_ind_d01, lon_ind_d01 = isel_ind(file_d01_raw, times, lats, lons)
time_ind_d02, lat_ind_d02, lon_ind_d02 = isel_ind(file_d02_raw, times, lats, lons)

# Raw datasets
start_time = time.perf_counter()
ds_d01 = open_ds(file_d01_raw,time_ind_d01,lat_ind_d01,lon_ind_d01)
ds_d02 = open_ds(file_d02_raw,time_ind_d02,lat_ind_d02,lon_ind_d02)
step1_time = time.perf_counter()
print('Dataset loaded \N{check mark}', step1_time-start_time, 'seconds')

# Coordinate dictionaries:
step2_time = time.perf_counter()
interp_P_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))

interp_P_levels_d01 = np.resize(interp_P_levels,(ds_d01.XLAT.shape[2],ds_d01.XLAT.shape[1],len(interp_P_levels)))
interp_P_levels_d01 = np.swapaxes(interp_P_levels_d01, 0, 2)
d01_coords = dict(
    XLAT=(('Time','south_north','west_east'),ds_d01.XLAT.values),
    XLONG=(('Time','south_north','west_east'),ds_d01.XLONG.values),
    bottom_top=(('bottom_top'),interp_P_levels),
    XTIME=('Time',ds_d01.XTIME.values),
    south_north=(('south_north'),ds_d01.XLAT[0,:,0].values),
    west_east=(('west_east'),ds_d01.XLONG[0,0,:].values)
    )
interp_P_levels_d02 = np.resize(interp_P_levels,(ds_d02.XLAT.shape[2],ds_d02.XLAT.shape[1],len(interp_P_levels)))
interp_P_levels_d02 = np.swapaxes(interp_P_levels_d02, 0, 2)
d02_coords = dict(
    XLAT=(('Time','south_north','west_east'),ds_d02.XLAT.values),
    XLONG=(('Time','south_north','west_east'),ds_d02.XLONG.values),
    bottom_top=(('bottom_top'),interp_P_levels),
    XTIME=('Time',ds_d02.XTIME.values),
    south_north=(('south_north'),ds_d02.XLAT[0,:,0].values),
    west_east=(('west_east'),ds_d02.XLONG[0,0,:].values)
    )

# Function that can removes the bottom_top dimension for 2-D datasets
def without_keys(d, keys):
	return {x: d[x] for x in d if x not in keys}
step1_time = time.perf_counter()
print('Created coordinate dictionaries \N{check mark}', step1_time-step2_time, 'seconds')

###########################################################
################# Load in the variables ###################
###########################################################

# Set the parameters for the specific island
    # Central Western Sumatra
start_coord		= [-1.8,103.8]
end_coord 		= [-5.8,99.8]
width			= 1.5
dx 				= 0.025
theta           = pi/4

##################### 3-D variables #######################

############ Interpolated zonal winds   [m/s] #############
step2_time = time.perf_counter()
# d01   # how I used to open data: ds = xr.open_dataset(file_d01_U).isel(Time=slice(0,t))
ds = open_ds(file_d01_U,time_ind_d01,lat_ind_d01,lon_ind_d01)
da_d01_U = ds['U'].compute()
da_d01_U = da_d01_U.assign_coords(d01_coords)
fill_value_f8 = da_d01_U.max()      # This is the fill_value meaning missing_data
# da_d01_U = da_d01_U.where(da_d01_U!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_U,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_U = ds['U'].compute()
# da_d02_U = da_d02_U.assign_coords(d02_coords)
# da_d02_U = da_d02_U.where(da_d02_U!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated zonal winds loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Interpolated Meridional winds   [m/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_V,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_V = ds['V'].compute()
# da_d01_V = da_d01_V.assign_coords(d01_coords)
# da_d01_V = da_d01_V.where(da_d01_V!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_V,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_V = ds['V'].compute()
# da_d02_V = da_d02_V.assign_coords(d02_coords)
# da_d02_V = da_d02_V.where(da_d02_V!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated meridional winds loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################################ Calculate Normal Wind ################################
# ## d01
# # Rotate the coordinate system w.r.t Sumatra's angle
# da_u, da_v = rotate_vec(da_d01_U, da_d01_V, theta)
# ################ Normal Wind - Cross-section Analysis ################
# da_cross_temp, all_line_coords = cross_section_multi(da_u, start_coord, end_coord, width, dx)
# # Create distance coordinate
# distance = np.linspace(0,dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]),da_cross_temp.shape[2])
# # # Mannually checked which indicies were closest to the coast for d01 (where nan's end)
# distance_d01 = distance - distance[16]
# # Create da with coordinates
# da_d01_NormalWind_cross = make_da_cross(da_d01_U, da_cross_temp, 'NormalWind', distance_d01, width, all_line_coords)
# da_d01_NormalWind_cross.to_netcdf('./d01_cross_NormalWind')

## d02
# da_u, da_v = rotate_vec(da_d02_U, da_d02_V, theta)
# ################ Normal Wind - Cross-section Analysis ################
# da_cross_temp, all_line_coords = cross_section_multi(da_u, start_coord, end_coord, width, dx)
# Create distance coordinate
# distance = np.linspace(0,dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]),da_cross_temp.shape[2])
# # Mannually checked which indicies were closest to the coast for d02 (where nan's end)
# distance_d02 = distance - distance[63]
# # Create da with coordinates
# da_d02_NormalWind_cross = make_da_cross(da_d02_U, da_cross_temp, 'NormalWind', distance_d02, width, all_line_coords)
# da_d02_NormalWind_cross.to_netcdf('./d02_cross_NormalWind')
# # Delete variables after to aliviate memory strain
# del da_d01_NormalWind_cross, da_d02_NormalWind_cross, da_d01_U, da_d01_V, da_d02_U, da_d02_V

# ############ Interpolated Vertical winds   [m/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_W,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_W = ds['W'].compute()
# da_d01_W = da_d01_W.assign_coords(d01_coords)
# da_d01_W = da_d01_W.where(da_d01_W!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_W,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_W = ds['W'].compute()
# da_d02_W = da_d02_W.assign_coords(d02_coords)
# da_d02_W = da_d02_W.where(da_d02_W!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated vertical winds loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############################ Vertical Wind - Cross-section Analysis ############################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_W, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_W_cross = make_da_cross(da_d01_W, da_cross_temp, 'W', distance_d01, width, all_line_coords)
# da_d01_W_cross.to_netcdf('./d01_cross_W')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_W, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_W_cross = make_da_cross(da_d02_W, da_cross_temp, 'W', distance_d02, width, all_line_coords)
# da_d02_W_cross.to_netcdf('./d02_cross_W')
# # Delete variables after to aliviate memory strain
# del da_d01_W_cross, da_d02_W_cross, da_d01_W, da_d02_W

# ############ Interpolated water vapor mixing ratio  [kg/kg] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_QV,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_QV = ds['QV'].compute()
# da_d01_QV = da_d01_QV.assign_coords(d01_coords)
# da_d01_QV = da_d01_QV.where(da_d01_QV!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_QV,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_QV = ds['QV'].compute()
# da_d02_QV = da_d02_QV.assign_coords(d02_coords)
# da_d02_QV = da_d02_QV.where(da_d02_QV!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated water vapor mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############################ Water Vapor - Cross-section Analysis ############################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_QV, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_QV_cross = make_da_cross(da_d01_QV, da_cross_temp, 'QV', distance_d01, width, all_line_coords)
# da_d01_QV_cross.to_netcdf('./d01_cross_QV')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_QV, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_QV_cross = make_da_cross(da_d02_QV, da_cross_temp, 'QV', distance_d02, width, all_line_coords)
# da_d02_QV_cross.to_netcdf('./d02_cross_QV')
# # Delete variables after to aliviate memory strain
# del da_d01_QV_cross, da_d02_QV_cross, da_d01_QV, da_d02_QV

# ############ Interpolated cloud water mixing ratio  [kg/kg] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_QC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_QC = ds['QC'].compute()
# da_d01_QC = da_d01_QC.assign_coords(d01_coords)
# da_d01_QC = da_d01_QC.where(da_d01_QC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_QC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_QC = ds['QC'].compute()
# da_d02_QC = da_d02_QC.assign_coords(d02_coords)
# da_d02_QC = da_d02_QC.where(da_d02_QC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############################ Cloud water - Cross-section Analysis ############################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_QC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_QC_cross = make_da_cross(da_d01_QC, da_cross_temp, 'QC', distance_d01, width, all_line_coords)
# da_d01_QC_cross.to_netcdf('./d01_cross_QC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_QC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_QC_cross = make_da_cross(da_d02_QC, da_cross_temp, 'QC', distance_d02, width, all_line_coords)
# da_d02_QC_cross.to_netcdf('./d02_cross_QC')
# # Delete variables after to aliviate memory strain
# del da_d01_QC_cross, da_d02_QC_cross, da_d01_QC, da_d02_QC

# ############ Interpolated rain water mixing ratio  [kg/kg] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_QR,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_QR = ds['QR'].compute()
# da_d01_QR = da_d01_QR.assign_coords(d01_coords)
# da_d01_QR = da_d01_QR.where(da_d01_QR!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_QR,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_QR = ds['QR'].compute()
# da_d02_QR = da_d02_QR.assign_coords(d02_coords)
# da_d02_QR = da_d02_QR.where(da_d02_QR!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############################ Rain water - Cross-section Analysis ############################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_QR, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_QR_cross = make_da_cross(da_d01_QR, da_cross_temp, 'QR', distance_d01, width, all_line_coords)
# da_d01_QR_cross.to_netcdf('./d01_cross_QR')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_QR, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_QR_cross = make_da_cross(da_d02_QR, da_cross_temp, 'QR', distance_d02, width, all_line_coords)
# da_d02_QR_cross.to_netcdf('./d02_cross_QR')
# # Delete variables after to aliviate memory strain
# del da_d01_QR_cross, da_d02_QR_cross, da_d01_QR, da_d02_QR

# ############ Interpolated ice mixing ratio  [kg/kg] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_QI,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_QI = ds['QI'].compute()
# da_d01_QI = da_d01_QI.assign_coords(d01_coords)
# da_d01_QI = da_d01_QI.where(da_d01_QI!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_QI,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_QI = ds['QI'].compute()
# da_d02_QI = da_d02_QI.assign_coords(d02_coords)
# da_d02_QI = da_d02_QI.where(da_d02_QI!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############################ Ice mixing ratio - Cross-section Analysis ############################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_QI, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_QI_cross = make_da_cross(da_d01_QI, da_cross_temp, 'QI', distance_d01, width, all_line_coords)
# da_d01_QI_cross.to_netcdf('./d01_cross_QI')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_QI, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_QI_cross = make_da_cross(da_d02_QI, da_cross_temp, 'QI', distance_d02, width, all_line_coords)
# da_d02_QI_cross.to_netcdf('./d02_cross_QI')
# # Delete variables after to aliviate memory strain
# del da_d01_QI_cross, da_d02_QI_cross, da_d01_QI, da_d02_QI

# ############ Interpolated snow mixing ratio  [kg/kg] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_QS,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_QS = ds['QS'].compute()
# da_d01_QS = da_d01_QS.assign_coords(d01_coords)
# da_d01_QS = da_d01_QS.where(da_d01_QS!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_QS,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_QS = ds['QS'].compute()
# da_d02_QS = da_d02_QS.assign_coords(d02_coords)
# da_d02_QS = da_d02_QS.where(da_d02_QS!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############################ Snow mixing ratio - Cross-section Analysis ############################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_QS, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_QS_cross = make_da_cross(da_d01_QS, da_cross_temp, 'QS', distance_d01, width, all_line_coords)
# da_d01_QS_cross.to_netcdf('./d01_cross_QS')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_QS, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_QS_cross = make_da_cross(da_d02_QS, da_cross_temp, 'QS', distance_d02, width, all_line_coords)
# da_d02_QS_cross.to_netcdf('./d02_cross_QS')
# # Delete variables after to aliviate memory strain
# del da_d01_QS_cross, da_d02_QS_cross, da_d01_QS, da_d02_QS

# ############ Interpolated graupel mixing ratio  [kg/kg] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_QG,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_QG = ds['QG'].compute()
# da_d01_QG = da_d01_QG.assign_coords(d01_coords)
# da_d01_QG = da_d01_QG.where(da_d01_QG!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_QG,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_QG = ds['QG'].compute()
# da_d02_QG = da_d02_QG.assign_coords(d02_coords)
# da_d02_QG = da_d02_QG.where(da_d02_QG!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated cloud water mixing ratio loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############################ Graupel mixing ratio - Cross-section Analysis ############################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_QG, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_QG_cross = make_da_cross(da_d01_QG, da_cross_temp, 'QG', distance_d01, width, all_line_coords)
# da_d01_QG_cross.to_netcdf('./d01_cross_QG')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_QG, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_QG_cross = make_da_cross(da_d02_QG, da_cross_temp, 'QG', distance_d02, width, all_line_coords)
# da_d02_QG_cross.to_netcdf('./d02_cross_QG')
# # Delete variables after to aliviate memory strain
# del da_d01_QG_cross, da_d02_QG_cross, da_d01_QG, da_d02_QG

# ############ Cloud Fraction [0-1] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_CLDFRA,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_CLDFRA = ds['CLDFRA'].compute()
# da_d01_CLDFRA = da_d01_CLDFRA.assign_coords(d01_coords)
# da_d01_CLDFRA = da_d01_CLDFRA.where(da_d01_CLDFRA!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_CLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_CLDFRA = ds['CLDFRA'].compute()
# da_d02_CLDFRA = da_d02_CLDFRA.assign_coords(d02_coords)
# da_d02_CLDFRA = da_d02_CLDFRA.where(da_d02_CLDFRA!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Cloud fraction loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################ Cloud Fraction - Cross-section Analysis ################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_CLDFRA, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_CLDFRA_cross = make_da_cross(da_d01_CLDFRA, da_cross_temp, 'CLDFRA', distance_d01, width, all_line_coords)
# da_d01_CLDFRA_cross.to_netcdf('./d01_cross_CLDFRA')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_CLDFRA, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_CLDFRA_cross = make_da_cross(da_d02_CLDFRA, da_cross_temp, 'CLDFRA', distance_d02, width, all_line_coords)
# da_d02_CLDFRA_cross.to_netcdf('./d02_cross_CLDFRA')
# # Delete variables after to aliviate memory strain
# del da_d01_CLDFRA_cross, da_d02_CLDFRA_cross, da_d01_CLDFRA, da_d02_CLDFRA

# ############ Latent Heating [K/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LH,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LH = ds['LH'].compute()
# da_d01_LH = da_d01_LH.assign_coords(d01_coords)
# da_d01_LH = da_d01_LH.where(da_d01_LH!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LH,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LH = ds['LH'].compute()
# da_d02_LH = da_d02_LH.assign_coords(d02_coords)
# da_d02_LH = da_d02_LH.where(da_d02_LH!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Latent heating loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################ Latent Heat - Cross-section Analysis ################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LH, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LH_cross = make_da_cross(da_d01_LH, da_cross_temp, 'LH', distance_d01, width, all_line_coords)
# da_d01_LH_cross.to_netcdf('./d01_cross_LH')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LH, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LH_cross = make_da_cross(da_d02_LH, da_cross_temp, 'LH', distance_d02, width, all_line_coords)
# da_d02_LH_cross.to_netcdf('./d02_cross_LH')
# # Delete variables after to aliviate memory strain
# del da_d01_LH_cross, da_d02_LH_cross, da_d01_LH, da_d02_LH

# ############ Potential Temperature [K] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_Theta,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_Theta = ds['Theta'].compute()
# da_d01_Theta = da_d01_Theta.assign_coords(d01_coords)
# da_d01_Theta = da_d01_Theta.where(da_d01_Theta!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_Theta,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_Theta = ds['Theta'].compute()
# da_d02_Theta = da_d02_Theta.assign_coords(d02_coords)
# da_d02_Theta = da_d02_Theta.where(da_d02_Theta!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Shortwave Clear-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################ Potential Temperature/Theta - Cross-section Analysis ################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_Theta, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_Theta_cross = make_da_cross(da_d01_Theta, da_cross_temp, 'Theta', distance_d01, width, all_line_coords)
# da_d01_Theta_cross.to_netcdf('./d01_cross_Theta')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_Theta, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_Theta_cross = make_da_cross(da_d02_Theta, da_cross_temp, 'Theta', distance_d02, width, all_line_coords)
# da_d02_Theta_cross.to_netcdf('./d02_cross_Theta')
# # Delete variables after to aliviate memory strain
# del da_d01_Theta_cross, da_d02_Theta_cross, da_d01_Theta, da_d02_Theta

# ############ Longwave Radiative Heating All-Sky [K/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWAll,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWAll = ds['LWAll'].compute()
# da_d01_LWAll = da_d01_LWAll.assign_coords(d01_coords)
# da_d01_LWAll = da_d01_LWAll.where(da_d01_LWAll!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWAll,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWAll = ds['LWAll'].compute()
# da_d02_LWAll = da_d02_LWAll.assign_coords(d02_coords)
# da_d02_LWAll = da_d02_LWAll.where(da_d02_LWAll!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Longwave All-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################ Longwave All - Cross-section Analysis ################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWAll, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWAll_cross = make_da_cross(da_d01_LWAll, da_cross_temp, 'LWAll', distance_d01, width, all_line_coords)
# da_d01_LWAll_cross.to_netcdf('./d01_cross_LWAll')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWAll, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWAll_cross = make_da_cross(da_d02_LWAll, da_cross_temp, 'LWAll', distance_d02, width, all_line_coords)
# da_d02_LWAll_cross.to_netcdf('./d02_cross_LWAll')
# # Delete variables after to aliviate memory strain
# del da_d01_LWAll, da_d02_LWAll

# ############ Longwave Radiative Heating Clear-Sky [K/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWClear,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWClear = ds['LWClear'].compute()
# da_d01_LWClear = da_d01_LWClear.assign_coords(d01_coords)
# da_d01_LWClear = da_d01_LWClear.where(da_d01_LWClear!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWClear,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWClear = ds['LWClear'].compute()
# da_d02_LWClear = da_d02_LWClear.assign_coords(d02_coords)
# da_d02_LWClear = da_d02_LWClear.where(da_d02_LWClear!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Longwave Clear-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################ Longwave Clear - Cross-section Analysis ################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWClear, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWClear_cross = make_da_cross(da_d01_LWClear, da_cross_temp, 'LWClear', distance_d01, width, all_line_coords)
# da_d01_LWClear_cross.to_netcdf('./d01_cross_LWClear')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWClear, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWClear_cross = make_da_cross(da_d02_LWClear, da_cross_temp, 'LWClear', distance_d02, width, all_line_coords)
# da_d02_LWClear_cross.to_netcdf('./d02_cross_LWClear')
# # Delete variables after to aliviate memory strain
# del da_d01_LWClear, da_d02_LWClear

# ################ Longwave CRF - Cross-section Analysis ################
# # Calculate the CRF
# da_d01_LWCRF_cross = da_d01_LWAll_cross - da_d01_LWClear_cross
# da_d01_LWCRF_cross = xr.DataArray(data=da_d01_LWCRF_cross, name='LWCRF')
# da_d01_LWCRF_cross.to_netcdf('./d01_cross_LWCRF')
# da_d02_LWCRF_cross = da_d02_LWAll_cross - da_d02_LWClear_cross
# da_d02_LWCRF_cross = xr.DataArray(data=da_d02_LWCRF_cross, name='LWCRF')
# da_d02_LWCRF_cross.to_netcdf('./d02_cross_LWCRF')
# # Delete variables after to aliviate memory strain
# del da_d01_LWAll_cross, da_d02_LWAll_cross, da_d01_LWClear_cross, da_d02_LWClear_cross

# ############ Shortwave Radiative Heating All-Sky [K/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWAll,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWAll = ds['SWAll'].compute()
# da_d01_SWAll = da_d01_SWAll.assign_coords(d01_coords)
# da_d01_SWAll = da_d01_SWAll.where(da_d01_SWAll!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWAll,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWAll = ds['SWAll'].compute()
# da_d02_SWAll = da_d02_SWAll.assign_coords(d02_coords)
# da_d02_SWAll = da_d02_SWAll.where(da_d02_SWAll!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Shortwave All-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################ Shortwave All - Cross-section Analysis ################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWAll, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWAll_cross = make_da_cross(da_d01_SWAll, da_cross_temp, 'SWAll', distance_d01, width, all_line_coords)
# da_d01_SWAll_cross.to_netcdf('./d01_cross_SWAll')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWAll, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWAll_cross = make_da_cross(da_d02_SWAll, da_cross_temp, 'SWAll', distance_d02, width, all_line_coords)
# da_d02_SWAll_cross.to_netcdf('./d02_cross_SWAll')
# # Delete variables after to aliviate memory strain
# del da_d01_SWAll, da_d02_SWAll

# ############ Shortwave Radiative Heating Clear-Sky [K/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWClear,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWClear = ds['SWClear'].compute()
# da_d01_SWClear = da_d01_SWClear.assign_coords(d01_coords)
# da_d01_SWClear = da_d01_SWClear.where(da_d01_SWClear!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWClear,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWClear = ds['SWClear'].compute()
# da_d02_SWClear = da_d02_SWClear.assign_coords(d02_coords)
# da_d02_SWClear = da_d02_SWClear.where(da_d02_SWClear!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Shortwave Clear-sky loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################ Shortwave Clear - Cross-section Analysis ################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWClear, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWClear_cross = make_da_cross(da_d01_SWClear, da_cross_temp, 'SWClear', distance_d01, width, all_line_coords)
# da_d01_SWClear_cross.to_netcdf('./d01_cross_SWClear')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWClear, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWClear_cross = make_da_cross(da_d02_SWClear, da_cross_temp, 'SWClear', distance_d02, width, all_line_coords)
# da_d02_SWClear_cross.to_netcdf('./d02_cross_SWClear')
# # Delete variables after to aliviate memory strain
# del da_d01_SWClear, da_d02_SWClear

# ################ Shortwave CRF - Cross-section Analysis ################
# # Calculate the CRF
# da_d01_SWCRF_cross = da_d01_SWAll_cross - da_d01_SWClear_cross
# da_d01_SWCRF_cross = xr.DataArray(data=da_d01_SWCRF_cross, name='SWCRF')
# da_d01_SWCRF_cross.to_netcdf('./d01_cross_SWCRF')
# da_d02_SWCRF_cross = da_d02_SWAll_cross - da_d02_SWClear_cross
# da_d02_SWCRF_cross = xr.DataArray(data=da_d02_SWCRF_cross, name='SWCRF')
# da_d02_SWCRF_cross.to_netcdf('./d02_cross_SWCRF')
# # Delete variables after to aliviate memory strain
# del da_d01_SWClear_cross, da_d02_SWClear_cross, da_d01_SWAll_cross, da_d02_SWAll_cross

# ################ Total CRF - Cross-section Analysis ################
# # Calculate the Total CRF
# da_d01_TotalCRF_cross = da_d01_SWCRF_cross + da_d01_LWCRF_cross
# da_d01_TotalCRF_cross = xr.DataArray(data=da_d01_TotalCRF_cross, name='TotalCRF')
# da_d01_TotalCRF_cross.to_netcdf('./d01_cross_TotalCRF')
# da_d02_TotalCRF_cross = da_d02_SWCRF_cross + da_d02_LWCRF_cross
# da_d02_TotalCRF_cross = xr.DataArray(data=da_d02_TotalCRF_cross, name='TotalCRF')
# da_d02_TotalCRF_cross.to_netcdf('./d02_cross_TotalCRF')
# # Delete variables after to aliviate memory strain
# del da_d01_TotalCRF_cross, da_d02_TotalCRF_cross, da_d01_SWCRF_cross, da_d01_LWCRF_cross, da_d02_SWCRF_cross, da_d02_LWCRF_cross

# ######################################################################################################################
# ##################### 2-D variables ##################################################################################
# ######################################################################################################################

# ############ Rain Rate     [mm/hr] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_RR,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_RR = ds['RR'].compute()
# da_d01_RR = da_d01_RR.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_RR = da_d01_RR.where(da_d01_RR!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_RR,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_RR = ds['RR'].compute()
# da_d02_RR = da_d02_RR.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_RR = da_d02_RR.where(da_d02_RR!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ################ Rain Rate - Cross-section Analysis #################
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_RR, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_RR_cross = make_da_cross(da_d01_RR, da_cross_temp, 'RR', distance_d01, width, all_line_coords)
# da_d01_RR_cross.to_netcdf('./d01_cross_RR')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_RR, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_RR_cross = make_da_cross(da_d02_RR, da_cross_temp, 'RR', distance_d02, width, all_line_coords)
# da_d02_RR_cross.to_netcdf('./d02_cross_RR')
# # Delete variables after to aliviate memory strain
# del da_d01_RR, da_d02_RR, da_d01_RR_cross, da_d02_RR_cross

############ Upward Heat Flux at Surface     [W/m^2] ############
step2_time = time.perf_counter()
# d01
ds = open_ds(file_d01_HFX,time_ind_d01,lat_ind_d01,lon_ind_d01)
da_d01_HFX = ds['HFX'].compute()
da_d01_HFX = da_d01_HFX.assign_coords(without_keys(d01_coords,'bottom_top'))
da_d01_HFX = da_d01_HFX.where(da_d01_HFX!=fill_value_f8)    # Change fill_value points to nans
# d02
ds = open_ds(file_d02_HFX,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_HFX = ds['HFX'].compute()
da_d02_HFX = da_d02_HFX.assign_coords(without_keys(d02_coords,'bottom_top'))
da_d02_HFX = da_d02_HFX.where(da_d02_HFX!=fill_value_f8)    # Change fill_value points to nans

step1_time = time.perf_counter()
print('Upward Heat Flux at Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')


################ Upward Heat Flux at Surface - Cross-section Analysis #################
# d01
da_cross_temp, all_line_coords = cross_section_multi(da_d01_HFX, start_coord, end_coord, width, dx)
da_cross_temp.shape

# Create distance coordinate
distance = np.linspace(0,dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]),da_cross_temp.shape[1])
# Mannually checked which indicies were closest to the coast for d01 (where nan's end)
distance_d01 = distance - distance[16]
# Create da with coordinates
da_d01_HFX_cross = make_da_cross(da_d01_HFX, da_cross_temp, 'HFX', distance_d01, width, all_line_coords)
da_d01_HFX_cross.to_netcdf('./d01_cross_HFX')
# d02
da_cross_temp, all_line_coords = cross_section_multi(da_d02_HFX, start_coord, end_coord, width, dx)
# Create distance coordinate
distance = np.linspace(0,dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]),da_cross_temp.shape[1])
# Mannually checked which indicies were closest to the coast for d02 (where nan's end)
distance_d02 = distance - distance[63]
# Create da with coordinates
da_d02_HFX_cross = make_da_cross(da_d02_HFX, da_cross_temp, 'HFX', distance_d02, width, all_line_coords)
da_d02_HFX_cross.to_netcdf('./d02_cross_HFX')
# Delete variables after to aliviate memory strain
del da_d01_HFX, da_d02_HFX, da_d01_HFX_cross, da_d02_HFX_cross

# ###########################################################################
# ################################# All-sky #################################
# ###########################################################################

# ############ Load Longwave Upwelling at TOA ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWUPT,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWUPT = ds['LWUPT'].compute()
# da_d01_LWUPT = da_d01_LWUPT.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_LWUPT = da_d01_LWUPT.where(da_d01_LWUPT!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWUPT,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWUPT = ds['LWUPT'].compute()
# da_d02_LWUPT = da_d02_LWUPT.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_LWUPT = da_d02_LWUPT.where(da_d02_LWUPT!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Longwave Upwelling at TOA ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWUPT, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWUPT_cross = make_da_cross(da_d01_LWUPT, da_cross_temp, 'LWUPT', distance_d01, width, all_line_coords)
# da_d01_LWUPT_cross.to_netcdf('./d01_cross_LWUPT')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWUPT, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWUPT_cross = make_da_cross(da_d02_LWUPT, da_cross_temp, 'LWUPT', distance_d02, width, all_line_coords)
# da_d02_LWUPT_cross.to_netcdf('./d02_cross_LWUPT')
# # Delete variables after to aliviate memory strain
# del da_d01_LWUPT, da_d02_LWUPT, da_d01_LWUPT_cross, da_d02_LWUPT_cross

# ############ Load Longwave Downwelling at TOA ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWDNT,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWDNT = ds['LWDNT'].compute()
# da_d01_LWDNT = da_d01_LWDNT.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_LWDNT = da_d01_LWDNT.where(da_d01_LWDNT!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWDNT,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWDNT = ds['LWDNT'].compute()
# da_d02_LWDNT = da_d02_LWDNT.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_LWDNT = da_d02_LWDNT.where(da_d02_LWDNT!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Longwave Downwelling at TOA ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWDNT, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWDNT_cross = make_da_cross(da_d01_LWDNT, da_cross_temp, 'LWDNT', distance_d01, width, all_line_coords)
# da_d01_LWDNT_cross.to_netcdf('./d01_cross_LWDNT')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWDNT, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWDNT_cross = make_da_cross(da_d02_LWDNT, da_cross_temp, 'LWDNT', distance_d02, width, all_line_coords)
# da_d02_LWDNT_cross.to_netcdf('./d02_cross_LWDNT')
# # Delete variables after to aliviate memory strain
# del da_d01_LWDNT, da_d02_LWDNT, da_d01_LWDNT_cross, da_d02_LWDNT_cross

# ############ Load Longwave Upwelling at SFC ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWUPB,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWUPB = ds['LWUPB'].compute()
# da_d01_LWUPB = da_d01_LWUPB.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_LWUPB = da_d01_LWUPB.where(da_d01_LWUPB!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWUPB,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWUPB = ds['LWUPB'].compute()
# da_d02_LWUPB = da_d02_LWUPB.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_LWUPB = da_d02_LWUPB.where(da_d02_LWUPB!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Longwave Upwelling at SFC ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWUPB, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWUPB_cross = make_da_cross(da_d01_LWUPB, da_cross_temp, 'LWUPB', distance_d01, width, all_line_coords)
# da_d01_LWUPB_cross.to_netcdf('./d01_cross_LWUPB')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWUPB, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWUPB_cross = make_da_cross(da_d02_LWUPB, da_cross_temp, 'LWUPB', distance_d02, width, all_line_coords)
# da_d02_LWUPB_cross.to_netcdf('./d02_cross_LWUPB')
# # Delete variables after to aliviate memory strain
# del da_d01_LWUPB, da_d02_LWUPB, da_d01_LWUPB_cross, da_d02_LWUPB_cross

# ############ Load Longwave Downwelling at SFC ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWDNB,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWDNB = ds['LWDNB'].compute()
# da_d01_LWDNB = da_d01_LWDNB.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_LWDNB = da_d01_LWDNB.where(da_d01_LWDNB!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWDNB,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWDNB = ds['LWDNB'].compute()
# da_d02_LWDNB = da_d02_LWDNB.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_LWDNB = da_d02_LWDNB.where(da_d02_LWDNB!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Longwave Downwelling at SFC ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWDNB, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWDNB_cross = make_da_cross(da_d01_LWDNB, da_cross_temp, 'LWDNB', distance_d01, width, all_line_coords)
# da_d01_LWDNB_cross.to_netcdf('./d01_cross_LWDNB')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWDNB, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWDNB_cross = make_da_cross(da_d02_LWDNB, da_cross_temp, 'LWDNB', distance_d02, width, all_line_coords)
# da_d02_LWDNB_cross.to_netcdf('./d02_cross_LWDNB')
# # Delete variables after to aliviate memory strain
# del da_d01_LWDNB, da_d02_LWDNB, da_d01_LWDNB_cross, da_d02_LWDNB_cross

# ############ Load Shortwave Upwelling at TOA ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWUPT,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWUPT = ds['SWUPT'].compute()
# da_d01_SWUPT = da_d01_SWUPT.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_SWUPT = da_d01_SWUPT.where(da_d01_SWUPT!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWUPT,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWUPT = ds['SWUPT'].compute()
# da_d02_SWUPT = da_d02_SWUPT.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_SWUPT = da_d02_SWUPT.where(da_d02_SWUPT!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Shortwave Upwelling at TOA ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWUPT, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWUPT_cross = make_da_cross(da_d01_SWUPT, da_cross_temp, 'SWUPT', distance_d01, width, all_line_coords)
# da_d01_SWUPT_cross.to_netcdf('./d01_cross_SWUPT')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWUPT, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWUPT_cross = make_da_cross(da_d02_SWUPT, da_cross_temp, 'SWUPT', distance_d02, width, all_line_coords)
# da_d02_SWUPT_cross.to_netcdf('./d02_cross_SWUPT')
# # Delete variables after to aliviate memory strain
# del da_d01_SWUPT, da_d02_SWUPT, da_d01_SWUPT_cross, da_d02_SWUPT_cross

# ############ Load Shortwave Downwelling at TOA ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWDNT,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWDNT = ds['SWDNT'].compute()
# da_d01_SWDNT = da_d01_SWDNT.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_SWDNT = da_d01_SWDNT.where(da_d01_SWDNT!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWDNT,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWDNT = ds['SWDNT'].compute()
# da_d02_SWDNT = da_d02_SWDNT.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_SWDNT = da_d02_SWDNT.where(da_d02_SWDNT!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Shortwave Downwelling at TOA ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWDNT, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWDNT_cross = make_da_cross(da_d01_SWDNT, da_cross_temp, 'SWDNT', distance_d01, width, all_line_coords)
# da_d01_SWDNT_cross.to_netcdf('./d01_cross_SWDNT')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWDNT, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWDNT_cross = make_da_cross(da_d02_SWDNT, da_cross_temp, 'SWDNT', distance_d02, width, all_line_coords)
# da_d02_SWDNT_cross.to_netcdf('./d02_cross_SWDNT')
# # Delete variables after to aliviate memory strain
# del da_d01_SWDNT, da_d02_SWDNT, da_d01_SWDNT_cross, da_d02_SWDNT_cross

# ############ Load Shortwave Upwelling at SFC ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWUPB,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWUPB = ds['SWUPB'].compute()
# da_d01_SWUPB = da_d01_SWUPB.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_SWUPB = da_d01_SWUPB.where(da_d01_SWUPB!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWUPB,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWUPB = ds['SWUPB'].compute()
# da_d02_SWUPB = da_d02_SWUPB.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_SWUPB = da_d02_SWUPB.where(da_d02_SWUPB!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Shortwave Upwelling at SFC ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWUPB, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWUPB_cross = make_da_cross(da_d01_SWUPB, da_cross_temp, 'SWUPB', distance_d01, width, all_line_coords)
# da_d01_SWUPB_cross.to_netcdf('./d01_cross_SWUPB')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWUPB, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWUPB_cross = make_da_cross(da_d02_SWUPB, da_cross_temp, 'SWUPB', distance_d02, width, all_line_coords)
# da_d02_SWUPB_cross.to_netcdf('./d02_cross_SWUPB')
# # Delete variables after to aliviate memory strain
# del da_d01_SWUPB, da_d02_SWUPB, da_d01_SWUPB_cross, da_d02_SWUPB_cross

# ############ Load Shortwave Downwelling at SFC ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWDNB,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWDNB = ds['SWDNB'].compute()
# da_d01_SWDNB = da_d01_SWDNB.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_SWDNB = da_d01_SWDNB.where(da_d01_SWDNB!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWDNB,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWDNB = ds['SWDNB'].compute()
# da_d02_SWDNB = da_d02_SWDNB.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_SWDNB = da_d02_SWDNB.where(da_d02_SWDNB!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Shortwave Downwelling at SFC ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWDNB, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWDNB_cross = make_da_cross(da_d01_SWDNB, da_cross_temp, 'SWDNB', distance_d01, width, all_line_coords)
# da_d01_SWDNB_cross.to_netcdf('./d01_cross_SWDNB')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWDNB, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWDNB_cross = make_da_cross(da_d02_SWDNB, da_cross_temp, 'SWDNB', distance_d02, width, all_line_coords)
# da_d02_SWDNB_cross.to_netcdf('./d02_cross_SWDNB')
# # Delete variables after to aliviate memory strain
# del da_d01_SWDNB, da_d02_SWDNB, da_d01_SWDNB_cross, da_d02_SWDNB_cross

# #############################################################################
# ################################# Clear-sky #################################
# #############################################################################

# ############ Load Longwave Upwelling at TOA ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWUPTC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWUPTC = ds['LWUPTC'].compute()
# da_d01_LWUPTC = da_d01_LWUPTC.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_LWUPTC = da_d01_LWUPTC.where(da_d01_LWUPTC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWUPTC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWUPTC = ds['LWUPTC'].compute()
# da_d02_LWUPTC = da_d02_LWUPTC.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_LWUPTC = da_d02_LWUPTC.where(da_d02_LWUPTC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Longwave Upwelling at TOA ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWUPTC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWUPTC_cross = make_da_cross(da_d01_LWUPTC, da_cross_temp, 'LWUPTC', distance_d01, width, all_line_coords)
# da_d01_LWUPTC_cross.to_netcdf('./d01_cross_LWUPTC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWUPTC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWUPTC_cross = make_da_cross(da_d02_LWUPTC, da_cross_temp, 'LWUPTC', distance_d02, width, all_line_coords)
# da_d02_LWUPTC_cross.to_netcdf('./d02_cross_LWUPTC')
# # Delete variables after to aliviate memory strain
# del da_d01_LWUPTC, da_d02_LWUPTC, da_d01_LWUPTC_cross, da_d02_LWUPTC_cross

# ############ Load Longwave Downwelling at TOA ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWDNTC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWDNTC = ds['LWDNTC'].compute()
# da_d01_LWDNTC = da_d01_LWDNTC.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_LWDNTC = da_d01_LWDNTC.where(da_d01_LWDNTC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWDNTC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWDNTC = ds['LWDNTC'].compute()
# da_d02_LWDNTC = da_d02_LWDNTC.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_LWDNTC = da_d02_LWDNTC.where(da_d02_LWDNTC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Longwave Downwelling at TOA ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWDNTC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWDNTC_cross = make_da_cross(da_d01_LWDNTC, da_cross_temp, 'LWDNTC', distance_d01, width, all_line_coords)
# da_d01_LWDNTC_cross.to_netcdf('./d01_cross_LWDNTC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWDNTC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWDNTC_cross = make_da_cross(da_d02_LWDNTC, da_cross_temp, 'LWDNTC', distance_d02, width, all_line_coords)
# da_d02_LWDNTC_cross.to_netcdf('./d02_cross_LWDNTC')
# # Delete variables after to aliviate memory strain
# del da_d01_LWDNTC, da_d02_LWDNTC, da_d01_LWDNTC_cross, da_d02_LWDNTC_cross

# ############ Load Longwave Upwelling at SFC ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWUPBC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWUPBC = ds['LWUPBC'].compute()
# da_d01_LWUPBC = da_d01_LWUPBC.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_LWUPBC = da_d01_LWUPBC.where(da_d01_LWUPBC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWUPBC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWUPBC = ds['LWUPBC'].compute()
# da_d02_LWUPBC = da_d02_LWUPBC.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_LWUPBC = da_d02_LWUPBC.where(da_d02_LWUPBC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Longwave Upwelling at SFC ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWUPBC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWUPBC_cross = make_da_cross(da_d01_LWUPBC, da_cross_temp, 'LWUPBC', distance_d01, width, all_line_coords)
# da_d01_LWUPBC_cross.to_netcdf('./d01_cross_LWUPBC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWUPBC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWUPBC_cross = make_da_cross(da_d02_LWUPBC, da_cross_temp, 'LWUPBC', distance_d02, width, all_line_coords)
# da_d02_LWUPBC_cross.to_netcdf('./d02_cross_LWUPBC')
# # Delete variables after to aliviate memory strain
# del da_d01_LWUPBC, da_d02_LWUPBC, da_d01_LWUPBC_cross, da_d02_LWUPBC_cross

# ############ Load Longwave Downwelling at SFC ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_LWDNBC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_LWDNBC = ds['LWDNBC'].compute()
# da_d01_LWDNBC = da_d01_LWDNBC.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_LWDNBC = da_d01_LWDNBC.where(da_d01_LWDNBC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_LWDNBC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_LWDNBC = ds['LWDNBC'].compute()
# da_d02_LWDNBC = da_d02_LWDNBC.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_LWDNBC = da_d02_LWDNBC.where(da_d02_LWDNBC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Longwave Downwelling at SFC ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_LWDNBC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_LWDNBC_cross = make_da_cross(da_d01_LWDNBC, da_cross_temp, 'LWDNBC', distance_d01, width, all_line_coords)
# da_d01_LWDNBC_cross.to_netcdf('./d01_cross_LWDNBC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_LWDNBC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_LWDNBC_cross = make_da_cross(da_d02_LWDNBC, da_cross_temp, 'LWDNBC', distance_d02, width, all_line_coords)
# da_d02_LWDNBC_cross.to_netcdf('./d02_cross_LWDNBC')
# # Delete variables after to aliviate memory strain
# del da_d01_LWDNBC, da_d02_LWDNBC, da_d01_LWDNBC_cross, da_d02_LWDNBC_cross

# ############ Load Shortwave Upwelling at TOA ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWUPTC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWUPTC = ds['SWUPTC'].compute()
# da_d01_SWUPTC = da_d01_SWUPTC.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_SWUPTC = da_d01_SWUPTC.where(da_d01_SWUPTC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWUPTC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWUPTC = ds['SWUPTC'].compute()
# da_d02_SWUPTC = da_d02_SWUPTC.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_SWUPTC = da_d02_SWUPTC.where(da_d02_SWUPTC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Shortwave Upwelling at TOA ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWUPTC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWUPTC_cross = make_da_cross(da_d01_SWUPTC, da_cross_temp, 'SWUPTC', distance_d01, width, all_line_coords)
# da_d01_SWUPTC_cross.to_netcdf('./d01_cross_SWUPTC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWUPTC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWUPTC_cross = make_da_cross(da_d02_SWUPTC, da_cross_temp, 'SWUPTC', distance_d02, width, all_line_coords)
# da_d02_SWUPTC_cross.to_netcdf('./d02_cross_SWUPTC')
# # Delete variables after to aliviate memory strain
# del da_d01_SWUPTC, da_d02_SWUPTC, da_d01_SWUPTC_cross, da_d02_SWUPTC_cross

# ############ Load Shortwave Downwelling at TOA ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWDNTC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWDNTC = ds['SWDNTC'].compute()
# da_d01_SWDNTC = da_d01_SWDNTC.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_SWDNTC = da_d01_SWDNTC.where(da_d01_SWDNTC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWDNTC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWDNTC = ds['SWDNTC'].compute()
# da_d02_SWDNTC = da_d02_SWDNTC.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_SWDNTC = da_d02_SWDNTC.where(da_d02_SWDNTC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Shortwave Downwelling at TOA ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWDNTC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWDNTC_cross = make_da_cross(da_d01_SWDNTC, da_cross_temp, 'SWDNTC', distance_d01, width, all_line_coords)
# da_d01_SWDNTC_cross.to_netcdf('./d01_cross_SWDNTC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWDNTC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWDNTC_cross = make_da_cross(da_d02_SWDNTC, da_cross_temp, 'SWDNTC', distance_d02, width, all_line_coords)
# da_d02_SWDNTC_cross.to_netcdf('./d02_cross_SWDNTC')
# # Delete variables after to aliviate memory strain
# del da_d01_SWDNTC, da_d02_SWDNTC, da_d01_SWDNTC_cross, da_d02_SWDNTC_cross

# ############ Load Shortwave Upwelling at SFC ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWUPBC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWUPBC = ds['SWUPBC'].compute()
# da_d01_SWUPBC = da_d01_SWUPBC.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_SWUPBC = da_d01_SWUPBC.where(da_d01_SWUPBC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWUPBC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWUPBC = ds['SWUPBC'].compute()
# da_d02_SWUPBC = da_d02_SWUPBC.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_SWUPBC = da_d02_SWUPBC.where(da_d02_SWUPBC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Shortwave Upwelling at SFC ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWUPBC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWUPBC_cross = make_da_cross(da_d01_SWUPBC, da_cross_temp, 'SWUPBC', distance_d01, width, all_line_coords)
# da_d01_SWUPBC_cross.to_netcdf('./d01_cross_SWUPBC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWUPBC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWUPBC_cross = make_da_cross(da_d02_SWUPBC, da_cross_temp, 'SWUPBC', distance_d02, width, all_line_coords)
# da_d02_SWUPBC_cross.to_netcdf('./d02_cross_SWUPBC')
# # Delete variables after to aliviate memory strain
# del da_d01_SWUPBC, da_d02_SWUPBC, da_d01_SWUPBC_cross, da_d02_SWUPBC_cross

# ############ Load Shortwave Downwelling at SFC ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_SWDNBC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_SWDNBC = ds['SWDNBC'].compute()
# da_d01_SWDNBC = da_d01_SWDNBC.assign_coords(without_keys(d01_coords,'bottom_top'))
# da_d01_SWDNBC = da_d01_SWDNBC.where(da_d01_SWDNBC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_SWDNBC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_SWDNBC = ds['SWDNBC'].compute()
# da_d02_SWDNBC = da_d02_SWDNBC.assign_coords(without_keys(d02_coords,'bottom_top'))
# da_d02_SWDNBC = da_d02_SWDNBC.where(da_d02_SWDNBC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Cross-sectional analysis of Shortwave Downwelling at SFC ############
# # d01
# da_cross_temp, all_line_coords = cross_section_multi(da_d01_SWDNBC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d01_SWDNBC_cross = make_da_cross(da_d01_SWDNBC, da_cross_temp, 'SWDNBC', distance_d01, width, all_line_coords)
# da_d01_SWDNBC_cross.to_netcdf('./d01_cross_SWDNBC')
# # d02
# da_cross_temp, all_line_coords = cross_section_multi(da_d02_SWDNBC, start_coord, end_coord, width, dx)
# # Create da with coordinates
# da_d02_SWDNBC_cross = make_da_cross(da_d02_SWDNBC, da_cross_temp, 'SWDNBC', distance_d02, width, all_line_coords)
# da_d02_SWDNBC_cross.to_netcdf('./d02_cross_SWDNBC')
# # Delete variables after to aliviate memory strain
# del da_d01_SWDNBC, da_d02_SWDNBC, da_d01_SWDNBC_cross, da_d02_SWDNBC_cross

