#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Author: Hrag Najarian
Date: January 2023
'''


# This code is meant to compare between the control run and the experiments when cloud-radiative interactions are turned off.

# In[1]:


from wrf import getvar, ALL_TIMES
import matplotlib as mpl
import cartopy.crs as ccrs
from flox.xarray import xarray_reduce
import glob
import dask
import wrf
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
import pandas as pd
import time
# import netCDF4 as nc
from math import cos, asin, sqrt, pi, atan, degrees
import scipy
from scipy.optimize import curve_fit

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# ### Set global variables
# Anything that will be used throughout the script should be assigned here.

# In[2]:


# Set the bounds you want to look at
lat_bound = [-5,5]			# South to North
lon_bound_d01 = [80,135]	# West to East
lon_bound_d02 = [90,125]	# West to East
lat_avg_bound = [-5,5]
d0x = 2						# number of domains


# ### List of various functions that are used throughout the script

# In[3]:


# Assumes cartesian coordinate system
def calculate_angle_between_points(p1, p2):
    # Calculate differences
    dy = p2[0] - p1[0]  # Lats
    dx = p2[1] - p1[1]  # Lons
    # Find the angle (radians)
    theta = math.atan(dy/dx)
    
    return theta

# # Example points
# point1 = [0, 0]
# point2 = [1, 1]

# theta = math.degrees(calculate_angle_between_points(point1, point2))
# print(f"The angle between the points is {theta} degrees")


# In[4]:


def vertical_integration(da):
	# Do a rolling mean to get the average between two pressure levels
	da = da.rolling(bottom_top=2).mean()
	# Create dp and make it the same shape as da.
	dp_levels = da.bottom_top.diff(dim='bottom_top').expand_dims(dim={'Time':len(da.Time),'Distance':len(da.Distance),'Spread':len(da.Spread)})
	# Now remove the first level of da since it's nan's
	da = da.isel(bottom_top=slice(1,None))

	da_vertically_integrated = (da * dp_levels).sum('bottom_top') / (-9.8)

	return da_vertically_integrated


# In[5]:


def round_to_two_significant_figures(number):
    """
    Rounds a float value to two significant figures using np.ceil or np.floor.

    Args:
        number (float): The input float value.

    Returns:
        float: The rounded value.
    """
    # Calculate the order of magnitude (power of 10) for the input number
    order_of_magnitude = np.floor(np.log10(np.abs(number)))

    # Calculate the factor to round to two significant figures
    factor = 10**(2 - order_of_magnitude)

    # Round the number using np.ceil or np.floor
    rounded_value = np.ceil(number * factor) / factor

    return rounded_value

def round_to_one_significant_figures(number):
    """
    Rounds a float value to two significant figures using np.ceil or np.floor.

    Args:
        number (float): The input float value.

    Returns:
        float: The rounded value.
    """
    # Calculate the order of magnitude (power of 10) for the input number
    order_of_magnitude = np.floor(np.log10(np.abs(number)))

    # Calculate the factor to round to two significant figures
    factor = 10**(1 - order_of_magnitude)

    # Round the number using np.ceil or np.floor
    rounded_value = np.ceil(number * factor) / factor

    return rounded_value

round_to_one_significant_figures(0.00001427)


# In[6]:


# Linear Regression Function based on the scipy.stats.linregress function

# Make sure the x and y values are 1-D matrices

# Inclusion of custom start and end lag inputs. 
	# If you don't care for the negative lags, then start at 0, or vise versa.
	# This way you're not wasting resources on lags you don't need.

def linreg(x, y, min_lag, max_lag):
	# Initialize matrices
	slopes = []
	yintercepts = []
	rvalues = []
	pvalues = []
	stderrors = []
	# How to disypher what lag relationships mean:
		# Negative lag implies that x at the moment correlates with your y in the future
		# Positive lag implies that x at the moment correlates with you y in the past
	for lag in range(min_lag,max_lag+1):	# itterate from min_lag to max_lag (+1 because of range function)
		if lag == 0:	
			slope, yintercept, rvalue, pvalue, stderror = scipy.stats.linregress(x, y)
		elif lag < 0:
			slope, yintercept, rvalue, pvalue, stderror = scipy.stats.linregress(x[:lag], y[-lag:])
		elif lag > 0:
			slope, yintercept, rvalue, pvalue, stderror = scipy.stats.linregress(x[lag:], y[:-lag])
		
		# Append the values!
		slopes.append(slope)
		yintercepts.append(yintercept)
		rvalues.append(rvalue)
		pvalues.append(pvalue)
		stderrors.append(stderror)

	# Compile data into a dataarray for easy management
	da_reg = xr.DataArray(
		data=np.arange(min_lag,max_lag+1),
		dims='lag',
		coords=dict(
			slope = ('lag', slopes),
			yintercept = ('lag', yintercepts),
			rvalue = ('lag', rvalues),
			pvalue = ('lag', pvalues),
			stderror= ('lag',stderrors)
			),
		name='lin_reg'
		)

	return da_reg


# In[7]:


# Calculate Temperature [K] via potential temperature equation [PT = T * (Po/P) ^ .286]
	# Link: https://glossary.ametsoc.org/wiki/Potential_temperature 
def theta_to_temp(theta):
	# PT = T * (P0/P)^(R/Cp)
		# PT = Potential temperature/theta
		# T = Temperature
		# P0 = 1000 hPa
		# P = Pressure
		# R = Gas constant for air (287.052874 J/(kg*K))
		# Cp = Specific heat capacity at constant pressure (1003.5 J/(kg*K))
			# R/Cp = 0.286
	# so
	# T = PT / (P0/P)^(0.286)
	temp = xr.zeros_like(theta)
	P = theta.bottom_top.values
	for i in range(len(P)):
		temp[:,i,...] = theta[:,i,...] / (1000/P[i])**(0.286)
	
	temp.name = 'Temperature'
	temp = temp.assign_attrs(Units='K')
	
	return temp


# In[8]:


# Calculate Potential Temperature [K] via potential temperature equation [PT = T * (Po/P) ^ .286]
def temp_to_theta(temp, psfc):
	# PT = T * (P0/P)^(R/Cp)
		# PT = Potential temperature/theta
		# T = Temperature
		# P0 = 1000 hPa
		# P = Pressure
		# R = Gas constant for air (287.052874 J/(kg*K))
		# Cp = Specific heat capacity at constant pressure (1003.5 J/(kg*K))
			# R/Cp = 0.286
	
	theta = xr.zeros_like(temp)
	theta = temp* ((1000/psfc)**(0.286))

	theta.name = 'Potential Temperature'
	theta = theta.assign_attrs(Units='K')

	return theta


# In[9]:


# Calculate Mixing Ratio [kg/kg]
def mixing_ratio(e):
	# Calculate mixing ratio
	# https://glossary.ametsoc.org/wiki/Mixing_ratio
	r = xr.zeros_like(e)
	P = e.bottom_top.values
	for i in range(len(P)):
		r[:,i,...] = 0.62197*((e[:,i,...])/(P[i]-e[:,i,...]))	# [kg/kg]
	
	r.name = 'Mixing Ratio'
	r = r.assign_attrs(Units='kg/kg')
	
	return r


# In[10]:


# Calculate Virtual Temperature [K] via Tv = T(1+(rv/eps)) / (1+rv)
def virt_temp(T, rv, eps):
	# Calculate virtual temperature
	# https://glossary.ametsoc.org/wiki/Virtual_temperature
	Tv = xr.zeros_like(T)
	Tv = T*(1+(rv/eps)) / (1+rv)

	Tv.name = 'Virtual Temperature'
	Tv = Tv.assign_attrs(Units='K')

	return Tv


# In[11]:


# Calculate Saturation Vapor Pressure 
def sat_vap_press(temperature):
	# Follow the Clausius-Clapeyron equation 
		# Link: https://geo.libretexts.org/Bookshelves/Meteorology_and_Climate_Science/Practical_Meteorology_(Stull)/04%3A_Water_Vapor/4.00%3A_Vapor_Pressure_at_Saturation
		# es = e0 * exp([L/Rv] * (1/T0 - 1/T))
	e0=6.113		# [hPa]
	Rv=461			# [J/(K*kg)]
	T0=273.15		# [K]
	Lv=2.5*(10**6)	# [J/kg]	# liquid water
	Ld=2.83*(10**6)	# [J/kg]	# ice water
	# Create matrix/dataarray da that has variable L values based on the 
		# temperature of the atmosphere where it changes based on if T < 0°C 
		# Divide by rv to create the constant L/Rv
	constant = temperature.where(temperature>273.15, Lv)/Rv
	constant = temperature.where(temperature<=273.15, Ld)/Rv

	es = e0 * np.exp(constant * ((1/T0)-(1/temperature)))

	es.name = 'Saturation Vapor Pressure'
	es = es.assign_attrs(Units='hPa')
	
	return es


# In[12]:


def sat_mixing_ratio(es):
	# Calculate saturated mixing ratio
	# https://glossary.ametsoc.org/wiki/Mixing_ratio
	ws = xr.zeros_like(es)
	P = es.bottom_top.values
	for i in range(len(P)):
		ws[:,i,...] = 0.62197*((es[:,i,...])/(P[i]-es[:,i,...]))	# [kg/kg]
	
	ws.name = 'Saturation Mixing Ratio'
	ws = ws.assign_attrs(Units='kg/kg')
	
	return ws


# In[13]:


def rel_hum(theta,w):
	# Calculate the relative humidity via observed potential temperature [K] and mixing ratio [kg/kg]
	temperature = theta_to_temp(theta)	# [K]
	es = sat_vap_press(temperature)		# [hPa]
	ws = sat_mixing_ratio(es)			# [kg/kg]
	RH = w/ws * 100						# [%]

	RH.name = 'Relative Humidity'
	RH = RH.assign_attrs(Units='%')

	return RH


# In[14]:


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


# In[15]:


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


# In[16]:


# This function finds the distance [km] between two coordinates in lat & lon. 
    # If your map projection is Mercator (check 'MAP_PROJ'), then this works. 
    # If Lambert, then you need to do conversions, look at ChatGPT logs ('Distance Calculation Approximation')
def dist(lat1, lon1, lat2, lon2):
    r = 6371 # km
    p = pi / 180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 2 * r * asin(sqrt(a))


# In[17]:


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


# In[18]:


# Purpose: Create an array with multiple cross-sectional data from WRFoutput.
# Input:
    # da = 			xr.data_array		 	works with both 2-D and 3-D variables!
    # start_coord = [latitude, longitude] 
    # end_coord = 	[latitude, longitude]
	# width = 		spread of cross-section in degrees i.e., 0.75° = 0.75
	# dx = 			distance between each cross-sectional line i.e., 0.05° = 0.05
# Output:
    # da_cross: 		matrix in time, height, distance, and # of lines
		# 					or time, distance, and # of lines if using a 2-D variable
	# all_line_coords:	
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

	# Create all_line_coords that holds all the coordinates for every line produced
	all_line_coords = np.zeros([line_coords.shape[0],line_coords.shape[1],spread.shape[0]])

	# Looping over all the lines
	for i in range(len(spread)):
		
		# Fix this part
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


# In[19]:


def hov_diurnal(da, LT):
    # Average over the bounds to create shape of (time,west_east)
    da_avg = da.where((da.XLAT>lat_avg_bound[0])&(da.XLAT<lat_avg_bound[1]),drop=True).mean(dim='south_north')

    if LT:
        # Create a np.array with shape (west_east,hour). Then populate each longitude column with its respective
            # diurnal cycle of rain rate.
        array = np.zeros((da_avg.shape[1],24))
        # Loop through each longitude, look through all the timesteps and group them by hour,
            # and then average the values at each hour. Output the 1x24 array into the empty array
        for i in range(array.shape[0]):
            array[i,:] = da_avg[:,i].groupby('LocalTime.hour').mean()
    else:
        array = da_avg.mean('west_east').groupby('XTIME.hour').mean()
    return array


# In[20]:


# Purpose: To alliviate the memory issue that arrises when you getvar() a big dataset.
# Input:
    # xrDataset = xr.open_dataset(filepath)
    # Dataset = Dataset(filepath)
    # varname = '<variablename>'
# Output:
    # da: This will be the xr.DataArray you are after
# Process:
    # First create an empty np.array that holds varname values.
    # Loop through each timestep to fill in the np.array.
    # Here's the trick, only getvar() one timestep, then expand the time dim.
        # This will repeate the values from the first timestep into the Time dim.
    # Assign the Time coordinates.
    # Replace the repeated values with the correct values from npArray.

# This method is ~6x more efficient than getvar(Dataset, varname, ALL_TIMES)

def wrf_np2da(xrDataset,Dataset,varname):
    da_times = getvar(Dataset, 'times', ALL_TIMES)			# Times in datetime64
    # Create empty numpy array
    npArray = np.empty((xrDataset.dims['Time'],xrDataset.dims['bottom_top'],
                        xrDataset.dims['south_north'],xrDataset.dims['west_east']),np.float32)
    for i in range(npArray.shape[0]):
        npArray[i,...] = getvar(Dataset, varname, timeidx=i, meta=False)
    da = getvar(Dataset, varname, timeidx=0)
    da = da.expand_dims(dim={'Time': da_times}, axis=0)
    da = da.assign_coords(dict(
        Time=('Time',da_times.values),
        XTIME=('Time',np.float32(xrDataset['ITIMESTEP'].values))
    ))
    da.values = npArray
    return da


# In[21]:


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


# In[22]:


# Function that can removes the bottom_top dimension for 2-D datasets
def without_keys(d, keys):
	return {x: d[x] for x in d if x not in keys}


# In[23]:


# Assumes cartesian/flat coordinate system
def calculate_angle_between_points(p1, p2):
    # Calculate differences
    dy = p2[0] - p1[0]  # Lats
    dx = p2[1] - p1[1]  # Lons
    # Find the angle (radians)
    theta = atan(dy/dx)

    # Negative if NW or SE direction
    # Positive if NE or SW direction
    
    return theta

start_coord		= [1.2,112.8]
end_coord 		= [5.2,108.8]
degrees(calculate_angle_between_points(start_coord, end_coord))


# In[24]:


# Makes making tick labels easier
# ticks are integers
def degree_labels(ticks, lat_or_lon):

	labels = []
	
	if lat_or_lon=='lat':
		for t in ticks:
			if t >= 0:
				labels.append(str(t)+u'\N{DEGREE SIGN}N')
			else:
				labels.append(str(t)+u'\N{DEGREE SIGN}S')
				
	elif lat_or_lon=='lon':
		for t in ticks:
			if t >= 0:
				labels.append(str(t)+u'\N{DEGREE SIGN}E')
			else:
				labels.append(str(t)+u'\N{DEGREE SIGN}W')
	return labels


# ## Load Data

# ### Control Data [Spatial]
# This section opens stitched wrfout .nc files. The raw .nc file is much to large, so I have extracted variables from the large file into smaller .nc files that only contains one variable. To get these smaller .nc files, please refer to extract_variable.py and interp_variable.py function I have created. This section opens those files up and assigned them to a variable.
# 
# Temporal 10.5-day (2015 11-22-12UTC -> 2015 12-03-00UTC)
# 
# Below the section where I assigned file names, there is an option for you to only open up a smaller domain (spatial and temporal) if you are interetsed in a select region. This will save you a lot of computational time.

# In[25]:


start_time = time.perf_counter()

parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26/2015-11-25-03--11-26-12'
file_d01_raw = parent_dir + '/raw/d01'
file_d02_raw = parent_dir + '/raw/d02'
# 2-D data
file_d02_HFX = parent_dir + '/L1/d02_HFX'			    # [W/m^2]
file_d02_LowCLDFRA = parent_dir + '/L1/d02_LowCLDFRA'   # [0->1]
file_d02_MidCLDFRA = parent_dir + '/L1/d02_MidCLDFRA'   # [0->1]
file_d02_HighCLDFRA = parent_dir + '/L1/d02_HighCLDFRA' # [0->1]
file_d02_LWDNB = parent_dir + '/L1/d02_LWDNB'		# [W/m^2]
file_d02_LWUPB = parent_dir + '/L1/d02_LWUPB'		# [W/m^2]
file_d02_SWDNB = parent_dir + '/L1/d02_SWDNB'		# [W/m^2]
file_d02_SWUPB = parent_dir + '/L1/d02_SWUPB'		# [W/m^2]
file_d02_SWDNBC = parent_dir + '/L1/d02_SWDNBC'		# [W/m^2]
file_d02_SWUPBC = parent_dir + '/L1/d02_SWUPBC'		# [W/m^2]
file_d02_RR = parent_dir + '/L1/d02_RR'				# [mm/dt]
file_d02_PSFC = parent_dir + '/L1/d02_PSFC'		    # [hPa]
file_d02_T2 = parent_dir + '/L1/d02_T2'		        # [K]
file_d02_U10 = parent_dir + '/L1/d02_U10'		    # [m/s]
file_d02_V10 = parent_dir + '/L1/d02_V10'		    # [m/s]
# Raw data
file_d01_P = parent_dir + '/L1/d01_P'				# [mm/dt]
file_d02_P = parent_dir + '/L1/d02_P'				# [mm/dt]
# Interpolated data 
file_d01_U = parent_dir + '/L2/d01_interp_U'	        # [m/s]
file_d02_U = parent_dir + '/L2/d02_interp_U'	        # [m/s]
file_d01_V = parent_dir + '/L2/d01_interp_V'	        # [m/s]
file_d02_V = parent_dir + '/L2/d02_interp_V'	        # [m/s]
file_d01_QV = parent_dir + '/L2/d01_interp_QV'	        # [kg/kg]
file_d02_QV = parent_dir + '/L2/d02_interp_QV'	        # [kg/kg]
file_d01_CLDFRA = parent_dir + '/L2/d01_interp_CLDFRA'	# 
file_d02_CLDFRA = parent_dir + '/L2/d02_interp_CLDFRA'	# 
file_d01_H_DIABATIC = parent_dir + '/L2/d01_interp_H_DIABATIC'	        # [K/s]
file_d02_H_DIABATIC = parent_dir + '/L2/d02_interp_H_DIABATIC'	        # [K/s]
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
#### All the data 
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-02T12')]
lats = [-20, 20]
lons = [80, 135]

#### Some of the data
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-03T00')]
times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-12-02T00')]        # NCRF Sunrise
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


##################### 3-D variables #######################

# ############ Interpolated zonal winds   [m/s] #############
# step2_time = time.perf_counter()
# # d01   # how I used to open data: ds = xr.open_dataset(file_d01_U).isel(Time=slice(0,t))
# ds = open_ds(file_d01_U,time_ind_d01,lat_ind_d01,lon_ind_d01)

# da_d01_U = ds['U'].compute()
# da_d01_U = da_d01_U.assign_coords(d01_coords)
# fill_value_f8 = da_d01_U.max()      # This is the fill_value meaning missing_data
# da_d01_U = da_d01_U.where(da_d01_U!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_U,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_U = ds['U'].compute()
# da_d02_U = da_d02_U.assign_coords(d02_coords)
# # fill_value = da_d02_U.max()      # This is the fill_value meaning missing_data
# da_d02_U = da_d02_U.where(da_d02_U!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated zonal winds loaded \N{check mark}', step1_time-step2_time, 'seconds')


# ############ Interpolated Meridional winds   [m/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_V,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_V = ds['V'].compute()
# da_d01_V = da_d01_V.assign_coords(d01_coords)
# # fill_value = da_d01_V.max()      # This is the fill_value meaning missing_data
# da_d01_V = da_d01_V.where(da_d01_V!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_V,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_V = ds['V'].compute()
# da_d02_V = da_d02_V.assign_coords(d02_coords)
# # fill_value = da_d02_V.max()      # This is the fill_value meaning missing_data
# da_d02_V = da_d02_V.where(da_d02_V!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated meridional winds loaded \N{check mark}', step1_time-step2_time, 'seconds')

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

############ Cloud Fraction [0-1] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_CLDFRA,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_CLDFRA = ds['CLDFRA']
# # Turn the fill value into nan's before averaging
# da_d01_CLDFRA = xr.where(da_d01_CLDFRA>1, np.nan, da_d01_CLDFRA)
# da_d01_CLDFRA = da_d01_CLDFRA.assign_coords(d01_coords)

# d02
# ds = open_ds(file_d02_CLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_CLDFRA = ds['CLDFRA']
# # Turn the fill value into nan's before averaging
# da_d02_CLDFRA = xr.where(da_d02_CLDFRA>1, np.nan, da_d02_CLDFRA)
# da_d02_CLDFRA = da_d02_CLDFRA.assign_coords(d02_coords)

# # Uncomment bellow if first time running
# 	# Low Cloud Fraction [1000-750 hPa]
# inds = np.where((interp_P_levels<=1000)&(interp_P_levels>=750))[0]
# da_d02_LowCLDFRA = da_d02_CLDFRA.isel(bottom_top=slice(inds[0],inds[-1])).mean('bottom_top', skipna=True)
# 	# Mid Cloud Fraction [750-500 hPa]
# inds = np.where((interp_P_levels<=750)&(interp_P_levels>=500))[0]
# da_d02_MidCLDFRA = da_d02_CLDFRA.isel(bottom_top=slice(inds[0],inds[-1])).mean('bottom_top', skipna=True)
# 	# High Cloud Fraction [500-200 hPa]
# inds = np.where((interp_P_levels<=500)&(interp_P_levels>=200))[0]
# da_d02_HighCLDFRA = da_d02_CLDFRA.isel(bottom_top=slice(inds[0],inds[-1])).mean('bottom_top', skipna=True)

# Save the computed cloud type fractions
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/L1'
# os.chdir(parent_dir)
# da_d02_LowCLDFRA.to_netcdf(path='d02_LowCLDFRA', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_MidCLDFRA.to_netcdf(path='d02_MidCLDFRA', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_HighCLDFRA.to_netcdf(path='d02_HighCLDFRA', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)

# ############ Latent Heating [K/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_H_DIABATIC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_H_DIABATIC = ds['H_DIABATIC'].compute()
# da_d01_H_DIABATIC = da_d01_H_DIABATIC.assign_coords(d01_coords)
# da_d01_H_DIABATIC = da_d01_H_DIABATIC.where(da_d01_H_DIABATIC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_H_DIABATIC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_H_DIABATIC = ds['H_DIABATIC'].compute()
# da_d02_H_DIABATIC = da_d02_H_DIABATIC.assign_coords(d02_coords)
# da_d02_H_DIABATIC = da_d02_H_DIABATIC.where(da_d02_H_DIABATIC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Latent heating loaded \N{check mark}', step1_time-step2_time, 'seconds')

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

# ############ Cloud-Radiative Forcing Calculations ############
# # Longwave
# da_d01_LWCRF = da_d01_LWAll - da_d01_LWClear
# da_d02_LWCRF = da_d02_LWAll - da_d02_LWClear
# # Shortwave
# da_d01_SWCRF = da_d01_SWAll - da_d01_SWClear
# da_d02_SWCRF = da_d02_SWAll - da_d02_SWClear
# # Total
# da_d01_TotalCRF = da_d01_LWCRF + da_d01_SWCRF
# da_d02_TotalCRF = da_d02_LWCRF + da_d02_SWCRF

##################### 2-D variables #######################

############ HFX     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_HFX,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_HFX = ds['HFX'].compute()
da_d02_HFX = da_d02_HFX.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('HFX loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ Low Cloud Fraction     [0->1] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_LowCLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_LowCLDFRA = ds['CLDFRA'].compute()
da_d02_LowCLDFRA = da_d02_LowCLDFRA.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Low Cloud Fraction loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ Mid Cloud Fraction     [0->1] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_MidCLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_MidCLDFRA = ds['CLDFRA'].compute()
da_d02_MidCLDFRA = da_d02_MidCLDFRA.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Mid Cloud Fraction loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ High Cloud Fraction     [0->1] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_HighCLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_HighCLDFRA = ds['CLDFRA'].compute()
da_d02_HighCLDFRA = da_d02_HighCLDFRA.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('High Cloud Fraction loaded \N{check mark}', step1_time-step2_time, 'seconds') 

############ LWDNB     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_LWDNB,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_LWDNB = ds['LWDNB'].compute()
da_d02_LWDNB = da_d02_LWDNB.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Longwave Downwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ LWUPB     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_LWUPB,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_LWUPB = ds['LWUPB'].compute()
da_d02_LWUPB = da_d02_LWUPB.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Longwave Upwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ LWSFC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
da_d02_LWSFC = da_d02_LWDNB - da_d02_LWUPB

step1_time = time.perf_counter()
print('Longwave Surface Flux calculated \N{check mark}', step1_time-step2_time, 'seconds')

############ SWDNB     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_SWDNB,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_SWDNB = ds['SWDNB'].compute()
da_d02_SWDNB = da_d02_SWDNB.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Shortwave Downwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ SWUPB     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_SWUPB,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_SWUPB = ds['SWUPB'].compute()
da_d02_SWUPB = da_d02_SWUPB.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Shortwave Upwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ SWSFC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
da_d02_SWSFC = da_d02_SWDNB - da_d02_SWUPB

step1_time = time.perf_counter()
print('Shortwave Surface Flux calculated \N{check mark}', step1_time-step2_time, 'seconds')

############ SWDNBC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_SWDNBC,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_SWDNBC = ds['SWDNBC'].compute()
da_d02_SWDNBC = da_d02_SWDNBC.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Clear Shortwave Downwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ SWUPBC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_SWUPBC,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_SWUPBC = ds['SWUPBC'].compute()
da_d02_SWUPBC = da_d02_SWUPBC.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Clear Shortwave Upwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ SWSFC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
da_d02_SWSFC_Clear = da_d02_SWDNBC - da_d02_SWUPBC

step1_time = time.perf_counter()
print('Clear Shortwave Surface Flux calculated \N{check mark}', step1_time-step2_time, 'seconds')

############ SWSFC CRF     [W/m^2] ############
step2_time = time.perf_counter()
# d02
da_d02_SWSFC_CRF = da_d02_SWSFC - da_d02_SWSFC_Clear

step1_time = time.perf_counter()
print('CRF Shortwave Surface Flux calculated \N{check mark}', step1_time-step2_time, 'seconds')

############ Rain Rate     [mm/hr] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_RR,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_RR = ds['RR'].compute()
da_d02_RR = da_d02_RR.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ Surface Pressure     [hPa] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_PSFC,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_PSFC = ds['PSFC'].compute()
da_d02_PSFC = da_d02_PSFC.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Surface Pressure loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ Detection of land & water  ############
step2_time = time.perf_counter()
# d01
da_d01_LANDMASK = ds_d01['LANDMASK'].sel(Time=slice(1)).compute().squeeze()   # Land = 1, Water = 0
# d02
da_d02_LANDMASK = ds_d02['LANDMASK'].sel(Time=slice(1)).compute().squeeze()   # Land = 1, Water = 0

step1_time = time.perf_counter()
print('Landmask loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ Terrain Height    [m]  ############
step2_time = time.perf_counter()
# d02
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()

step1_time = time.perf_counter()
print('Terrain Height loaded \N{check mark}', step1_time-step2_time, 'seconds')

print('Domain d01 & d02 \N{check mark}', step1_time-start_time, 'seconds')


# In[26]:


# d02
# da_d02_LANDMASK = ds_d02['LANDMASK'].sel(Time=slice(1)).compute().squeeze()   # Land = 1, Water = 0


# ### NCRF Sunrise Data [Spatial]

# In[ ]:


start_time = time.perf_counter()

parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'

# Raw data
file_d02_sunrise = parent_dir + '/raw/d02_sunrise'
file_d02_sunset = parent_dir + '/raw/d02_sunset'
# 2-D data
file_d02_sunrise_RR = parent_dir + '/L1/d02_sunrise_RR'				# [mm/dt]
file_d02_sunset_RR = parent_dir + '/L1/d02_sunset_RR'				# [mm/dt]
file_d02_sunrise_HFX = parent_dir + '/L1/d02_sunrise_HFX'			# [W/m^2]
file_d02_sunset_HFX = parent_dir + '/L1/d02_sunset_HFX'				# [W/m^2]
file_d02_sunrise_LowCLDFRA = parent_dir + '/L1/d02_sunrise_LowCLDFRA'		# [0->1]
file_d02_sunset_LowCLDFRA = parent_dir + '/L1/d02_sunset_LowCLDFRA'			# [0->1]
file_d02_sunrise_MidCLDFRA = parent_dir + '/L1/d02_sunrise_MidCLDFRA'		# [0->1]
file_d02_sunset_MidCLDFRA = parent_dir + '/L1/d02_sunset_MidCLDFRA'			# [0->1]
file_d02_sunrise_HighCLDFRA = parent_dir + '/L1/d02_sunrise_HighCLDFRA'		# [0->1]
file_d02_sunset_HighCLDFRA = parent_dir + '/L1/d02_sunset_HighCLDFRA'		# [0->1]
file_d02_sunrise_LWDNB = parent_dir + '/L1/d02_sunrise_LWDNB'		# [W/m^2]
file_d02_sunset_LWDNB = parent_dir + '/L1/d02_sunset_LWDNB'			# [W/m^2]
file_d02_sunrise_LWUPB = parent_dir + '/L1/d02_sunrise_LWUPB'		# [W/m^2]
file_d02_sunset_LWUPB = parent_dir + '/L1/d02_sunset_LWUPB'			# [W/m^2]
file_d02_sunrise_SWDNB = parent_dir + '/L1/d02_sunrise_SWDNB'		# [W/m^2]
file_d02_sunset_SWDNB = parent_dir + '/L1/d02_sunset_SWDNB'			# [W/m^2]
file_d02_sunrise_SWUPB = parent_dir + '/L1/d02_sunrise_SWUPB'		# [W/m^2]
file_d02_sunset_SWUPB = parent_dir + '/L1/d02_sunset_SWUPB'			# [W/m^2]
file_d02_sunrise_SWDNBC = parent_dir + '/L1/d02_sunrise_SWDNBC'		# [W/m^2]
file_d02_sunset_SWDNBC = parent_dir + '/L1/d02_sunset_SWDNBC'			# [W/m^2]
file_d02_sunrise_SWUPBC = parent_dir + '/L1/d02_sunrise_SWUPBC'		# [W/m^2]
file_d02_sunset_SWUPBC = parent_dir + '/L1/d02_sunset_SWUPBC'			# [W/m^2]
# 3-D data
file_d02_sunrise_CLDFRA = parent_dir + '/L2/d02_sunrise_interp_CLDFRA'		# [0->1]
file_d02_sunset_CLDFRA = parent_dir + '/L2/d02_sunset_interp_CLDFRA'	    # [0->1]

######################################################################################
################ Declare the bounds you want to specifically look at #################
#### All the data 
# times = [np.datetime64('2015-11-22T12'), np.datetime64('2015-12-02T12')]
lats = [-20, 20]
lons = [80, 135]

#### Some of the data
# times = [np.datetime64('2015-11-23T13'), np.datetime64('2015-12-02T12')]      # NCRF Sunset
times = [np.datetime64('2015-11-23T01'), np.datetime64('2015-12-02T00')]        # NCRF Sunrise
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


##################### 3-D variables #######################

# ############ Interpolated zonal winds   [m/s] #############
# step2_time = time.perf_counter()
# # d01   # how I used to open data: ds = xr.open_dataset(file_d01_U).isel(Time=slice(0,t))
# ds = open_ds(file_d01_U,time_ind_d01,lat_ind_d01,lon_ind_d01)

# da_d01_U = ds['U'].compute()
# da_d01_U = da_d01_U.assign_coords(d01_coords)
# fill_value_f8 = da_d01_U.max()      # This is the fill_value meaning missing_data
# da_d01_U = da_d01_U.where(da_d01_U!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_U,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_U = ds['U'].compute()
# da_d02_U = da_d02_U.assign_coords(d02_coords)
# # fill_value = da_d02_U.max()      # This is the fill_value meaning missing_data
# da_d02_U = da_d02_U.where(da_d02_U!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated zonal winds loaded \N{check mark}', step1_time-step2_time, 'seconds')


# ############ Interpolated Meridional winds   [m/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_V,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_V = ds['V'].compute()
# da_d01_V = da_d01_V.assign_coords(d01_coords)
# # fill_value = da_d01_V.max()      # This is the fill_value meaning missing_data
# da_d01_V = da_d01_V.where(da_d01_V!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_V,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_V = ds['V'].compute()
# da_d02_V = da_d02_V.assign_coords(d02_coords)
# # fill_value = da_d02_V.max()      # This is the fill_value meaning missing_data
# da_d02_V = da_d02_V.where(da_d02_V!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Interpolated meridional winds loaded \N{check mark}', step1_time-step2_time, 'seconds')

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

# ############ Cloud Fraction [0->1] ############
# step2_time = time.perf_counter()
# # d02
# ds = open_ds(file_d02_sunrise_CLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_sunrise_CLDFRA = ds['CLDFRA']
# # Turn the fill value into nan's before averaging
# da_d02_sunrise_CLDFRA = xr.where(da_d02_sunrise_CLDFRA>1, np.nan, da_d02_sunrise_CLDFRA)
# da_d02_sunrise_CLDFRA = da_d02_sunrise_CLDFRA.assign_coords(d02_coords)

# # # Uncomment bellow if first time running
# # 	# Low Cloud Fraction [1000-750 hPa]
# # inds = np.where((interp_P_levels<=1000)&(interp_P_levels>=750))[0]
# # da_d02_sunrise_LowCLDFRA = da_d02_sunrise_CLDFRA.isel(bottom_top=slice(inds[0],inds[-1])).mean('bottom_top', skipna=True)
# # 	# Mid Cloud Fraction [750-500 hPa]
# # inds = np.where((interp_P_levels<=750)&(interp_P_levels>=500))[0]
# # da_d02_sunrise_MidCLDFRA = da_d02_sunrise_CLDFRA.sel(bottom_top=slice(inds[0],inds[-1])).mean('bottom_top', skipna=True)
# # 	# High Cloud Fraction [500-200 hPa]
# # inds = np.where((interp_P_levels<=500)&(interp_P_levels>=200))[0]
# # da_d02_sunrise_HighCLDFRA = da_d02_sunrise_CLDFRA.sel(bottom_top=slice(inds[0],inds[-1])).mean('bottom_top', skipna=True)

# # # Save the computed cloud type fractions
# # parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff/L1'
# # os.chdir(parent_dir)
# # da_d02_sunrise_LowCLDFRA.to_netcdf(path='d02_sunrise_LowCLDFRA', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# # da_d02_sunrise_MidCLDFRA.to_netcdf(path='d02_sunrise_MidCLDFRA', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# # da_d02_sunrise_HighCLDFRA.to_netcdf(path='d02_sunrise_HighCLDFRA', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)

# # step1_time = time.perf_counter()
# # print('Cloud fraction loaded \N{check mark}', step1_time-step2_time, 'seconds')


# ############ Latent Heating [K/s] ############
# step2_time = time.perf_counter()
# # d01
# ds = open_ds(file_d01_H_DIABATIC,time_ind_d01,lat_ind_d01,lon_ind_d01)
# da_d01_H_DIABATIC = ds['H_DIABATIC'].compute()
# da_d01_H_DIABATIC = da_d01_H_DIABATIC.assign_coords(d01_coords)
# da_d01_H_DIABATIC = da_d01_H_DIABATIC.where(da_d01_H_DIABATIC!=fill_value_f8)    # Change fill_value points to nans
# # d02
# ds = open_ds(file_d02_H_DIABATIC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_H_DIABATIC = ds['H_DIABATIC'].compute()
# da_d02_H_DIABATIC = da_d02_H_DIABATIC.assign_coords(d02_coords)
# da_d02_H_DIABATIC = da_d02_H_DIABATIC.where(da_d02_H_DIABATIC!=fill_value_f8)    # Change fill_value points to nans

# step1_time = time.perf_counter()
# print('Latent heating loaded \N{check mark}', step1_time-step2_time, 'seconds')

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

# ############ Cloud-Radiative Forcing Calculations ############
# # Longwave
# da_d01_LWCRF = da_d01_LWAll - da_d01_LWClear
# da_d02_LWCRF = da_d02_LWAll - da_d02_LWClear
# # Shortwave
# da_d01_SWCRF = da_d01_SWAll - da_d01_SWClear
# da_d02_SWCRF = da_d02_SWAll - da_d02_SWClear
# # Total
# da_d01_TotalCRF = da_d01_LWCRF + da_d01_SWCRF
# da_d02_TotalCRF = da_d02_LWCRF + da_d02_SWCRF

##################### 2-D variables #######################

############ HFX     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_HFX,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_HFX = ds['HFX'].compute()
da_d02_sunrise_HFX = da_d02_sunrise_HFX.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('HFX loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ Low Cloud Fraction     [0->1] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_LowCLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_LowCLDFRA = ds['CLDFRA'].compute()
da_d02_sunrise_LowCLDFRA = da_d02_sunrise_LowCLDFRA.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Low Cloud Fraction loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ Mid Cloud Fraction     [0->1] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_MidCLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_MidCLDFRA = ds['CLDFRA'].compute()
da_d02_sunrise_MidCLDFRA = da_d02_sunrise_MidCLDFRA.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Mid Cloud Fraction loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ High Cloud Fraction     [0->1] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_HighCLDFRA,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_HighCLDFRA = ds['CLDFRA'].compute()
da_d02_sunrise_HighCLDFRA = da_d02_sunrise_HighCLDFRA.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('High Cloud Fraction loaded \N{check mark}', step1_time-step2_time, 'seconds') 

############ LWDNB     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_LWDNB,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_LWDNB = ds['LWDNB'].compute()
da_d02_sunrise_LWDNB = da_d02_sunrise_LWDNB.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Longwave Downwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ LWUPB     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_LWUPB,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_LWUPB = ds['LWUPB'].compute()
da_d02_sunrise_LWUPB = da_d02_sunrise_LWUPB.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Longwave Upwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ LWSFC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
da_d02_sunrise_LWSFC = da_d02_sunrise_LWDNB - da_d02_sunrise_LWUPB

step1_time = time.perf_counter()
print('Longwave Surface Flux calculated \N{check mark}', step1_time-step2_time, 'seconds')

############ SWDNB     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_SWDNB,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_SWDNB = ds['SWDNB'].compute()
da_d02_sunrise_SWDNB = da_d02_sunrise_SWDNB.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Shortwave Downwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ SWUPB     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_SWUPB,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_SWUPB = ds['SWUPB'].compute()
da_d02_sunrise_SWUPB = da_d02_sunrise_SWUPB.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Shortwave Upwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ SWSFC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
da_d02_sunrise_SWSFC = da_d02_sunrise_SWDNB - da_d02_sunrise_SWUPB

step1_time = time.perf_counter()
print('Shortwave Surface Flux calculated \N{check mark}', step1_time-step2_time, 'seconds')

############ SWDNBC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_SWDNBC,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_SWDNBC = ds['SWDNBC'].compute()
da_d02_sunrise_SWDNBC = da_d02_sunrise_SWDNBC.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Clear Shortwave Downwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ SWUPBC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_SWUPBC,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_SWUPBC = ds['SWUPBC'].compute()
da_d02_sunrise_SWUPBC = da_d02_sunrise_SWUPBC.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Clear Shortwave Upwelling Surface loaded \N{check mark}', step1_time-step2_time, 'seconds')

############ SWSFC     [W/m^2] ############
step2_time = time.perf_counter()
# d02
da_d02_sunrise_SWSFC_Clear = da_d02_sunrise_SWDNBC - da_d02_sunrise_SWUPBC

step1_time = time.perf_counter()
print('Clear Shortwave Surface Flux calculated \N{check mark}', step1_time-step2_time, 'seconds')

############ SWSFC CRF     [W/m^2] ############
step2_time = time.perf_counter()
# d02
da_d02_sunrise_SWSFC_CRF = da_d02_sunrise_SWSFC - da_d02_sunrise_SWSFC_Clear

step1_time = time.perf_counter()
print('CRF Shortwave Surface Flux calculated \N{check mark}', step1_time-step2_time, 'seconds')

############ Rain Rate     [mm/hr] ############
step2_time = time.perf_counter()
# d02
ds = open_ds(file_d02_sunrise_RR,time_ind_d02,lat_ind_d02,lon_ind_d02)
da_d02_sunrise_RR = ds['RR'].compute()
da_d02_sunrise_RR = da_d02_sunrise_RR.assign_coords(without_keys(d02_coords,'bottom_top'))

step1_time = time.perf_counter()
print('Rain rates loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Surface Pressure     [hPa] ############
# step2_time = time.perf_counter()
# # d02
# ds = open_ds(file_d02_PSFC,time_ind_d02,lat_ind_d02,lon_ind_d02)
# da_d02_PSFC = ds['PSFC'].compute()
# da_d02_PSFC = da_d02_PSFC.assign_coords(without_keys(d02_coords,'bottom_top'))

# step1_time = time.perf_counter()
# print('Surface Pressure loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Detection of land & water  ############
# step2_time = time.perf_counter()
# # d02
# da_d02_LANDMASK = ds_d02['LANDMASK'].sel(Time=slice(1)).compute().squeeze()   # Land = 1, Water = 0

# step1_time = time.perf_counter()
# print('Landmask loaded \N{check mark}', step1_time-step2_time, 'seconds')

# ############ Terrain Height    [m]  ############
# step2_time = time.perf_counter()
# # d02
# da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()

# step1_time = time.perf_counter()
# print('Terrain Height loaded \N{check mark}', step1_time-step2_time, 'seconds')


print('Domain d01 & d02 \N{check mark}', step1_time-start_time, 'seconds')


# ### Cross-Section data

# #### Control

# In[ ]:


# cd into the directory with your cross-sectional datasets to make the file paths when using xr.open_dataset a lot shorter
parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/L3/'

# Select which cross-sectional data you'd like to observe
cross_dir = 'Sumatra_northwest'

# This are global parameters used throughout cross-section analysis
if cross_dir == 'Sumatra_mid_central': cross_sec, cross_title = (1,'Sumatra Mid Central')
elif cross_dir == 'Sumatra_northwest': cross_sec, cross_title = (2,'Sumatra Northwest')
elif cross_dir == 'Borneo_northwest': cross_sec, cross_title = (3,'Borneo Northwest')

# cd into the specific cross-section directory
os.chdir(parent_dir+cross_dir)

## Load in the variables
start_time = time.perf_counter()
# Normal Wind [m/s]
da_d02_cross_NormalWind_cntl= xr.open_dataset('d02_cross_NormalWind')['NormalWind'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Vertical Wind [m/s]
da_d02_cross_W_cntl= xr.open_dataset('d02_cross_W')['W'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Rain Rate [mm/day]]
da_d02_cross_RR_cntl= xr.open_dataset('d02_cross_RR')['RR'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Upward Sensible Heat Flux @ Surface [W/m^2]
da_d02_cross_HFX_cntl= xr.open_dataset('d02_cross_HFX')['HFX'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Latent Heat Flux @ Surface [W/m^2]
da_d02_cross_LH_cntl= xr.open_dataset('d02_cross_LH')['LH'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Upward Moisture Flux @ Surface [kg/(m^2 s)]
da_d02_cross_QFX_cntl= xr.open_dataset('d02_cross_QFX')['QFX'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Water vapor [kg/kg]
da_d02_cross_QV_cntl= xr.open_dataset('d02_cross_QV')['QV'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# # Cloud water [kg/kg]
# da_d02_cross_QC_cntl= xr.open_dataset('d02_cross_QC')['QC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# # Rain water [kg/kg]
# da_d02_cross_QR_cntl= xr.open_dataset('d02_cross_QR')['QR'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# # Ice water [kg/kg]
# da_d02_cross_QI_cntl= xr.open_dataset('d02_cross_QI')['QI'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# # Snow water [kg/kg]
# da_d02_cross_QS_cntl= xr.open_dataset('d02_cross_QS')['QS'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# # Graupel water [kg/kg]
# da_d02_cross_QG_cntl= xr.open_dataset('d02_cross_QG')['QG'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# # Total water [kg/kg]
# da_d02_cross_QTotal_cntl= da_d02_cross_QV_cntl+da_d02_cross_QC_cntl+da_d02_cross_QR_cntl+da_d02_cross_QI_cntl+da_d02_cross_QS_cntl+da_d02_cross_QG_cntl
# Total water [kg/kg]
da_d02_cross_QTotal_cntl= xr.open_dataset('d02_cross_QTotal')['QTotal'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Cloud Fraction [0->1]
da_d02_cross_CLDFRA_cntl= xr.open_dataset('d02_cross_CLDFRA')['CLDFRA'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
	# Low Cloud Fraction [1000-750 hPa]
da_d02_cross_LowCLDFRA_cntl=da_d02_cross_CLDFRA_cntl.sel(bottom_top=slice(1000,750)).mean('bottom_top')
	# Mid Cloud Fraction [750-500 hPa]
da_d02_cross_MidCLDFRA_cntl=da_d02_cross_CLDFRA_cntl.sel(bottom_top=slice(750,500)).mean('bottom_top')
	# High Cloud Fraction [500-200 hPa]
da_d02_cross_HighCLDFRA_cntl=da_d02_cross_CLDFRA_cntl.sel(bottom_top=slice(500,200)).mean('bottom_top')
# Latent Heating [K/s]
da_d02_cross_H_DIABATIC_cntl= xr.open_dataset('d02_cross_H_DIABATIC')['H_DIABATIC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Potential Temperature [K]
da_d02_cross_Theta_cntl= xr.open_dataset('d02_cross_Theta')['Theta'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Temperature @ 2m [K]
da_d02_cross_T2_cntl= xr.open_dataset('d02_cross_T2')['T2'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Surface Pressure [hPa]
da_d02_cross_PSFC_cntl= xr.open_dataset('d02_cross_PSFC')['PSFC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Potential Temperature @ 2m [K]
da_d02_cross_Theta2_cntl= temp_to_theta(da_d02_cross_T2_cntl,da_d02_cross_PSFC_cntl)
# Virtual Temperature [K]
da_d02_cross_Tv_cntl = virt_temp(theta_to_temp(da_d02_cross_Theta_cntl), mixing_ratio(da_d02_cross_QV_cntl), 0.622)
# # Relative Humidity [%]
# da_d02_cross_RH_cntl = rel_hum(da_d02_cross_Theta_cntl, da_d02_cross_QTotal_cntl)
# CAPE [J/kg]
da_d02_cross_CAPE_cntl= xr.open_dataset('d02_cross_CAPE')['CAPE'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# CIN [J/kg]
da_d02_cross_CIN_cntl= xr.open_dataset('d02_cross_CIN')['CIN'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()


## Vertically-resolved Radiative heating
# Longwave All-sky Heating [K/s]
da_d02_cross_LWAll_cntl= xr.open_dataset('d02_cross_LWAll')['LWAll'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave Clear-sky Heating [K/s]
da_d02_cross_LWClear_cntl= xr.open_dataset('d02_cross_LWClear')['LWClear'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave CRF [K/s]
da_d02_cross_LWCRF_cntl= xr.open_dataset('d02_cross_LWCRF')['LWCRF'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave All-sky Heating [K/s]
da_d02_cross_SWAll_cntl= xr.open_dataset('d02_cross_SWAll')['SWAll'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Clear-sky Heating [K/s]
da_d02_cross_SWClear_cntl= xr.open_dataset('d02_cross_SWClear')['SWClear'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave CRF [K/s]
da_d02_cross_SWCRF_cntl= xr.open_dataset('d02_cross_SWCRF')['SWCRF'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Total All-sky Heating [K/s]
da_d02_cross_NetAll_cntl= da_d02_cross_LWAll_cntl+da_d02_cross_SWAll_cntl
# Total CRF [K/s]
da_d02_cross_NetCRF_cntl= xr.open_dataset('d02_cross_TotalCRF')['TotalCRF'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()


## Vertically-integrated Radiative heating
# Longwave Downwelling at Surface [W/m^2]
da_d02_cross_LWDownSfc_cntl= xr.open_dataset('d02_cross_LWDNB')['LWDNB'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave Downwelling at Surface, Clear-sky [W/m^2]
da_d02_cross_LWDownSfcClear_cntl= xr.open_dataset('d02_cross_LWDNBC')['LWDNBC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave Downwelling at Surface [W/m^2]
da_d02_cross_LWUpSfc_cntl= xr.open_dataset('d02_cross_LWUPB')['LWUPB'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave Downwelling at Surface, Clear-sky [W/m^2]
da_d02_cross_LWUpSfcClear_cntl= xr.open_dataset('d02_cross_LWUPBC')['LWUPBC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave Downwelling at Toa [W/m^2]
da_d02_cross_LWDownToa_cntl= xr.open_dataset('d02_cross_LWDNT')['LWDNT'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave Downwelling at Toa, Clear-sky [W/m^2]
da_d02_cross_LWDownToaClear_cntl= xr.open_dataset('d02_cross_LWDNTC')['LWDNTC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave Downwelling at Toa [W/m^2]
da_d02_cross_LWUpToa_cntl= xr.open_dataset('d02_cross_LWUPT')['LWUPT'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Longwave Downwelling at Toa, Clear-sky [W/m^2]
da_d02_cross_LWUpToaClear_cntl= xr.open_dataset('d02_cross_LWUPTC')['LWUPTC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Downwelling at Surface [W/m^2]
da_d02_cross_SWDownSfc_cntl= xr.open_dataset('d02_cross_SWDNB')['SWDNB'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Downwelling at Surface, Clear-sky [W/m^2]
da_d02_cross_SWDownSfcClear_cntl= xr.open_dataset('d02_cross_SWDNBC')['SWDNBC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Downwelling at Surface [W/m^2]
da_d02_cross_SWUpSfc_cntl= xr.open_dataset('d02_cross_SWUPB')['SWUPB'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Downwelling at Surface, Clear-sky [W/m^2]
da_d02_cross_SWUpSfcClear_cntl= xr.open_dataset('d02_cross_SWUPBC')['SWUPBC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Downwelling at Toa [W/m^2]
da_d02_cross_SWDownToa_cntl= xr.open_dataset('d02_cross_SWDNT')['SWDNT'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Downwelling at Toa, Clear-sky [W/m^2]
da_d02_cross_SWDownToaClear_cntl= xr.open_dataset('d02_cross_SWDNTC')['SWDNTC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Downwelling at Toa [W/m^2]
da_d02_cross_SWUpToa_cntl= xr.open_dataset('d02_cross_SWUPT')['SWUPT'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
# Shortwave Downwelling at Toa, Clear-sky [W/m^2]
da_d02_cross_SWUpToaClear_cntl= xr.open_dataset('d02_cross_SWUPTC')['SWUPTC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()


## Integrated Atmospheric Cloud-radiative Forcing Calculations [W/m^2]
# Longwave Atmospheric Radiative Heating [W/m^2]
da_d02_cross_LWAtm_cntl= (da_d02_cross_LWDownToa_cntl-da_d02_cross_LWUpToa_cntl) + (da_d02_cross_LWUpSfc_cntl-da_d02_cross_LWDownSfc_cntl)
# Longwave Atmospheric Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_LWAtmClear_cntl= (da_d02_cross_LWDownToaClear_cntl-da_d02_cross_LWUpToaClear_cntl) + (da_d02_cross_LWUpSfcClear_cntl-da_d02_cross_LWDownSfcClear_cntl)
# Shortwave Atmospheric Radiative Heating [W/m^2]
da_d02_cross_SWAtm_cntl= (da_d02_cross_SWDownToa_cntl-da_d02_cross_SWUpToa_cntl) + (da_d02_cross_SWUpSfc_cntl-da_d02_cross_SWDownSfc_cntl)
# Shortwave Atmospheric Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_SWAtmClear_cntl= (da_d02_cross_SWDownToaClear_cntl-da_d02_cross_SWUpToaClear_cntl) + (da_d02_cross_SWUpSfcClear_cntl-da_d02_cross_SWDownSfcClear_cntl)
# Net Atmospheric Radiative Heating [W/m^2]
da_d02_cross_NetAtm_cntl= da_d02_cross_LWAtm_cntl + da_d02_cross_SWAtm_cntl
# Net Atmospheric Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_NetAtmClear_cntl= da_d02_cross_LWAtmClear_cntl + da_d02_cross_SWAtmClear_cntl
# Net Atmospheric Cloud-Radiative Forcing [W/m^2]
da_d02_cross_NetAtmCRF_cntl= da_d02_cross_NetAtm_cntl - da_d02_cross_NetAtmClear_cntl


## Integrated Surface Cloud-radiative Forcing Calculations [W/m^2]
# Longwave Surface Radiative Heating [W/m^2]
da_d02_cross_LWSfc_cntl= (da_d02_cross_LWDownSfc_cntl-da_d02_cross_LWUpSfc_cntl)
# Longwave Surface Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_LWSfcClear_cntl= (da_d02_cross_LWDownSfcClear_cntl-da_d02_cross_LWUpSfcClear_cntl)
# Longwave Surface Cloud-Radiative Forcing [W/m^2]
da_d02_cross_LWSfcCRF_cntl= da_d02_cross_LWSfc_cntl - da_d02_cross_LWSfcClear_cntl
# Shortwave Surface Radiative Heating [W/m^2]
da_d02_cross_SWSfc_cntl= (da_d02_cross_SWDownSfc_cntl-da_d02_cross_SWUpSfc_cntl)
# Shortwave Surface Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_SWSfcClear_cntl= (da_d02_cross_SWDownSfcClear_cntl-da_d02_cross_SWUpSfcClear_cntl)
# Shortwave Surface Cloud-Radiative Forcing [W/m^2]
da_d02_cross_SWSfcCRF_cntl= da_d02_cross_SWSfc_cntl - da_d02_cross_SWSfcClear_cntl
# Net Surface Radiative Heating [W/m^2]
da_d02_cross_NetSfc_cntl= da_d02_cross_LWSfc_cntl + da_d02_cross_SWSfc_cntl
# Net Surface Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_NetSfcClear_cntl= da_d02_cross_LWSfcClear_cntl + da_d02_cross_SWSfcClear_cntl
# Net Surface Cloud-Radiative Forcing [W/m^2]
da_d02_cross_NetSfcCRF_cntl= da_d02_cross_NetSfc_cntl - da_d02_cross_NetSfcClear_cntl

## Integrated Top of Atmosphere Cloud-radiative Forcing Calculations [W/m^2]
# Longwave Top of Atmosphere Radiative Heating [W/m^2]
da_d02_cross_LWToa_cntl= (da_d02_cross_LWDownToa_cntl-da_d02_cross_LWUpToa_cntl)
# Longwave Top of Atmosphere Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_LWToaClear_cntl= (da_d02_cross_LWDownToaClear_cntl-da_d02_cross_LWUpToaClear_cntl)
# Longwave Top of Atmosphere Cloud-Radiative Forcing [W/m^2]
da_d02_cross_LWToaCRF_cntl= da_d02_cross_LWToa_cntl - da_d02_cross_LWToaClear_cntl
# Shortwave Top of Atmosphere Radiative Heating [W/m^2]
da_d02_cross_SWToa_cntl= (da_d02_cross_SWDownToa_cntl-da_d02_cross_SWUpToa_cntl)
# Shortwave Top of Atmosphere Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_SWToaClear_cntl= (da_d02_cross_SWDownToaClear_cntl-da_d02_cross_SWUpToaClear_cntl)
# Shortwave Top of Atmosphere Cloud-Radiative Forcing [W/m^2]
da_d02_cross_SWToaCRF_cntl= da_d02_cross_SWToa_cntl - da_d02_cross_SWToaClear_cntl
# Net Top of Atmosphere Radiative Heating [W/m^2]
da_d02_cross_NetToa_cntl= da_d02_cross_LWToa_cntl + da_d02_cross_SWToa_cntl
# Net Top of Atmosphere Radiative Heating, Clear-sky [W/m^2]
da_d02_cross_NetToaClear_cntl= da_d02_cross_LWToaClear_cntl + da_d02_cross_SWToaClear_cntl
# Net Top of Atmosphere Cloud-Radiative Forcing [W/m^2]
da_d02_cross_NetToaCRF_cntl= da_d02_cross_NetToa_cntl - da_d02_cross_NetToaClear_cntl

# # If first time calculating the Integrated Atmospheric values, run the section of code below
# da_d02_cross_QTotal_cntl.name = 'QTotal'
# da_d02_cross_QTotal_cntl.to_netcdf(path='d02_cross_QTotal', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_cross_LWAtm_cntl.name = 'LWAtm'
# da_d02_cross_LWAtm_cntl.to_netcdf(path='d02_cross_LWAtm', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_cross_LWAtmClear_cntl.name = 'LWAtmClear'
# da_d02_cross_LWAtmClear_cntl.to_netcdf(path='d02_cross_LWAtmClear', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_cross_SWAtm_cntl.name = 'SWAtm'
# da_d02_cross_SWAtm_cntl.to_netcdf(path='d02_cross_SWAtm', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_cross_SWAtmClear_cntl.name = 'SWAtmClear'
# da_d02_cross_SWAtmClear_cntl.to_netcdf(path='d02_cross_SWAtmClear', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_cross_NetAtm_cntl.name = 'NetAtm'
# da_d02_cross_NetAtm_cntl.to_netcdf(path='d02_cross_NetAtm', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_cross_NetAtmClear_cntl.name = 'NetAtmClear'
# da_d02_cross_NetAtmClear_cntl.to_netcdf(path='d02_cross_NetAtmClear', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# da_d02_cross_NetAtmCRF_cntl.name = 'NetAtmCRF'
# da_d02_cross_NetAtmCRF_cntl.to_netcdf(path='d02_cross_NetAtmCRF', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)

step1_time = time.perf_counter()
print('Control cross-section data loaded \N{check mark}', step1_time-start_time, 'seconds')

# # Terrain	Takes about 3 min
# # Updated method on plotting terrain
# 	# I take each line from the cross section, capture the surface pressure at each line,
# 	# and then average it over all the lines to get the AVERAGE TERRAIN HEIGHT.)
# terrain_height_d02 = np.zeros([da_d02_cross_NormalWind_cntl.shape[0],da_d02_cross_NormalWind_cntl.shape[2],da_d02_cross_NormalWind_cntl.shape[3]])
# for i in range(int(len(da_d02_cross_NormalWind_cntl.Spread))):
# 	for j in range(da_d02_cross_NormalWind_cntl.shape[0]):
# 		terrain_height_d02[j,:,i] = da_d02_PSFC[j,...].interp(south_north=da_d02_cross_NormalWind_cntl.Lat[:,i], west_east=da_d02_cross_NormalWind_cntl.Lon[:,i], method="linear")

# # Turn into dataArray
# d02_cross_PSFC = xr.DataArray(
# 	data=terrain_height_d02,
# 	dims=['Time','Distance','Spread'],
# 	coords=dict(
# 		Time=ds_d02.XTIME.values,
# 		Distance=da_d02_cross_NormalWind_cntl.Distance.values,
# 		Spread=da_d02_cross_NormalWind_cntl.Spread.values,
# 		Lat=(['Distance','Spread'],da_d02_cross_NormalWind_cntl.Lat.values),
# 		Lon=(['Distance','Spread'],da_d02_cross_NormalWind_cntl.Lon.values)
# 	)
# )
# Name and save the dataArray
# d02_cross_PSFC.name = 'PSFC'
# d02_cross_PSFC.to_netcdf(path='d02_cross_PSFC', mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
d02_cross_PSFC = xr.open_dataset('d02_cross_PSFC')['PSFC'].isel(Time=slice(time_ind_d01[0],time_ind_d01[1])).compute()
step2_time = time.perf_counter()
print('Terrain data loaded \N{check mark}', step2_time-step1_time, 'seconds')


# #### CRF Off Data

# In[ ]:


# Load in the cross-sectional data that has been concatinated with all the simulations where CRF is turned off.
	# These dataarrays will hold x-amount of simulations and are created from the code above ^

# cd into the directory with your cross-sectional datasets to make the file paths when using xr.open_dataset a lot shorter
parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff/'

# cd into the specific cross-section directory
os.chdir(parent_dir+cross_dir)

# Sumatra_mid_central, Sumatra_northwest, Borneo_northwest

## Load in the variables
start_time = time.perf_counter()
# Normal Wind [m/s]
da_d02_cross_NormalWind_CRFoff= xr.open_dataset('d02_cross_NormalWind_CRFoff')['NormalWind'].compute()
# Vertical Wind [m/s]
# da_d02_cross_W_CRFoff= xr.open_dataset('d02_cross_W_CRFoff')['W'].compute()
# Rain Rate [mm/day]
da_d02_cross_RR_CRFoff= xr.open_dataset('d02_cross_RR_CRFoff')['RR'].compute()
# Upward Sensible Heat Flux @ Surface [W/m^2]
da_d02_cross_HFX_CRFoff= xr.open_dataset('d02_cross_HFX_CRFoff')['HFX'].compute()
# Latent Heat Flux @ Surface [W/m^2]
da_d02_cross_LH_CRFoff= xr.open_dataset('d02_cross_LH_CRFoff')['LH'].compute()
# Upward Moisture Flux @ Surface [kg/(m^2 s)]
da_d02_cross_QFX_CRFoff= xr.open_dataset('d02_cross_QFX_CRFoff')['QFX'].compute()
# Water vapor [kg/kg]
da_d02_cross_QV_CRFoff= xr.open_dataset('d02_cross_QV_CRFoff')['QV'].compute()
# # Cloud water [kg/kg]
# da_d02_cross_QC_CRFoff= xr.open_dataset('d02_cross_QC_CRFoff')['QC']
# # Rain water [kg/kg]
# da_d02_cross_QR_CRFoff= xr.open_dataset('d02_cross_QR_CRFoff')['QR']
# # Ice water [kg/kg]
# da_d02_cross_QI_CRFoff= xr.open_dataset('d02_cross_QI_CRFoff')['QI']
# # Snow water [kg/kg]
# da_d02_cross_QS_CRFoff= xr.open_dataset('d02_cross_QS_CRFoff')['QS']
# # Graupel water [kg/kg]
# da_d02_cross_QG_CRFoff= xr.open_dataset('d02_cross_QG_CRFoff')['QG']
# Total water [kg/kg]
da_d02_cross_QTotal_CRFoff= xr.open_dataset('d02_cross_QTotal_CRFoff')['QTotal'].compute()
# Cloud Fraction [0->1]
da_d02_cross_CLDFRA_CRFoff= xr.open_dataset('d02_cross_CLDFRA_CRFoff')['CLDFRA'].compute()
	# Low Cloud Fraction [1000-750 hPa]
da_d02_cross_LowCLDFRA_CRFoff=da_d02_cross_CLDFRA_CRFoff.sel(bottom_top=slice(1000,750)).mean('bottom_top')
	# Mid Cloud Fraction [750-500 hPa]
da_d02_cross_MidCLDFRA_CRFoff=da_d02_cross_CLDFRA_CRFoff.sel(bottom_top=slice(750,500)).mean('bottom_top')
	# High Cloud Fraction [500-200 hPa]
da_d02_cross_HighCLDFRA_CRFoff=da_d02_cross_CLDFRA_CRFoff.sel(bottom_top=slice(500,200)).mean('bottom_top')
# Latent Heating [K/s]
da_d02_cross_H_DIABATIC_CRFoff= xr.open_dataset('d02_cross_H_DIABATIC_CRFoff')['H_DIABATIC'].compute()
# Potential Temperature [K]
da_d02_cross_Theta_CRFoff= xr.open_dataset('d02_cross_Theta_CRFoff')['Theta'].compute()
# Temperature 2m [K]
da_d02_cross_T2_CRFoff= xr.open_dataset('d02_cross_T2_CRFoff')['T2'].compute()
# Surface Pressure [hPa]
da_d02_cross_PSFC_CRFoff= xr.open_dataset('d02_cross_PSFC_CRFoff')['PSFC'].compute()
# Potential Temperature @ 2m [K]
da_d02_cross_Theta2_CRFoff= temp_to_theta(da_d02_cross_T2_CRFoff,da_d02_cross_PSFC_CRFoff)
# Virtual Temperature [K]
da_d02_cross_Tv_CRFoff = virt_temp(theta_to_temp(da_d02_cross_Theta_CRFoff), mixing_ratio(da_d02_cross_QV_CRFoff), 0.622)
# # Relative Humidity [%]
# da_d02_cross_RH_CRFoff = rel_hum(da_d02_cross_Theta_CRFoff, da_d02_cross_QTotal_CRFoff)
# CAPE [J/kg]
da_d02_cross_CAPE_CRFoff= xr.open_dataset('d02_cross_CAPE_CRFoff')['CAPE'].compute()
# CIN [J/kg]
da_d02_cross_CIN_CRFoff= xr.open_dataset('d02_cross_CIN_CRFoff')['CIN'].compute()


## Vertically-resolved Radiative heating
# Longwave All-sky Heating [K/s]
da_d02_cross_LWAll_CRFoff= xr.open_dataset('d02_cross_LWAll_CRFoff')['LWAll'].compute()
# Shortwave All-sky Heating [K/s]
da_d02_cross_SWAll_CRFoff= xr.open_dataset('d02_cross_SWAll_CRFoff')['SWAll'].compute()
# Net All-sky Heating [K/s]
da_d02_cross_NetAll_CRFoff= da_d02_cross_LWAll_CRFoff+da_d02_cross_SWAll_CRFoff


## Vertically-INTEGRATED Radiative heating
# Longwave Downwelling at Surface [W/m^2]
da_d02_cross_LWDownSfc_CRFoff= xr.open_dataset('d02_cross_LWDNB_CRFoff')['LWDNB'].compute()
# Longwave Downwelling at Surface [W/m^2]
da_d02_cross_LWUpSfc_CRFoff= xr.open_dataset('d02_cross_LWUPB_CRFoff')['LWUPB'].compute()
# Longwave Downwelling at Toa [W/m^2]
da_d02_cross_LWDownToa_CRFoff= xr.open_dataset('d02_cross_LWDNT_CRFoff')['LWDNT'].compute()
# Longwave Downwelling at Toa [W/m^2]
da_d02_cross_LWUpToa_CRFoff= xr.open_dataset('d02_cross_LWUPT_CRFoff')['LWUPT'].compute()
# Shortwave Downwelling at Surface [W/m^2]
da_d02_cross_SWDownSfc_CRFoff= xr.open_dataset('d02_cross_SWDNB_CRFoff')['SWDNB'].compute()
# Shortwave Downwelling at Surface [W/m^2]
da_d02_cross_SWUpSfc_CRFoff= xr.open_dataset('d02_cross_SWUPB_CRFoff')['SWUPB'].compute()
# Shortwave Downwelling at Toa [W/m^2]
da_d02_cross_SWDownToa_CRFoff= xr.open_dataset('d02_cross_SWDNT_CRFoff')['SWDNT'].compute()
# Shortwave Downwelling at Toa [W/m^2]
da_d02_cross_SWUpToa_CRFoff= xr.open_dataset('d02_cross_SWUPT_CRFoff')['SWUPT'].compute()
# LW Atmospheric Heating [W/m^2]
da_d02_cross_LWAtm_CRFoff= xr.open_dataset('d02_cross_LWAtm_CRFoff')['LWAtm'].compute()
# SW Atmospheric Heating [W/m^2]
da_d02_cross_SWAtm_CRFoff= xr.open_dataset('d02_cross_SWAtm_CRFoff')['SWAtm'].compute()
# Net Atmospheric Heating [W/m^2]
da_d02_cross_NetAtm_CRFoff= xr.open_dataset('d02_cross_NetAtm_CRFoff')['NetAtm'].compute()

## Integrated Surface Cloud-radiative Forcing Calculations [W/m^2]
# Longwave Surface Radiative Heating [W/m^2]
da_d02_cross_LWSfc_CRFoff= (da_d02_cross_LWDownSfc_CRFoff-da_d02_cross_LWUpSfc_CRFoff)
# Shortwave Surface Radiative Heating [W/m^2]
da_d02_cross_SWSfc_CRFoff= (da_d02_cross_SWDownSfc_CRFoff-da_d02_cross_SWUpSfc_CRFoff)
# Net Surface Radiative Heating [W/m^2]
da_d02_cross_NetSfc_CRFoff= da_d02_cross_LWSfc_CRFoff + da_d02_cross_SWSfc_CRFoff

## Integrated Top of Atmosphere Cloud-radiative Forcing Calculations [W/m^2]
# Longwave Top of Atmosphere Radiative Heating [W/m^2]
da_d02_cross_LWToa_CRFoff= (da_d02_cross_LWDownToa_CRFoff-da_d02_cross_LWUpToa_CRFoff)
# Shortwave Top of Atmosphere Radiative Heating [W/m^2]
da_d02_cross_SWToa_CRFoff= (da_d02_cross_SWDownToa_CRFoff-da_d02_cross_SWUpToa_CRFoff)
# Net Top of Atmosphere Radiative Heating [W/m^2]
da_d02_cross_NetToa_CRFoff= da_d02_cross_LWToa_CRFoff + da_d02_cross_SWToa_CRFoff


step1_time = time.perf_counter()
print('CRF-off simulation cross-section data loaded \N{check mark}', step1_time-start_time, 'seconds')

# Convert W/m^2 to K/day
# interp_dp_levels = -d02_cross_PSFC
# cp = 1004		# [J/kg K]
# g = -9.8		# [m/s^2]
# da_d02_cross_NetAtm_cntl/(cp*(1/g)*interp_dp_levels)*86400	# [J/sm^2 / J/kgK / s^2/m / kg/ms^2] == [K/s]


# ##### Create Ensemble CRF Off Data
# Combine cross-sectional ensemble data and save them as .nc files
# 
# Uncomment and run if you need the ensemble .nc files

# In[ ]:


# def save_ens_cross_data(parent_dir,cross_dir,var_list,num_sims,sim_time_length):
# 	for i in var_list:		# Loop through the variables

# 		# Create the pathname based on the variable, then save the ensemble cross-sectional data as an .nc file
# 		file_names = sorted(glob.glob(parent_dir + '/2015*/L3/'+ cross_dir + '/d02_cross_' + i))

# 		# Open the multiple files, and concat them over the Time dimension
# 		da = xr.open_mfdataset(file_names, concat_dim='Time', combine='nested', data_vars='all', coords='all')[i]	# Only grab the variable from var_list

# 		# Is data 3-D or 2-D?
# 		if 'bottom_top' in da.dims:		# If the dataset is 3-D
# 			# Use numpy to reshape the data into dimensions of [num_sims X time X height X line_length X #_of_lines]
# 				# This is needed since da Time dimension is the length of each simulation * num_sims
# 			da_np = np.reshape(da.values, (num_sims, sim_time_length, len(da.bottom_top), len(da.Distance), len(da.Spread)))
# 			# Create a new dataArray that holds da_np
# 			da = xr.DataArray(
# 				name=i,
# 				data=da_np,
# 				dims=['Lead','Time','bottom_top','Distance','Spread'],
# 				coords=dict(
# 					Lead = np.arange(0,num_sims,1),
# 					EnsTime = (['Lead','Time'], da.Time.values.reshape(num_sims,sim_time_length)),
# 					bottom_top = da.bottom_top.values,
# 					Distance = da.Distance.values,
# 					Spread = da.Spread.values,
# 					Lat = (['Time','Distance','Spread'], da.Lat.values[:sim_time_length,:,:]),
# 					Lon = (['Time','Distance','Spread'], da.Lon.values[:sim_time_length,:,:])
# 					)
# 			)
# 			# Transpose to the way you want the dims to be ordered
# 			da = da.transpose('Time','bottom_top','Distance','Spread','Lead')

# 		else:	# If the dataset is 2-D
# 			# Use numpy to reshape the data into dimensions of [num_sims X time X line_length X #_of_lines]
# 				# This is needed since da Time dimension is the length of each simulation * num_sims
# 			da_np = np.reshape(da.values, (num_sims, sim_time_length, len(da.Distance), len(da.Spread)))
# 			# Create a new dataArray that holds da_np
# 			da = xr.DataArray(
# 				name=i,
# 				data=da_np,
# 				dims=['Lead','Time','Distance','Spread'],
# 				coords=dict(
# 					Lead = np.arange(0,num_sims,1),
# 					EnsTime = (['Lead','Time'], da.Time.values.reshape(num_sims,sim_time_length)),
# 					Distance = da.Distance.values,
# 					Spread = da.Spread.values,
# 					Lat = (['Time','Distance','Spread'], da.Lat.values[:sim_time_length,:,:]),
# 					Lon = (['Time','Distance','Spread'], da.Lon.values[:sim_time_length,:,:])
# 					)
# 			)
# 			# Transpose to the way you want the dims to be ordered
# 			da = da.transpose('Time','Distance','Spread','Lead')
# 		# Where you'd like to save the dataset as an .nc file
# 		path_name = parent_dir + '/' + cross_dir +'/d02_cross_' + i + '_CRFoff'
# 		da.to_netcdf(path=path_name, mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)
# 		print(i + ' has been saved.')
# 	return

# # Directory where all the simulation sub-directories are kept
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'
# # List of variables that you would like saved
# var_list = ['NormalWind','QFX', 'HFX', 'T2', 'PSFC', 'LH', 'CLDFRA', 'H_DIABATIC', 'Theta','CAPE','CIN', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'LWAll', 'SWAll', 'RR', 'LWDNB', 'LWUPB', 'LWDNT', 'LWUPT', 'SWDNB', 'SWUPB', 'SWDNT', 'SWUPT']
# # var_list = ['?']
# num_sims = 18			# How many sims
# sim_time_length = 36	# How many time steps each sim (all sims should have the same time length)

# cross_dir = 'Sumatra_mid_central', 'Sumatra_northwest', 'Borneo_northwest'

# save_ens_cross_data(parent_dir, cross_dir, var_list, num_sims, sim_time_length)

# ################################################################
# ################ Save the rest of the variables ################
# ################################################################

# # Calculate cross-sectional data and save them as .nc files
# 	# QTotal, LWAtm, LWAtmClear, SWAtm, SWAtmClear, NetAtm, NetAtmClear, NetAtmCRF
# cross_dir = 'Sumatra_northwest'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff/' 
# target_dir = parent_dir + cross_dir

# ## QTotal ##
# file_names = sorted(glob.glob(target_dir + '/d02_cross_Q*'))
# # Sum all the Q's
# da = xr.open_dataset(file_names[0])['QC'] + xr.open_dataset(file_names[2])['QG'] + xr.open_dataset(file_names[3])['QI'] + xr.open_dataset(file_names[4])['QR'] + xr.open_dataset(file_names[5])['QS'] + xr.open_dataset(file_names[6])['QV']
# da.name = 'QTotal'
# # Save the file
# path_name = target_dir + '/d02_cross_QTotal_CRFoff'
# da.to_netcdf(path=path_name, mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)

# # No need to calculate Clear values and CRF because icloud=0 so values are already Clear
# ## LWAtm ##
# da = (xr.open_dataset(target_dir +'/d02_cross_LWDNT_CRFoff')['LWDNT']-xr.open_dataset(target_dir+'/d02_cross_LWUPT_CRFoff')['LWUPT']) + (xr.open_dataset(target_dir+'/d02_cross_LWUPB_CRFoff')['LWUPB']-xr.open_dataset(target_dir+'/d02_cross_LWDNB_CRFoff')['LWDNB'])
# da.name = 'LWAtm'
# path_name = target_dir + '/d02_cross_LWAtm_CRFoff'
# da.to_netcdf(path=path_name, mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)

# ## SWAtm ##
# da = (xr.open_dataset(target_dir +'/d02_cross_SWDNT_CRFoff')['SWDNT']-xr.open_dataset(target_dir+'/d02_cross_SWUPT_CRFoff')['SWUPT']) + (xr.open_dataset(target_dir+'/d02_cross_SWUPB_CRFoff')['SWUPB']-xr.open_dataset(target_dir+'/d02_cross_SWDNB_CRFoff')['SWDNB'])
# da.name = 'SWAtm'
# path_name = target_dir + '/d02_cross_SWAtm_CRFoff'
# da.to_netcdf(path=path_name, mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)

# ## NetAtm ##
# da = xr.open_dataset(target_dir +'/d02_cross_LWAtm_CRFoff')['LWAtm'] + xr.open_dataset(target_dir+'/d02_cross_SWAtm_CRFoff')['SWAtm'] 
# da.name = 'NetAtm'
# path_name = target_dir + '/d02_cross_NetAtm_CRFoff'
# da.to_netcdf(path=path_name, mode='w', format='NETCDF4', unlimited_dims='Time', compute=True)


# ## Sensitivity testing data

# In[ ]:


# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26'

# # 3-D Variables

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_NormalWind')['NormalWind']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_NormalWind')['NormalWind']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_NormalWind')['NormalWind']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_NormalWind')['NormalWind']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_NormalWind')['NormalWind']
# da_d02_NormalWind_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_CLDFRA')['CLDFRA']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_CLDFRA')['CLDFRA']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_CLDFRA')['CLDFRA']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_CLDFRA')['CLDFRA']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_CLDFRA')['CLDFRA']
# da_d02_CLDFRA_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_H_DIABATIC')['H_DIABATIC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_H_DIABATIC')['H_DIABATIC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_H_DIABATIC')['H_DIABATIC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_H_DIABATIC')['H_DIABATIC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_H_DIABATIC')['H_DIABATIC']
# da_d02_H_DIABATIC_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_Theta')['Theta']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_Theta')['Theta']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_Theta')['Theta']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_Theta')['Theta']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_Theta')['Theta']
# da_d02_Theta_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_QV')['QV']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_QV')['QV']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_QV')['QV']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_QV')['QV']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_QV')['QV']
# da_d02_QV_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_QC')['QC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_QC')['QC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_QC')['QC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_QC')['QC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_QC')['QC']
# da_d02_QC_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_QR')['QR']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_QR')['QR']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_QR')['QR']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_QR')['QR']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_QR')['QR']
# da_d02_QR_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_QI')['QI']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_QI')['QI']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_QI')['QI']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_QI')['QI']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_QI')['QI']
# da_d02_QI_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_QS')['QS']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_QS')['QS']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_QS')['QS']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_QS')['QS']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_QS')['QS']
# da_d02_QS_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_QG')['QG']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_QG')['QG']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_QG')['QG']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_QG')['QG']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_QG')['QG']
# da_d02_QG_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# da_d02_QTotal_cross_CRFoff = da_d02_QV_cross_CRFoff + da_d02_QC_cross_CRFoff + da_d02_QR_cross_CRFoff + da_d02_QI_cross_CRFoff + da_d02_QS_cross_CRFoff + da_d02_QG_cross_CRFoff

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWAll')['LWAll']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWAll')['LWAll']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWAll')['LWAll']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWAll')['LWAll']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWAll')['LWAll']
# da_d02_LWAll_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWClear')['LWClear']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWClear')['LWClear']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWClear')['LWClear']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWClear')['LWClear']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWClear')['LWClear']
# da_d02_LWClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWCRF')['LWCRF']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWCRF')['LWCRF']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWCRF')['LWCRF']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWCRF')['LWCRF']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWCRF')['LWCRF']
# da_d02_LWCRF_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWAll')['SWAll']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWAll')['SWAll']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWAll')['SWAll']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWAll')['SWAll']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWAll')['SWAll']
# da_d02_SWAll_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWClear')['SWClear']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWClear')['SWClear']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWClear')['SWClear']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWClear')['SWClear']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWClear')['SWClear']
# da_d02_SWClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWCRF')['SWCRF']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWCRF')['SWCRF']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWCRF')['SWCRF']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWCRF')['SWCRF']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWCRF')['SWCRF']
# da_d02_SWCRF_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_TotalCRF')['TotalCRF']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_TotalCRF')['TotalCRF']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_TotalCRF')['TotalCRF']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_TotalCRF')['TotalCRF']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_TotalCRF')['TotalCRF']
# da_d02_NetCRF_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','bottom_top','Distance','Spread','Lead')

# # 2-D Variables
# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_RR')['RR']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_RR')['RR']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_RR')['RR']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_RR')['RR']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_RR')['RR']
# da_d02_RR_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWDNB')['LWDNB']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWDNB')['LWDNB']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWDNB')['LWDNB']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWDNB')['LWDNB']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWDNB')['LWDNB']
# da_d02_LWDownSfc_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWDNBC')['LWDNBC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWDNBC')['LWDNBC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWDNBC')['LWDNBC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWDNBC')['LWDNBC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWDNBC')['LWDNBC']
# da_d02_LWDownSfcClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWUPB')['LWUPB']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWUPB')['LWUPB']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWUPB')['LWUPB']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWUPB')['LWUPB']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWUPB')['LWUPB']
# da_d02_LWUpSfc_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWUPBC')['LWUPBC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWUPBC')['LWUPBC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWUPBC')['LWUPBC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWUPBC')['LWUPBC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWUPBC')['LWUPBC']
# da_d02_LWUpSfcClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWDNT')['LWDNT']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWDNT')['LWDNT']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWDNT')['LWDNT']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWDNT')['LWDNT']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWDNT')['LWDNT']
# da_d02_LWDownToa_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWDNTC')['LWDNTC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWDNTC')['LWDNTC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWDNTC')['LWDNTC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWDNTC')['LWDNTC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWDNTC')['LWDNTC']
# da_d02_LWDownToaClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWUPT')['LWUPT']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWUPT')['LWUPT']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWUPT')['LWUPT']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWUPT')['LWUPT']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWUPT')['LWUPT']
# da_d02_LWUpToa_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_LWUPTC')['LWUPTC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_LWUPTC')['LWUPTC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_LWUPTC')['LWUPTC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_LWUPTC')['LWUPTC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_LWUPTC')['LWUPTC']
# da_d02_LWUpToaClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWDNB')['SWDNB']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWDNB')['SWDNB']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWDNB')['SWDNB']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWDNB')['SWDNB']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWDNB')['SWDNB']
# da_d02_SWDownSfc_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWDNBC')['SWDNBC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWDNBC')['SWDNBC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWDNBC')['SWDNBC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWDNBC')['SWDNBC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWDNBC')['SWDNBC']
# da_d02_SWDownSfcClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWUPB')['SWUPB']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWUPB')['SWUPB']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWUPB')['SWUPB']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWUPB')['SWUPB']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWUPB')['SWUPB']
# da_d02_SWUpSfc_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWUPBC')['SWUPBC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWUPBC')['SWUPBC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWUPBC')['SWUPBC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWUPBC')['SWUPBC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWUPBC')['SWUPBC']
# da_d02_SWUpSfcClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWDNT')['SWDNT']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWDNT')['SWDNT']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWDNT')['SWDNT']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWDNT')['SWDNT']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWDNT')['SWDNT']
# da_d02_SWDownToa_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWDNTC')['SWDNTC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWDNTC')['SWDNTC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWDNTC')['SWDNTC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWDNTC')['SWDNTC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWDNTC')['SWDNTC']
# da_d02_SWDownToaClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWUPT')['SWUPT']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWUPT')['SWUPT']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWUPT')['SWUPT']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWUPT')['SWUPT']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWUPT')['SWUPT']
# da_d02_SWUpToa_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# var1 = xr.open_dataset(parent_dir + '/2015-11-25-03--11-26-12/L3/d02_cross_SWUPTC')['SWUPTC']
# var2 = xr.open_dataset(parent_dir + '/2015-11-25-06--11-26-12/L3/d02_cross_SWUPTC')['SWUPTC']
# var3 = xr.open_dataset(parent_dir + '/2015-11-25-09--11-26-12/L3/d02_cross_SWUPTC')['SWUPTC']
# var4 = xr.open_dataset(parent_dir + '/2015-11-25-12--11-26-12/L3/d02_cross_SWUPTC')['SWUPTC']
# var5 = xr.open_dataset(parent_dir + '/2015-11-25-15--11-26-12/L3/d02_cross_SWUPTC')['SWUPTC']
# da_d02_SWUpToaClear_cross_CRFoff = xr.concat([var1,var2,var3,var4,var5],dim='Lead',data_vars='all',coords='all',compat='broadcast_equals').transpose('Time','Distance','Spread','Lead').compute()

# # Longwave Atmospheric Radiative Heating [W/m^2]
# da_d02_LWAtm_cross_CRFoff= (da_d02_LWDownToa_cross_CRFoff-da_d02_LWUpToa_cross_CRFoff) + (da_d02_LWUpSfc_cross_CRFoff-da_d02_LWDownSfc_cross_CRFoff)
# # Longwave Atmospheric Radiative Heating, Clear-sky [W/m^2]
# da_d02_LWAtmClear_cross_CRFoff= (da_d02_LWDownToaClear_cross_CRFoff-da_d02_LWUpToaClear_cross_CRFoff) + (da_d02_LWUpSfcClear_cross_CRFoff-da_d02_LWDownSfcClear_cross_CRFoff)
# # Shortwave Atmospheric Radiative Heating [W/m^2]
# da_d02_SWAtm_cross_CRFoff= (da_d02_SWDownToa_cross_CRFoff-da_d02_SWUpToa_cross_CRFoff) + (da_d02_SWUpSfc_cross_CRFoff-da_d02_SWDownSfc_cross_CRFoff)
# # Shortwave Atmospheric Radiative Heating, Clear-sky [W/m^2]
# da_d02_SWAtmClear_cross_CRFoff= (da_d02_SWDownToaClear_cross_CRFoff-da_d02_SWUpToaClear_cross_CRFoff) + (da_d02_SWUpSfcClear_cross_CRFoff-da_d02_SWDownSfcClear_cross_CRFoff)
# # Net Atmospheric Radiative Heating [W/m^2]
# da_d02_NetAtm_cross_CRFoff= da_d02_LWAtm_cross_CRFoff + da_d02_SWAtm_cross_CRFoff
# # Net Atmospheric Radiative Heating, Clear-sky [W/m^2]
# da_d02_NetAtmClear_cross_CRFoff= da_d02_LWAtmClear_cross_CRFoff + da_d02_SWAtmClear_cross_CRFoff
# # Net Atmospheric Cloud-Radiative Forcing [W/m^2]
# da_d02_NetAtmCRF_cross_CRFoff= da_d02_NetAtm_cross_CRFoff - da_d02_NetAtmClear_cross_CRFoff

# # Save the variables
# os.chdir('/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26')

# da_d02_NormalWind_cross_CRFoff.to_netcdf(path='da_d02_NormalWind_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_CLDFRA_cross_CRFoff.to_netcdf(path='da_d02_CLDFRA_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_H_DIABATIC_cross_CRFoff.to_netcdf(path='da_d02_H_DIABATIC_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_Theta_cross_CRFoff.to_netcdf(path='da_d02_Theta_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_QV_cross_CRFoff.to_netcdf(path='da_d02_QV_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_QC_cross_CRFoff.to_netcdf(path='da_d02_QC_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_QR_cross_CRFoff.to_netcdf(path='da_d02_QR_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_QI_cross_CRFoff.to_netcdf(path='da_d02_QI_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_QS_cross_CRFoff.to_netcdf(path='da_d02_QS_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_QG_cross_CRFoff.to_netcdf(path='da_d02_QG_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_QTotal_cross_CRFoff.to_netcdf(path='da_d02_QTotal_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWAll_cross_CRFoff.to_netcdf(path='da_d02_LWAll_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWClear_cross_CRFoff.to_netcdf(path='da_d02_LWClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWCRF_cross_CRFoff.to_netcdf(path='da_d02_LWCRF_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWAll_cross_CRFoff.to_netcdf(path='da_d02_SWAll_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWClear_cross_CRFoff.to_netcdf(path='da_d02_SWClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWCRF_cross_CRFoff.to_netcdf(path='da_d02_SWCRF_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_NetCRF_cross_CRFoff.to_netcdf(path='da_d02_NetCRF_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_RR_cross_CRFoff.to_netcdf(path='da_d02_RR_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWDownSfc_cross_CRFoff.to_netcdf(path='da_d02_LWDownSfc_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWDownSfcClear_cross_CRFoff.to_netcdf(path='da_d02_LWDownSfcClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWUpSfc_cross_CRFoff.to_netcdf(path='da_d02_LWUpSfc_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWUpSfcClear_cross_CRFoff.to_netcdf(path='da_d02_LWUpSfcClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWDownToa_cross_CRFoff.to_netcdf(path='da_d02_LWDownToa_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWDownToaClear_cross_CRFoff.to_netcdf(path='da_d02_LWDownToaClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWUpToa_cross_CRFoff.to_netcdf(path='da_d02_LWUpToa_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWUpToaClear_cross_CRFoff.to_netcdf(path='da_d02_LWUpToaClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWDownSfc_cross_CRFoff.to_netcdf(path='da_d02_SWDownSfc_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWDownSfcClear_cross_CRFoff.to_netcdf(path='da_d02_SWDownSfcClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWUpSfc_cross_CRFoff.to_netcdf(path='da_d02_SWUpSfc_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWUpSfcClear_cross_CRFoff.to_netcdf(path='da_d02_SWUpSfcClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWDownToa_cross_CRFoff.to_netcdf(path='da_d02_SWDownToa_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWDownToaClear_cross_CRFoff.to_netcdf(path='da_d02_SWDownToaClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWUpToa_cross_CRFoff.to_netcdf(path='da_d02_SWUpToa_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWUpToaClear_cross_CRFoff.to_netcdf(path='da_d02_SWUpToaClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWAtm_cross_CRFoff.to_netcdf(path='da_d02_LWAtm_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_LWAtmClear_cross_CRFoff.to_netcdf(path='da_d02_LWAtmClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWAtm_cross_CRFoff.to_netcdf(path='da_d02_SWAtm_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_SWAtmClear_cross_CRFoff.to_netcdf(path='da_d02_SWAtmClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_NetAtm_cross_CRFoff.to_netcdf(path='da_d02_NetAtm_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_NetAtmClear_cross_CRFoff.to_netcdf(path='da_d02_NetAtmClear_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)
# da_d02_NetAtmCRF_cross_CRFoff.to_netcdf(path='da_d02_NetAtmCRF_cross_CRFoff',mode='w',format='NETCDF4',unlimited_dims='Time',compute=True)


# # AMS Tropical 36 Post Figures

# Western Central Sumatra cross-sectional view

# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)
fs = 18
d02_coords = dict(
    south_north=('south_north',ds_d02.XLAT[0,:,0].values),
    west_east=('west_east',ds_d02.XLONG[0,0,:].values)
    )
# Prepare the Terrain Height    [m]
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()
x = da_d02_TOPO.assign_coords(d02_coords)
lat = [da_d02_cross_NormalWind_cntl.Lat.min()-0.5, da_d02_cross_NormalWind_cntl.Lat.max()+0.5]
lon = [da_d02_cross_NormalWind_cntl.Lon.min()-0.5, da_d02_cross_NormalWind_cntl.Lon.max()+0.5]

# Yokoi et al. 2017-2019 domain:
x = x.sel(
	south_north=slice(lat[0],lat[1]),
	west_east=slice(lon[0],lon[1]))

# x_ticks = np.array([100,102,104])
# x_tick_labels = [u'100\N{DEGREE SIGN}E',
#                  u'102\N{DEGREE SIGN}E', u'104\N{DEGREE SIGN}E']
# y_ticks = np.array([-6,-4,-2])
# y_tick_labels = [u'6\N{DEGREE SIGN}S',
#                  u'4\N{DEGREE SIGN}S', u'2\N{DEGREE SIGN}S']

# Plot your terrain
ax1 = fig.add_subplot(gs[0,0])
cf1 = x.plot.contourf(
	cmap='terrain',
	# levels=np.arange(0,4250,250),
	levels = np.append(0,np.logspace(0,3.65,50)),

)

# Plot the individual cross-sectioned lines
for i in range(4,int(da_d02_cross_NormalWind_cntl.shape[3])):
	plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,i],da_d02_cross_NormalWind_cntl.Lat[:,i],'r',linewidth=0.5)
# Plot the center line
plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],da_d02_cross_NormalWind_cntl.Lat[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(da_d02_cross_NormalWind_cntl.Lon[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],da_d02_cross_NormalWind_cntl.Lat[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='r', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='r', label='BMKG Observatory')

# # Plot Assumed Coastline
plt.scatter(da_d02_cross_NormalWind_cntl.Lon[np.where(da_d02_cross_NormalWind_cntl.Distance==0)],da_d02_cross_NormalWind_cntl.Lat[np.where(da_d02_cross_NormalWind_cntl.Distance==0)],s=3)


# Plot the individual cross-sectioned lines
for i in range(int(da_d02_cross_NormalWind_cntl_old.shape[3])-4):
	plt.plot(da_d02_cross_NormalWind_cntl_old.Lon[:,i],da_d02_cross_NormalWind_cntl_old.Lat[:,i],'k',linewidth=0.5)
# Plot the center line
plt.plot(da_d02_cross_NormalWind_cntl_old.Lon[:,int(da_d02_cross_NormalWind_cntl_old.shape[3]/2)],da_d02_cross_NormalWind_cntl_old.Lat[:,int(da_d02_cross_NormalWind_cntl_old.shape[3]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(da_d02_cross_NormalWind_cntl_old.Lon[:,int(da_d02_cross_NormalWind_cntl_old.shape[3]/2)],da_d02_cross_NormalWind_cntl_old.Lat[:,int(da_d02_cross_NormalWind_cntl_old.shape[3]/2)],s=3)

# Plot Assumed Coastline
plt.scatter(da_d02_cross_NormalWind_cntl_old.Lon[np.where(da_d02_cross_NormalWind_cntl_old.Distance==0)],da_d02_cross_NormalWind_cntl_old.Lat[np.where(da_d02_cross_NormalWind_cntl_old.Distance==0)],s=3)


cbar=cf1.colorbar
cbar.set_label('Terrain Height [m]',fontsize=fs)
cbar.set_ticks([0,10,100,1000,2000,3000])
cbar.minorticks_off()
cbar.set_ticklabels([0,10,100,1000,2000,3000],fontsize=fs)

ax1.set_xlabel('',fontsize=18)
ax1.set_ylabel('',fontsize=18)
ax1.set_title('Western Central Sumatra',fontsize=fs+4)
# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(x_tick_labels,fontsize=fs)
# ax1.set_yticks(y_ticks)
# ax1.set_yticklabels(y_tick_labels,fontsize=fs)
ax1.legend(fontsize='x-large', markerscale=1.7, edgecolor='0')
# fig.savefig('/home/hragnajarian/PhD/plots/Domain_cross.png', transparent=True)


# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)
fs = 18
d02_coords = dict(
    south_north=('south_north',ds_d02.XLAT[0,:,0].values),
    west_east=('west_east',ds_d02.XLONG[0,0,:].values)
    )
# Prepare the Terrain Height    [m]
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()
x = da_d02_TOPO.assign_coords(d02_coords)
lat = [da_d02_cross_NormalWind_cntl_old.Lat.min()-0.5, da_d02_cross_NormalWind_cntl_old.Lat.max()+0.5]
lon = [da_d02_cross_NormalWind_cntl_old.Lon.min()-0.5, da_d02_cross_NormalWind_cntl_old.Lon.max()+0.5]

# Yokoi et al. 2017-2019 domain:
x = x.sel(
	south_north=slice(lat[0],lat[1]),
	west_east=slice(lon[0],lon[1]))

# x_ticks = np.array([100,102,104])
# x_tick_labels = [u'100\N{DEGREE SIGN}E',
#                  u'102\N{DEGREE SIGN}E', u'104\N{DEGREE SIGN}E']
# y_ticks = np.array([-6,-4,-2])
# y_tick_labels = [u'6\N{DEGREE SIGN}S',
#                  u'4\N{DEGREE SIGN}S', u'2\N{DEGREE SIGN}S']

# Plot your terrain
ax1 = fig.add_subplot(gs[0,0])
cf1 = x.plot.contourf(
	cmap='terrain',
	# levels=np.arange(0,4250,250),
	levels = np.append(0,np.logspace(0,3.65,50)),

)

# Plot the individual cross-sectioned lines
for i in range(int(da_d02_cross_NormalWind_cntl_old.shape[3])):
	plt.plot(da_d02_cross_NormalWind_cntl_old.Lon[:,i],da_d02_cross_NormalWind_cntl_old.Lat[:,i],'r',linewidth=0.5)
# Plot the center line
plt.plot(da_d02_cross_NormalWind_cntl_old.Lon[:,int(da_d02_cross_NormalWind_cntl_old.shape[3]/2)],da_d02_cross_NormalWind_cntl_old.Lat[:,int(da_d02_cross_NormalWind_cntl_old.shape[3]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(da_d02_cross_NormalWind_cntl_old.Lon[:,int(da_d02_cross_NormalWind_cntl_old.shape[3]/2)],da_d02_cross_NormalWind_cntl_old.Lat[:,int(da_d02_cross_NormalWind_cntl_old.shape[3]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='r', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='r', label='BMKG Observatory')

# Plot Assumed Coastline
plt.scatter(da_d02_cross_NormalWind_cntl_old.Lon[np.where(da_d02_cross_NormalWind_cntl_old.Distance==0)],da_d02_cross_NormalWind_cntl_old.Lat[np.where(da_d02_cross_NormalWind_cntl_old.Distance==0)],s=3)

cbar=cf1.colorbar
cbar.set_label('Terrain Height [m]',fontsize=fs)
cbar.set_ticks([0,10,100,1000,2000,3000])
cbar.minorticks_off()
cbar.set_ticklabels([0,10,100,1000,2000,3000],fontsize=fs)

ax1.set_xlabel('',fontsize=18)
ax1.set_ylabel('',fontsize=18)
ax1.set_title('Western Central Sumatra',fontsize=fs+4)
# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(x_tick_labels,fontsize=fs)
# ax1.set_yticks(y_ticks)
# ax1.set_yticklabels(y_tick_labels,fontsize=fs)
ax1.legend(fontsize='x-large', markerscale=1.7, edgecolor='0')
# fig.savefig('/home/hragnajarian/PhD/plots/Domain_cross.png', transparent=True)


# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)
fs = 18
d02_coords = dict(
    south_north=('south_north',ds_d02.XLAT[0,:,0].values),
    west_east=('west_east',ds_d02.XLONG[0,0,:].values)
    )
# Prepare the Terrain Height    [m]
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()
x = da_d02_TOPO.assign_coords(d02_coords)
lat = [da_d02_cross_NormalWind_CRFoff.Lat.min()-0.5, da_d02_cross_NormalWind_CRFoff.Lat.max()+0.5]
lon = [da_d02_cross_NormalWind_CRFoff.Lon.min()-0.5, da_d02_cross_NormalWind_CRFoff.Lon.max()+0.5]

# Yokoi et al. 2017-2019 domain:
x = x.sel(
	south_north=slice(lat[0],lat[1]),
	west_east=slice(lon[0],lon[1]))

# x_ticks = np.array([100,102,104])
# x_tick_labels = [u'100\N{DEGREE SIGN}E',
#                  u'102\N{DEGREE SIGN}E', u'104\N{DEGREE SIGN}E']
# y_ticks = np.array([-6,-4,-2])
# y_tick_labels = [u'6\N{DEGREE SIGN}S',
#                  u'4\N{DEGREE SIGN}S', u'2\N{DEGREE SIGN}S']

# Plot your terrain
ax1 = fig.add_subplot(gs[0,0])
cf1 = x.plot.contourf(
	cmap='terrain',
	# levels=np.arange(0,4250,250),
	levels = np.append(0,np.logspace(0,3.65,50)),

)

# Plot the individual cross-sectioned lines
for i in range(int(da_d02_cross_NormalWind_CRFoff.shape[3])):
	plt.plot(da_d02_cross_NormalWind_CRFoff.Lon[0,:,i],da_d02_cross_NormalWind_CRFoff.Lat[0,:,i],'r',linewidth=0.5)
# Plot the center line
plt.plot(da_d02_cross_NormalWind_CRFoff.Lon[0,:,int(da_d02_cross_NormalWind_CRFoff.shape[3]/2)],da_d02_cross_NormalWind_CRFoff.Lat[0,:,int(da_d02_cross_NormalWind_CRFoff.shape[3]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(da_d02_cross_NormalWind_CRFoff.Lon[0,:,int(da_d02_cross_NormalWind_CRFoff.shape[3]/2)],da_d02_cross_NormalWind_CRFoff.Lat[0,:,int(da_d02_cross_NormalWind_CRFoff.shape[3]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='r', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='r', label='BMKG Observatory')

# Plot Assumed Coastline
plt.scatter(da_d02_cross_NormalWind_CRFoff.Lon[0][np.where(da_d02_cross_NormalWind_CRFoff.Distance==0)],da_d02_cross_NormalWind_CRFoff.Lat[0][np.where(da_d02_cross_NormalWind_CRFoff.Distance==0)])

cbar=cf1.colorbar
cbar.set_label('Terrain Height [m]',fontsize=fs)
cbar.set_ticks([0,10,100,1000,2000,3000])
cbar.minorticks_off()
cbar.set_ticklabels([0,10,100,1000,2000,3000],fontsize=fs)

ax1.set_xlabel('',fontsize=18)
ax1.set_ylabel('',fontsize=18)
ax1.set_title('Western Central Sumatra',fontsize=fs+4)
# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(x_tick_labels,fontsize=fs)
# ax1.set_yticks(y_ticks)
# ax1.set_yticklabels(y_tick_labels,fontsize=fs)
ax1.legend(fontsize='x-large', markerscale=1.7, edgecolor='0')
# fig.savefig('/home/hragnajarian/PhD/plots/Domain_cross.png', transparent=True)


# North Western Sumatra cross-sectional view

# In[ ]:


# cross_section_multi(da, start_coord, end_coord, width, dx)
    # North West Sumatra
start_coord		= [5.2,96.4]
end_coord 		= [1.2,92.4]
width			= 1.5
dx 				= 0.025

da_cross_temp, all_line_coords = cross_section_multi(da_d01_RR, start_coord, end_coord, width, dx)


# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)

fs=18

d02_coords = dict(
    south_north=('south_north',ds_d02.XLAT[0,:,0].values),
    west_east=('west_east',ds_d02.XLONG[0,0,:].values)
    )
# Prepare the Terrain Height    [m]
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()
x = da_d02_TOPO.assign_coords(d02_coords)
lat = [all_line_coords[0].min()-0.5, all_line_coords[0].max()+0.5]
lon = [all_line_coords[1].min()-0.5, all_line_coords[1].max()+0.5]

# Yokoi et al. 2017-2019 domain:
x = x.sel(
	south_north=slice(lat[0],lat[1]),
	west_east=slice(lon[0],lon[1]))

x_ticks = np.array([92,94,96])
x_tick_labels = [u'92\N{DEGREE SIGN}E',
                 u'94\N{DEGREE SIGN}E', u'96\N{DEGREE SIGN}E']
y_ticks = np.array([0,2,4,6])
y_tick_labels = [u'0\N{DEGREE SIGN}',
                 u'2\N{DEGREE SIGN}N', u'4\N{DEGREE SIGN}N', u'6\N{DEGREE SIGN}N']

# Plot your terrain
ax1 = fig.add_subplot(gs[0,0])
cf1 = x.plot.contourf(
	cmap='terrain',
	# levels=np.arange(0,4250,250),
	levels = np.append(0,np.logspace(0,3.65,50)),

)

# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords.shape[2])):
	plt.plot(all_line_coords[1,:,i],all_line_coords[0,:,i],'r',linewidth=0.5)
# Plot the center line
plt.plot(all_line_coords[1][:,int(all_line_coords.shape[2]/2)],all_line_coords[0][:,int(all_line_coords.shape[2]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords[1][:,int(all_line_coords.shape[2]/2)],all_line_coords[0][:,int(all_line_coords.shape[2]/2)],s=3)

cbar=cf1.colorbar
cbar.set_label('Terrain Height [m]',fontsize=fs)
cbar.set_ticks([0,10,100,1000,2000,3000])
cbar.minorticks_off()
cbar.set_ticklabels([0,10,100,1000,2000,3000],fontsize=fs)

ax1.set_xlabel('',fontsize=18)
ax1.set_ylabel('',fontsize=18)
ax1.set_title('North Western Sumatra',fontsize=fs+4)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels,fontsize=fs)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels,fontsize=fs)
# ax1.legend(fontsize='x-large', markerscale=1.7, edgecolor='0')
# fig.savefig('/home/hragnajarian/PhD/plots/Domain_cross.png', transparent=True)


# North Western Borneo cross-sectional view

# In[ ]:


# cross_section_multi(da, start_coord, end_coord, width, dx)
    # North West Borneo
start_coord		= [1.2,112.8]
end_coord 		= [5.9,108.1]
width			= 1.5
dx 				= 0.025

da_cross_temp, all_line_coords = cross_section_multi(da_d01_RR, start_coord, end_coord, width, dx)
da_cross_temp.shape


# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))

gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)

fs=18

d02_coords = dict(
    south_north=('south_north',ds_d02.XLAT[0,:,0].values),
    west_east=('west_east',ds_d02.XLONG[0,0,:].values)
    )
# Prepare the Terrain Height    [m]
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()
x = da_d02_TOPO.assign_coords(d02_coords)
lat = [all_line_coords[0].min()-0.5, all_line_coords[0].max()+0.5]
lon = [all_line_coords[1].min()-0.5, all_line_coords[1].max()+0.5]

x = x.sel(
	south_north=slice(lat[0],lat[1]),
	west_east=slice(lon[0],lon[1]))

x_ticks = np.array([108,110,112,114])
x_tick_labels = [u'108\N{DEGREE SIGN}E',u'110\N{DEGREE SIGN}E',
                 u'112\N{DEGREE SIGN}E', u'114\N{DEGREE SIGN}E']
y_ticks = np.array([1,3,5,7])
y_tick_labels = [u'1\N{DEGREE SIGN}N',u'3\N{DEGREE SIGN}N',
                 u'5\N{DEGREE SIGN}N', u'7\N{DEGREE SIGN}N']

# Plot your terrain
ax1 = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree(central_longitude=0))
cf1 = x.plot.contourf(
	cmap='terrain',
	# levels=np.arange(0,4250,250),
	levels = np.append(0,np.logspace(0,3.65,50)),

)

# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords.shape[2])):
# for i in range(1):
	plt.plot(all_line_coords[1,:,i],all_line_coords[0,:,i],'r',linewidth=0.5)
# Plot the center line
plt.plot(all_line_coords[1,:,int(all_line_coords.shape[2]/2)],all_line_coords[0,:,int(all_line_coords.shape[2]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords[1,:,int(all_line_coords.shape[2]/2)],all_line_coords[0,:,int(all_line_coords.shape[2]/2)],s=3)

cbar=cf1.colorbar
cbar.set_label('Terrain Height [m]',fontsize=fs)
cbar.set_ticks([0,10,100,1000,2000,3000])
cbar.minorticks_off()
cbar.set_ticklabels([0,10,100,1000,2000,3000],fontsize=fs)

ax1.set_xlabel('',fontsize=18)
ax1.set_ylabel('',fontsize=18)
ax1.set_title('North Western Borneo',fontsize=fs+4)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels,fontsize=fs)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels,fontsize=fs)
# ax1.legend(fontsize='x-large', markerscale=1.7, edgecolor='0')
# fig.savefig('/home/hragnajarian/PhD/plots/Domain_cross.png', transparent=True)


# All three cross sections

# In[28]:


d01_coords = dict(
    XLAT=(('Time','south_north','west_east'),ds_d01.XLAT[0:2].values),
    XLONG=(('Time','south_north','west_east'),ds_d01.XLONG[0:2].values),
    bottom_top=(('bottom_top'),interp_P_levels),
    Time=('Time',ds_d01.XTIME.values[0:2]),
    south_north=(('south_north'),ds_d01.XLAT[0,:,0].values),
    west_east=(('west_east'),ds_d01.XLONG[0,0,:].values)
    )


# In[29]:


# Western Central Sumatra
start_coord		= [-2.2,103.2]
end_coord 		= [-6.7,98.7]
width			= 1.5
dx 				= 0.025
theta1           = calculate_angle_between_points(start_coord, end_coord)    # Degrees
# da_cross_temp1, all_line_coords1 = cross_section_multi(da_d01_RR, start_coord, end_coord, width, dx)
# da_cross_temp1, all_line_coords1 = cross_section_multi(da_d01_LANDMASK.expand_dims(dim={'Time': 216}).assign_coords(without_keys(d01_coords,'bottom_top')), start_coord, end_coord, width, dx)

da_cross_temp1, all_line_coords1 = cross_section_multi(da_d01_LANDMASK.expand_dims(dim={'Time': 2}).assign_coords(without_keys(d01_coords,'bottom_top')), start_coord, end_coord, width, dx)

# North West Sumatra
start_coord		= [5.2,96.4]
end_coord 		= [1.2,92.4]
width			= 1.5
dx 				= 0.025
theta2           = calculate_angle_between_points(start_coord, end_coord)    # Degrees
# da_cross_temp2, all_line_coords2 = cross_section_multi(da_d01_RR, start_coord, end_coord, width, dx)
# da_cross_temp2, all_line_coords2 = cross_section_multi(da_d01_LANDMASK.expand_dims(dim={'Time': 1}).assign_coords(without_keys(d01_coords,'bottom_top')), start_coord, end_coord, width, dx)

da_cross_temp2, all_line_coords2 = cross_section_multi(da_d01_LANDMASK.expand_dims(dim={'Time': 2}).assign_coords(without_keys(d01_coords,'bottom_top')), start_coord, end_coord, width, dx)

# North West Borneo
start_coord		= [1.2,112.8]
end_coord 		= [5.9,108.1]
width			= 1.5
dx 				= 0.025
theta3           = calculate_angle_between_points(start_coord, end_coord)    # Degrees
# da_cross_temp3, all_line_coords3 = cross_section_multi(da_d01_RR, start_coord, end_coord, width, dx)
da_cross_temp3, all_line_coords3 = cross_section_multi(da_d01_LANDMASK.expand_dims(dim={'Time': 2}).assign_coords(without_keys(d01_coords,'bottom_top')), start_coord, end_coord, width, dx)

da_cross_temp3, all_line_coords3 = cross_section_multi(da_d01_LANDMASK.expand_dims(dim={'Time': 2}).assign_coords(without_keys(d01_coords,'bottom_top')), start_coord, end_coord, width, dx)


# In[ ]:


# Create distance coordinate
distance_d01 = np.linspace(0,dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]),da_cross_temp1.shape[-2])
mid_cross_ind = int(da_cross_temp1.shape[2]/2)	# Find middle cross-section index
if da_cross_temp1[0,0,mid_cross_ind]==1:     # Figure out if the start is over land or ocean
	coast_ind = np.where(da_cross_temp1[0,:,mid_cross_ind]==0)[0][0]	# First 0 (land->ocean)
else:
	coast_ind = np.where(da_cross_temp1[0,:,mid_cross_ind]==1)[0][0]	# First 1 (ocean->land)
distance_d01_1 = distance_d01 - distance_d01[coast_ind]   # Negative values is land

# Create distance coordinate
distance_d01 = np.linspace(0,dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]),da_cross_temp2.shape[-2])
mid_cross_ind = int(da_cross_temp2.shape[2]/2)	# Find middle cross-section index
if da_cross_temp2[0,0,mid_cross_ind]==0:     # Figure out if the start is over land or ocean
	coast_ind = np.where(da_cross_temp2[0,:,mid_cross_ind]==1)[0][0]	# First 1 (ocean->land)
else:
	coast_ind = np.where(da_cross_temp2[0,:,mid_cross_ind]==0)[0][0]	# First 0 (land->ocean)
distance_d01_2 = distance_d01 - distance_d01[coast_ind]   # Negative values is land

# Create distance coordinate
distance_d01 = np.linspace(0,dist(start_coord[0], start_coord[1], end_coord[0], end_coord[1]),da_cross_temp3.shape[-2])
mid_cross_ind = int(da_cross_temp3.shape[2]/2)	# Find middle cross-section index
if da_cross_temp3[0,0,mid_cross_ind]==1:     # Figure out if the start is over land or ocean
	coast_ind = np.where(da_cross_temp3[0,:,mid_cross_ind]==0)[0][0]	# First 0 (land->ocean)
else:
	coast_ind = np.where(da_cross_temp3[0,:,mid_cross_ind]==1)[0][0]	# First 1 (ocean->land)
distance_d01_3 = distance_d01 - distance_d01[coast_ind]   # Negative values is land

a = len(distance_d01_1)-np.where(distance_d01_1==0)[0][0]
b = len(distance_d01_2)-np.where(distance_d01_2==0)[0][0]
c = len(distance_d01_3)-np.where(distance_d01_3==0)[0][0]

print(a, b, c)

print(len(distance_d01_1), len(distance_d01_2), len(distance_d01_3))

print(degrees(theta1), degrees(theta2), degrees(theta3))


# In[ ]:


fig = plt.figure(figsize=(16,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)

fs=18

d02_coords = dict(
    south_north=('south_north',ds_d02.XLAT[0,:,0].values),
    west_east=('west_east',ds_d02.XLONG[0,0,:].values)
    )
# Prepare the Terrain Height    [m]
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()
x = da_d02_TOPO.assign_coords(d02_coords)

x_ticks = np.array([90,100,110,120])
x_tick_labels = [u'90\N{DEGREE SIGN}E', u'100\N{DEGREE SIGN}E',
                 u'110\N{DEGREE SIGN}E',u'120\N{DEGREE SIGN}E']
y_ticks = np.array([-10,-5,0,5,10])
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',u'0\N{DEGREE SIGN}',
                 u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot your terrain
ax1 = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree(central_longitude=0))
cf1 = x.plot.contourf(
	cmap='terrain',
	# levels=np.arange(0,4250,250),
	levels = np.append(0,np.logspace(0,3.65,50)),
)

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'r',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# Plot the off-shore radar (R/V Mirai of JAMSTEC)
RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# Plot the on-shore observatory in Bengkulu city (BMKG observatory)
BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'r',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'r',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

cbar=cf1.colorbar
cbar.set_label('Terrain Height [m]',fontsize=fs)
cbar.set_ticks([0,10,100,1000,2000,3000])
cbar.minorticks_off()
cbar.set_ticklabels([0,10,100,1000,2000,3000],fontsize=fs)

ax1.set_xlabel('',fontsize=18)
ax1.set_ylabel('',fontsize=18)
ax1.set_title('',fontsize=fs+4)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels,fontsize=fs)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels,fontsize=fs)
ax1.legend(fontsize='x-large', markerscale=1.7, edgecolor='0')
# fig.savefig('/home/hragnajarian/PhD/plots/Domain_cross.png', transparent=True)


# Cross Section Domain

# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)
fs = 18
d02_coords = dict(
    south_north=('south_north',ds_d02.XLAT[0,:,0].values),
    west_east=('west_east',ds_d02.XLONG[0,0,:].values)
    )
# Prepare the Terrain Height    [m]
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()
x = da_d02_TOPO.assign_coords(d02_coords)
lat = [da_d02_cross_RR_cntl.Lat.min()-0.5, da_d02_cross_RR_cntl.Lat.max()+0.5]
lon = [da_d02_cross_RR_cntl.Lon.min()-0.5, da_d02_cross_RR_cntl.Lon.max()+0.5]

# Yokoi et al. 2017-2019 domain:
x = x.sel(
	south_north=slice(lat[0],lat[1]),
	west_east=slice(lon[0],lon[1]))

x_ticks = np.arange(int(np.floor(lon[0])),int(np.ceil(lon[1])),2)
x_tick_labels = degree_labels(x_ticks, 'lon')

y_ticks = np.arange(int(np.floor(lat[0])),int(np.ceil(lat[1])),2)
y_tick_labels = degree_labels(y_ticks, 'lat')

# Plot your terrain
ax1 = fig.add_subplot(gs[0,0])
cf1 = x.plot.contourf(
	cmap='terrain',
	# levels=np.arange(0,4250,250),
	levels = np.append(0,np.logspace(0,3.65,50)),
)

# Plot the individual cross-sectioned lines
for i in range(int(da_d02_cross_RR_cntl.shape[2])):
	plt.plot(da_d02_cross_RR_cntl.Lon[:,i],da_d02_cross_RR_cntl.Lat[:,i],'r',linewidth=0.5)
# Plot the center line
plt.plot(da_d02_cross_RR_cntl.Lon[:,int(da_d02_cross_RR_cntl.shape[2]/2)],da_d02_cross_RR_cntl.Lat[:,int(da_d02_cross_RR_cntl.shape[2]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(da_d02_cross_RR_cntl.Lon[:,int(da_d02_cross_RR_cntl.shape[2]/2)],da_d02_cross_RR_cntl.Lat[:,int(da_d02_cross_RR_cntl.shape[2]/2)],s=3)

# Plot Assumed Coastline
plt.scatter(da_d02_cross_RR_cntl.Lon[np.where(da_d02_cross_RR_cntl.Distance==0)],da_d02_cross_RR_cntl.Lat[np.where(da_d02_cross_RR_cntl.Distance==0)],s=2, label='Assumed Coastline')

# Plot Distance away from coast
ind = -15
dist = np.round(da_d02_cross_RR_cntl.Distance[np.where(da_d02_cross_RR_cntl.Distance==0)[0]+(ind)].values,1)
plt.scatter(da_d02_cross_RR_cntl.Lon[np.where(da_d02_cross_RR_cntl.Distance==0)[0]+(ind)],da_d02_cross_RR_cntl.Lat[np.where(da_d02_cross_RR_cntl.Distance==0)[0]+(ind)],s=2, label=f'Coastline at {dist[0]}km')

if cross_title == 'Sumatra Mid Central':	# Plot the off-shore radar (R/V Mirai of JAMSTEC)
	RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='r', label='R/V Mirai')
	# Plot the on-shore observatory in Bengkulu city (BMKG observatory)
	BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='r', label='BMKG Observatory')

cbar=cf1.colorbar
cbar.set_label('Terrain Height [m]',fontsize=fs)
cbar.set_ticks([0,10,100,1000,2000,3000])
cbar.minorticks_off()
cbar.set_ticklabels([0,10,100,1000,2000,3000],fontsize=fs)

ax1.set_xlabel('',fontsize=18)
ax1.set_ylabel('',fontsize=18)
ax1.set_title(cross_title,fontsize=fs+4)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels,fontsize=fs)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels,fontsize=fs)
ax1.set_xlim([lon[0],lon[1]])
ax1.set_ylim([lat[0],lat[1]])
ax1.legend(fontsize='x-large', markerscale=1.7, edgecolor='0')
# fig.savefig('/home/hragnajarian/PhD/plots/Domain_cross.png', transparent=True)


# Rain Rate Evolution

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

title = 'Normalized Diurnal Rain Rate Difference from Control over ' + cross_title

fig.suptitle(title, fontsize=26)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_RR_CRFoff.sel(Lead=slice(0,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(2,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0:2,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_RR_CRFoff.sel(Lead=slice(1,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(14,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:14,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
# Concat starting zeros from x3 to the trailing end of x2
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = xr.concat([x2,x3[:11]],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,4.5,0.25),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2[:38].plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)
x2[37:].plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,3.5,0.5),
	cmap='binary',
	extend='neither'
)

# Plot the cross-sectional data
cf3 = x3[13:].plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)
x3[:14].plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,3.5,0.5),
	cmap='binary',
	extend='neither'
)

# Fontsize
fs = 20

# Plot phase speed lines
if cross_title == 'Borneo Northwest':
	# Borneo phase speed lines
	ax1.plot([-205,-43],[13.5,28.5] , color='r')
	ax1.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,65],[13.5,28.5] , color='r')
	ax1.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,530],[13.5,30.5] , color='r')
	ax1.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax1.plot([-205,-43],[37.5,52.5] , color='r')
	ax1.text(-145, 44, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,65],[37.5,52.5] , color='r')
	ax1.text(25, 43.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,530],[37.5,54.5] , color='r')
	ax1.text(145, 40, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax2.plot([-205,-43],[13.5,28.5] , color='r')
	ax2.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-205,65],[13.5,28.5] , color='r')
	ax2.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-205,530],[13.5,30.5] , color='r')
	ax2.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-205,-43],[13.5,28.5] , color='r')
	ax3.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,65],[13.5,28.5] , color='r')
	ax3.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,530],[13.5,30.5] , color='r')
	ax3.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-205,-43],[37.5,52.5] , color='r')
	ax3.text(-145, 44, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,65],[37.5,52.5] , color='r')
	ax3.text(25, 43.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,530],[37.5,54.5] , color='r')
	ax3.text(145, 40, '12 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.set_yticks(x1.hour[0::3].values+0.5-1)
	ax2.set_yticks(x1.hour[0::3].values+0.5)
	ax3.set_yticks(x1.hour[0::3].values+0.5)
else:
	ax1.plot([-45,117],[9,24] , color='r')
	ax1.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,225],[9,24] , color='r')
	ax1.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,603],[9,24] , color='r')
	ax1.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax1.plot([-45,117],[33,48] , color='r')
	ax1.text(60, 43, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,225],[33,48] , color='r')
	ax1.text(180, 39, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,603],[33,48] , color='r')
	ax1.text(300, 37, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax2.plot([-45,117],[9,24] , color='r')
	ax2.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-45,225],[9,24] , color='r')
	ax2.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-45,603],[9,24] , color='r')
	ax2.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-45,117],[9,24] , color='r')
	ax3.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,225],[9,24] , color='r')
	ax3.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,603],[9,24] , color='r')
	ax3.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-45,117],[33,48] , color='r')
	ax3.text(60, 43, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,225],[33,48] , color='r')
	ax3.text(180, 39, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,603],[33,48] , color='r')
	ax3.text(300, 37, '12 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.set_yticks(x1.hour[0::3].values+0.5)
	ax2.set_yticks(x1.hour[0::3].values+0.5)
	ax3.set_yticks(x1.hour[0::3].values+0.5)

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--', linewidth=2)
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax3.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('')
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_ylabel('Lead Time', fontsize=fs)
ax1.set_xticks(np.arange(-200,600,100))
ax2.set_xticks(np.arange(-200,600,100))
ax3.set_xticks(np.arange(-200,600,100))
ax1.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax3.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax1.set_yticklabels(np.concatenate((np.arange(7,25,3),np.arange(1,25,3),np.arange(1,7,3))),fontsize=fs-6)
ax2.set_yticklabels(np.arange(0,48,3),fontsize=fs-6)
ax3.set_yticklabels(np.arange(-12,36,3),fontsize=fs-6)
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
ax3.set_title('NCRF Sunset', loc='center', fontsize=fs)
ax3.set_title('c)', loc='left', fontsize=fs)

ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,5,1))
cbar.set_ticklabels(np.arange(0,5,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm/day$]', fontsize=fs-6)

cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_ticklabels(np.arange(-3,4,1), fontsize=fs-6)
cbar.set_label("[$(RR-RR_{cntl})/\sigma_{cntl}$]", fontsize=fs-6)

# Save figure
# fig.savefig('/home/hragnajarian/PhD/plots/NormDiff_RR.png', transparent=True)


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

title = 'Normalized Diurnal Rain Rate over ' + cross_title

fig.suptitle(title, fontsize=26)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_RR_CRFoff.sel(Lead=slice(0,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(2,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0:2,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_RR_CRFoff.sel(Lead=slice(1,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# # Switch to local time
# x3 = x3.assign_coords({'Time':x3.Time + np.timedelta64(7,'h')})
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(14,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:14,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3) / x1.std('hour')

# Normalize the data
x1 = (x1) / x1.std('hour')
# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,6.5,0.5),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,6.5,0.5),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,6.5,0.5),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot phase speed lines
if cross_title == 'Borneo Northwest':
	# Borneo phase speed lines
	ax1.plot([-205,-43],[13.5,28.5] , color='r')
	ax1.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,65],[13.5,28.5] , color='r')
	ax1.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,530],[13.5,30.5] , color='r')
	ax1.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax1.plot([-205,-43],[37.5,52.5] , color='r')
	ax1.text(-145, 44, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,65],[37.5,52.5] , color='r')
	ax1.text(25, 43.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,530],[37.5,54.5] , color='r')
	ax1.text(145, 40, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax2.plot([-205,-43],[13.5,28.5] , color='r')
	ax2.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-205,65],[13.5,28.5] , color='r')
	ax2.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-205,530],[13.5,30.5] , color='r')
	ax2.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-205,-43],[13.5,28.5] , color='r')
	ax3.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,65],[13.5,28.5] , color='r')
	ax3.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,530],[13.5,30.5] , color='r')
	ax3.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-205,-43],[37.5,52.5] , color='r')
	ax3.text(-145, 44, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,65],[37.5,52.5] , color='r')
	ax3.text(25, 43.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,530],[37.5,54.5] , color='r')
	ax3.text(145, 40, '12 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.set_yticks(x1.hour[0::3].values+0.5-1)
	ax2.set_yticks(x1.hour[0::3].values+0.5)
	ax3.set_yticks(x1.hour[0::3].values+0.5)
else:
	ax1.plot([-45,117],[9,24] , color='r')
	ax1.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,225],[9,24] , color='r')
	ax1.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,603],[9,24] , color='r')
	ax1.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax1.plot([-45,117],[33,48] , color='r')
	ax1.text(60, 43, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,225],[33,48] , color='r')
	ax1.text(180, 39, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,603],[33,48] , color='r')
	ax1.text(300, 37, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax2.plot([-45,117],[9,24] , color='r')
	ax2.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-45,225],[9,24] , color='r')
	ax2.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-45,603],[9,24] , color='r')
	ax2.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-45,117],[9,24] , color='r')
	ax3.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,225],[9,24] , color='r')
	ax3.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,603],[9,24] , color='r')
	ax3.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-45,117],[33,48] , color='r')
	ax3.text(60, 43, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,225],[33,48] , color='r')
	ax3.text(180, 39, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,603],[33,48] , color='r')
	ax3.text(300, 37, '12 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.set_yticks(x1.hour[0::3].values+0.5)
	ax2.set_yticks(x1.hour[0::3].values+0.5)
	ax3.set_yticks(x1.hour[0::3].values+0.5)


# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--', linewidth=2)
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax3.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('')
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_ylabel('Lead Time', fontsize=fs)
ax1.set_xticks(np.arange(-200,600,100))
ax2.set_xticks(np.arange(-200,600,100))
ax3.set_xticks(np.arange(-200,600,100))
ax1.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax3.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax1.set_yticklabels(np.concatenate((np.arange(7,25,3),np.arange(1,25,3),np.arange(1,7,3))),fontsize=fs-6)
ax2.set_yticklabels(np.arange(0,48,3),fontsize=fs-6)
ax3.set_yticklabels(np.arange(-12,36,3),fontsize=fs-6)
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
ax3.set_title('NCRF Sunset', loc='center', fontsize=fs)
ax3.set_title('c)', loc='left', fontsize=fs)

ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Normalized Rain Rate [RR/$\sigma_{cntl}$]')


# Domain stuff

# In[77]:


time_type = 'LT'

#### Load Data

if time_type == 'UTC':

	### UTC

	## Control

	# RR
	RR_cntl = da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
		# Land
	RR_cntl_land = RR_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	RR_cntl_ocean = RR_cntl.where(da_d02_LANDMASK==0)
		# Net
	RR_cntl_net = RR_cntl

	# LowCLDFRA
	LowCLDFRA_cntl = da_d02_LowCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
		# Land
	LowCLDFRA_cntl_land = LowCLDFRA_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	LowCLDFRA_cntl_ocean = LowCLDFRA_cntl.where(da_d02_LANDMASK==0)
		# Net
	LowCLDFRA_cntl_net = LowCLDFRA_cntl

	# MidCLDFRA
	MidCLDFRA_cntl = da_d02_MidCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
		# Land
	MidCLDFRA_cntl_land = MidCLDFRA_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	MidCLDFRA_cntl_ocean = MidCLDFRA_cntl.where(da_d02_LANDMASK==0)
		# Net
	MidCLDFRA_cntl_net = MidCLDFRA_cntl

	# HighCLDFRA
	HighCLDFRA_cntl = da_d02_HighCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
		# Land
	HighCLDFRA_cntl_land = HighCLDFRA_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	HighCLDFRA_cntl_ocean = HighCLDFRA_cntl.where(da_d02_LANDMASK==0)
		# Net
	HighCLDFRA_cntl_net = HighCLDFRA_cntl

	## SWCRF
	SWSFC_cntl = da_d02_SWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
	# Remove the diurnal mean
	for i in np.arange(0,24):
		SWSFC_cntl[SWSFC_cntl.Time.dt.hour.isin(i)] = SWSFC_cntl[SWSFC_cntl.Time.dt.hour.isin(i)] - SWSFC_cntl[SWSFC_cntl.Time.dt.hour.isin(i)].mean()

		# Land
	SWSFC_cntl_land = SWSFC_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	SWSFC_cntl_ocean = SWSFC_cntl.where(da_d02_LANDMASK==0)
		# Net
	SWSFC_cntl_net = SWSFC_cntl

	## NCRF Sunrise

	# RR
	RR_sunrise = da_d02_sunrise_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
	inds = np.argwhere(RR_sunrise.to_numpy()<0)	# Find the inds where RR is negative b/c of stitching many simulations together
	RR_sunrise[inds[:,0]] = (RR_sunrise[inds[:,0]-1,...]+RR_sunrise[inds[:,0]+1,...])/2
		# Land
	RR_sunrise_land = RR_sunrise.where(da_d02_LANDMASK==1)
		# Ocean
	RR_sunrise_ocean = RR_sunrise.where(da_d02_LANDMASK==0)
		# Net
	RR_sunrise_net = RR_sunrise

		# LowCLDFRA
	LowCLDFRA_sunrise = da_d02_sunrise_LowCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	LowCLDFRA_sunrise_land = LowCLDFRA_sunrise.where(da_d02_LANDMASK==1)
		# Ocean
	LowCLDFRA_sunrise_ocean = LowCLDFRA_sunrise.where(da_d02_LANDMASK==0)
		# Net
	LowCLDFRA_sunrise_net = LowCLDFRA_sunrise

	# MidCLDFRA
	MidCLDFRA_sunrise = da_d02_sunrise_MidCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	MidCLDFRA_sunrise_land = MidCLDFRA_sunrise.where(da_d02_LANDMASK==1)
		# Ocean
	MidCLDFRA_sunrise_ocean = MidCLDFRA_sunrise.where(da_d02_LANDMASK==0)
		# Net
	MidCLDFRA_sunrise_net = MidCLDFRA_sunrise

	# HighCLDFRA
	HighCLDFRA_sunrise = da_d02_sunrise_HighCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	HighCLDFRA_sunrise_land = HighCLDFRA_sunrise.where(da_d02_LANDMASK==1)
		# Ocean
	HighCLDFRA_sunrise_ocean = HighCLDFRA_sunrise.where(da_d02_LANDMASK==0)
		# Net
	HighCLDFRA_sunrise_net = HighCLDFRA_sunrise

	## SWCRF
	SWSFC_sunrise = da_d02_sunrise_SWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
	# Remove the diurnal mean
	for i in np.arange(0,24):
		SWSFC_sunrise[SWSFC_sunrise.Time.dt.hour.isin(i)] = SWSFC_sunrise[SWSFC_sunrise.Time.dt.hour.isin(i)] - SWSFC_sunrise[SWSFC_sunrise.Time.dt.hour.isin(i)].mean()
		# Land
	SWSFC_sunrise_land = SWSFC_sunrise.where(da_d02_LANDMASK==1)
		# Ocean
	SWSFC_sunrise_ocean = SWSFC_sunrise.where(da_d02_LANDMASK==0)
		# Net
	SWSFC_sunrise_net = SWSFC_sunrise

else:

	### Local Solar Time

	## Control

	# RR
	RR_cntl = assign_LT_coord(da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	RR_cntl_land = RR_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	RR_cntl_ocean = RR_cntl.where(da_d02_LANDMASK==0)
		# Net
	RR_cntl_net = RR_cntl

	# LowCLDFRA
	LowCLDFRA_cntl = assign_LT_coord(da_d02_LowCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	LowCLDFRA_cntl_land = LowCLDFRA_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	LowCLDFRA_cntl_ocean = LowCLDFRA_cntl.where(da_d02_LANDMASK==0)
		# Net
	LowCLDFRA_cntl_net = LowCLDFRA_cntl

	# MidCLDFRA
	MidCLDFRA_cntl = assign_LT_coord(da_d02_MidCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	MidCLDFRA_cntl_land = MidCLDFRA_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	MidCLDFRA_cntl_ocean = MidCLDFRA_cntl.where(da_d02_LANDMASK==0)
		# Net
	MidCLDFRA_cntl_net = MidCLDFRA_cntl

	# HighCLDFRA
	HighCLDFRA_cntl = assign_LT_coord(da_d02_HighCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	HighCLDFRA_cntl_land = HighCLDFRA_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	HighCLDFRA_cntl_ocean = HighCLDFRA_cntl.where(da_d02_LANDMASK==0)
		# Net
	HighCLDFRA_cntl_net = HighCLDFRA_cntl

	# # SW SFC
	# SWSFC_cntl = assign_LT_coord(da_d02_SWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
	# ## Remove the diurnal mean
	# npSWSFC_cntl = np.empty(SWSFC_cntl.shape)	# Create numpy array to take place xr data since 2-D boolean indexing is not supported in xarray
	# for i in np.arange(0,24):
	# 	bool_array = SWSFC_cntl.LocalTime.dt.hour.isin(i)
	# 	npSWSFC_cntl[bool_array] = SWSFC_cntl.values[bool_array] - SWSFC_cntl.values[bool_array].mean()
	# 	print(i)
	# SWSFC_cntl.data = npSWSFC_cntl
	# 	# Land
	# SWSFC_cntl_land = SWSFC_cntl.where(da_d02_LANDMASK==1)
	# 	# Ocean
	# SWSFC_cntl_ocean = SWSFC_cntl.where(da_d02_LANDMASK==0)
	# 	# Net
	# SWSFC_cntl_net = SWSFC_cntl

	# # SW SFC CRF
	# SWSFC_CRF_cntl = assign_LT_coord(da_d02_SWSFC_CRF.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
	# ## Remove the diurnal mean
	# npSWSFC_CRF_cntl = np.empty(SWSFC_CRF_cntl.shape)	# Create numpy array to take place xr data since 2-D boolean indexing is not supported in xarray
	# for i in np.arange(0,24):
	# 	bool_array = SWSFC_CRF_cntl.LocalTime.dt.hour.isin(i)
	# 	npSWSFC_CRF_cntl[bool_array] = SWSFC_CRF_cntl.values[bool_array] - SWSFC_CRF_cntl.values[bool_array].mean()
	# 	print(i)
	# SWSFC_CRF_cntl.data = npSWSFC_CRF_cntl
	# 	# Land
	# SWSFC_CRF_cntl_land = SWSFC_CRF_cntl.where(da_d02_LANDMASK==1)
	# 	# Ocean
	# SWSFC_CRF_cntl_ocean = SWSFC_CRF_cntl.where(da_d02_LANDMASK==0)
	# 	# Net
	# SWSFC_CRF_cntl_net = SWSFC_CRF_cntl

	# SW Surface CRF
	SWSFC_CRF_cntl = assign_LT_coord(da_d02_SWSFC_CRF.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	SWSFC_CRF_cntl_land = SWSFC_CRF_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	SWSFC_CRF_cntl_ocean = SWSFC_CRF_cntl.where(da_d02_LANDMASK==0)
		# Net
	SWSFC_CRF_cntl_net = SWSFC_CRF_cntl

	# SW Surface Clear
	SWSFC_Clear_cntl = assign_LT_coord(da_d02_SWSFC_Clear.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	SWSFC_Clear_cntl_land = SWSFC_Clear_cntl.where(da_d02_LANDMASK==1)
		# Ocean
	SWSFC_Clear_cntl_ocean = SWSFC_Clear_cntl.where(da_d02_LANDMASK==0)
		# Net
	SWSFC_Clear_cntl_net = SWSFC_Clear_cntl



	## NCRF Sunrise

	# RR
	RR_sunrise = assign_LT_coord(da_d02_sunrise_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
	# Adjust the values that are negative
	inds = np.argwhere(RR_sunrise.mean(['south_north','west_east']).to_numpy()<0)[:,0]	# Find the inds where RR is negative b/c of stitching many simulations together
	RR_sunrise[inds] = (RR_sunrise[inds-1].values+RR_sunrise[inds+1].values)/2
		# Land
	RR_sunrise_land = RR_sunrise.where(da_d02_LANDMASK==1).mean('south_north',skipna=True).groupby('LocalTime').mean()
		# Ocean
	RR_sunrise_ocean = RR_sunrise.where(da_d02_LANDMASK==0).mean('south_north',skipna=True).groupby('LocalTime').mean()
		# Net
	RR_sunrise_net = RR_sunrise.mean('south_north',skipna=True).groupby('LocalTime').mean()

	# LowCLDFRA
	LowCLDFRA_sunrise = assign_LT_coord(da_d02_sunrise_LowCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	LowCLDFRA_sunrise_land = LowCLDFRA_sunrise.where(da_d02_LANDMASK==1).mean('south_north',skipna=True).groupby('LocalTime').mean()
		# Ocean
	LowCLDFRA_sunrise_ocean = LowCLDFRA_sunrise.where(da_d02_LANDMASK==0).mean('south_north',skipna=True).groupby('LocalTime').mean()
		# Net
	LowCLDFRA_sunrise_net = LowCLDFRA_sunrise.mean('south_north',skipna=True).groupby('LocalTime').mean()

	# MidCLDFRA
	MidCLDFRA_sunrise = assign_LT_coord(da_d02_sunrise_MidCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	MidCLDFRA_sunrise_land = MidCLDFRA_sunrise.where(da_d02_LANDMASK==1).mean('south_north',skipna=True).groupby('LocalTime').mean()
		# Ocean
	MidCLDFRA_sunrise_ocean = MidCLDFRA_sunrise.where(da_d02_LANDMASK==0).mean('south_north',skipna=True).groupby('LocalTime').mean()
		# Net
	MidCLDFRA_sunrise_net = MidCLDFRA_sunrise.mean('south_north',skipna=True).groupby('LocalTime').mean()

	# HighCLDFRA
	HighCLDFRA_sunrise = assign_LT_coord(da_d02_sunrise_HighCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	HighCLDFRA_sunrise_land = HighCLDFRA_sunrise.where(da_d02_LANDMASK==1).mean('south_north',skipna=True).groupby('LocalTime').mean()
		# Ocean
	HighCLDFRA_sunrise_ocean = HighCLDFRA_sunrise.where(da_d02_LANDMASK==0).mean('south_north',skipna=True).groupby('LocalTime').mean()
		# Net
	HighCLDFRA_sunrise_net = HighCLDFRA_sunrise.mean('south_north',skipna=True).groupby('LocalTime').mean()

	# # SWSFC
	# SWSFC_sunrise = assign_LT_coord(da_d02_sunrise_SWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
	# SWSFC_sunrise = SWSFC_sunrise.mean('south_north')	# Setup for removing diurnal mean
	# # Remove the diurnal mean
	# npSWSFC_sunrise = np.empty(SWSFC_sunrise.shape)	# Create numpy array to take place xr data since 2-D boolean indexing is not supported in xarray
	# for i in np.arange(0,24):
	# 	npSWSFC_sunrise[SWSFC_sunrise.LocalTime.dt.hour.isin(i)] = SWSFC_sunrise.values[SWSFC_sunrise.LocalTime.dt.hour.isin(i)] - SWSFC_sunrise.values[SWSFC_sunrise.LocalTime.dt.hour.isin(i)].mean()
	# SWSFC_sunrise.data = npSWSFC_sunrise
	# 	# Land
	# SWSFC_sunrise_land = SWSFC_sunrise.where(da_d02_LANDMASK==1).groupby('LocalTime').mean()
	# 	# Ocean
	# SWSFC_sunrise_ocean = SWSFC_sunrise.where(da_d02_LANDMASK==0).groupby('LocalTime').mean()
	# 	# Net
	# SWSFC_sunrise_net = SWSFC_sunrise.groupby('LocalTime').mean()


# In[166]:


# LST
time_start = 9
time_end = 15
lag_plot = 0				# The lag relationship plotted on the first row
cloud_type = 'Low'

# Cloud Fraction [0->1] 																Between specific times & where it's land/ocean or both
if cloud_type=='Low':
	x1 = LowCLDFRA_cntl_land.values[(LowCLDFRA_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(LowCLDFRA_cntl_land.values))].flatten()
elif cloud_type=='Mid':
	x1 = MidCLDFRA_cntl_land.values[(MidCLDFRA_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(MidCLDFRA_cntl_land.values))].flatten()
else:
	x1 = HighCLDFRA_cntl_land.values[(HighCLDFRA_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(HighCLDFRA_cntl_land.values))].flatten()

# Rain rate [mm/hr]
if cloud_type =='Low':
	# Rain Rate [mm/day] LOW CLOUDS
	x2 = RR_cntl_land.values[(RR_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(LowCLDFRA_cntl_land.values))].flatten()
else:
	# Rain Rate [mm/day]
	x2 = RR_cntl_land.values[(RR_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(RR_cntl_land.values))].flatten()

# SW Surface Flux  [W/m^2] 
if cloud_type =='Low':
	# SW Surface CRF Flux  [W/m^2] LOW CLOUDS
	x3 = SWSFC_CRF_cntl_land.values[(SWSFC_CRF_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(LowCLDFRA_cntl_land.values))].flatten()
		# Normalize by theoretical max flux
	x3 = x3/SWSFC_Clear_cntl_land.values[(SWSFC_Clear_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(LowCLDFRA_cntl_land.values))].flatten()
else:
	# SW Surface CRF Flux  [W/m^2] 
	x3 = SWSFC_CRF_cntl_land.values[(SWSFC_CRF_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(SWSFC_CRF_cntl_land.values))].flatten()
		# Normalize by theoretical max flux
	x3 = x3/SWSFC_Clear_cntl_land.values[(SWSFC_Clear_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(SWSFC_Clear_cntl_land.values))].flatten()


# In[ ]:


plt.hist(x1, density=True)


# In[ ]:


plt.hist(x2, bins=np.arange(-.5,15.5,1), density=True)


# In[ ]:


x2.max()


# In[ ]:


x2.min()


# In[ ]:


# plt.hist(x3, bins=np.arange(-850,450,100), density=True)
plt.hist(x3, density=True)
# plt.hist(SWSFC_Clear_cntl_land.values[(SWSFC_Clear_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(SWSFC_Clear_cntl_land.values))].flatten(), density=True)


# In[ ]:


x3.max()


# In[ ]:


x3[~np.isnan(x3)].max() 


# In[167]:


nitter=np.random.randint(low=0,high=len(x1),size=100000)


# In[168]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[0.87,0.03], width_ratios=[0.85,0.15], hspace=0.3, wspace=0.05)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

fs = 18

# LST
# time_start = 7
# time_end = 12

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# # Cloud Fraction [0->1] 
# x1 = HighCLDFRA_cntl_land.values[(HighCLDFRA_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(HighCLDFRA_cntl_land.values))].flatten()

# # Rain Rate [mm/day]
# x2 = RR_cntl_land.values[(RR_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(RR_cntl_land.values))].flatten()

# # SW Surface Flux  [W/m^2] 
# x3 = SWSFC_cntl_land.values[(SWSFC_cntl_land.LocalTime.dt.hour.isin(np.arange(time_start,time_end)))&(~np.isnan(SWSFC_cntl_land.values))].flatten()


x1_range = [-.05,1.05]
x2_range = [0,7]
# x3_range = [-900,300]
# x3_range = [-1000,100]
x3_range = [-1.05,.05]

# Only plot a select number of scatters
# nitter=1000000
# nitter=np.random.randint(low=0,high=len(x1),size=100000)

# s1 = ax1.scatter(x1[:nitter],x3[:nitter],c=x2[:nitter],s=15,cmap='gray_r',edgecolors='0.5',alpha=0.5,linewidths=.15, vmin=x2_range[0], vmax=x2_range[1])
s1 = ax1.scatter(x1[nitter],x3[nitter],c=x2[nitter],s=15,cmap='gray_r',edgecolors='0.5',alpha=0.25,linewidths=.01, vmin=x2_range[0], vmax=x2_range[1])

# Linear Regression
# min_lag = 0
# max_lag = 0
# lag_plot = 0				# The lag relationship plotted on the first row
# da2 = linreg(x1, x3, min_lag, max_lag)
# l2 = ax1.plot([-200,200],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-200, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*200],'r',linestyle='--')
# slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
# yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
# r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
# ax1.text(.6, -.2, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

# # Polynomial Fit
# poly_fit = np.polyfit(x1, x3, deg=3)
# z = np.poly1d(poly_fit)
# l2 = ax1.plot(np.arange(0,1.01,.01), z(np.arange(0,1.01,.01)))

# # Exponential Fit
# # Prepare dataset
# bin_edges = np.arange(0, 1.05, 0.05)  # bins from 0 to 100, with a bin width of 10
# # Digitize x-values: determine which bin each x-value falls into
# bin_indices = np.digitize(x1, bin_edges)
# # Initialize arrays to hold binned results
# bin_means = np.zeros(len(bin_edges) - 1)
# bin_counts = np.zeros(len(bin_edges) - 1)
# # Calculate mean y-values in each bin
# for i in range(1, len(bin_edges)):
#     bin_mask = bin_indices == i
#     bin_means[i-1] = x3[bin_mask].mean() if np.any(bin_mask) else np.nan
#     bin_counts[i-1] = np.sum(bin_mask)
# # Define the exponential function
# def exponential_func(x, a, b):
#     return a * np.exp(b * x)
# # Fit the exponential curve
# params, covariance = curve_fit(exponential_func, bin_edges[:-1], bin_means)
# # Extract the fitting parameters
# a_fit, b_fit = params
# # Generate y values based on the fitted parameters
# y_fit = exponential_func(bin_edges[:-1], *params)
# l2 = ax1.plot(bin_edges[:-1], y_fit, color='red', label=f'Fitted Curve: y = {a_fit:.2f} * exp({b_fit:.2f} * x)')


ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
# ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Normalized SW Surface CRF & {cloud_type} Cloud Fraction\n over land n={len(x1):.0f} between {time_start}-{time_end} LST'
ax1.set_title(title, fontsize=12)
ax1.set_xlabel('Cloud Fraction',fontsize=10)
ax1.set_ylabel('(SW SFC CRF) / (SW SFC Clear)',fontsize=10)
ax1.set_xlim([x1_range[0],x1_range[1]])
ax1.set_ylim([x3_range[0],x3_range[1]])
ax1.grid(linestyle='--', axis='both', linewidth=1)

cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(s1,cax=cax1, orientation='horizontal', pad=0, extend='max')
cbar.set_ticks(np.arange(x2_range[0],x2_range[1]+1,1))
cbar.set_ticklabels(np.arange(x2_range[0],x2_range[1]+1,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm/hour$]')


# Create rain rate plot to the right
dy = .025			# Bin size
rain_rate_bins = np.arange(-1,0+dy,dy)
# rain_rate_bins = np.arange(0,1+dy,dy)
rain_rate = np.zeros(len(rain_rate_bins))
for i in range(len(rain_rate_bins)-1):
	RR = np.nanmean(np.where((x3>rain_rate_bins[i])&(x3<=rain_rate_bins[i+1]),x2, np.nan))
	#RR = x2.where((x3>rain_rate_bins[i])&(x3<=rain_rate_bins[i+1]),drop=True).mean().values
	if ~np.isnan(RR):
		rain_rate[i]=RR

l1 = ax2.plot(rain_rate,rain_rate_bins, color='k')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.set_xlabel('[$mm/hour$]',fontsize=10)
ax2.set_ylabel('',fontsize=10)
ax2.set_yticklabels([])
# ax2.xaxis.set_tick_params(left = False)
ax2.set_xlim([0,3])
ax2.set_ylim([x3_range[0],x3_range[1]])
ax2.grid(linestyle='--', axis='both', linewidth=1)


# Cross-section Rain Rate

# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

# Coastal/Custom Average
land_dist_thresh=-100
ocean_dist_thresh=100

# Domain Average
# land_dist_thresh=da_d02_cross_RR_cntl.Distance.min().values
# ocean_dist_thresh=da_d02_cross_RR_cntl.Distance.max().values

# Load Data
	# Control
x1 = da_d02_cross_RR_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))	# allow for 12-hr spin up
x1 = x1.groupby('Time.hour').mean()
		# Average
x1avg = x1.mean(['Distance','Spread'])
x1avg = xr.concat([x1avg,x1avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land only
x1land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1land = xr.concat([x1land,x1land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x1ocean = x1.where((x1.Distance>0&(x1.Distance<ocean_dist_thresh))).mean(['Distance','Spread'])
x1ocean = xr.concat([x1ocean,x1ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# 00 UTC icloud off
x2 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
		# Average
x2avg = x2.mean('Distance')
x2avg = xr.concat([x2avg,x2avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land
x2land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean('Distance')
x2land = xr.concat([x2land,x2land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
x2ocean = x2.where((x2.Distance>0&(x2.Distance<ocean_dist_thresh))).mean('Distance')
x2ocean = xr.concat([x2ocean,x2ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# 12 UTC icloud off
x3 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(1,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x3 = x3.mean('Spread')
da = x3.drop_vars(['Lead','Time','Times'])
x3 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x3 = x3.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
		# Average
x3avg = x3.mean('Distance')
x3avg = xr.concat([x3avg,x3avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land
x3land = x3.where((x3.Distance<0)&(x3.Distance>land_dist_thresh)).mean('Distance')
x3land = xr.concat([x3land,x3land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
x3ocean = x3.where((x3.Distance>0&(x3.Distance<ocean_dist_thresh))).mean('Distance')
x3ocean = xr.concat([x3ocean,x3ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Control
avg1 = x1avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '-',
	label='$Control_{avg}$'
)
land1 = x1land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='$Control_{land}$'
)
ocean1 = x1ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='$Control_{ocean}$'
)
# 00UTC CRF off
avg2 = x2avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{avg}$'
)
land2 = x2land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{land}$'
)
ocean2 = x2ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{ocean}$'
)
# 12UTC CRF off
avg3 = x3avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{avg}$'
)
land3 = x3land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{land}$'
)
ocean3 = x3ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{ocean}$'
)

ax1.set_xlim([0,24])
ax1.set_ylim([0,5])
ax1.set_xticks(np.arange(0,27,3))
title = f'Domain Averaged Diurnal Rain Rate: {cross_title:s}\n Land: 0-{abs(int(land_dist_thresh))}km Ocean: 0-{int(ocean_dist_thresh)}km'
ax1.set_title(title, fontsize=fs-4)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Rain Rate [$mm d^{-1}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncols=3)


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

# Coastal/Custom Average
land_dist_thresh = -242
land_coast_dist = -100
ocean_dist_thresh = 100
ocean_coast_dist = 0

# Domain Average
# land_dist_thresh=da_d02_cross_RR_cntl.Distance.min().values
# ocean_dist_thresh=da_d02_cross_RR_cntl.Distance.max().values

# Load Data
	# Control
x1 = da_d02_cross_RR_cntl.sel(Time=slice('2015-11-23T02','2015-12-02T00'))	# allow for 12-hr spin up & start when NCRF Sunrise starts
x1 = x1.groupby('Time.hour').mean()
		# Average
x1avg = x1.mean(['Distance','Spread'])
x1avg = xr.concat([x1avg,x1avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land only
x1land = x1.where((x1.Distance<land_coast_dist)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1land = xr.concat([x1land,x1land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x1ocean = x1.where((x1.Distance>ocean_coast_dist)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1ocean = xr.concat([x1ocean,x1ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# 00 UTC icloud off
x2 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
		# Average
x2avg = x2.mean('Distance')
x2avg = xr.concat([x2avg,x2avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land
x2land = x2.where((x2.Distance<land_coast_dist)&(x2.Distance>land_dist_thresh)).mean('Distance')
x2land = xr.concat([x2land,x2land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
x2ocean = x2.where((x2.Distance>ocean_coast_dist)&(x2.Distance<ocean_dist_thresh)).mean('Distance')
x2ocean = xr.concat([x2ocean,x2ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# 12 UTC icloud off
x3 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(1,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x3 = x3.mean('Spread')
da = x3.drop_vars(['Lead','Time','Times'])
x3 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x3 = x3.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
		# Average
x3avg = x3.mean('Distance')
x3avg = xr.concat([x3avg,x3avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land
x3land = x3.where((x3.Distance<land_coast_dist)&(x3.Distance>land_dist_thresh)).mean('Distance')
x3land = xr.concat([x3land,x3land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
x3ocean = x3.where((x3.Distance>ocean_coast_dist)&(x3.Distance<ocean_dist_thresh)).mean('Distance')
x3ocean = xr.concat([x3ocean,x3ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Control
avg1 = x1avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '-',
	label='$Control_{avg}$'
)
land1 = x1land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='$Control_{land}$'
)
ocean1 = x1ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='$Control_{ocean}$'
)
# 00UTC CRF off
avg2 = x2avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{avg}$'
)
land2 = x2land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{land}$'
)
ocean2 = x2ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{ocean}$'
)
# 12UTC CRF off
avg3 = x3avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{avg}$'
)
land3 = x3land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{land}$'
)
ocean3 = x3ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{ocean}$'
)

ax1.set_xlim([0,24])
ax1.set_ylim([0,5])
ax1.set_xticks(np.arange(0,27,3))
title = f'Domain Averaged Diurnal Rain Rate: {cross_title:s}\n Land: {int(land_coast_dist)}->{int(land_dist_thresh)}km Ocean: {int(ocean_coast_dist)}->{int(ocean_dist_thresh)}km'
ax1.set_title(title, fontsize=fs-4)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Rain Rate [$mm d^{-1}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncols=3)


# Domain Rain Rate

# In[ ]:


# RR
RR_cntl = assign_LT_coord(da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=2)
	# Land
RR_cntl_land = xarray_reduce(RR_cntl.where(da_d02_LANDMASK==1).mean('south_north'), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
RR_cntl_land_2d = xr.concat([RR_cntl_land,RR_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))


# In[ ]:


# RR
RR_cntl = assign_LT_coord(da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
	# Land
RR_cntl_land = xarray_reduce(RR_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
RR_cntl_land_3d = xr.concat([RR_cntl_land,RR_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))


# In[ ]:


RR_cntl_land_3d-RR_cntl_land_2d


# In[ ]:


time_type = 'LST'

#### Load Data

if time_type == 'UTC':

	### UTC

	## Control

	# RR
	RR_cntl = da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	RR_cntl_land = RR_cntl.where(da_d02_LANDMASK==1).mean(['south_north','west_east']).groupby('Time.hour').mean()
	RR_cntl_land = xr.concat([RR_cntl_land,RR_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))	# Concat to make the end the same as the start
		# Ocean
	RR_cntl_ocean = RR_cntl.where(da_d02_LANDMASK==0).mean(['south_north','west_east']).groupby('Time.hour').mean()
	RR_cntl_ocean = xr.concat([RR_cntl_ocean,RR_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	RR_cntl_net = RR_cntl.mean(['south_north','west_east']).groupby('Time.hour').mean()
	RR_cntl_net = xr.concat([RR_cntl_net,RR_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	## NCRF Sunrise

	# RR
	RR_sunrise = da_d02_sunrise_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()	# copy ensures no data manipulation occurs on the OG dataarray
	inds = np.argwhere(RR_sunrise.mean(['south_north','west_east']).to_numpy()<0)[:,0]			# Find the inds where RR is negative (errors) b/c of stitching many simulations together
	RR_sunrise[inds] = (RR_sunrise[inds-1].values+RR_sunrise[inds+1].values)/2	# Average the RR between the times where RR is negative
		# Land
	RR_sunrise_land = RR_sunrise.where(da_d02_LANDMASK==1).mean(['south_north','west_east']).groupby('Time.hour').mean()
	RR_sunrise_land = xr.concat([RR_sunrise_land,RR_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
	# RR_sunrise_land[1] = np.mean([RR_sunrise_land[0],RR_sunrise_land[2]])
		# Ocean
	RR_sunrise_ocean = RR_sunrise.where(da_d02_LANDMASK==0).mean(['south_north','west_east']).groupby('Time.hour').mean()
	RR_sunrise_ocean = xr.concat([RR_sunrise_ocean,RR_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
	# RR_sunrise_ocean[1] = np.mean([RR_sunrise_ocean[0],RR_sunrise_ocean[2]])
		# Net
	RR_sunrise_net = RR_sunrise.mean(['south_north','west_east']).groupby('Time.hour').mean()
	RR_sunrise_net = xr.concat([RR_sunrise_net,RR_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
	# RR_sunrise_net[1] = np.mean([RR_sunrise_net[0],RR_sunrise_net[2]])

else:

	### Local Solar Time

	## Control

	# RR
	RR_cntl = assign_LT_coord(da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	RR_cntl_land = xarray_reduce(RR_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	RR_cntl_land = xr.concat([RR_cntl_land,RR_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	RR_cntl_ocean = xarray_reduce(RR_cntl.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	RR_cntl_ocean = xr.concat([RR_cntl_ocean,RR_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	RR_cntl_net = xarray_reduce(RR_cntl, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	RR_cntl_net = xr.concat([RR_cntl_net,RR_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	## NCRF Sunrise

	# RR
	RR_sunrise = assign_LT_coord(da_d02_sunrise_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy(), dim_num=3)
	inds = np.argwhere(RR_sunrise.mean(['south_north','west_east']).to_numpy()<0)[:,0]			# Find the inds where RR is negative b/c of stitching many simulations together
	RR_sunrise[inds] = (RR_sunrise[inds-1].values+RR_sunrise[inds+1].values)/2				# Average the RR between the times where RR is negative
		# Land
	RR_sunrise_land = xarray_reduce(RR_sunrise.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	RR_sunrise_land = xr.concat([RR_sunrise_land,RR_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	RR_sunrise_ocean = xarray_reduce(RR_sunrise.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	RR_sunrise_ocean = xr.concat([RR_sunrise_ocean,RR_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	RR_sunrise_net = xarray_reduce(RR_sunrise, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	RR_sunrise_net = xr.concat([RR_sunrise_net,RR_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))


# In[75]:


fig = plt.figure(figsize=(14,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

fs = 18

# RR-land
l1 = RR_cntl_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='RR-cntl-land'
)
l2 = RR_sunrise_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='RR-sunrise-land'
)
l3 = RR_cntl_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='RR-cntl-ocean'
)
l4 = RR_sunrise_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='RR-sunrise-ocean'
)
l5 = RR_cntl_net.plot.line(
    ax=ax1,
    color='k',
    linewidth=2,
    linestyle = '-',
	label='RR-cntl-net'
)
l6 = RR_sunrise_net.plot.line(
    ax=ax1,
    color='k',
    linewidth=2,
    linestyle = '--',
	label='RR-sunrise-net'
)

fs = 20

ax1.set_xticks(np.arange(0,25,3))
ax1.set_yticks(np.arange(0,3,1))
title = f'Diurnal Composite of Rain Rate over the MC'
ax1.set_title(title, fontsize=fs)
ax1.set_xlabel(f'{time_type}', fontsize=fs)
ax1.set_ylabel('Rain Rate [$mm d^{-1}$]', fontsize=fs)
ax1.set_xticklabels(np.arange(0,25,3), fontsize=fs-6)
ax1.set_yticklabels(np.arange(0,3,1), fontsize=fs-6)
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.legend(ncol=3, fontsize=fs-6)
ax1.set_xlim([0,24])
ax1.set_ylim([0,2])

# Save figure
# fig.savefig('/home/hragnajarian/PhD/plots/DC_HFXandRR.png', transparent=True)


# Normal Wind Evolution

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 950
title = f'Normalized Diurnal Normal Wind Difference from Control over ' + cross_title + f' at {level}hPa'

fig.suptitle(title, fontsize=26)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_NormalWind_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_NormalWind_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_NormalWind_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Fontsize
fs = 20

# Plot phase speed lines
if cross_title == 'Borneo Northwest':
	# Borneo phase speed lines
	ax1.plot([-205,-43],[13.5,28.5] , color='r')
	ax1.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,65],[13.5,28.5] , color='r')
	ax1.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,530],[13.5,30.5] , color='r')
	ax1.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax1.plot([-205,-43],[37.5,52.5] , color='r')
	ax1.text(-145, 44, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,65],[37.5,52.5] , color='r')
	ax1.text(25, 43.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,530],[37.5,54.5] , color='r')
	ax1.text(145, 40, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax2.plot([-205,-43],[13.5,28.5] , color='r')
	ax2.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-205,65],[13.5,28.5] , color='r')
	ax2.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-205,530],[13.5,30.5] , color='r')
	ax2.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-205,-43],[13.5,28.5] , color='r')
	ax3.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,65],[13.5,28.5] , color='r')
	ax3.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,530],[13.5,30.5] , color='r')
	ax3.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-205,-43],[37.5,52.5] , color='r')
	ax3.text(-145, 44, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,65],[37.5,52.5] , color='r')
	ax3.text(25, 43.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-205,530],[37.5,54.5] , color='r')
	ax3.text(145, 40, '12 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.set_yticks(x1.hour[0::3].values+0.5-1)
	ax2.set_yticks(x1.hour[0::3].values+0.5)
	ax3.set_yticks(x1.hour[0::3].values+0.5)
else:
	ax1.plot([-45,117],[9,24] , color='r')
	ax1.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,225],[9,24] , color='r')
	ax1.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,603],[9,24] , color='r')
	ax1.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax1.plot([-45,117],[33,48] , color='r')
	ax1.text(60, 43, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,225],[33,48] , color='r')
	ax1.text(180, 39, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,603],[33,48] , color='r')
	ax1.text(300, 37, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax2.plot([-45,117],[9,24] , color='r')
	ax2.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-45,225],[9,24] , color='r')
	ax2.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax2.plot([-45,603],[9,24] , color='r')
	ax2.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-45,117],[9,24] , color='r')
	ax3.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,225],[9,24] , color='r')
	ax3.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,603],[9,24] , color='r')
	ax3.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

	ax3.plot([-45,117],[33,48] , color='r')
	ax3.text(60, 43, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,225],[33,48] , color='r')
	ax3.text(180, 39, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax3.plot([-45,603],[33,48] , color='r')
	ax3.text(300, 37, '12 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.set_yticks(x1.hour[0::3].values+0.5)
	ax2.set_yticks(x1.hour[0::3].values+0.5)
	ax3.set_yticks(x1.hour[0::3].values+0.5)

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--', linewidth=2)
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax3.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('')
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_ylabel('Lead Time', fontsize=fs)
ax1.set_xticks(np.arange(-200,600,100))
ax2.set_xticks(np.arange(-200,600,100))
ax3.set_xticks(np.arange(-200,600,100))
ax1.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax3.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax1.set_yticklabels(np.concatenate((np.arange(7,25,3),np.arange(1,25,3),np.arange(1,7,3))),fontsize=fs-6)
ax2.set_yticklabels(np.arange(0,48,3),fontsize=fs-6)
ax3.set_yticklabels(np.arange(-12,36,3),fontsize=fs-6)
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
ax3.set_title('NCRF Sunset', loc='center', fontsize=fs)
ax3.set_title('c)', loc='left', fontsize=fs)

ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-4,5,1))
cbar.set_ticklabels(np.arange(-4,5,1), fontsize=fs-6)
cbar.set_label('Normal wind [$m/s$]', fontsize=fs-6)

cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_ticklabels(np.arange(-3,4,1), fontsize=fs-6)
cbar.set_label("[$(U_{Normal}-U_{Normal_{cntl}})/\sigma_{cntl}$]", fontsize=fs-6)


# Net CRF at Surface and Upward Heat Flux

# In[ ]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[0.87,0.03], width_ratios=[0.85,0.15], hspace=0.3, wspace=0.05)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

fs = 18

# Set thresholds
land_dist_thresh = -242
land_coast_dist = -100
ocean_dist_thresh = 100
ocean_coast_dist = 0

time_start = 0
time_end = 9
lag_plot = 0				# The lag relationship plotted on the first row

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# Upward Sensible Heat Flux @ Surface [W/m^2] 
x1 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1 = x1.sel(Time=x1.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<land_coast_dist),drop=True).mean('Spread')
# Anomalize
x1mean = x1.groupby('Time.hour').mean()
for i in np.arange(time_start,time_end):
	x1[x1.Time.dt.hour.isin(i)] = x1[x1.Time.dt.hour.isin(i)]-x1mean.sel(hour=i)	# Divide by the mean recorded at each hour to remove diurnal variations

# Rain Rate [mm/day]
x2 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x2 = x2.sel(Time=x2.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<land_coast_dist),drop=True).mean('Spread')
# x2mean = x2.groupby('Time.hour').mean()
# for i in np.arange(time_start,time_end):
# 	x2[x2.Time.dt.hour.isin(i)] = x2[x2.Time.dt.hour.isin(i)]-x2mean.sel(hour=i)	# Divide by the mean recorded at each hour to remove diurnal variations

# Net Surface CRF [W/m^2] 
x3 = da_d02_cross_NetSfcCRF_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x3 = x3.sel(Time=x3.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<land_coast_dist),drop=True).mean('Spread')
# Anomalize
x3mean = x3.groupby('Time.hour').mean()
for i in np.arange(time_start,time_end):
	x3[x3.Time.dt.hour.isin(i)] = x3[x3.Time.dt.hour.isin(i)]-x3mean.sel(hour=i)	# Divide by the mean recorded at each hour to remove diurnal variations

## Calculate linear regressions	##
min_lag = 0
max_lag = 0

da2 = linreg(x1.values[~np.isnan(x1)], x3.values[~np.isnan(x1)], min_lag, max_lag)

## Plot linear regressions	##

s1 = ax1.scatter(x1,x3,c=x2,s=15,cmap='gray_r',edgecolors='0.5',alpha=0.5,linewidths=.15, vmin=0, vmax=7)

l2 = ax1.plot([-200,200],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-200, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*200],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(-100, 325, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Anomalized Hourly Net Surface CRF & Sensible Heat Flux Up\n {land_coast_dist}->{land_dist_thresh:.0f} km, {time_start:.0f}->{time_end-1:.0f}UTC, n={len(x1.values[~np.isnan(x1)]):.0f}, over {cross_title:s}'
ax1.set_title(title, fontsize=12)
ax1.set_xlabel('Anomalous HFX [$W/m^{2}$]',fontsize=10)
ax1.set_ylabel('Anomalous Net SFC CRF [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([-150,149.99])
ax1.set_ylim([-425,425])
ax1.grid(linestyle='--', axis='both', linewidth=1)

cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(s1,cax=cax1, orientation='horizontal', pad=0, extend='max')
cbar.set_ticks(np.arange(0,8,1))
cbar.set_ticklabels(np.arange(0,8,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm/day$]')


# Create rain rate plot to the right
dy = 25			# Bin size
rain_rate_bins = np.arange(-425,450,dy)
rain_rate = np.zeros(len(rain_rate_bins))
for i in range(len(rain_rate_bins)-1):
	RR = x2.where((x3>rain_rate_bins[i])&(x3<rain_rate_bins[i+1]),drop=True).mean().values
	if ~np.isnan(RR):
		rain_rate[i]=RR

l1 = ax2.plot(rain_rate,rain_rate_bins, color='k')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.set_xlabel('[$mm/day$]',fontsize=10)
ax2.set_ylabel('',fontsize=10)
ax2.set_yticklabels([])
# ax2.xaxis.set_tick_params(left = False)
ax2.set_xlim([0,7])
ax2.set_ylim([-425,425])
ax2.grid(linestyle='--', axis='both', linewidth=1)


# Upward Heat Flux and SW CRF at Surface

# In[ ]:


fig = plt.figure(figsize=(14,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
# ax2 = ax1.twinx()

fs = 18
# land_dist_thresh = -165
land_dist_thresh = da_d02_cross_HFX_cntl.Distance.min().values
# ocean_dist_thresh = -65
ocean_dist_thresh = -100

local_time = np.concatenate((np.arange(7,25,3),np.arange(1,10,3)))
# UTC_time = np.concatenate((np.arange(0,24,3),[0]))

# Load Data
## Sensible Heat Flux Up
	# Control
x1 = da_d02_cross_HFX_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.where((x1.Distance<ocean_dist_thresh)&(x1.Distance>land_dist_thresh),drop=True).mean(['Distance','Spread'])
x1std = x1.groupby('Time.hour').std()
x1 = x1.groupby('Time.hour').mean()
x1std = xr.concat([x1std,x1std[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1 = xr.concat([x1,x1[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# NCRF Sunrise 
x2 = da_d02_cross_HFX_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='HFX',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2 = x2.where((x2.Distance<ocean_dist_thresh)&(x2.Distance>land_dist_thresh),drop=True).mean(['Distance'])
x2std = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').std()
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
x2std = xr.concat([x2std,x2std[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

## SW SFC FLUX
    # Control
x3 = da_d02_cross_SWSfc_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x3 = x3.where((x3.Distance<ocean_dist_thresh)&(x3.Distance>land_dist_thresh),drop=True).mean(['Distance','Spread'])
x3std = x3.groupby('Time.hour').std()
x3 = x3.groupby('Time.hour').mean()
x3std = xr.concat([x3std,x3std[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x3 = xr.concat([x3,x3[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

    # NCRF Sunrise
x4 = da_d02_cross_SWSfc_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x4 = x4.mean('Spread')
da = x4.drop_vars(['Lead','Time','Times'])
x4 = xr.DataArray(
				name='SWSFC',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x4 = x4.where((x4.Distance<ocean_dist_thresh)&(x4.Distance>land_dist_thresh),drop=True).mean(['Distance'])
x4std = x4.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').std()
x4 = x4.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
x4std = xr.concat([x4std,x4std[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x4 = xr.concat([x4,x4[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

## LW SFC FLUX
    # Control
x5 = da_d02_cross_LWSfc_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x5 = x5.where((x5.Distance<ocean_dist_thresh)&(x5.Distance>land_dist_thresh),drop=True).mean(['Distance','Spread'])
x5std = x5.groupby('Time.hour').std()
x5 = x5.groupby('Time.hour').mean()
x5std = xr.concat([x5std,x5std[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x5 = xr.concat([x5,x5[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

    # NCRF Sunrise
x6 = da_d02_cross_LWSfc_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x6 = x6.mean('Spread')
da = x6.drop_vars(['Lead','Time','Times'])
x6 = xr.DataArray(
				name='LWSFC',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x6 = x6.where((x6.Distance<ocean_dist_thresh)&(x6.Distance>land_dist_thresh),drop=True).mean(['Distance'])
x6std = x6.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').std()
x6 = x6.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
x6std = xr.concat([x6std,x6std[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x6 = xr.concat([x6,x6[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# HFX
l1 = x1.plot.line(
    ax=ax1,
    color='k',
    linewidth=2,
    linestyle = '-',
	label='Sensible$_{Control}$'
)
l2 = x2.plot.line(
    ax=ax1,
    color='k',
    linewidth=2,
    linestyle = '--',
	label='Sensible$_{NCRF Sunrise}$'
)
# SW
l3 = x3.plot.line(
    ax=ax1,
    color='r',
    linewidth=2,
    linestyle = '-',
	label='SW SFC$_{Control}$'
)
l4 = x4.plot.line(
    ax=ax1,
    color='r',
    linewidth=2,
    linestyle = '--',
	label='SW SFC$_{NCRF Sunrise}$'
)
# LW
l5 = x5.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='LW SFC$_{Control}$'
)
l6 = x6.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='LW SFC$_{NCRF Sunrise}$'
)
# Plot the shading
# ax1.fill_between(np.arange(0,25,1),x1.values-x1std.values,x1.values+x1std.values, alpha=0.15, color='k')
# ax1.fill_between(np.arange(0,25,1),x2.values-x2std.values,x2.values+x2std.values, alpha=0.15, color='k')
# ax1.fill_between(np.arange(0,25,1),x3.values-x3std.values,x3.values+x3std.values, alpha=0.15, color='r')
# ax1.fill_between(np.arange(0,25,1),x4.values-x4std.values,x4.values+x4std.values, alpha=0.15, color='r')
# ax1.fill_between(np.arange(0,25,1),x5.values-x5std.values,x5.values+x5std.values, alpha=0.15, color='r')
# ax1.fill_between(np.arange(0,25,1),x6.values-x6std.values,x6.values+x6std.values, alpha=0.15, color='r')

# fs = 20

if cross_title == 'Borneo Northwest': ax1.set_xticks(np.arange(0,27,3)-1)
else: ax1.set_xticks(np.arange(0,27,3))
ax1.set_yticks(np.concatenate(([-100],np.arange(0,1200,200))))
title = f'Diurnal Composite of Flux at Surface\n {ocean_dist_thresh:.0f}->{land_dist_thresh:.0f} km over {cross_title:s}'
ax1.set_title(title, fontsize=fs-4)
ax1.set_xlabel('Local Time', fontsize=fs)
ax1.set_ylabel('Flux [$Wm^{-2}$]', fontsize=fs)
ax1.set_xticklabels(local_time, fontsize=fs-6)
ax1.set_yticklabels(np.concatenate(([-100],np.arange(0,1200,200))), fontsize=fs-6)
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.legend(ncol=3, fontsize=fs-6)
ax1.set_xlim([0,24])
ax1.set_ylim([-150,1000])

# Save figure
# fig.savefig('/home/hragnajarian/PhD/plots/DC_HFXandSWSFC.png', transparent=True)


# In[78]:


#### Load Data

time_type = 'LST'

if time_type == 'UST':

	### UST

	## Control

	# Sensible Heat Flux Up
	HFX_cntl = da_d02_HFX.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	HFX_cntl_land = HFX_cntl.where(da_d02_LANDMASK==1).mean(['south_north','west_east']).groupby('Time.hour').mean()
	HFX_cntl_land = xr.concat([HFX_cntl_land,HFX_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	HFX_cntl_ocean = HFX_cntl.where(da_d02_LANDMASK==0).mean(['south_north','west_east']).groupby('Time.hour').mean()
	HFX_cntl_ocean = xr.concat([HFX_cntl_ocean,HFX_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	HFX_cntl_net = HFX_cntl.mean(['south_north','west_east']).groupby('Time.hour').mean()
	HFX_cntl_net = xr.concat([HFX_cntl_net,HFX_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# LWSFC
	LWSFC_cntl = da_d02_LWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	LWSFC_cntl_land = LWSFC_cntl.where(da_d02_LANDMASK==1).mean(['south_north','west_east']).groupby('Time.hour').mean()
	LWSFC_cntl_land = xr.concat([LWSFC_cntl_land,LWSFC_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	LWSFC_cntl_ocean = LWSFC_cntl.where(da_d02_LANDMASK==0).mean(['south_north','west_east']).groupby('Time.hour').mean()
	LWSFC_cntl_ocean = xr.concat([LWSFC_cntl_ocean,LWSFC_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	LWSFC_cntl_net = LWSFC_cntl.mean(['south_north','west_east']).groupby('Time.hour').mean()
	LWSFC_cntl_net = xr.concat([LWSFC_cntl_net,LWSFC_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# SWSFC
	SWSFC_cntl = da_d02_SWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	SWSFC_cntl_land = SWSFC_cntl.where(da_d02_LANDMASK==1).mean(['south_north','west_east']).groupby('Time.hour').mean()
	SWSFC_cntl_land = xr.concat([SWSFC_cntl_land,SWSFC_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	SWSFC_cntl_ocean = SWSFC_cntl.where(da_d02_LANDMASK==0).mean(['south_north','west_east']).groupby('Time.hour').mean()
	SWSFC_cntl_ocean = xr.concat([SWSFC_cntl_ocean,SWSFC_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	SWSFC_cntl_net = SWSFC_cntl.mean(['south_north','west_east']).groupby('Time.hour').mean()
	SWSFC_cntl_net = xr.concat([SWSFC_cntl_net,SWSFC_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	## NCRF Sunrise

	# Sensible Heat Flux Up
	HFX_sunrise = da_d02_sunrise_HFX.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	HFX_sunrise_land = HFX_sunrise.where(da_d02_LANDMASK==1).mean(['south_north','west_east']).groupby('Time.hour').mean()
	HFX_sunrise_land = xr.concat([HFX_sunrise_land,HFX_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	HFX_sunrise_ocean = HFX_sunrise.where(da_d02_LANDMASK==0).mean(['south_north','west_east']).groupby('Time.hour').mean()
	HFX_sunrise_ocean = xr.concat([HFX_sunrise_ocean,HFX_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	HFX_sunrise_net = HFX_sunrise.mean(['south_north','west_east']).groupby('Time.hour').mean()
	HFX_sunrise_net = xr.concat([HFX_sunrise_net,HFX_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# LWSFC
	LWSFC_sunrise = da_d02_sunrise_LWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	LWSFC_sunrise_land = LWSFC_sunrise.where(da_d02_LANDMASK==1).mean(['south_north','west_east']).groupby('Time.hour').mean()
	LWSFC_sunrise_land = xr.concat([LWSFC_sunrise_land,LWSFC_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	LWSFC_sunrise_ocean = LWSFC_sunrise.where(da_d02_LANDMASK==0).mean(['south_north','west_east']).groupby('Time.hour').mean()
	LWSFC_sunrise_ocean = xr.concat([LWSFC_sunrise_ocean,LWSFC_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	LWSFC_sunrise_net = LWSFC_sunrise.mean(['south_north','west_east']).groupby('Time.hour').mean()
	LWSFC_sunrise_net = xr.concat([LWSFC_sunrise_net,LWSFC_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# SWSFC
	SWSFC_sunrise = da_d02_sunrise_SWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00'))
		# Land
	SWSFC_sunrise_land = SWSFC_sunrise.where(da_d02_LANDMASK==1).mean(['south_north','west_east']).groupby('Time.hour').mean()
	SWSFC_sunrise_land = xr.concat([SWSFC_sunrise_land,SWSFC_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	SWSFC_sunrise_ocean = SWSFC_sunrise.where(da_d02_LANDMASK==0).mean(['south_north','west_east']).groupby('Time.hour').mean()
	SWSFC_sunrise_ocean = xr.concat([SWSFC_sunrise_ocean,SWSFC_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	SWSFC_sunrise_net = SWSFC_sunrise.mean(['south_north','west_east']).groupby('Time.hour').mean()
	SWSFC_sunrise_net = xr.concat([SWSFC_sunrise_net,SWSFC_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

else:


	# RR
	RR_cntl = assign_LT_coord(da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	RR_cntl_land = xarray_reduce(RR_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	RR_cntl_land = xr.concat([RR_cntl_land,RR_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))


	### LST

	## Control

	# Sensible Heat Flux Up
	HFX_cntl = assign_LT_coord(da_d02_HFX.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	HFX_cntl_land = xarray_reduce(HFX_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HFX_cntl_land = xr.concat([HFX_cntl_land,HFX_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	HFX_cntl_ocean = xarray_reduce(HFX_cntl.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HFX_cntl_ocean = xr.concat([HFX_cntl_ocean,HFX_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	HFX_cntl_net = xarray_reduce(HFX_cntl, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HFX_cntl_net = xr.concat([HFX_cntl_net,HFX_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# Low Cloud Fraction
	LowCLDFRA_cntl = assign_LT_coord(da_d02_LowCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	LowCLDFRA_cntl_land = xarray_reduce(LowCLDFRA_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LowCLDFRA_cntl_land = xr.concat([LowCLDFRA_cntl_land,LowCLDFRA_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	LowCLDFRA_cntl_ocean = xarray_reduce(LowCLDFRA_cntl.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LowCLDFRA_cntl_ocean = xr.concat([LowCLDFRA_cntl_ocean,LowCLDFRA_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	LowCLDFRA_cntl_net = xarray_reduce(LowCLDFRA_cntl, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LowCLDFRA_cntl_net = xr.concat([LowCLDFRA_cntl_net,LowCLDFRA_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# Mid Cloud Fraction
	MidCLDFRA_cntl = assign_LT_coord(da_d02_MidCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	MidCLDFRA_cntl_land = xarray_reduce(MidCLDFRA_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	MidCLDFRA_cntl_land = xr.concat([MidCLDFRA_cntl_land,MidCLDFRA_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	MidCLDFRA_cntl_ocean = xarray_reduce(MidCLDFRA_cntl.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	MidCLDFRA_cntl_ocean = xr.concat([MidCLDFRA_cntl_ocean,MidCLDFRA_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	MidCLDFRA_cntl_net = xarray_reduce(MidCLDFRA_cntl, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	MidCLDFRA_cntl_net = xr.concat([MidCLDFRA_cntl_net,MidCLDFRA_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# High Cloud Fraction
	HighCLDFRA_cntl = assign_LT_coord(da_d02_HighCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	HighCLDFRA_cntl_land = xarray_reduce(HighCLDFRA_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HighCLDFRA_cntl_land = xr.concat([HighCLDFRA_cntl_land,HighCLDFRA_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	HighCLDFRA_cntl_ocean = xarray_reduce(HighCLDFRA_cntl.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HighCLDFRA_cntl_ocean = xr.concat([HighCLDFRA_cntl_ocean,HighCLDFRA_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	HighCLDFRA_cntl_net = xarray_reduce(HighCLDFRA_cntl, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HighCLDFRA_cntl_net = xr.concat([HighCLDFRA_cntl_net,HighCLDFRA_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# LWSFC
	LWSFC_cntl = assign_LT_coord(da_d02_LWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	LWSFC_cntl_land = xarray_reduce(LWSFC_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LWSFC_cntl_land = xr.concat([LWSFC_cntl_land,LWSFC_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	LWSFC_cntl_ocean = xarray_reduce(LWSFC_cntl.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LWSFC_cntl_ocean = xr.concat([LWSFC_cntl_ocean,LWSFC_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	LWSFC_cntl_net = xarray_reduce(LWSFC_cntl, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LWSFC_cntl_net = xr.concat([LWSFC_cntl_net,LWSFC_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# SWSFC
	SWSFC_cntl = assign_LT_coord(da_d02_SWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	SWSFC_cntl_land = xarray_reduce(SWSFC_cntl.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	SWSFC_cntl_land = xr.concat([SWSFC_cntl_land,SWSFC_cntl_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	SWSFC_cntl_ocean = xarray_reduce(SWSFC_cntl.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	SWSFC_cntl_ocean = xr.concat([SWSFC_cntl_ocean,SWSFC_cntl_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	SWSFC_cntl_net = xarray_reduce(SWSFC_cntl, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	SWSFC_cntl_net = xr.concat([SWSFC_cntl_net,SWSFC_cntl_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))



	## NCRF Sunrise

	# Sensible Heat Flux Up
	HFX_sunrise = assign_LT_coord(da_d02_sunrise_HFX.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	HFX_sunrise_land = xarray_reduce(HFX_sunrise.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HFX_sunrise_land = xr.concat([HFX_sunrise_land,HFX_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	HFX_sunrise_ocean = xarray_reduce(HFX_sunrise.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HFX_sunrise_ocean = xr.concat([HFX_sunrise_ocean,HFX_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	HFX_sunrise_net = xarray_reduce(HFX_sunrise, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HFX_sunrise_net = xr.concat([HFX_sunrise_net,HFX_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# Low Cloud Fraction
	LowCLDFRA_sunrise = assign_LT_coord(da_d02_sunrise_LowCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	LowCLDFRA_sunrise_land = xarray_reduce(LowCLDFRA_sunrise.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LowCLDFRA_sunrise_land = xr.concat([LowCLDFRA_sunrise_land,LowCLDFRA_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	LowCLDFRA_sunrise_ocean = xarray_reduce(LowCLDFRA_sunrise.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LowCLDFRA_sunrise_ocean = xr.concat([LowCLDFRA_sunrise_ocean,LowCLDFRA_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	LowCLDFRA_sunrise_net = xarray_reduce(LowCLDFRA_sunrise, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LowCLDFRA_sunrise_net = xr.concat([LowCLDFRA_sunrise_net,LowCLDFRA_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# Mid Cloud Fraction
	MidCLDFRA_sunrise = assign_LT_coord(da_d02_sunrise_MidCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	MidCLDFRA_sunrise_land = xarray_reduce(MidCLDFRA_sunrise.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	MidCLDFRA_sunrise_land = xr.concat([MidCLDFRA_sunrise_land,MidCLDFRA_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	MidCLDFRA_sunrise_ocean = xarray_reduce(MidCLDFRA_sunrise.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	MidCLDFRA_sunrise_ocean = xr.concat([MidCLDFRA_sunrise_ocean,MidCLDFRA_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	MidCLDFRA_sunrise_net = xarray_reduce(MidCLDFRA_sunrise, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	MidCLDFRA_sunrise_net = xr.concat([MidCLDFRA_sunrise_net,MidCLDFRA_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# High Cloud Fraction
	HighCLDFRA_sunrise = assign_LT_coord(da_d02_sunrise_HighCLDFRA.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	HighCLDFRA_sunrise_land = xarray_reduce(HighCLDFRA_sunrise.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HighCLDFRA_sunrise_land = xr.concat([HighCLDFRA_sunrise_land,HighCLDFRA_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	HighCLDFRA_sunrise_ocean = xarray_reduce(HighCLDFRA_sunrise.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HighCLDFRA_sunrise_ocean = xr.concat([HighCLDFRA_sunrise_ocean,HighCLDFRA_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	HighCLDFRA_sunrise_net = xarray_reduce(HighCLDFRA_sunrise, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	HighCLDFRA_sunrise_net = xr.concat([HighCLDFRA_sunrise_net,HighCLDFRA_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# LWSFC
	LWSFC_sunrise = assign_LT_coord(da_d02_sunrise_LWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	LWSFC_sunrise_land = xarray_reduce(LWSFC_sunrise.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LWSFC_sunrise_land = xr.concat([LWSFC_sunrise_land,LWSFC_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	LWSFC_sunrise_ocean = xarray_reduce(LWSFC_sunrise.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LWSFC_sunrise_ocean = xr.concat([LWSFC_sunrise_ocean,LWSFC_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	LWSFC_sunrise_net = xarray_reduce(LWSFC_sunrise, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	LWSFC_sunrise_net = xr.concat([LWSFC_sunrise_net,LWSFC_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))

	# SWSFC
	SWSFC_sunrise = assign_LT_coord(da_d02_sunrise_SWSFC.sel(Time=slice('2015-11-23T02','2015-12-02T00')), dim_num=3)
		# Land
	SWSFC_sunrise_land = xarray_reduce(SWSFC_sunrise.where(da_d02_LANDMASK==1), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	SWSFC_sunrise_land = xr.concat([SWSFC_sunrise_land,SWSFC_sunrise_land[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
	SWSFC_sunrise_ocean = xarray_reduce(SWSFC_sunrise.where(da_d02_LANDMASK==0), 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	SWSFC_sunrise_ocean = xr.concat([SWSFC_sunrise_ocean,SWSFC_sunrise_ocean[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
	SWSFC_sunrise_net = xarray_reduce(SWSFC_sunrise, 'LocalTime.hour', func='mean', expected_groups=(np.arange(0,24)), isbin=[False])
	SWSFC_sunrise_net = xr.concat([SWSFC_sunrise_net,SWSFC_sunrise_net[0]], dim='hour').assign_coords(hour=np.arange(0,25,1))


# In[83]:


fig = plt.figure(figsize=(14,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
# ax2 = ax1.twinx()

fs = 18

l1 = HFX_cntl_land.plot.line(
    ax=ax1,
    color='k',
    linewidth=2,
    linestyle = '-',
	label='HFX-cntl-land'
)
l2 = HFX_sunrise_land.plot.line(
    ax=ax1,
    color='k',
    linewidth=2,
    linestyle = '--',
	label='HFX-sunrise-land'
)
# SW
l3 = SWSFC_cntl_land.plot.line(
    ax=ax1,
    color='r',
    linewidth=2,
    linestyle = '-',
	label='SWSFC-cntl-land'
)
l4 = SWSFC_sunrise_land.plot.line(
    ax=ax1,
    color='r',
    linewidth=2,
    linestyle = '--',
	label='SWSFC-sunrise-land'
)
# LW
l5 = LWSFC_cntl_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='LWSFC-cntl-land'
)
l6 = LWSFC_sunrise_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='LWSFC-sunrise-land'
)
# # CLDFRA
# l7 = LowCLDFRA_cntl_land.plot.line(
#     ax=ax2,
#     color='aqua',
#     linewidth=2,
#     linestyle = '-',
# 	label='LowCLDFRA-cntl-land'
# )
# l8 = LowCLDFRA_sunrise_land.plot.line(
#     ax=ax2,
#     color='aqua',
#     linewidth=2,
#     linestyle = '--',
# 	label='LowCLDFRA-sunrise-land'
# )
# # CLDFRA
# l9 = MidCLDFRA_cntl_land.plot.line(
#     ax=ax2,
#     color='violet',
#     linewidth=2,
#     linestyle = '-',
# 	label='MidCLDFRA-cntl-land'
# )
# l10 = MidCLDFRA_sunrise_land.plot.line(
#     ax=ax2,
#     color='violet',
#     linewidth=2,
#     linestyle = '--',
# 	label='MidCLDFRA-sunrise-land'
# )
# # CLDFRA
# l11 = HighCLDFRA_cntl_land.plot.line(
#     ax=ax2,
#     color='royalblue',
#     linewidth=2,
#     linestyle = '-',
# 	label='HighCLDFRA-cntl-land'
# )
# l12 = HighCLDFRA_sunrise_land.plot.line(
#     ax=ax2,
#     color='royalblue',
#     linewidth=2,
#     linestyle = '--',
# 	label='HighCLDFRA-sunrise-land'
# )


fs = 20

ax1.set_xticks(np.arange(0,25,3))
ax1.set_yticks(np.concatenate(([-100],np.arange(0,1200,200))))
ax2.set_yticks(np.arange(0,.3,.1))
title = f'Diurnal Composite of Surface Heat Flux over the MC'
ax1.set_title(title, fontsize=fs)
ax2.set_title('', fontsize=fs)
ax1.set_xlabel(f'{time_type}', fontsize=fs)
ax1.set_ylabel('Flux [$Wm^{-2}$]', fontsize=fs)
ax2.set_ylabel('Cloud Fraction', fontsize=fs)
ax1.set_xticklabels(np.arange(0,25,3), fontsize=fs-6)
ax1.set_yticklabels(np.concatenate(([-100],np.arange(0,1200,200))), fontsize=fs-6)
ax2.set_yticklabels(np.arange(0,.3,.1), fontsize=fs-6)
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.legend(ncol=3, fontsize=fs-9, loc='upper left')
ax2.legend(ncol=3, fontsize=fs-12, loc='upper right')
ax1.set_xlim([0,24])
ax1.set_ylim([-150,1000])
ax2.set_ylim([-.03,.2])

# Save figure
# fig.savefig('/home/hragnajarian/PhD/plots/DC_HFXandSWSFC.png', transparent=True)


# Vertically Integrated Moisture

# In[ ]:


fig = plt.figure(figsize=(8.5,10))
gs = gridspec.GridSpec(nrows=2, ncols=1,  hspace=0.2, height_ratios=[0.96,0.04])
ax1 = fig.add_subplot(gs[0,0])

levels = [950,200]

# fig.suptitle(f'Vertically integrated {levels[0]:.0f}-{levels[1]:.0f}hPa Normalized Total Q Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QTotal_cntl[13:-12].sel(bottom_top=slice(levels[0],levels[1]))
# Vertically integrate
x1 = vertical_integration(x1)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(0,18,2), bottom_top=slice(levels[0],levels[1])).mean('Lead')
# Vertically integrate
x2 = vertical_integration(x2)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean('Spread')
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(1,18,2), bottom_top=slice(levels[0],levels[1])).mean('Lead')
# Vertically integrate
x3 = vertical_integration(x3)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean('Spread')
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the datasets
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = xr.concat([x2,x3[:11]],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))
# x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf2 = x2[:38].plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='BrBG',
	center=0,
	extend='both'
)
cf3 = x2[37:].plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,3.5,0.5),
	cmap='binary',
	extend='neither'
)

fs = 20

if cross_title == 'Borneo Northwest':
	# Borneo phase speed lines
	ax1.plot([-205,-43],[13.5,28.5] , color='r')
	ax1.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,65],[13.5,28.5] , color='r')
	ax1.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-205,530],[13.5,30.5] , color='r')
	ax1.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.set_yticks(x1.hour[0::3].values+0.5-1)
else:
	ax1.plot([-45,117],[9,24] , color='r')
	ax1.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,225],[9,24] , color='r')
	ax1.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.plot([-45,603],[9,24] , color='r')
	ax1.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)
	ax1.set_yticks(x1.hour[0::3].values+0.5)

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax1.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs)
ax1.set_xticks(np.arange(-200,600,100))
ax1.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
# y_ticks made in the if else statement above ^
ax1.set_yticklabels(np.concatenate((np.arange(7,25,3),np.arange(1,25,3),np.arange(1,7,3))),fontsize=fs-6)
ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax1.set_ylim([0,46.5])
ax1.invert_xaxis()
# Set titles/labels
title = 'Normalized QTotal|$_{200hPa}^{950hPa}$ Difference from Control over ' + cross_title
ax1.set_title(title, fontsize=fs-4)

# Plot the colorbar
	# Vertical Wind colorbar

cax2 = fig.add_subplot(gs[1,0])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_ticklabels(np.arange(-3,4,1), fontsize=fs-6)
cbar.set_label('[$(QTotal-QTotal_{cntl})/\sigma_{cntl}$]', fontsize=fs-6)

# Save figure
# fig.savefig('/home/hragnajarian/PhD/plots/NormDiff_QTotal.png', transparent=True)


# Doppler Shift Anlaysis

# In[ ]:


# Subjective times of off-shore propagation 
if cross_title == 'Borneo Northwest':
	start_LT = 2
	end_LT = 8
	UTC_to_LT = +8
else:
	start_LT = 18
	end_LT = 0
	UTC_to_LT = +7


ptop = 200

# Control
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
NormalWind_cntl = da_d02_cross_NormalWind_cntl[13:-24]
NormalWind_cntl['Time'] = NormalWind_cntl['Time']+np.timedelta64(UTC_to_LT,'h')	# Change to LT
NormalWind_cntl = NormalWind_cntl.groupby('Time.hour').mean().mean('Spread')
if end_LT < start_LT:
	NormalWind_cntl = NormalWind_cntl.sel(hour=np.concatenate([np.arange(0,end_LT+1),np.arange(start_LT,24)])).mean('hour')
else:
	NormalWind_cntl = NormalWind_cntl.sel(hour=slice(start_LT,end_LT)).mean('hour')

# NCRF Sunrise
# # Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
# NormalWind_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# # Average over all the simulations and cross-sections, group by Time and then average 
# NormalWind_NCRF = NormalWind_NCRF.mean(['Lead','Spread'])


# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
NormalWind_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1+UTC_to_LT,24),np.arange(0,UTC_to_LT+1)))))
# Average over all the simulations and cross-sections, group by Time and then average 
NormalWind_NCRF = NormalWind_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
if end_LT < start_LT:
	NormalWind_NCRF = NormalWind_NCRF.sel(hour=np.concatenate([np.arange(0,end_LT+1),np.arange(start_LT,24)])).mean('hour')
else:
	NormalWind_NCRF = NormalWind_NCRF.sel(hour=slice(start_LT,end_LT)).mean('hour')

NormalWind_Diff = NormalWind_NCRF - NormalWind_cntl
# NormalWind_Diff = NormalWind_cntl

# Smooth the data a little bit
smoothing_num = 3
NormalWind_Diff = NormalWind_Diff.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

fig = plt.figure(figsize=(8.5,10))
# fig = plt.figure(figsize=(9,7.5))

gs = gridspec.GridSpec(nrows=2, ncols=1, hspace=0.2, height_ratios=[0.96,0.04])
ax1 = fig.add_subplot(gs[0,0])

# Plot terrains
y = d02_cross_PSFC.max(axis=(0,2))
ax1.plot(NormalWind_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
y = d02_cross_PSFC.min(axis=(0,2))
ax1.plot(NormalWind_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
y = d02_cross_PSFC.mean(axis=(0,2))
ax1.plot(NormalWind_cntl.Distance,y,color='black',linewidth=2)

# Plot the cross-sectional data
cf1 = NormalWind_Diff.plot.contourf(
	ax=ax1,
	add_colorbar=False,
	# levels=np.arange(-10,10.5,.5),
	levels=np.arange(-2,2.25,.25),
	cmap='RdBu_r',
	yscale='log',
	ylim=[ptop,1000],
	extend='both'
)

fs = 20
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax1.set_ylabel('Pressure Levels [hPa]', fontsize=fs)
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)

string = f'Normal Wind Difference from Control: {cross_title:s}\n (NCRF - CNTL) {start_LT:.0f}-{end_LT:.0f}LT'
ax1.set_title(string, fontsize=fs-4)
yticks = np.linspace(1000,100,10)
ax1.set_yticks(yticks)
ax1.set_yticklabels(yticks, fontsize=fs-6)
xticks = np.arange(-200,600,100)
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticks, fontsize=fs-6)
ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.set_ylim([1000,ptop])
ax1.set_xlim([NormalWind_cntl.Distance[0],NormalWind_cntl.Distance[-1]])
# ax1.invert_yaxis()
ax1.invert_xaxis()

ax3 = fig.add_subplot(gs[1,0])
cbar = plt.colorbar(cf1, cax=ax3, orientation='horizontal', pad=0 , aspect=100, extend='max')
cbar.set_ticks(np.arange(-2,2.5,0.5))
cbar.set_ticklabels(np.arange(-2,2.5,0.5), fontsize=fs-6)
cbar.set_label("Normal Wind' [$m/s$]", fontsize=fs-6)
cbar.minorticks_off()

# Save figure
# fig.savefig('/home/hragnajarian/PhD/plots/DailyAvgDiff_NormalWind.png', transparent=True)


# ## Spatial Analysis

# ### Diurnal SW SFC Flux Amplitude

# In[31]:


# Data
rolls = 5
lon_bound = [90,125]
lat_bound = [-10,10]
time_bound = ['2015-11-23T02','2015-12-02T00']

# Moving average to "denoise"
data = da_d02_SWSFC.sel(Time=slice(time_bound[0],time_bound[1])).rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
data = data.groupby('Time.hour').mean('Time')
# Find the max and min values
x = data.max(dim='hour', keep_attrs=True)
y = data.min(dim='hour', keep_attrs=True)
amplitude_SWSFC_cntl = x-y
amplitude_SWSFC_cntl_land = xr.where(da_d02_LANDMASK==1, amplitude_SWSFC_cntl, np.nan)
amplitude_SWSFC_cntl_ocean = xr.where(da_d02_LANDMASK==0, amplitude_SWSFC_cntl, np.nan)

# Moving average to "denoise"
data = da_d02_sunrise_SWSFC.rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
data = data.groupby('Time.hour').mean('Time')
# Find the max and min values
x = data.max(dim='hour', keep_attrs=True)
y = data.min(dim='hour', keep_attrs=True)
amplitude_SWSFC_sunrise = x-y
amplitude_SWSFC_sunrise_land = xr.where(da_d02_LANDMASK==1, amplitude_SWSFC_sunrise, np.nan)
amplitude_SWSFC_sunrise_ocean = xr.where(da_d02_LANDMASK==0, amplitude_SWSFC_sunrise, np.nan)

# Difference
amplitude_SWSFC_diff =  amplitude_SWSFC_sunrise - amplitude_SWSFC_cntl
amplitude_SWSFC_diff_land = xr.where(da_d02_LANDMASK==1, amplitude_SWSFC_diff, np.nan)
amplitude_SWSFC_diff_ocean = xr.where(da_d02_LANDMASK==0, amplitude_SWSFC_diff, np.nan)


# In[133]:


###############################################################################################################################
###### This version groupby is done day-by-day while the above version does a diurnal composite. Both perform similarly. ######
###############################################################################################################################
# Data
rolls = 1
lon_bound = [90,125]
lat_bound = [-10,10]
time_bound = ['2015-11-23T02','2015-12-02T00']

# Moving average to "denoise"
data = da_d02_SWSFC.sel(Time=slice(time_bound[0],time_bound[1]))#.rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
# Find the max and min values
x = data.groupby('Time.day').max('Time')
y = data.groupby('Time.day').min('Time')
amplitude_SWSFC_cntl = x-y
amplitude_SWSFC_cntl_land = xr.where(da_d02_LANDMASK==1, amplitude_SWSFC_cntl, np.nan)
amplitude_SWSFC_cntl_ocean = xr.where(da_d02_LANDMASK==0, amplitude_SWSFC_cntl, np.nan)

# Moving average to "denoise"
data = da_d02_sunrise_SWSFC#.rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
# Find the max and min values
x = data.groupby('Time.day').max('Time')
y = data.groupby('Time.day').min('Time')
amplitude_SWSFC_sunrise = x-y
amplitude_SWSFC_sunrise_land = xr.where(da_d02_LANDMASK==1, amplitude_SWSFC_sunrise, np.nan)
amplitude_SWSFC_sunrise_ocean = xr.where(da_d02_LANDMASK==0, amplitude_SWSFC_sunrise, np.nan)

# Difference
amplitude_SWSFC_diff =  amplitude_SWSFC_sunrise - amplitude_SWSFC_cntl
amplitude_SWSFC_diff_land = xr.where(da_d02_LANDMASK==1, amplitude_SWSFC_diff, np.nan)
amplitude_SWSFC_diff_ocean = xr.where(da_d02_LANDMASK==0, amplitude_SWSFC_diff, np.nan)


# In[34]:


probability_density = True

a = plt.hist(amplitude_SWSFC_diff.values.flatten(),bins=np.arange(-112.5,412.5,25), label=f'Domain n ={len(amplitude_SWSFC_diff.values.flatten())}', density=probability_density, histtype='step', color='k')
b = plt.hist(amplitude_SWSFC_diff_ocean.values.flatten(), bins=np.arange(-112.5,412.5,25), label=f'Ocean n = {np.nansum(~np.isnan(amplitude_SWSFC_diff_ocean.values.flatten()))}', density=probability_density, histtype='step', color='dodgerblue')
c = plt.hist(amplitude_SWSFC_diff_land.values.flatten(), bins=np.arange(-112.5,412.5,25), label=f'Land n = {np.nansum(~np.isnan(amplitude_SWSFC_diff_land.values.flatten()))}', density=probability_density, histtype='step', color='peru')
plt.title(f'Diurnal SW SFC Amplitude Difference (NCRF - CNTL)\nRolling n: {rolls}')
plt.xlabel('(W/m^2)')
if probability_density: plt.ylabel('Fractional count')
else: plt.ylabel('# of grids')
plt.legend()


# ### Diurnal HFX Amplitude

# In[37]:


# Data
rolls = 5
lon_bound = [90,125]
lat_bound = [-10,10]
time_bound = ['2015-11-23T02','2015-12-02T00']

# Moving average to "denoise"
data = da_d02_HFX.sel(Time=slice(time_bound[0],time_bound[1])).rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
data = data.groupby('Time.hour').mean('Time')
# Find the max and min values
x = data.max(dim='hour', keep_attrs=True)
y = data.min(dim='hour', keep_attrs=True)
amplitude_HFX_cntl = x-y
amplitude_HFX_cntl_land = xr.where(da_d02_LANDMASK==1, amplitude_HFX_cntl, np.nan)
amplitude_HFX_cntl_ocean = xr.where(da_d02_LANDMASK==0, amplitude_HFX_cntl, np.nan)

# Moving average to "denoise"
data = da_d02_sunrise_HFX.rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
data = data.groupby('Time.hour').mean('Time')
# Find the max and min values
x = data.max(dim='hour', keep_attrs=True)
y = data.min(dim='hour', keep_attrs=True)
amplitude_HFX_sunrise = x-y
amplitude_HFX_sunrise_land = xr.where(da_d02_LANDMASK==1, amplitude_HFX_sunrise, np.nan)
amplitude_HFX_sunrise_ocean = xr.where(da_d02_LANDMASK==0, amplitude_HFX_sunrise, np.nan)

# Difference
amplitude_HFX_diff = amplitude_HFX_sunrise - amplitude_HFX_cntl
amplitude_HFX_diff_land = xr.where(da_d02_LANDMASK==1, amplitude_HFX_diff, np.nan)
amplitude_HFX_diff_ocean = xr.where(da_d02_LANDMASK==0, amplitude_HFX_diff, np.nan)


# In[132]:


###############################################################################################################################
###### This version groupby is done day-by-day while the above version does a diurnal composite. Both perform similarly. ######
###############################################################################################################################
# Data
rolls = 1
lon_bound = [90,125]
lat_bound = [-10,10]
time_bound = ['2015-11-23T02','2015-12-02T00']

# Moving average to "denoise"
data = da_d02_HFX.sel(Time=slice(time_bound[0],time_bound[1]))#.rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
# Find the max and min values
x = data.groupby('Time.day').max('Time')
y = data.groupby('Time.day').min('Time')
amplitude_HFX_cntl = x-y
amplitude_HFX_cntl_land = xr.where(da_d02_LANDMASK==1, amplitude_HFX_cntl, np.nan)
amplitude_HFX_cntl_ocean = xr.where(da_d02_LANDMASK==0, amplitude_HFX_cntl, np.nan)

# Moving average to "denoise"
data = da_d02_sunrise_HFX#.rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
# Find the max and min values
x = data.groupby('Time.day').max('Time')
y = data.groupby('Time.day').min('Time')
amplitude_HFX_sunrise = x-y
amplitude_HFX_sunrise_land = xr.where(da_d02_LANDMASK==1, amplitude_HFX_sunrise, np.nan)
amplitude_HFX_sunrise_ocean = xr.where(da_d02_LANDMASK==0, amplitude_HFX_sunrise, np.nan)

# Difference
amplitude_HFX_diff =  amplitude_HFX_sunrise - amplitude_HFX_cntl
amplitude_HFX_diff_land = xr.where(da_d02_LANDMASK==1, amplitude_HFX_diff, np.nan)
amplitude_HFX_diff_ocean = xr.where(da_d02_LANDMASK==0, amplitude_HFX_diff, np.nan)


# In[40]:


probability_density = True

a = plt.hist(amplitude_HFX_diff.values.flatten(),bins=np.arange(-10,210,20), label=f'Domain n ={len(amplitude_HFX_diff.values.flatten())}', density=probability_density, histtype='step', color='k')
b = plt.hist(amplitude_HFX_diff_ocean.values.flatten(), bins=np.arange(-10,210,20), label=f'Ocean n = {np.nansum(~np.isnan(amplitude_HFX_diff_ocean.values.flatten()))}', density=probability_density, histtype='step', color='dodgerblue')
c = plt.hist(amplitude_HFX_diff_land.values.flatten(), bins=np.arange(-10,210,20), label=f'Land n = {np.nansum(~np.isnan(amplitude_HFX_diff_land.values.flatten()))}', density=probability_density, histtype='step', color='peru')
plt.title(f'Diurnal HFX Amplitude Difference (NCRF - CNTL)\nRolling n: {rolls}')
plt.xlabel('(W/m^2)')
if probability_density: plt.ylabel('Fractional count')
else: plt.ylabel('# of grids')
plt.legend()


# In[41]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

cf = amplitude_HFX_cntl.plot.contourf(
        ax=ax1,
        add_colorbar=False,
        cmap='gray_r',
        levels=np.arange(0,825,25),
        vmin=0, vmax=825,
        x='west_east',
        y='south_north',
        xlim=[lon_bound[0], lon_bound[1]],
        ylim=[lat_bound[0], lat_bound[1]],
        extend='max',
    )

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
# ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title('Control Diurnal HFX Amplitude', loc='center')
ax1.set_title(f'Center smooth # of grids: {rolls}', loc='right')

ax2 = fig.add_subplot(gs[1,1])
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.arange(0,900,100))
cbar.set_label('Diurnal HFX Amplitude (W/m^2)')


# In[42]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

cf = amplitude_HFX_diff.plot.contourf(
        ax=ax1,
        add_colorbar=False,
        cmap='RdBu_r',
        levels=np.arange(-100,120,20),
        x='west_east',
        y='south_north',
        xlim=[lon_bound[0], lon_bound[1]],
        ylim=[lat_bound[0], lat_bound[1]],
        extend='both',
    )

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title('Diurnal HFX Amplitude Difference (NCRF - CNTL)', loc='center')
ax1.set_title(f'Center smooth # of grids: {rolls}', loc='right')

ax2 = fig.add_subplot(gs[1,1])
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(-100,150,50))
cbar.set_label('Diurnal HFX Amplitude (W/m^2)')


# ### Diurnal RR Amplitude

# In[46]:


# Data
rolls = 5
lon_bound = [90,125]
lat_bound = [-10,10]
time_bound = ['2015-11-23T02','2015-12-02T00']

# Moving average to "denoise"
RR_cntl = da_d02_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy().chunk({'Time':5}).rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
RR_cntl = assign_LT_coord(RR_cntl,dim_num=3)	# Create a local time coordinte
# Group by local time and the 'dim' parameter ensures only the Time dimension is mean'd/averaged into the 'hour' dimension
RR_cntl = xarray_reduce(RR_cntl, 'LocalTime.hour', func='mean', dim='Time', expected_groups=(np.arange(0,24)), isbin=[False]).transpose('hour','west_east','south_north')
# Find the max and min values
x = RR_cntl.max(dim='hour', keep_attrs=True)
y = RR_cntl.min(dim='hour', keep_attrs=True)
amplitude_RR_cntl = x-y
amplitude_RR_cntl_land = xr.where(da_d02_LANDMASK==1, amplitude_RR_cntl, np.nan)
amplitude_RR_cntl_ocean = xr.where(da_d02_LANDMASK==0, amplitude_RR_cntl, np.nan)

# First adjust data where there are negative RR's.
RR_sunrise = da_d02_sunrise_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy().chunk({'Time':5})
inds = np.unique(np.argwhere(RR_sunrise.values<0)[:,0])	# Time steps where RR is negative due to stitching over many simulations
RR_sunrise[inds] = (RR_sunrise[inds-1].values + RR_sunrise[inds+1].values) / 2
RR_sunrise = assign_LT_coord(RR_sunrise,dim_num=3).rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()	# Create a local time coordinte
# Group by local time and the 'dim' parameter ensures only the Time dimension is mean'd/averaged into the 'hour' dimension
RR_sunrise = xarray_reduce(RR_sunrise, 'LocalTime.hour', func='mean', dim='Time', expected_groups=(np.arange(0,24)), isbin=[False]).transpose('hour','west_east','south_north')
# Find the max and min values
x = RR_sunrise.max(dim='hour', keep_attrs=True)
y = RR_sunrise.min(dim='hour', keep_attrs=True)
amplitude_RR_sunrise = x-y
amplitude_RR_sunrise_land = xr.where(da_d02_LANDMASK==1, amplitude_RR_sunrise, np.nan)
amplitude_RR_sunrise_ocean = xr.where(da_d02_LANDMASK==0, amplitude_RR_sunrise, np.nan)

# Difference
amplitude_RR_diff =  amplitude_RR_sunrise - amplitude_RR_cntl
amplitude_RR_diff_land = xr.where(da_d02_LANDMASK==1, amplitude_RR_diff, np.nan)
amplitude_RR_diff_ocean = xr.where(da_d02_LANDMASK==0, amplitude_RR_diff, np.nan)


# In[ ]:


###############################################################################################################################
###### This version groupby is done day-by-day while the above version does a diurnal composite. Both perform similarly. ######
###############################################################################################################################

# Data
rolls = 5
lon_bound = [90,125]
lat_bound = [-10,10]
time_bound = ['2015-11-23T02','2015-12-02T00']

# Moving average to "denoise"
RR_cntl = da_d02_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy().chunk({'Time':5}).rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
RR_cntl = assign_LT_coord(RR_cntl,dim_num=3)	# Create a local time coordinte
# Group by local time and the 'dim' parameter ensures only the Time dimension is mean'd/averaged into the 'day' dimension
RR_cntl = xarray_reduce(RR_cntl, 'LocalTime.day', func='mean', dim='Time', expected_groups=(np.arange(0,24)), isbin=[False]).transpose('day','west_east','south_north')
# Find the max and min values
x = RR_cntl.max(dim='day', keep_attrs=True)
y = RR_cntl.min(dim='day', keep_attrs=True)
amplitude_RR_cntl = x-y
amplitude_RR_cntl_land = xr.where(da_d02_LANDMASK==1, amplitude_RR_cntl, np.nan)
amplitude_RR_cntl_ocean = xr.where(da_d02_LANDMASK==0, amplitude_RR_cntl, np.nan)

# First adjust data where there are negative RR's.
RR_sunrise = da_d02_sunrise_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy().chunk({'Time':5})
inds = np.unique(np.argwhere(RR_sunrise.values<0)[:,0])	# Time steps where RR is negative due to stitching over many simulations
RR_sunrise[inds] = (RR_sunrise[inds-1].values + RR_sunrise[inds+1].values) / 2
RR_sunrise = assign_LT_coord(RR_sunrise,dim_num=3).rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()	# Create a local time coordinte
# Group by local time and the 'dim' parameter ensures only the Time dimension is mean'd/averaged into the 'day' dimension
RR_sunrise = xarray_reduce(RR_sunrise, 'LocalTime.day', func='mean', dim='Time', expected_groups=(np.arange(0,24)), isbin=[False]).transpose('day','west_east','south_north')
# Find the max and min values
x = RR_sunrise.max(dim='day', keep_attrs=True)
y = RR_sunrise.min(dim='day', keep_attrs=True)
amplitude_RR_sunrise = x-y
amplitude_RR_sunrise_land = xr.where(da_d02_LANDMASK==1, amplitude_RR_sunrise, np.nan)
amplitude_RR_sunrise_ocean = xr.where(da_d02_LANDMASK==0, amplitude_RR_sunrise, np.nan)


# In[47]:


probability_density = True

a = plt.hist(amplitude_RR_diff.values.flatten(),bins=np.arange(-11,21,2), label=f'Domain n ={len(amplitude_RR_diff.values.flatten())}',density=probability_density, histtype='step', color='k')
b = plt.hist(amplitude_RR_diff_ocean.values.flatten(),bins=np.arange(-11,21,2), label=f'Ocean n = {np.nansum(~np.isnan(amplitude_RR_diff_ocean.values.flatten()))}',density=probability_density, histtype='step', color='dodgerblue')
c = plt.hist(amplitude_RR_diff_land.values.flatten(),bins=np.arange(-11,21,2), label=f'Land n = {np.nansum(~np.isnan(amplitude_RR_diff_land.values.flatten()))}',density=probability_density, histtype='step', color='peru')
plt.title(f'Diurnal RR Amplitude Difference (NCRF - CNTL)\nRolling n: {rolls}')
plt.xlabel('(mm/hr)')
if probability_density: plt.ylabel('Fractional count')
else: plt.ylabel('# of grids')
plt.legend()


# In[58]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

cf = amplitude_RR_cntl.plot.contourf(
        ax=ax1,
        add_colorbar=False,
        cmap='gray_r',
        levels=np.arange(0-.5,11+1+.5,1),
        vmin=0, vmax=11,
        x='west_east',
        y='south_north',
        xlim=[lon_bound[0], lon_bound[1]],
        ylim=[lat_bound[0], lat_bound[1]],
    )

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title('Control Diurnal Amplitude', loc='center')
ax1.set_title(f'Center smooth # of grids: {rolls}', loc='right')

ax2 = fig.add_subplot(gs[1,1])
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.arange(0,11+1,1))
cbar.set_label('Diurnal Amplitude (mm/day)')


# In[59]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

levels = [-10, 10]
dl = 1

cf = amplitude_RR_diff.plot.contourf(
    ax=ax1,
    add_colorbar=False,
    cmap='RdBu_r',
    # levels=np.arange(-10,11,1),
    levels=np.arange(levels[0]-(dl/2),levels[1]+dl+(dl/2),dl),
    vmin=levels[0], vmax=levels[1],
    x='west_east',
    y='south_north',
	extend='both',
    xlim=[lon_bound[0], lon_bound[1]],
    ylim=[lat_bound[0], lat_bound[1]],
)

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title('Diurnal Amplitude Difference (NCRF - CNTL)', loc='center')
ax1.set_title(f'Center smooth # of grids: {rolls}', loc='right')

ax2 = fig.add_subplot(gs[1,1])
# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(-10,12,2))
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(levels[0],levels[1]+1,dl+1))
cbar.minorticks_off()
cbar.set_label('Diurnal Amplitude Anomaly (mm/day)')


# ### Diurnal RR Phase Maximum

# In[60]:


# Data
rolls = 5
lon_bound = [90,125]
lat_bound = [-10,10]
time_bound = ['2015-11-23T02','2015-12-02T00']

# Moving average to "denoise"
data = da_d02_RR.sel(Time=slice(time_bound[0],time_bound[1])).rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
data = data.groupby('Time.hour').mean('Time')
# Find the time it maximizes
phase_max_cntl = data.idxmax('hour',keep_attrs=True)		# makes it 0-23

# Adjust data where there are negative RR's.
data = da_d02_sunrise_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy()
inds = np.unique(np.argwhere(data.values<0)[:,0])	# Time steps where RR is negative due to stitching over many simulations
data[inds] = (data[inds-1].values + data[inds+1].values) / 2
# Moving average to "denoise"
data = data.rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
data = data.groupby('Time.hour').mean('Time')
# Find the time it maximizes
phase_max_sunrise = data.idxmax('hour',keep_attrs=True)		# makes it 0-23

phase_max_diff =   phase_max_sunrise - phase_max_cntl

# When taking the difference of times, you need to make sure the distribution is between -12->11 or -11->12
	# depending on the order of the difference
phase_max_diff_adj = xr.where((phase_max_diff<=-13), phase_max_diff+24, phase_max_diff)
phase_max_diff_adj = xr.where((phase_max_diff>=12),phase_max_diff-24, phase_max_diff_adj)

phase_max_diff_adj_land = xr.where(da_d02_LANDMASK==1,phase_max_diff_adj,np.nan)
phase_max_diff_adj_ocean = xr.where(da_d02_LANDMASK==0,phase_max_diff_adj,np.nan)


# In[61]:


probability_density = True

a = plt.hist(phase_max_diff_adj.values.flatten(),bins=np.arange(-14.5,13.5,1), label=f'Domain n = {len(phase_max_diff_adj.values.flatten())}',density=probability_density, histtype='step', color='k')
b = plt.hist(phase_max_diff_adj_ocean.values.flatten(),bins=np.arange(-14.5,13.5,1), label=f'Ocean n = {np.nansum(~np.isnan(phase_max_diff_adj_ocean.values.flatten()))}',density=probability_density, histtype='step', color='dodgerblue')
c = plt.hist(phase_max_diff_adj_land.values.flatten(),bins=np.arange(-14.5,13.5,1), label=f'Land n = {np.nansum(~np.isnan(phase_max_diff_adj_land.values.flatten()))}',density=probability_density, histtype='step', color='peru')
plt.title(f'Diurnal Maximum Timing Difference (NCRF - CNTL)\nRolling n: {rolls}')
plt.xlabel('Hours')
if probability_density: plt.ylabel('Fractional count')
else: plt.ylabel('# of grids')
plt.legend()


# In[62]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

levels = [0,23]
dl = 1
cf = phase_max_cntl.plot.contourf(
    ax=ax1,
    add_colorbar=False,
    cmap='hsv',
    # levels=np.arange(levels[0],levels[1],dl),
    levels=np.arange(levels[0]-(dl/2),levels[1]+1+(dl/2),dl),
    # vmin=0, vmax=levels[1]-dl,
    vmin=levels[0], vmax=levels[1],
    x='west_east',
    y='south_north',
    xlim=[lon_bound[0], lon_bound[1]],
    ylim=[lat_bound[0], lat_bound[1]],
		extend='neither'
    )

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title('Control Diurnal Phase Maximum', loc='center')
ax1.set_title(f'Center smooth # of grids: {rolls}', loc='right')

ax2 = fig.add_subplot(gs[1,1])
# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.arange(0,24,3))
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='neither', ticks=np.arange(levels[0],levels[1]+1,dl+2))
cbar.minorticks_off()
cbar.set_label('UTC')


# In[63]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

levels = [-6,6]
dl = 1
cf = phase_max_diff_adj.plot.contourf(
    ax=ax1,
    add_colorbar=False,
    cmap='RdBu_r',
    levels=np.arange(levels[0]-(dl/2),levels[1]+1+(dl/2),dl),
    vmin=levels[0], vmax=levels[1],
    x='west_east',
    y='south_north',
    xlim=[lon_bound[0], lon_bound[1]],
    ylim=[lat_bound[0], lat_bound[1]],
		extend='both'
    )


# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title('Diurnal Maximum Timing Difference (NCRF - CNTL)', loc='center')
ax1.set_title(f'Center smooth # of grids: {rolls}', loc='right')

ax2 = fig.add_subplot(gs[1,1])
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(levels[0],levels[1]+1,dl+1))
cbar.minorticks_off()
cbar.set_label('Hours')


# ### Diurnal RR Phase >1mm/hr

# In[32]:


# Data
rolls = 5
lon_bound = [90,125]
lat_bound = [-10,10]
time_bound = ['2015-11-23T02','2015-12-02T00']

# Moving average to "denoise"
data = da_d02_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy().rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
data = data.groupby('Time.hour').mean('Time')

# Find the time it maximizes
phase_max_cntl = data.idxmax('hour',keep_attrs=True)
# Find the time it rains more than the set threshold rain rate while rain rate is increasing d(RR)/dt>0
	# Find rain rates >=threshold_RR, sort it going from lowest to highest, and extract the first index/hour
threshhold_RR = 1
threshold_dRRdt = 0
dRR_dt = xr.concat([data,data[0]], dim='hour').diff(dim='hour')
RR_above_thresh = xr.where((data>=threshhold_RR)&(dRR_dt>threshold_dRRdt), data, 999).values
initiation_hours = np.argsort(RR_above_thresh,axis=0)[0,:,:]
# If the RR is the fill value AND it is the first index of time, it is an area that did not recieve >threshold_RR rain rate
phase_initiation_cntl = np.where((RR_above_thresh[0,:,:]==999)&(initiation_hours==0), np.nan, initiation_hours)
phase_initiation_cntl = xr.DataArray(
	data=phase_initiation_cntl,
	dims=['south_north','west_east'],
	coords=dict(
		south_north = ('south_north', d02_coords['south_north'][1]),
		west_east = ('west_east', d02_coords['west_east'][1])),
)

# Moving average to "denoise"
data = da_d02_sunrise_RR.sel(Time=slice(time_bound[0],time_bound[1])).copy().rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
inds = np.unique(np.argwhere(data.values<0)[:,0])	# Time steps where RR is negative due to stitching over many simulations
data[inds] = (data[inds-1].values + data[inds+1].values) / 2
data = data.groupby('Time.hour').mean('Time')
# Find the time it maximizes
phase_max_sunrise = data.idxmax('hour',keep_attrs=True)
# Find the time it rains more than threshold rain rate
	# Find rain rates >=threshold_RR, sort it going from lowest to highest, and extract the first index/hour
dRR_dt = xr.concat([data,data[0]], dim='hour').diff(dim='hour')
RR_above_thresh = xr.where((data>=threshhold_RR)&(dRR_dt>threshold_dRRdt), data, 999).values
initiation_hours = np.argsort(RR_above_thresh,axis=0)[0,:,:]
# If the RR is the fill value AND it is the first index of time, it is an area that did not recieve >threshold_RR rain rate
phase_initiation_sunrise = np.where((RR_above_thresh[0,:,:]==999)&((initiation_hours==0)), np.nan, initiation_hours)
phase_initiation_sunrise = xr.DataArray(
	data=phase_initiation_sunrise,
	dims=['south_north','west_east'],
	coords=dict(
		south_north = ('south_north', d02_coords['south_north'][1]),
		west_east = ('west_east', d02_coords['west_east'][1])),
)

phase_initiation_diff = phase_initiation_sunrise - phase_initiation_cntl

# When taking the difference of times, you need to make sure the distribution is between -12->11 or -11->12
	# depending on the order of the difference
phase_initiation_diff_adj = xr.where((phase_initiation_diff<=-12), phase_initiation_diff+24, phase_initiation_diff)
phase_initiation_diff_adj = xr.where((phase_initiation_diff>=13), phase_initiation_diff-24, phase_initiation_diff_adj)

phase_initiation_diff_adj_land = xr.where(da_d02_LANDMASK==1,phase_initiation_diff_adj,np.nan)
phase_initiation_diff_adj_ocean = xr.where(da_d02_LANDMASK==0,phase_initiation_diff_adj,np.nan)


# In[96]:


probability_density = True

a = plt.hist(phase_initiation_diff_adj.values.flatten(),bins=np.arange(-13.5,14.5,1), label=f'Domain n = {np.nansum(~np.isnan(phase_initiation_diff_adj.values.flatten()))}',density=probability_density, histtype='step', color='k')
b = plt.hist(phase_initiation_diff_adj_ocean.values.flatten(),bins=np.arange(-13.5,14.5,1), label=f'Ocean n = {np.nansum(~np.isnan(phase_initiation_diff_adj_ocean.values.flatten()))}',density=probability_density, histtype='step', color='dodgerblue')
c = plt.hist(phase_initiation_diff_adj_land.values.flatten(),bins=np.arange(-13.5,14.5,1), label=f'Land n = {np.nansum(~np.isnan(phase_initiation_diff_adj_land.values.flatten()))}',density=probability_density, histtype='step', color='peru')
plt.title(f'Diurnal Initiation Timing Difference (NCRF - CNTL)\n Rolling n: {rolls}, Threshold: >{threshhold_RR}mm/d')
plt.xlabel('Hours')

if probability_density: plt.ylabel('Fractional count'), plt.ylim([0,.2])
else: plt.ylabel('# of grids')
plt.legend()


# In[66]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

levels = [0,23]
dl = 1
cf = phase_initiation_cntl.plot.contourf(
    ax=ax1,
    add_colorbar=False,
    cmap='hsv',
    # levels=np.arange(levels[0],levels[1],dl),
    levels=np.arange(levels[0]-(dl/2),levels[1]+1+(dl/2),dl),
    # vmin=0, vmax=levels[1]-dl,
    vmin=levels[0], vmax=levels[1],
    x='west_east',
    y='south_north',
    xlim=[lon_bound[0], lon_bound[1]],
    ylim=[lat_bound[0], lat_bound[1]],
		extend='neither'
    )

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title(f'Control Diurnal Initiation Time (>{threshhold_RR}mm/d)', loc='center')
ax1.set_title(f'Center smooth # of grids: {rolls}', loc='right')

ax2 = fig.add_subplot(gs[1,1])
# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.arange(0,24,3))
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='neither', ticks=np.arange(levels[0],levels[1]+1,dl+2))
cbar.minorticks_off()
cbar.set_label('UTC')


# In[186]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

levels = [-12,12]
levels = [-6,6]
dl = 1
rolls = 1
x = phase_initiation_diff_adj.rolling({'south_north':rolls, 'west_east':rolls}, min_periods=1, center=True).mean()
cf = x.plot.contourf(
    ax=ax1,
    add_colorbar=False,
    cmap='RdBu_r',
    levels=np.arange(levels[0]-(dl/2),levels[1]+1+(dl/2),dl),
    vmin=levels[0], vmax=levels[1],
    x='west_east',
    y='south_north',
    xlim=[lon_bound[0], lon_bound[1]],
    ylim=[lat_bound[0], lat_bound[1]],
		extend='both'
    )


# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title(f'Diurnal Initiation Timing Difference\n(>{threshhold_RR}mm/d) (NCRF - CNTL)', loc='center')
ax1.set_title(f'Center smooth # of grids: {rolls}', loc='right')

ax2 = fig.add_subplot(gs[1,1])
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(levels[0],levels[1]+1,dl+1))
cbar.minorticks_off()
cbar.set_label('Hours')


# ### Diurnal RR Vs HFX Amplitude

# In[100]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

fs = 18


# Diurnal Composites
	# Rain Rate Amplitude (RR)
	# Sensible Heat Flux (HFX)

## Calculate linear regressions	##
min_lag = 0
max_lag = 0
lag_plot = 0				# The lag relationship plotted on the first row
domain = 'Land'

if domain=='Land':
	# Land
	da2 = linreg(amplitude_RR_diff_land.values.flatten()[~np.isnan(amplitude_RR_diff_land.values.flatten())], amplitude_HFX_diff_land.values.flatten()[~np.isnan(amplitude_HFX_diff_land.values.flatten())], min_lag, max_lag)
elif domain=='Ocean':
	# Ocean
	da2 = linreg(amplitude_RR_diff_ocean.values.flatten()[~np.isnan(amplitude_RR_diff_ocean.values.flatten())], amplitude_HFX_diff_ocean.values.flatten()[~np.isnan(amplitude_HFX_diff_ocean.values.flatten())], min_lag, max_lag)
elif domain=='Net':
	# Net
	da2 = linreg(amplitude_RR_diff.values.flatten()[~np.isnan(amplitude_RR_diff.values.flatten())], amplitude_HFX_diff.values.flatten()[~np.isnan(amplitude_HFX_diff.values.flatten())], min_lag, max_lag)

## Plot linear regressions	##
if domain=='Land':
	s1 = ax1.scatter(amplitude_RR_diff_land, amplitude_HFX_diff_land,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Ocean':
	s1 = ax1.scatter(amplitude_RR_diff_ocean, amplitude_HFX_diff_ocean,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Net':
	s1 = ax1.scatter(amplitude_RR_diff, amplitude_HFX_diff,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)

l2 = ax1.plot([-200,200],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-200, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*200],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(-7.5,255, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Diurnal RR and HFX Amplitude Relationship\nover the MC {domain}'
ax1.set_title(title, fontsize=fs)
ax1.set_xlabel('Anomalous RR Amplitude [$mm/hr$]',fontsize=10)
ax1.set_ylabel('Anomalous HFX Ampltude [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([-10,20])
ax1.set_ylim([-50,300])
ax1.grid(linestyle='--', axis='both', linewidth=1)


# In[101]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

fs = 18

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# Diurnal Composites
	# Rain Rate Amplitude (RR)
	# Sensible Heat Flux (HFX)

## Calculate linear regressions	##
min_lag = 0
max_lag = 0
lag_plot = 0				# The lag relationship plotted on the first row
domain = 'Land'

if domain=='Land':
	# Land
	da2 = linreg(amplitude_SWSFC_diff_land.values.flatten()[~np.isnan(amplitude_SWSFC_diff_land.values.flatten())], amplitude_HFX_diff_land.values.flatten()[~np.isnan(amplitude_HFX_diff_land.values.flatten())], min_lag, max_lag)
elif domain=='Ocean':
	# Ocean
	da2 = linreg(amplitude_SWSFC_diff_ocean.values.flatten()[~np.isnan(amplitude_SWSFC_diff_ocean.values.flatten())], amplitude_HFX_diff_ocean.values.flatten()[~np.isnan(amplitude_HFX_diff_ocean.values.flatten())], min_lag, max_lag)
elif domain=='Net':
	# Net
	da2 = linreg(amplitude_SWSFC_diff.values.flatten()[~np.isnan(amplitude_SWSFC_diff.values.flatten())], amplitude_HFX_diff.values.flatten()[~np.isnan(amplitude_HFX_diff.values.flatten())], min_lag, max_lag)

## Plot linear regressions	##
if domain=='Land':
	s1 = ax1.scatter(amplitude_SWSFC_diff_land, amplitude_HFX_diff_land,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Ocean':
	s1 = ax1.scatter(amplitude_SWSFC_diff_ocean, amplitude_HFX_diff_ocean,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Net':
	s1 = ax1.scatter(amplitude_SWSFC_diff, amplitude_HFX_diff,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)

l2 = ax1.plot([-50,1000],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-50, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*1000],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(25, 275, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
# ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Diurnal SW SFC and HFX Amplitude Relationship\nover the MC {domain}'
ax1.set_title(title, fontsize=fs)
ax1.set_xlabel('Anomalous SW SFC Amplitude [$W/m^{2}$]',fontsize=10)
ax1.set_ylabel('Anomalous HFX Amplitude [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([0,700])
ax1.set_ylim([-50,300])
ax1.grid(linestyle='--', axis='both', linewidth=1)


# In[60]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

fs = 18

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# Diurnal Composites
	# Rain Rate Amplitude (RR)
	# Sensible Heat Flux (HFX)

## Calculate linear regressions	##
min_lag = 0
max_lag = 0
lag_plot = 0				# The lag relationship plotted on the first row
domain = 'Land'

if domain=='Land':
	# Land
	da2 = linreg(amplitude_SWSFC_diff_land.values.flatten()[~np.isnan(amplitude_SWSFC_diff_land.values.flatten())], amplitude_HFX_diff_land.values.flatten()[~np.isnan(amplitude_HFX_diff_land.values.flatten())], min_lag, max_lag)
elif domain=='Ocean':
	# Ocean
	da2 = linreg(amplitude_SWSFC_diff_ocean.values.flatten()[~np.isnan(amplitude_SWSFC_diff_ocean.values.flatten())], amplitude_HFX_diff_ocean.values.flatten()[~np.isnan(amplitude_HFX_diff_ocean.values.flatten())], min_lag, max_lag)
elif domain=='Net':
	# Net
	da2 = linreg(amplitude_SWSFC_diff.values.flatten()[~np.isnan(amplitude_SWSFC_diff.values.flatten())], amplitude_HFX_diff.values.flatten()[~np.isnan(amplitude_HFX_diff.values.flatten())], min_lag, max_lag)

## Plot linear regressions	##
if domain=='Land':
	s1 = ax1.scatter(amplitude_SWSFC_diff_land, amplitude_HFX_diff_land,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Ocean':
	s1 = ax1.scatter(amplitude_SWSFC_diff_ocean, amplitude_HFX_diff_ocean,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Net':
	s1 = ax1.scatter(amplitude_SWSFC_diff, amplitude_HFX_diff,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)

l2 = ax1.plot([-50,1000],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-50, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*1000],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(25, 425, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
# ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Daily SW SFC and HFX Amplitude Relationship\nover the MC {domain}'
ax1.set_title(title, fontsize=fs)
ax1.set_xlabel('Anomalous SW SFC Amplitude [$W/m^{2}$]',fontsize=10)
ax1.set_ylabel('Anomalous HFX Ampltude [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([0,800])
ax1.set_ylim([-50,300])
ax1.grid(linestyle='--', axis='both', linewidth=1)


# ### RR Amplitude vs Initiation

# In[65]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[0.9,0.05], hspace=0.3, wspace=0.05)
ax1 = fig.add_subplot(gs[0,0])

fs = 18

# Diurnal Composites
	# Rain Rate Amplitude (RR)
	# Sensible Heat Flux (HFX)

## Calculate linear regressions	##
min_lag = 0
max_lag = 0
lag_plot = 0				# The lag relationship plotted on the first row
domain = 'Land'
probability_density = True

y_levels = [-10, 30]
y_dl = 1

if probability_density:
	c_levels = [10**-6,0.02]
	c_dl = .002
else:
	c_levels = [0,5000]
	c_dl = 500

if domain=='Land':
	# Land
	x=phase_initiation_diff_adj_land.values.flatten()[~np.isnan(phase_initiation_diff_adj_land.values.flatten())]
	y=amplitude_RR_diff_land.values.flatten()[~np.isnan(phase_initiation_diff_adj_land.values.flatten())]
	da2 = linreg(x, y, min_lag, max_lag)
elif domain=='Ocean':
	# Ocean
	da2 = linreg(phase_initiation_diff_adj_ocean.values.flatten()[~np.isnan(phase_initiation_diff_adj_ocean.values.flatten())], amplitude_RR_diff_ocean.values.flatten()[~np.isnan(amplitude_RR_diff_ocean.values.flatten())], min_lag, max_lag)
elif domain=='Net':
	# Net
	da2 = linreg(phase_initiation_diff_adj.values.flatten()[~np.isnan(phase_initiation_diff_adj.values.flatten())], amplitude_RR_diff.values.flatten()[~np.isnan(amplitude_RR_diff.values.flatten())], min_lag, max_lag)

## Plot linear regressions	##
if domain=='Land':
	H, xedges, yedges = np.histogram2d(x=x, y=y, bins=[np.arange(-11.5,12.5,1), np.arange(y_levels[0],y_levels[1]+y_dl,y_dl)], density=probability_density)
	X, Y = np.meshgrid(xedges, yedges)
	cf = ax1.pcolormesh(X, Y, H.T, cmap='OrRd', norm='log', shading='auto', vmin=c_levels[0], vmax=c_levels[1])
	# cf = ax1.pcolormesh(X, Y, H.T, cmap='OrRd', norm='log', vmin=c_levels[0], vmax=c_levels[1])
	# s1 = ax1.scatter(phase_initiation_diff_adj_land, amplitude_RR_diff_land,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Ocean':
	s1 = ax1.scatter(phase_initiation_diff_adj_ocean, amplitude_RR_diff_ocean,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Net':
	s1 = ax1.scatter(phase_initiation_diff_adj, amplitude_RR_diff,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)

# l2 = ax1.plot([-200,200],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-200, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*200],'r',linestyle='--')
# slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
# yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
# r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
# ax1.text(-35,255, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Diurnal RR and Early Initiation Difference (NCRF - CNTL)\nover the MC {domain}'
ax1.set_title(title, fontsize=fs)
ax1.set_ylabel('Anomalous RR Amplitude [$mm/hr$]',fontsize=10)
ax1.set_xlabel('Initiation Difference [$hr$]',fontsize=10)
ax1.set_xticks(np.arange(-12,13,2))
ax1.set_xlim([-11,11])
ax1.set_ylim([y_levels[0],y_levels[1]])
ax1.grid(linestyle='--', axis='both', linewidth=1)


ax2 = fig.add_subplot(gs[1,0])
# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.arange(c_levels[0],c_levels[1]+c_dl, c_dl))
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.array([0,10**-6,10**-5,10**-4,10**-3,10**-2]))
cbar.minorticks_off()
if probability_density:
	cbar.set_label('Fractional Frequnecy')
else:
	cbar.set_label('Frequnecy')


# In[122]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[0.9,0.05], hspace=0.3, wspace=0.05)
ax1 = fig.add_subplot(gs[0,0])

fs = 18


# Diurnal Composites
	# Rain Rate Amplitude (RR)
	# Sensible Heat Flux (HFX)

## Calculate linear regressions	##
min_lag = 0
max_lag = 0
lag_plot = 0				# The lag relationship plotted on the first row
domain = 'Land'
probability_density = False

y_levels = [0, 30]
y_dl = 1

if probability_density:
	c_levels = [10**-6,0.02]
	c_dl = .002
else:
	c_levels = [0,5000]
	c_dl = 500

if domain=='Land':
	# Land
	x=xr.where(da_d02_LANDMASK==1,phase_initiation_cntl,np.nan).values.flatten()[~np.isnan(phase_initiation_cntl.values.flatten())]
	y=amplitude_RR_cntl_land.values.flatten()[~np.isnan(phase_initiation_cntl.values.flatten())]
	da2 = linreg(x, y, min_lag, max_lag)
elif domain=='Ocean':
	# Ocean
	da2 = linreg(phase_initiation_diff_adj_ocean.values.flatten()[~np.isnan(phase_initiation_diff_adj_ocean.values.flatten())], amplitude_RR_diff_ocean.values.flatten()[~np.isnan(amplitude_RR_diff_ocean.values.flatten())], min_lag, max_lag)
elif domain=='Net':
	# Net
	da2 = linreg(phase_initiation_diff_adj.values.flatten()[~np.isnan(phase_initiation_diff_adj.values.flatten())], amplitude_RR_diff.values.flatten()[~np.isnan(amplitude_RR_diff.values.flatten())], min_lag, max_lag)

## Plot linear regressions	##
if domain=='Land':
	H_cntl, xedges, yedges = np.histogram2d(x=x, y=y, bins=[np.arange(-0.5,24.5,1), np.arange(y_levels[0],y_levels[1]+y_dl,y_dl)], density=probability_density)
	X, Y = np.meshgrid(xedges, yedges)
	if probability_density:
		cf = ax1.pcolormesh(X, Y, H_cntl.T, cmap='OrRd', norm='log', shading='auto', vmin=c_levels[0], vmax=c_levels[1])
	else:
		cf = ax1.pcolormesh(X, Y, H_cntl.T, cmap='OrRd', norm='linear', shading='auto', vmin=c_levels[0], vmax=c_levels[1])
elif domain=='Ocean':
	s1 = ax1.scatter(phase_initiation_diff_adj_ocean, amplitude_RR_diff_ocean,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Net':
	s1 = ax1.scatter(phase_initiation_diff_adj, amplitude_RR_diff,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)

# l2 = ax1.plot([-200,200],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-200, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*200],'r',linestyle='--')
# slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
# yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
# r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
# ax1.text(-35,255, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
# ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Diurnal RR Amplitude and Initiation Time: CNTL\nover the MC {domain}'
ax1.set_title(title, fontsize=fs)
ax1.set_ylabel('RR Amplitude [$mm/hr$]',fontsize=10)
ax1.set_xlabel('Initiation Time (UTC)',fontsize=10)
ax1.set_xticks(np.arange(0,25,3))
ax1.set_xlim([0,23])
ax1.set_ylim([y_levels[0],y_levels[1]])
ax1.grid(linestyle='--', axis='both', linewidth=1)


ax2 = fig.add_subplot(gs[1,0])

if probability_density:
	cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.array([0,10**-6,10**-5,10**-4,10**-3,10**-2]))
	cbar.set_label('Fractional Frequnecy')
else:
	cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.arange(c_levels[0],c_levels[1]+c_dl, c_dl))
	cbar.set_label('Frequnecy')
# cbar.minorticks_off()


# In[120]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[0.9,0.05], hspace=0.3, wspace=0.05)
ax1 = fig.add_subplot(gs[0,0])

fs = 18


# Diurnal Composites
	# Rain Rate Amplitude (RR)
	# Sensible Heat Flux (HFX)

## Calculate linear regressions	##
min_lag = 0
max_lag = 0
lag_plot = 0				# The lag relationship plotted on the first row
domain = 'Land'
probability_density = False

y_levels = [0, 30]
y_dl = 1

if probability_density:
	c_levels = [10**-6,0.012]
	c_dl = .002
else:
	c_levels = [0,5000]
	c_dl = 500

if domain=='Land':
	# Land
	x=xr.where(da_d02_LANDMASK==1,phase_initiation_sunrise,np.nan).values.flatten()[~np.isnan(phase_initiation_sunrise.values.flatten())]
	y=amplitude_RR_sunrise_land.values.flatten()[~np.isnan(phase_initiation_sunrise.values.flatten())]
	da2 = linreg(x, y, min_lag, max_lag)
elif domain=='Ocean':
	# Ocean
	da2 = linreg(phase_initiation_diff_adj_ocean.values.flatten()[~np.isnan(phase_initiation_diff_adj_ocean.values.flatten())], amplitude_RR_diff_ocean.values.flatten()[~np.isnan(amplitude_RR_diff_ocean.values.flatten())], min_lag, max_lag)
elif domain=='Net':
	# Net
	da2 = linreg(phase_initiation_diff_adj.values.flatten()[~np.isnan(phase_initiation_diff_adj.values.flatten())], amplitude_RR_diff.values.flatten()[~np.isnan(amplitude_RR_diff.values.flatten())], min_lag, max_lag)

## Plot linear regressions	##
if domain=='Land':
	H_sunrise, xedges, yedges = np.histogram2d(x=x, y=y, bins=[np.arange(-0.5,24.5,1), np.arange(y_levels[0],y_levels[1]+y_dl,y_dl)], density=probability_density)
	X, Y = np.meshgrid(xedges, yedges)
	if probability_density:
		cf = ax1.pcolormesh(X, Y, H_sunrise.T, cmap='OrRd', norm='log', shading='auto', vmin=c_levels[0], vmax=c_levels[1])
	else:
		cf = ax1.pcolormesh(X, Y, H_sunrise.T, cmap='OrRd', norm='linear', shading='auto', vmin=c_levels[0], vmax=c_levels[1])
	# cf = ax1.pcolormesh(X, Y, H.T, cmap='OrRd', norm='log', vmin=c_levels[0], vmax=c_levels[1])
	# s1 = ax1.scatter(phase_initiation_diff_adj_land, amplitude_RR_diff_land,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Ocean':
	s1 = ax1.scatter(phase_initiation_diff_adj_ocean, amplitude_RR_diff_ocean,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Net':
	s1 = ax1.scatter(phase_initiation_diff_adj, amplitude_RR_diff,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)

# l2 = ax1.plot([-200,200],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-200, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*200],'r',linestyle='--')
# slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
# yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
# r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
# ax1.text(-35,255, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
# ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Diurnal RR Amplitude and Initiation Time: NCRF\nover the MC {domain}'
ax1.set_title(title, fontsize=fs)
ax1.set_ylabel('RR Amplitude [$mm/hr$]',fontsize=10)
ax1.set_xlabel('Initiation Time (UTC)',fontsize=10)
ax1.set_xticks(np.arange(0,25,3))
ax1.set_xlim([0,23])
ax1.set_ylim([y_levels[0],y_levels[1]])
ax1.grid(linestyle='--', axis='both', linewidth=1)


ax2 = fig.add_subplot(gs[1,0])
# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.arange(c_levels[0],c_levels[1]+c_dl, c_dl))
# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.array([0,10**-6,10**-5,10**-4,10**-3,10**-2]))
# cbar.minorticks_off()
if probability_density:
	cbar.set_label('Fractional Frequnecy')
	cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.array([0,10**-6,10**-5,10**-4,10**-3,10**-2]))
else:
	cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='max', ticks=np.arange(c_levels[0],c_levels[1]+c_dl, c_dl))
	cbar.set_label('Frequnecy')


# In[121]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[0.9,0.05], hspace=0.3, wspace=0.05)
ax1 = fig.add_subplot(gs[0,0])

fs = 18


# Diurnal Composites
	# Rain Rate Amplitude (RR)
	# Sensible Heat Flux (HFX)

## Calculate linear regressions	##
min_lag = 0
max_lag = 0
lag_plot = 0				# The lag relationship plotted on the first row
domain = 'Land'
probability_density = True

y_levels = [0, 30]
y_dl = 1

if probability_density:
	c_levels = [-0.01,0.01]
	c_dl = .002
else:
	c_levels = [0,5000]
	c_dl = 500

if domain=='Land':
	# Land
	x=xr.where(da_d02_LANDMASK==1,phase_initiation_sunrise,np.nan).values.flatten()[~np.isnan(phase_initiation_sunrise.values.flatten())]
	y=amplitude_RR_sunrise_land.values.flatten()[~np.isnan(phase_initiation_sunrise.values.flatten())]
	da2 = linreg(x, y, min_lag, max_lag)
elif domain=='Ocean':
	# Ocean
	da2 = linreg(phase_initiation_diff_adj_ocean.values.flatten()[~np.isnan(phase_initiation_diff_adj_ocean.values.flatten())], amplitude_RR_diff_ocean.values.flatten()[~np.isnan(amplitude_RR_diff_ocean.values.flatten())], min_lag, max_lag)
elif domain=='Net':
	# Net
	da2 = linreg(phase_initiation_diff_adj.values.flatten()[~np.isnan(phase_initiation_diff_adj.values.flatten())], amplitude_RR_diff.values.flatten()[~np.isnan(amplitude_RR_diff.values.flatten())], min_lag, max_lag)

## Plot linear regressions	##
if domain=='Land':
	H, xedges, yedges = np.histogram2d(x=x, y=y, bins=[np.arange(-0.5,24.5,1), np.arange(y_levels[0],y_levels[1]+y_dl,y_dl)], density=probability_density)
	X, Y = np.meshgrid(xedges, yedges)
	cf = ax1.pcolormesh(X, Y, H_sunrise.T-H_cntl.T, cmap='RdBu_r', norm=mpl.colors.SymLogNorm(linthresh=0.001, linscale=1,
                                              vmin=c_levels[0], vmax=c_levels[1], base=10), shading='auto')
	# cf = ax1.pcolormesh(X, Y, H.T, cmap='OrRd', norm='log', vmin=c_levels[0], vmax=c_levels[1])
	# s1 = ax1.scatter(phase_initiation_diff_adj_land, amplitude_RR_diff_land,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Ocean':
	s1 = ax1.scatter(phase_initiation_diff_adj_ocean, amplitude_RR_diff_ocean,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Net':
	s1 = ax1.scatter(phase_initiation_diff_adj, amplitude_RR_diff,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)

# l2 = ax1.plot([-200,200],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-200, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*200],'r',linestyle='--')
# slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
# yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
# r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
# ax1.text(-35,255, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
# ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Diurnal RR Amplitude and Initiation Time : NCRF-CNTL\nover the MC {domain}'
ax1.set_title(title, fontsize=fs)
ax1.set_ylabel('RR Amplitude [$mm/hr$]',fontsize=10)
ax1.set_xlabel('Initiation Time (UTC)',fontsize=10)
ax1.set_xticks(np.arange(0,25,3))
ax1.set_xlim([0,23])
ax1.set_ylim([y_levels[0],y_levels[1]])
ax1.grid(linestyle='--', axis='both', linewidth=1)


ax2 = fig.add_subplot(gs[1,0])
if probability_density:
	
	cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.array([-10**-2,-10**-3,0,10**-3,10**-2]))
	# cbar.set_ticklabels(np.array(c_levels[0]))
	# cbar.set_ticklabels(np.arange(levels[0],levels[1]+dl,dl*5), fontsize=fs-4)
	cbar.set_label('Fractional Frequnecy')
else:
	# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(c_levels[0],c_levels[1]+c_dl, c_dl))
	# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.array([-10**-1,-10**-2,-10**-3,-10**-4,-10**-5,-10**-6,0,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]))
	# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(-.01,.01+.002, .002))
	cbar.set_label('Frequnecy')


# In[117]:


fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[0.9,0.05], hspace=0.3, wspace=0.05)
ax1 = fig.add_subplot(gs[0,2])

fs = 18

# Diurnal Composites
	# Rain Rate Amplitude (RR)
	# Sensible Heat Flux (HFX)

## Calculate linear regressions	##
min_lag = 0
max_lag = 0
lag_plot = 0				# The lag relationship plotted on the first row
domain = 'Land'
probability_density = True

y_levels = [0, 30]
y_dl = 1

if probability_density:
	c_levels = [-0.01,0.01]
	c_dl = .002
else:
	c_levels = [0,5000]
	c_dl = 500

if domain=='Land':
	# Land
	x=xr.where(da_d02_LANDMASK==1,phase_initiation_sunrise,np.nan).values.flatten()[~np.isnan(phase_initiation_sunrise.values.flatten())]
	y=amplitude_RR_sunrise_land.values.flatten()[~np.isnan(phase_initiation_sunrise.values.flatten())]
	da2 = linreg(x, y, min_lag, max_lag)
elif domain=='Ocean':
	# Ocean
	da2 = linreg(phase_initiation_diff_adj_ocean.values.flatten()[~np.isnan(phase_initiation_diff_adj_ocean.values.flatten())], amplitude_RR_diff_ocean.values.flatten()[~np.isnan(amplitude_RR_diff_ocean.values.flatten())], min_lag, max_lag)
elif domain=='Net':
	# Net
	da2 = linreg(phase_initiation_diff_adj.values.flatten()[~np.isnan(phase_initiation_diff_adj.values.flatten())], amplitude_RR_diff.values.flatten()[~np.isnan(amplitude_RR_diff.values.flatten())], min_lag, max_lag)

## Plot linear regressions	##
if domain=='Land':
	H, xedges, yedges = np.histogram2d(x=x, y=y, bins=[np.arange(-0.5,24.5,1), np.arange(y_levels[0],y_levels[1]+y_dl,y_dl)], density=probability_density)
	X, Y = np.meshgrid(xedges, yedges)
	cf = ax1.pcolormesh(X, Y, H_sunrise.T-H_cntl.T, cmap='RdBu_r', norm=mpl.colors.SymLogNorm(linthresh=0.001, linscale=1,
                                              vmin=c_levels[0], vmax=c_levels[1], base=10), shading='auto')
	# cf = ax1.pcolormesh(X, Y, H.T, cmap='OrRd', norm='log', vmin=c_levels[0], vmax=c_levels[1])
	# s1 = ax1.scatter(phase_initiation_diff_adj_land, amplitude_RR_diff_land,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Ocean':
	s1 = ax1.scatter(phase_initiation_diff_adj_ocean, amplitude_RR_diff_ocean,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)
elif domain=='Net':
	s1 = ax1.scatter(phase_initiation_diff_adj, amplitude_RR_diff,s=15,edgecolors='0.5', alpha=0.5, linewidths=.15)

# l2 = ax1.plot([-200,200],[da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*-200, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*200],'r',linestyle='--')
# slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
# yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
# r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
# ax1.text(-35,255, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
# ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
title = f'Diurnal RR Amplitude and Initiation Time : NCRF-CNTL\nover the MC {domain}'
ax1.set_title(title, fontsize=fs)
ax1.set_ylabel('RR Amplitude [$mm/hr$]',fontsize=10)
ax1.set_xlabel('Initiation Time (UTC)',fontsize=10)
ax1.set_xticks(np.arange(0,25,3))
ax1.set_xlim([0,23])
ax1.set_ylim([y_levels[0],y_levels[1]])
ax1.grid(linestyle='--', axis='both', linewidth=1)


ax2 = fig.add_subplot(gs[1,2])
if probability_density:
	
	cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.array([-10**-2,-10**-3,0,10**-3,10**-2]))
	# cbar.set_ticklabels(np.array(c_levels[0]))
	# cbar.set_ticklabels(np.arange(levels[0],levels[1]+dl,dl*5), fontsize=fs-4)
	cbar.set_label('Fractional Frequnecy')
else:
	# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(c_levels[0],c_levels[1]+c_dl, c_dl))
	# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.array([-10**-1,-10**-2,-10**-3,-10**-4,-10**-5,-10**-6,0,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1]))
	# cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(-.01,.01+.002, .002))
	cbar.set_label('Frequnecy')


# ### Distance from Coast

# In[ ]:


from scipy.ndimage import label, sum as ndi_sum


# In[169]:


# d02
da_d02_LU_INDEX = ds_d02['LU_INDEX'].sel(Time=slice(1)).compute().squeeze()   # Land = 1, Water = 0
da_d02_LU_INDEX = da_d02_LU_INDEX.assign_coords(south_north=('south_north',da_d02_LU_INDEX.XLAT[:,0].values),
	west_east=('west_east',da_d02_LU_INDEX.XLONG[0,:].values))
# d02
da_d02_LAKEMASK = ds_d02['LAKEMASK'].sel(Time=slice(1)).compute().squeeze()   # Lake = 1, Non-Lake = 0
da_d02_LAKEMASK = da_d02_LAKEMASK.assign_coords(south_north=('south_north',da_d02_LAKEMASK.XLAT[:,0].values),
	west_east=('west_east',da_d02_LAKEMASK.XLONG[0,:].values))
# d02
da_d02_LANDMASK = ds_d02['LANDMASK'].sel(Time=slice(1)).compute().squeeze()   # Land = 1, Water = 0
da_d02_LANDMASK = xr.where((da_d02_LU_INDEX==17), 0, 1)		# Index 17 is specifically ocean
da_d02_LANDMASK = xr.where((da_d02_LAKEMASK==1), 1, da_d02_LANDMASK)	# If it's a lake, change change landmask to land since it's surrounded by land and no ocean.
da_d02_LANDMASK = da_d02_LANDMASK.assign_coords(south_north=('south_north',da_d02_LANDMASK.XLAT[:,0].values),
	west_east=('west_east',da_d02_LANDMASK.XLONG[0,:].values))


# In[170]:


## Removes small islands
from scipy.ndimage import label, sum as ndi_sum

def remove_small_clusters(matrix, min_cluster_size):
    # Label the connected components (clusters) in the matrix
    labeled_matrix, num_features = label(matrix)

    # Calculate the size of each cluster
    cluster_sizes = ndi_sum(matrix, labeled_matrix, index=np.arange(1, num_features + 1))

    # Remove clusters smaller than the min_cluster_size
    for cluster_num, size in enumerate(cluster_sizes, start=1):
        if size < min_cluster_size:
            matrix[labeled_matrix == cluster_num] = 0

    return matrix, cluster_sizes

min_cluster_size = 100  # 600 removes nearly all small islands but results don't indicate a huge change in off-shore features
da_d02_LANDMASK.data, cluster_sizes = remove_small_clusters(da_d02_LANDMASK.values, min_cluster_size)


distance_from_coast = da_d02_LANDMASK.copy(deep=True)
# scipy.ndimage.distance_transform_edt only works on values of 1 (land)
	# Where it is 1 (land), do the transform as usual, where it is 0 (ocean), change 0's to 1's and 1's to 0's, and then do the transform
	# Multiply by the horizontal resolution to get accurate distance from coast values
distance_from_coast.data = -xr.where(distance_from_coast==1, scipy.ndimage.distance_transform_edt(da_d02_LANDMASK)*3, scipy.ndimage.distance_transform_edt(xr.where(da_d02_LANDMASK==0, 1, 0))*-3)# multipy by 3 km to match distance from coast

# distance_from_coast.data = -xr.where(distance_from_coast==1, scipy.ndimage.distance_transform_bf(da_d02_LANDMASK, metric='taxicab', return_indices=False)*3, scipy.ndimage.distance_transform_bf(xr.where(da_d02_LANDMASK==0, 1, 0), metric='taxicab', return_indices=False)*-3)# multipy by 3 km to match distance from coast

distance_from_coast.name = 'DistFromCoast'
distance_from_coast = distance_from_coast.assign_attrs(Units='km', description='Negative for land, Postive for ocean')
distance_from_coast = distance_from_coast.assign_coords(south_north=('south_north',distance_from_coast.XLAT[:,0].values),
	west_east=('west_east',distance_from_coast.XLONG[0,:].values))

# smoothing_num=40
# distance_from_coast = distance_from_coast.rolling(south_north=smoothing_num, west_east=smoothing_num, min_periods=1, center=True).mean()


# In[ ]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))
fs=18

lon_bound = [90,125]
lat_bound = [-10,10]

levels = [-250, 250]
dl = 10

cf = distance_from_coast.plot.contourf(
  ax=ax1,
  add_colorbar=False,
  cmap='RdBu',
  levels=np.arange(levels[0],levels[1]+dl,dl),
  x='west_east',
  y='south_north',
  vmin=levels[0], vmax=levels[1],
  xlim=[lon_bound[0], lon_bound[1]],
  ylim=[lat_bound[0], lat_bound[1]],
  extend='both'
  # extend='neither'
  )

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels, fontsize=fs-4)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels, fontsize=fs-4)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title(f'Distance from Coast with <{min_cluster_size*(3**2)}km^2 islands removed', loc='center', fontsize=fs)

ax2 = fig.add_subplot(gs[1,1])
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(levels[0],levels[1]+dl,dl*5))
cbar.set_ticklabels(np.arange(levels[0],levels[1]+dl,dl*5), fontsize=fs-4)
cbar.minorticks_off()
cbar.set_label('Kilometers', fontsize=fs-4)


# In[ ]:


l = plt.hist(distance_from_coast.values.flatten(), bins=np.arange(-300-(100/2),1100+(100/2),100), density=True)


# In[ ]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

lon_bound = [90,125]
lat_bound = [-10,10]

# levels = [-250, 250]
levels = [0, 17]
dl = 1

cf = da_d02_LU_INDEX.plot.contourf(
  ax=ax1,
  add_colorbar=False,
  cmap='jet',
  levels=np.arange(levels[0],levels[1]+dl,dl),
  x='west_east',
  y='south_north',
  vmin=-250, vmax=250,
  xlim=[lon_bound[0], lon_bound[1]],
  ylim=[lat_bound[0], lat_bound[1]],
  extend='both'
  # extend='neither'
  )

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title(f'Land Use Index', loc='center')

ax2 = fig.add_subplot(gs[1,1])
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(levels[0],levels[1]+dl, dl))
cbar.minorticks_off()
cbar.set_label('Km')


# In[ ]:


fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0, height_ratios=[.95,.05],width_ratios=[.15,.7,.15])
ax1 = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree(central_longitude=0))

lon_bound = [90,125]
lat_bound = [-10,10]

# levels = [-250, 250]
levels = [0, 1]
dl = .5

cf = da_d02_LAKEMASK.plot.contourf(
  ax=ax1,
  add_colorbar=False,
  cmap='RdBu',
  levels=np.arange(levels[0],levels[1]+dl,dl),
  x='west_east',
  y='south_north',
  vmin=0, vmax=1,
  xlim=[lon_bound[0], lon_bound[1]],
  ylim=[lat_bound[0], lat_bound[1]],
  # extend='both'
  extend='neither'
  )

# Western Central Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords1.shape[2])):
	plt.plot(all_line_coords1[1,:,i],all_line_coords1[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords1[1][:,int(all_line_coords1.shape[2]/2)],all_line_coords1[0][:,int(all_line_coords1.shape[2]/2)],s=3)
# # Plot the off-shore radar (R/V Mirai of JAMSTEC)
# RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='g', label='R/V Mirai')
# # Plot the on-shore observatory in Bengkulu city (BMKG observatory)
# BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='g', label='BMKG Observatory')

# North Western Sumatra
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords2.shape[2])):
	plt.plot(all_line_coords2[1,:,i],all_line_coords2[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords2[1][:,int(all_line_coords2.shape[2]/2)],all_line_coords2[0][:,int(all_line_coords2.shape[2]/2)],s=3)

# North Western Borneo
# Plot the individual cross-sectioned lines
for i in range(int(all_line_coords3.shape[2])):
	plt.plot(all_line_coords3[1,:,i],all_line_coords3[0,:,i],'k',linewidth=0.25)
# Plot the center line
plt.plot(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],'k',linewidth=1)
# Plot the grid resolution
plt.scatter(all_line_coords3[1][:,int(all_line_coords3.shape[2]/2)],all_line_coords3[0][:,int(all_line_coords3.shape[2]/2)],s=3)

# Set parameters and labels
x_ticks = [90,100,110,120]
x_tick_labels = [u'90\N{DEGREE SIGN}E',
                 u'100\N{DEGREE SIGN}E', u'110\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E']
y_ticks = [-10,-5,0,5,10]
y_tick_labels = [u'10\N{DEGREE SIGN}S',u'5\N{DEGREE SIGN}S',
		 		u'0\N{DEGREE SIGN}',
				u'5\N{DEGREE SIGN}N',u'10\N{DEGREE SIGN}N']

# Plot the coast lines
ax1.coastlines(linewidth=1, color='k', resolution='50m')  # cartopy function
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')

ax1.set_title(f'Lake or Non-Lake', loc='center')

ax2 = fig.add_subplot(gs[1,1])
cbar = plt.colorbar(cf, cax=ax2, orientation='horizontal', extend='both', ticks=np.arange(levels[0],levels[1]+dl, dl))
cbar.minorticks_off()
cbar.set_label('Index')


# ### Hovmoller Rain Rate

# #### Diurnal Composite

# In[172]:


## Load Data

# Control

RR_cntl = da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
	# Assign Local Time
RR_cntl = assign_LT_coord(RR_cntl)
	# Assign distance from coast coordinate
RR_cntl = RR_cntl.assign_coords(distance_from_coast=(('south_north','west_east'), distance_from_coast.values))
	# Group by local time and by distance from coast via flox
	# Average by distance from coast
RR_cntl = xarray_reduce(RR_cntl,
			'LocalTime.hour',
			'distance_from_coast',
			func='mean',
			expected_groups=(np.arange(0,24), np.arange(np.floor(RR_cntl.distance_from_coast.min().values), np.ceil(RR_cntl.distance_from_coast.max().values)+3,3)),
			isbin=[False, True]
			)
RR_cntl = RR_cntl.assign_coords(distance_from_coast_bins=pd.arrays.IntervalArray(RR_cntl.distance_from_coast_bins.values).left.values)
RR_cntl = RR_cntl.rename(distance_from_coast_bins='distance_from_coast')
# Concat to make a more continuous plot
RR_cntl = xr.concat([RR_cntl,RR_cntl], dim='hour', data_vars='all')
RR_cntl = RR_cntl.assign_coords(hour=(['hour'], np.arange(-0.5,47.5)))


# Sunrise CRF off @ 00UTC	[Morning]
RR_sunrise = da_d02_sunrise_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
inds = np.argwhere(RR_sunrise.mean(['south_north','west_east']).to_numpy()<0)	# Find the inds where RR is negative b/c of stitching many simulations together
RR_sunrise[inds[:,0],...] = (RR_sunrise[inds[:,0]-1,...].to_numpy()+RR_sunrise[inds[:,0]+1,...].to_numpy())/2
	# Assign Local Time
RR_sunrise = assign_LT_coord(RR_sunrise)
	# Assign distance from coast coordinate
RR_sunrise = RR_sunrise.assign_coords(distance_from_coast=(('south_north','west_east'), distance_from_coast.values))
	# Group by local time and by distance from coast via flox
	# Average by distance from coast
RR_sunrise = xarray_reduce(RR_sunrise,
			'LocalTime.hour',
			'distance_from_coast',
			func='mean',
			expected_groups=(np.arange(0,24), np.arange(np.floor(RR_sunrise.distance_from_coast.min().values), np.ceil(RR_sunrise.distance_from_coast.max().values)+3,3)),
			isbin=[False, True]
			)
RR_sunrise = RR_sunrise.assign_coords(distance_from_coast_bins=pd.arrays.IntervalArray(RR_sunrise.distance_from_coast_bins.values).left.values)
RR_sunrise = RR_sunrise.rename(distance_from_coast_bins='distance_from_coast')
# Concat to make a more continuous plot
RR_sunrise = xr.concat([RR_sunrise,RR_sunrise], dim='hour', data_vars='all')
RR_sunrise = RR_sunrise.assign_coords(hour=(['hour'], np.arange(-0.5,47.5)))


## Smooth the data
smoothing_num=3
RR_cntl = RR_cntl.rolling(distance_from_coast=smoothing_num, min_periods=1, center=True).mean()
RR_sunrise = RR_sunrise.rolling(distance_from_coast=smoothing_num, min_periods=1, center=True).mean()


# In[173]:


fig = plt.figure(figsize=(13,8))
gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.25, height_ratios=[0.825,0.03], wspace=0.15, width_ratios=[.5,.5])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
# Fontsize
fs = 20

title = 'Rain Rate Diurnal Composite over MC'
fig.suptitle(title, fontsize=fs+2)

dl=.1
# Plot the cross-sectional data
cf1 = RR_cntl.plot.contourf(
	ax=ax1,
	x = 'distance_from_coast',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,4+dl,dl),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = RR_sunrise.plot.contourf(
	ax=ax2,
	x = 'distance_from_coast',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,4+dl,dl),
	cmap='gray_r',
	center=0,
	extend='max'
)

## Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs-2)
ax2.set_ylabel('Lead Time', fontsize=fs-2)
ax1.set_xticks(np.arange(-300,700,100))
ax2.set_xticks(np.arange(-300,700,100))
ax1.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax1.set_yticks(RR_cntl.hour[0::3].values+0.5)
ax2.set_yticks(RR_cntl.hour[0::3].values+0.5)
# ax1.set_yticklabels(np.concatenate((np.arange(7,25,3),np.arange(1,25,3),np.arange(1,7,3))),fontsize=fs-6)
ax1.set_yticklabels(np.concatenate((np.arange(0,23,3), np.arange(0,23,3))),fontsize=fs-6)
ax2.set_yticklabels(np.arange(0,48,3),fontsize=fs-6)
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)

# ax1.set_xlim([RR_cntl.distance_from_coast[0],RR_cntl.distance_from_coast[-1]])
# ax2.set_xlim([RR_sunrise.distance_from_coast[0],RR_sunrise.distance_from_coast[-1]])
ax1.set_xlim([RR_cntl.distance_from_coast[0], 600])
ax2.set_xlim([RR_sunrise.distance_from_coast[0], 600])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax1.invert_xaxis()
ax2.invert_xaxis()


## Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,5,1))
cbar.set_ticklabels(np.arange(0,5,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm hr^{-1}$]', fontsize=fs-6)


# In[174]:


fig = plt.figure(figsize=(13,8))
gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.25, height_ratios=[0.825,0.05], wspace=0.15, width_ratios=[.5,.5])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
# Fontsize
fs = 20

title = 'Normalized Diurnal Rain Rate Difference from Control over the MC'
fig.suptitle(title, fontsize=fs+2)

dl=.125
# Plot the cross-sectional data
cf1 = RR_cntl.plot.contourf(
	ax=ax1,
	x = 'distance_from_coast',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,4+dl,dl),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
RR_sunrise_diff = (RR_sunrise-RR_cntl)/RR_cntl.std('hour')
cf2 = RR_sunrise_diff.plot.contourf(
	ax=ax2,
	x = 'distance_from_coast',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3+dl*2,dl*2),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

## Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs-2)
ax2.set_ylabel('Lead Time', fontsize=fs-2)
ax1.set_xticks(np.arange(-300,700,100))
ax2.set_xticks(np.arange(-300,700,100))
ax1.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax1.set_yticks(RR_cntl.hour[0::3].values+0.5)
ax2.set_yticks(RR_cntl.hour[0::3].values+0.5)
# ax1.set_yticklabels(np.concatenate((np.arange(7,25,3),np.arange(1,25,3),np.arange(1,7,3))),fontsize=fs-6)
ax1.set_yticklabels(np.concatenate((np.arange(0,23,3), np.arange(0,23,3))),fontsize=fs-6)
ax2.set_yticklabels(np.arange(0,48,3),fontsize=fs-6)
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)

# ax1.set_xlim([RR_cntl.distance_from_coast[0],RR_cntl.distance_from_coast[-1]])
# ax2.set_xlim([RR_sunrise.distance_from_coast[0],RR_sunrise.distance_from_coast[-1]])
ax1.set_xlim([RR_cntl.distance_from_coast[0], 600])
ax2.set_xlim([RR_sunrise.distance_from_coast[0], 600])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax1.invert_xaxis()
ax2.invert_xaxis()


## Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,5,1))
cbar.set_ticklabels(np.arange(0,5,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm hr^{-1}$]', fontsize=fs-6)
	# Rain rate colorbar
cax2 = fig.add_subplot(gs[1, 1])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,3+1,1))
cbar.set_ticklabels(np.arange(-3,3+1,1), fontsize=fs-6)
cbar.set_label('[$(RR-RR_{cntl})/\sigma_{cntl}$]', fontsize=fs-6)


# #### Time Series

# In[175]:


## Load Data

# Control

RR_cntl = da_d02_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
	# Assign Local Time
RR_cntl = assign_LT_coord(RR_cntl)
	# Assign distance from coast coordinate
RR_cntl = RR_cntl.assign_coords(distance_from_coast=(('south_north','west_east'), distance_from_coast.values))
	# Group by local time and by distance from coast via flox
	# Average by distance from coast
RR_cntl = xarray_reduce(RR_cntl,
			'LocalTime',
			'distance_from_coast',
			func='mean',
			expected_groups=(np.unique(RR_cntl.LocalTime.values), np.arange(np.floor(RR_cntl.distance_from_coast.min().values), np.ceil(RR_cntl.distance_from_coast.max().values)+3,3)),
			isbin=[False, True]
			)
RR_cntl = RR_cntl.assign_coords(distance_from_coast_bins=pd.arrays.IntervalArray(RR_cntl.distance_from_coast_bins.values).left.values)
RR_cntl = RR_cntl.rename(distance_from_coast_bins='distance_from_coast')

# Sunrise CRF off @ 00UTC	[Morning]
RR_sunrise = da_d02_sunrise_RR.sel(Time=slice('2015-11-23T02','2015-12-02T00')).copy()
inds = np.argwhere(RR_sunrise.mean(['south_north','west_east']).to_numpy()<0)	# Find the inds where RR is negative b/c of stitching many simulations together
RR_sunrise[inds[:,0],...] = (RR_sunrise[inds[:,0]-1,...].to_numpy()+RR_sunrise[inds[:,0]+1,...].to_numpy())/2
	# Assign Local Time
RR_sunrise = assign_LT_coord(RR_sunrise)
	# Assign distance from coast coordinate
RR_sunrise = RR_sunrise.assign_coords(distance_from_coast=(('south_north','west_east'), distance_from_coast.values))
	# Group by local time and by distance from coast via flox
	# Average by distance from coast
RR_sunrise = xarray_reduce(RR_sunrise,
			'LocalTime',
			'distance_from_coast',
			func='mean',
			expected_groups=(np.unique(RR_sunrise.LocalTime.values), np.arange(np.floor(RR_sunrise.distance_from_coast.min().values), np.ceil(RR_sunrise.distance_from_coast.max().values)+3, 3)),
			isbin=[False, True]
			)
RR_sunrise = RR_sunrise.assign_coords(distance_from_coast_bins=pd.arrays.IntervalArray(RR_sunrise.distance_from_coast_bins.values).left.values)
RR_sunrise = RR_sunrise.rename(distance_from_coast_bins='distance_from_coast')


## Smooth the data
smoothing_num=1

RR_cntl = RR_cntl.rolling(distance_from_coast=smoothing_num, min_periods=1, center=True).mean()
RR_sunrise = RR_sunrise.rolling(distance_from_coast=smoothing_num, min_periods=1, center=True).mean()


# In[179]:


fig = plt.figure(figsize=(14,10))
gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.2, height_ratios=[0.825,0.03], wspace=0.05, width_ratios=[.5,.5])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
fs = 20

title = 'Rain Rate Time Series over the MC'
fig.suptitle(title, fontsize=fs+2)


dl = .1
# Control
cf1 = RR_cntl.plot.contourf(
	ax=ax1,
	x = 'distance_from_coast',
    y = 'LocalTime',
	add_colorbar=False,
	levels=np.arange(0,4+dl,dl),
	# levels=np.append(0,np.logspace(0,0.9,20)),
	cmap='gray_r',
	center=0,
)

# Sunrise NCRF 00 UTC icloud=0
cf2 = RR_sunrise.plot.contourf(
	ax=ax2,
	x = 'distance_from_coast',
    y = 'LocalTime',
	add_colorbar=False,
	levels=np.arange(0,4+dl,dl),
	# levels=np.append(0,np.logspace(0,0.9,20)),
	cmap='gray_r',
	center=0,
)

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')

# for i in range(len(RR_cntl.Time[::24].values)-1):
# 	ax2.axhline(y=RR_cntl.Time[::24].values[i], color='r', linestyle='--',)
# 	ax3.axhline(y=RR_cntl.Time[12::24].values[i], color='r', linestyle='--')

ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs-2)
ax2.set_ylabel('', fontsize=fs)
ax1.set_xticks(np.arange(-300,700,100))
ax2.set_xticks(np.arange(-300,700,100))
ax1.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax1.set_yticks(RR_cntl.LocalTime[::24].values)
ax2.set_yticks(RR_cntl.LocalTime[::24].values)
ax1.set_yticklabels(RR_cntl.LocalTime[::24].dt.strftime("%m/%d %H").values, fontsize=fs-10)
ax2.set_yticklabels('')
# ax1.set_xlim([RR_cntl.distance_from_coast[0],RR_cntl.distance_from_coast[-1]])
# ax2.set_xlim([RR_sunrise.distance_from_coast[0],RR_sunrise.distance_from_coast[-1]])
ax1.set_xlim([RR_cntl.distance_from_coast[0], 600])
ax2.set_xlim([RR_sunrise.distance_from_coast[0], 600])
ax2.set_ylim([RR_cntl.LocalTime[0].values,RR_cntl.LocalTime[-1].values])
ax1.invert_xaxis()
ax2.invert_xaxis()
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)


# Plot the colorbar
	# Rain rate colorbar
ax2 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100, extend='max')
cbar.set_ticks(np.arange(0,5,1))
cbar.set_ticklabels(np.arange(0,5,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm hr^{-1}$]', fontsize=fs-6)


# In[180]:


fig = plt.figure(figsize=(14,10))
gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.2, height_ratios=[0.825,0.03], wspace=0.05, width_ratios=[.5,.5])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
fs = 20

title = 'Rain Rate Difference From Control Time Series over the MC'
fig.suptitle(title, fontsize=fs+2)


dl = .1
# Control
cf1 = RR_cntl.plot.contourf(
	ax=ax1,
	x = 'distance_from_coast',
    y = 'LocalTime',
	add_colorbar=False,
	levels=np.arange(0,4+dl,dl),
	cmap='gray_r',
	center=0,
)

RR_sunrise_diff = RR_sunrise-RR_cntl
# Sunrise NCRF 00 UTC icloud=0
cf2 = RR_sunrise_diff.plot.contourf(
	ax=ax2,
	x = 'distance_from_coast',
    y = 'LocalTime',
	add_colorbar=False,
	levels=np.arange(-3,3+dl*2,dl*2),
	cmap='RdBu_r',
	center=0,
)

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')

# for i in range(len(RR_cntl.Time[::24].values)-1):
# 	ax2.axhline(y=RR_cntl.Time[::24].values[i], color='r', linestyle='--',)
# 	ax3.axhline(y=RR_cntl.Time[12::24].values[i], color='r', linestyle='--')

ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs-2)
ax2.set_ylabel('', fontsize=fs)
ax1.set_xticks(np.arange(-300,700,100))
ax2.set_xticks(np.arange(-300,700,100))
ax1.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax1.set_yticks(RR_cntl.LocalTime[::24].values)
ax2.set_yticks(RR_cntl.LocalTime[::24].values)
ax1.set_yticklabels(RR_cntl.LocalTime[::24].dt.strftime("%m/%d %H").values, fontsize=fs-10)
ax2.set_yticklabels('')
# ax1.set_xlim([RR_cntl.distance_from_coast[0],RR_cntl.distance_from_coast[-1]])
# ax2.set_xlim([RR_sunrise.distance_from_coast[0],RR_sunrise.distance_from_coast[-1]])
ax1.set_xlim([RR_cntl.distance_from_coast[0], 600])
ax2.set_xlim([RR_sunrise.distance_from_coast[0], 600])
ax2.set_ylim([RR_cntl.LocalTime[0].values,RR_cntl.LocalTime[-1].values])
ax1.invert_xaxis()
ax2.invert_xaxis()
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)


# Plot the colorbar
	# Rain rate colorbar
ax3 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=ax3, orientation='horizontal', pad=0 , aspect=100, extend='max')
cbar.set_ticks(np.arange(0,4+1,1))
cbar.set_ticklabels(np.arange(0,4+1,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm hr^{-1}$]', fontsize=fs-6)

	# Rain rate colorbar
ax4 = fig.add_subplot(gs[1, 1])
cbar2 = plt.colorbar(cf2, cax=ax4, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar2.set_ticks(np.arange(-3,3+1,1))
cbar2.set_ticklabels(np.arange(-3,3+1,1), fontsize=fs-6)
cbar2.set_label('Rain Rate Difference [$mm hr^{-1}$]', fontsize=fs-6)


# In[182]:


fig = plt.figure(figsize=(14,10))
gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.2, height_ratios=[0.825,0.03], wspace=0.05, width_ratios=[.5,.5])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
fs = 20

title = 'Normalized Diurnal Rain Rate Difference from Control\nTime Series over the MC'
fig.suptitle(title, fontsize=fs+2)


dl = .1
# Control
cf1 = RR_cntl.plot.contourf(
	ax=ax1,
	x = 'distance_from_coast',
    y = 'LocalTime',
	add_colorbar=False,
	levels=np.arange(0,4+dl,dl),
	cmap='gray_r',
	center=0,
)

RR_sunrise_diff = (RR_sunrise-RR_cntl)/RR_cntl.std('LocalTime')
# Sunrise NCRF 00 UTC icloud=0
cf2 = RR_sunrise_diff.plot.contourf(
	ax=ax2,
	x = 'distance_from_coast',
    y = 'LocalTime',
	add_colorbar=False,
	levels=np.arange(-3,3+dl*2,dl*2),
	cmap='RdBu_r',
	center=0,
)

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')

# for i in range(len(RR_cntl.Time[::24].values)-1):
# 	ax2.axhline(y=RR_cntl.Time[::24].values[i], color='r', linestyle='--',)
# 	ax3.axhline(y=RR_cntl.Time[12::24].values[i], color='r', linestyle='--')

ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs-2)
ax2.set_ylabel('', fontsize=fs)
ax1.set_xticks(np.arange(-300,700,100))
ax2.set_xticks(np.arange(-300,700,100))
ax1.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-300,700,100),fontsize=fs-6)
ax1.set_yticks(RR_cntl.LocalTime[::24].values)
ax2.set_yticks(RR_cntl.LocalTime[::24].values)
ax1.set_yticklabels(RR_cntl.LocalTime[::24].dt.strftime("%m/%d %H").values, fontsize=fs-10)
ax2.set_yticklabels('')
# ax1.set_xlim([RR_cntl.distance_from_coast[0],RR_cntl.distance_from_coast[-1]])
# ax2.set_xlim([RR_sunrise.distance_from_coast[0],RR_sunrise.distance_from_coast[-1]])
ax1.set_xlim([RR_cntl.distance_from_coast[0], 600])
ax2.set_xlim([RR_sunrise.distance_from_coast[0], 600])
ax2.set_ylim([RR_cntl.LocalTime[0].values,RR_cntl.LocalTime[-1].values])
ax1.invert_xaxis()
ax2.invert_xaxis()
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)


# Plot the colorbar
	# Rain rate colorbar
ax3 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=ax3, orientation='horizontal', pad=0 , aspect=100, extend='max')
cbar.set_ticks(np.arange(0,4+1,1))
cbar.set_ticklabels(np.arange(0,4+1,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm hr^{-1}$]', fontsize=fs-6)

	# Rain rate colorbar
ax4 = fig.add_subplot(gs[1, 1])
cbar2 = plt.colorbar(cf2, cax=ax4, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar2.set_ticks(np.arange(-3,3+1,1))
cbar2.set_ticklabels(np.arange(-3,3+1,1), fontsize=fs-6)
cbar2.set_label('[$(RR-RR_{cntl})/\sigma_{cntl}$]', fontsize=fs-6)


# ## Cross-sectional Analysis

# ### Plot Domain

# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)

d02_coords = dict(
    south_north=('south_north',ds_d02.XLAT[0,:,0].values),
    west_east=('west_east',ds_d02.XLONG[0,0,:].values)
    )
# Prepare the Terrain Height    [m]
da_d02_TOPO = ds_d02['HGT'].sel(Time=slice(1)).compute().squeeze()
x = da_d02_TOPO.assign_coords(d02_coords)
lat = [da_d02_cross_NormalWind_cntl.Lat.max()+0.5,da_d02_cross_NormalWind_cntl.Lat.min()-0.5]
lon = [da_d02_cross_NormalWind_cntl.Lon.min()-0.5,da_d02_cross_NormalWind_cntl.Lon.max()+0.5]

# Yokoi et al. 2017-2019 domain:
x = x.sel(
	south_north=slice(lat[1],lat[0]),
	west_east=slice(lon[0],lon[1]))

x_ticks = np.array([100,102,104])
x_tick_labels = [u'100\N{DEGREE SIGN}E',
                 u'102\N{DEGREE SIGN}E', u'104\N{DEGREE SIGN}E']
y_ticks = np.array([-6,-4,-2])
y_tick_labels = [u'6\N{DEGREE SIGN}S',
                 u'4\N{DEGREE SIGN}S', u'2\N{DEGREE SIGN}S']

# Plot your terrain
ax1 = fig.add_subplot(gs[0,0])
cf1 = x.plot.contourf(
	cmap='terrain',
	# levels=np.arange(0,4250,250),
	levels = np.append(0,np.logspace(0,3.65,50)),

)

# Plot the individual cross-sectioned lines
for i in range(int(da_d02_cross_NormalWind_cntl.shape[3])):
	plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,i],da_d02_cross_NormalWind_cntl.Lat[:,i],'r',linewidth=0.5)
# Plot the center line
plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],da_d02_cross_NormalWind_cntl.Lat[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],'r',linewidth=1)
# Plot the grid resolution
plt.scatter(da_d02_cross_NormalWind_cntl.Lon[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],da_d02_cross_NormalWind_cntl.Lat[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],s=3)
# Plot the off-shore radar (R/V Mirai of JAMSTEC)
RV = plt.scatter(101.90,-4.07,s=100,marker='*',c='r', label='R/V Mirai')
# Plot the on-shore observatory in Bengkulu city (BMKG observatory)
BMKG = plt.scatter(102.34,-3.86,s=100,marker='o',c='r', label='BMKG Observatory')

cbar=cf1.colorbar
cbar.set_label('Terrain Height [m]',fontsize=fs)
cbar.set_ticks([0,10,100,1000,2000,3000])
cbar.minorticks_off()
cbar.set_ticklabels([0,10,100,1000,2000,3000],fontsize=fs)

# ax1.set_xlabel('Longitude',fontsize=18)
# ax1.set_ylabel('Latitude',fontsize=18)
ax1.set_title('Western Central Sumatra',fontsize=fs)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels,fontsize=14)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_tick_labels,fontsize=14)

ax1.legend(fontsize='x-large', markerscale=1.75)


# ### Time Series Evolution

# #### Hovmoller Rain Rate

# In[ ]:


fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, height_ratios=[0.875,0.03], wspace=0.075, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fs = 20
title = 'Rain Rate Evolution over ' + cross_title

fig.suptitle(title, fontsize=26)

# Load Data
	# Control
x1 = da_d02_cross_RR_cntl.sel(Time=slice('2015-11-23T00','2015-12-03T00'))
x1 = x1.mean('Spread')
	# 00 UTC icloud off
x2 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
	# 12 UTC icloud off
x3 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(1,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x3 = x3.mean('Spread')

# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	# levels=np.append(0,np.logspace(0,0.9,20)),
	cmap='gray_r',
	center=0,
)
# 00 UTC icloud=0
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'EnsTime',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	# levels=np.append(0,np.logspace(0,0.9,20)),
	cmap='gray_r',
	center=0,
)
# 12 UTC icloud=0
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'EnsTime',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	# levels=np.append(0,np.logspace(0,0.9,20)),
	cmap='gray_r',
	center=0,
)

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')

# for i in range(len(x1.Time[::24].values)-1):
# 	ax2.axhline(y=x1.Time[::24].values[i], color='r', linestyle='--',)
# 	ax3.axhline(y=x1.Time[12::24].values[i], color='r', linestyle='--')

ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax3.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('UTC', fontsize=fs)
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.set_xticks(np.arange(-200,600,100))
ax2.set_xticks(np.arange(-200,600,100))
ax3.set_xticks(np.arange(-200,600,100))
ax1.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax3.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax1.set_yticks(x1.Time[::24].values)
ax2.set_yticks(x1.Time[::24].values)
ax3.set_yticks(x1.Time[::24].values)
ax1.set_yticklabels(x1.Time[::24].dt.strftime("%m-%d %H").values, fontsize=fs-10)
ax2.set_yticklabels('')
ax3.set_yticklabels('')
ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax2.set_ylim([x1.Time[0].values,x1.Time[-1].values])
ax3.set_ylim([x1.Time[0].values,x1.Time[-1].values])
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
ax3.set_title('NCRF Sunset', loc='center', fontsize=fs)
ax3.set_title('c)', loc='left', fontsize=fs)
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)


# Plot the colorbar
	# Rain rate colorbar
ax2 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100, extend='max')
cbar.set_ticks(np.arange(0,5,1))
cbar.set_ticklabels(np.arange(0,5,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm d^{-1}$]', fontsize=fs-6)


# In[ ]:


fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, height_ratios=[0.875,0.03], wspace=0.075, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

fs = 20
title = 'Rain Rate Evolution over ' + cross_title
fig.suptitle(title, fontsize=26)

# Load Data
	# Control
x1 = da_d02_cross_RR_cntl.sel(Time=slice('2015-11-23T00','2015-12-03T00'))
x1 = x1.mean('Spread')
	# 00 UTC icloud off
x2 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2 = x2-x1
	# 12 UTC icloud off
x3 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(1,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x3 = x3.mean('Spread')
da = x3.drop_vars(['Lead','Time','Times'])
x3 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x3 = x3-x1

# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	# levels=np.append(0,np.logspace(0,0.9,20)),
	cmap='gray_r',
	center=0,
)
# 00 UTC icloud=0
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-2,2.25,0.25),
	# levels=np.append(0,np.logspace(0,0.9,20)),
	cmap='RdBu_r',
	center=0,
)
# 12 UTC icloud=0
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-2,2.25,0.25),
	# levels=np.append(0,np.logspace(0,0.9,20)),
	cmap='RdBu_r',
	center=0,
)

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax3.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('UTC', fontsize=fs-6)
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax2.set_ylim([x1.Time[0].values,x1.Time[-1].values])
ax3.set_ylim([x1.Time[0].values,x1.Time[-1].values])
ax1.set_yticks(x1.Time[::24].values)
ax2.set_yticks(x1.Time[::24].values)
ax3.set_yticks(x1.Time[::24].values)
ax1.set_yticklabels(x1.Time[::24].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels('')
ax3.set_yticklabels('')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
ax3.set_title('NCRF Sunset', loc='center', fontsize=fs)
ax3.set_title('c)', loc='left', fontsize=fs)


# Plot the colorbar
	# Rain rate colorbar
ax4 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=ax4, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(0,5,1))
cbar.set_label('Rain Rate [$mm d^{-1}$]', fontsize=fs-6)
	# Anomalous rain rate colorbar
ax5 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=ax5, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-2,3,1))
cbar.set_label('Anomalous Rain Rate [$mm d^{-1}$]', fontsize=fs-6)


# In[ ]:


fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, height_ratios=[0.875,0.03], wspace=0.075, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Anomalous Rain Rate Evolution over Western Central Sumatra Coast', fontsize=14)

# Load Data
	# 00 UTC icloud off
x2 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
	# 12 UTC icloud off
x3 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(1,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x3 = x3.mean('Spread')
da = x3.drop_vars(['Lead','Time','Times'])
x3 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
	# Difference between 00UTC CRF off and 12UTC CRF off
x1 = x2-x3

# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-2,2.25,0.25),
	cmap='RdBu_r',
	center=0,
)
# 00 UTC icloud=0
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	cmap='gray_r',
	center=0,
)
# 12 UTC icloud=0
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	cmap='gray_r',
	center=0,
)
# For ticks and such
x1 = da_d02_cross_RR_cntl.sel(Time=slice('2015-11-23T00','2015-12-03T00'))
x1 = x1.mean('Spread')

# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax2.set_ylim([x1.Time[0].values,x1.Time[-1].values])
ax3.set_ylim([x1.Time[0].values,x1.Time[-1].values])
ax1.set_yticks(x1.Time[::24].values)
ax2.set_yticks(x1.Time[::24].values)
ax3.set_yticks(x1.Time[::24].values)
ax1.set_yticklabels(x1.Time[::24].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels('')
ax3.set_yticklabels('')
# Set titles/labels
ax1.set_title('00UTC minus 12UTC', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)


# Plot the colorbar
	# Rain rate colorbar
ax4 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=ax4, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-2,3,1))
cbar.set_label('Anomalous Rain Rate [$mm d^{-1}$]')
	# Anomalous rain rate colorbar
ax5 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=ax5, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(0,5,1))
cbar.set_label('Rain Rate [$mm d^{-1}$]')


# #### Domain Average

# ##### Rain Rate & Cloud Frac

# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
ax2 = ax1.twinx()

land_dist_thresh = -100		# km
ocean_dist_thresh = 100	# km

# Rain Rate [00 UTC CRF Off]
    # remove first 12 hrs (spin-up), average over Spread, then concat each 24 hour section together
da = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Distance')
# x2 = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')

# Rain Rate [control]
x1 = da_d02_cross_RR_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Distance')
# x1 = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean('Distance')

# Low Cloud Fraction
x3 = da_d02_cross_LowCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Distance')
# x3 = x3.where((x3.Distance>0)&(x3.Distance<ocean_dist_thresh)).mean('Distance')

l1 = x1.plot.line(
    ax=ax1,
	x='Time',
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='RR Control'
)

l2 = x2.plot.line(
    ax=ax1,
	x='Time',
    color='r',
    linewidth=2,
    linestyle = '-',
	label='RR CRF Off 00 UTC'
)

l3 = x3.plot.line(
    ax=ax2,
    x='Time',
    color='k',
    linewidth=2,
    linestyle = ':',
	label='Low Cloud Frac'
)
ax1.axhline(y=0, color='k', linestyle='-')

ax1.set_xlim([x1.Time[0],x1.Time[-1]])
ax1.set_ylim([0,7])
ax2.set_ylim([0,0.25])
ax1.set_xticks(np.arange('2015-11-23T12', '2015-12-02T12', np.timedelta64(12,'h'), dtype='datetime64[h]'))
ax1.set_xticklabels(x1.Time[10::12].dt.strftime("%m/%d %H").values)
ax1.set_title(f'Rain Rate and Low Cloud Faction Evolution between 0->{land_dist_thresh:.0f} km', fontsize=14)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Rain Rate [$mm d^{-1}$]')
ax2.set_ylabel('Cloud Fraction')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend()


# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
ax2 = ax1.twinx()

land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km

# Rain Rate [00 UTC CRF Off]
    # remove first 12 hrs (spin-up), average over Spread, then concat each 24 hour section together
da = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Distance')

# Rain Rate [control]
x1 = da_d02_cross_RR_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Distance')

# Low Cloud Fraction
x3 = da_d02_cross_LowCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Distance')

# CRFOff rain rate anomalies with the control dicomposite removed
# x2 = x2.groupby('Time.hour')-x1.groupby('Time.hour').mean()
x2 = x2 - x1


l1 = x1.plot.line(
    ax=ax1,
	x='Time',
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='RR Control'
)

l2 = x2.plot.line(
    ax=ax1,
	x='Time',
    color='r',
    linewidth=2,
    linestyle = '-',
	label='RR CRF Off 00 UTC'
)

l3 = x3.plot.line(
    ax=ax2,
    x='Time',
    color='k',
    linewidth=2,
    linestyle = ':',
	label='Low Cloud Frac'
)
ax1.axhline(y=0, color='k', linestyle='-')

ax1.set_xlim([x1.Time[0],x1.Time[-1]])
ax1.set_ylim([-3,7])
ax2.set_ylim([0,0.25])
ax1.set_xticks(np.arange('2015-11-23T12', '2015-12-02T12', np.timedelta64(12,'h'), dtype='datetime64[h]'))
ax1.set_xticklabels(x1.Time[10::12].dt.strftime("%m/%d %H").values)
ax1.set_title(f'Rain Rate difference from control and Low Cloud Faction Evolution between 0->{land_dist_thresh:.0f} km', fontsize=14)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Rain Rate Anomaly [$mm d^{-1}$]', color='r')
ax2.set_ylabel('Cloud Fraction')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend()


# ##### 2m Temperature

# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
# ax2 = ax1.twinx()

land_dist_thresh = -100		# km
ocean_dist_thresh = 100		# km

# Temperature 2m [00 UTC CRF Off]
    # remove first 12 hrs (spin-up), average over Spread, then concat each 24 hour section together
da = da_d02_cross_T2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
# Isolate over a specific region/time
x2_land = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Distance')
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')

# Temperature 2m [control]
x1 = da_d02_cross_T2_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x1_land = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Distance')
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean('Distance')

# # Low Cloud Fraction
# x3 = da_d02_cross_LowCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# # Isolate over a specific region/time
# x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Distance')
# # x3 = x3.where((x3.Distance>0)&(x3.Distance<ocean_dist_thresh)).mean('Distance')

# Control
l1_land = x1_land.plot.line(
    ax=ax1,
	x='Time',
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='$Control_{land}$'
)
l1_ocean = x1_ocean.plot.line(
    ax=ax1,
	x='Time',
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='$Control_{ocean}$'
)
# CRF Off
l2_land = x2_land.plot.line(
    ax=ax1,
	x='Time',
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='$CRFoff_{land}$'
)
# CRF Off
l2_ocean = x2_ocean.plot.line(
    ax=ax1,
	x='Time',
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='$CRFoff_{ocean}$'
)

# l3 = x3.plot.line(
#     ax=ax2,
#     x='Time',
#     color='k',
#     linewidth=2,
#     linestyle = ':',
# 	label='Low Cloud Frac'
# )

# ax1.axhline(y=0, color='k', linestyle='-',linewidth=1)
ax1.set_xlim([x1.Time[0],x1.Time[-1]])
# ax1.set_ylim([0,7])
# ax2.set_ylim([0,0.25])
ax1.set_xticks(np.arange('2015-11-23T12', '2015-12-02T12', np.timedelta64(12,'h'), dtype='datetime64[h]'))
ax1.set_xticklabels(x1.Time[10::12].dt.strftime("%m/%d %H").values)
ax1.set_title(f'2m Temperature Evolution\n Land: 0->{land_dist_thresh:.0f}km and Ocean: 0->{ocean_dist_thresh:.0f}km', fontsize=14)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(loc='upper right',ncol=2)


# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
ax2 = ax1.twinx()

land_dist_thresh = -100		# km
ocean_dist_thresh = 100		# km

# Temperature 2m [00 UTC CRF Off]
    # remove first 12 hrs (spin-up), average over Spread, then concat each 24 hour section together
da = da_d02_cross_T2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
# Isolate over a specific region/time
x2_land = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Distance')
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean('Distance')

# Temperature 2m [control]
x1 = da_d02_cross_T2_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x1_land = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Distance')
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean('Distance')
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean('Distance')

# Calculate Anomalies
x_land_diff = x2_land-x1_land
x_ocean_diff = x2_ocean-x1_ocean
x_diff = x2-x1

# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x3_land= x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Distance')
x3_ocean = x3.where((x3.Distance>0)&(x3.Distance<ocean_dist_thresh)).mean('Distance')
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<ocean_dist_thresh)).mean('Distance')

# Mid Cloud Fraction
x4 = da_d02_cross_MidCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x4_land= x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0)).mean('Distance')
x4_ocean = x4.where((x4.Distance>0)&(x4.Distance<ocean_dist_thresh)).mean('Distance')
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<ocean_dist_thresh)).mean('Distance')

# Low Cloud Fraction
x5 = da_d02_cross_LowCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x5_land= x5.where((x5.Distance>land_dist_thresh)&(x5.Distance<0)).mean('Distance')
x5_ocean = x5.where((x5.Distance>0)&(x5.Distance<ocean_dist_thresh)).mean('Distance')
x5 = x5.where((x5.Distance>land_dist_thresh)&(x5.Distance<ocean_dist_thresh)).mean('Distance')

# Difference (CRFoff-Control)
    # Average
l_avg = x_diff.plot.line(
    ax=ax1,
	x='Time',
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Difference$_{avg}$'
)
#     # Land
# l_land = x_land_diff.plot.line(
#     ax=ax1,
# 	x='Time',
#     color='peru',
#     linewidth=2,
#     linestyle = '-',
# 	label='Difference$_{land}$'
# )
#     # Ocean
# l_ocean = x_ocean_diff.plot.line(
#     ax=ax1,
# 	x='Time',
#     color='dodgerblue',
#     linewidth=2,
#     linestyle = '-',
# 	label='Difference$_{ocean}$'
# )
# High loud Fraction
    # Average
l3_avg = x3.plot.line(
    ax=ax2,
    x='Time',
    color='0.3',
    linewidth=2,
    linestyle = ':',
	label='High Cloud Frac$_{avg}$'
)
#     # Land
# l3_land = x3_land.plot.line(
#     ax=ax2,
#     x='Time',
#     color='peru',
#     linewidth=2,
#     linestyle = ':',
# 	label='High Cloud Frac$_{land}$'
# )
#     # Ocean
# l3_ocean = x3_ocean.plot.line(
#     ax=ax2,
#     x='Time',
#     color='dodgerblue',
#     linewidth=2,
#     linestyle = ':',
# 	label='High Cloud Frac$_{ocean}$'
# )
# Mid Cloud Fraction
    # Average
l4_avg = x4.plot.line(
    ax=ax2,
    x='Time',
    color='.3',
    linewidth=2,
    linestyle = '--',
	label='Mid Cloud Frac$_{avg}$'
)
# Low Cloud Fraction
    # Average
l5_avg = x5.plot.line(
    ax=ax2,
    x='Time',
    color='.3',
    linewidth=2,
    linestyle = '-',
	label='Low Cloud Frac$_{avg}$'
)

ax1.axhline(y=0, color='k', linestyle='-',linewidth=1)
ax1.set_xlim([x1.Time[0],x1.Time[-1]])
ax1.set_ylim([-2,2])
ax2.set_ylim([0,1])
ax1.set_xticks(np.arange('2015-11-23T12', '2015-12-02T12', np.timedelta64(12,'h'), dtype='datetime64[h]'))
ax1.set_xticklabels(x1.Time[10::12].dt.strftime("%m/%d %H").values)
ax1.set_title(f'2m Temperature Anomaly (NCRF-Control) Evolution\n Land: 0->{land_dist_thresh:.0f}km and Ocean: 0->{ocean_dist_thresh:.0f}km', fontsize=14)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Temperature Anomaly [$K$]')
ax2.set_ylabel('Cloud Fraction')
# ax1.set_ylabel('Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(loc='upper left',ncol=1)
ax2.legend(loc='upper right',ncol=1)


# ##### 2m Potential Temperature

# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
# ax2 = ax1.twinx()

land_dist_thresh = -100		# km
ocean_dist_thresh = 100		# km

# Potential Temperature 2m [00 UTC CRF Off]
    # remove first 12 hrs (spin-up), average over Spread, then concat each 24 hour section together
da = da_d02_cross_Theta2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
# Isolate over a specific region/time
x2_land = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Distance')
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')

# Potential Temperature 2m [control]
x1 = da_d02_cross_Theta2_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x1_land = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Distance')
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean('Distance')

# # Low Cloud Fraction
# x3 = da_d02_cross_LowCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# # Isolate over a specific region/time
# x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Distance')
# # x3 = x3.where((x3.Distance>0)&(x3.Distance<ocean_dist_thresh)).mean('Distance')

# Control
l1_land = x1_land.plot.line(
    ax=ax1,
	x='Time',
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='$Control_{land}$'
)
l1_ocean = x1_ocean.plot.line(
    ax=ax1,
	x='Time',
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='$Control_{ocean}$'
)
# CRF Off
l2_land = x2_land.plot.line(
    ax=ax1,
	x='Time',
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='$CRFoff_{land}$'
)
# CRF Off
l2_ocean = x2_ocean.plot.line(
    ax=ax1,
	x='Time',
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='$CRFoff_{ocean}$'
)

# l3 = x3.plot.line(
#     ax=ax2,
#     x='Time',
#     color='k',
#     linewidth=2,
#     linestyle = ':',
# 	label='Low Cloud Frac'
# )

# ax1.axhline(y=0, color='k', linestyle='-',linewidth=1)
ax1.set_xlim([x1.Time[0],x1.Time[-1]])
# ax1.set_ylim([0,7])
# ax2.set_ylim([0,0.25])
ax1.set_xticks(np.arange('2015-11-23T12', '2015-12-02T12', np.timedelta64(12,'h'), dtype='datetime64[h]'))
ax1.set_xticklabels(x1.Time[10::12].dt.strftime("%m/%d %H").values)
ax1.set_title(f'2m Potential Temperature Evolution\n Land: 0->{land_dist_thresh:.0f}km and Ocean: 0->{ocean_dist_thresh:.0f}km', fontsize=14)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Potential Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(loc='upper right',ncol=2)


# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
ax2 = ax1.twinx()

land_dist_thresh = -100		# km
ocean_dist_thresh = 100		# km

# Potential Temperature 2m [00 UTC CRF Off]
    # remove first 12 hrs (spin-up), average over Spread, then concat each 24 hour section together
da = da_d02_cross_Theta2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2_land = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Distance')
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean('Distance')

# Potential Temperature 2m [control]
x1 = da_d02_cross_Theta2_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x1_land = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Distance')
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean('Distance')
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean('Distance')

# Calculate Anomalies
x_land_diff = x2_land-x1_land
x_ocean_diff = x2_ocean-x1_ocean
x_avg_diff = x2_avg-x1_avg

# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x3_land= x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Distance')
x3_ocean = x3.where((x3.Distance>0)&(x3.Distance<ocean_dist_thresh)).mean('Distance')
x3_avg = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<ocean_dist_thresh)).mean('Distance')

# Mid Cloud Fraction
x4 = da_d02_cross_MidCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x4_land= x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0)).mean('Distance')
x4_ocean = x4.where((x4.Distance>0)&(x4.Distance<ocean_dist_thresh)).mean('Distance')
x4_avg = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<ocean_dist_thresh)).mean('Distance')

# Low Cloud Fraction
x5 = da_d02_cross_LowCLDFRA_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).mean('Spread')	# remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x5_land= x5.where((x5.Distance>land_dist_thresh)&(x5.Distance<0)).mean('Distance')
x5_ocean = x5.where((x5.Distance>0)&(x5.Distance<ocean_dist_thresh)).mean('Distance')
x5_avg = x5.where((x5.Distance>land_dist_thresh)&(x5.Distance<ocean_dist_thresh)).mean('Distance')

# Difference (CRFoff-Control)
    # Average
l_avg = x_avg_diff.plot.line(
    ax=ax1,
	x='Time',
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Difference$_{avg}$'
)
#     # Land
# l_land = x_land_diff.plot.line(
#     ax=ax1,
# 	x='Time',
#     color='peru',
#     linewidth=2,
#     linestyle = '-',
# 	label='Difference$_{land}$'
# )
#     # Ocean
# l_ocean = x_ocean_diff.plot.line(
#     ax=ax1,
# 	x='Time',
#     color='dodgerblue',
#     linewidth=2,
#     linestyle = '-',
# 	label='Difference$_{ocean}$'
# )
# High loud Fraction
    # Average
l3_avg = x3_avg.plot.line(
    ax=ax2,
    x='Time',
    color='0.3',
    linewidth=2,
    linestyle = ':',
	label='High Cloud Frac$_{avg}$'
)
#     # Land
# l3_land = x3_land.plot.line(
#     ax=ax2,
#     x='Time',
#     color='peru',
#     linewidth=2,
#     linestyle = ':',
# 	label='High Cloud Frac$_{land}$'
# )
#     # Ocean
# l3_ocean = x3_ocean.plot.line(
#     ax=ax2,
#     x='Time',
#     color='dodgerblue',
#     linewidth=2,
#     linestyle = ':',
# 	label='High Cloud Frac$_{ocean}$'
# )
# Mid Cloud Fraction
    # Average
l4_avg = x4_avg.plot.line(
    ax=ax2,
    x='Time',
    color='.3',
    linewidth=2,
    linestyle = '--',
	label='Mid Cloud Frac$_{avg}$'
)
# Low Cloud Fraction
    # Average
l5_avg = x5_avg.plot.line(
    ax=ax2,
    x='Time',
    color='.3',
    linewidth=2,
    linestyle = '-',
	label='Low Cloud Frac$_{avg}$'
)

ax1.axhline(y=0, color='k', linestyle='-',linewidth=1)
ax1.set_xlim([x1.Time[0],x1.Time[-1]])
ax1.set_ylim([-2,2])
ax2.set_ylim([0,1])
ax1.set_xticks(np.arange('2015-11-23T12', '2015-12-02T12', np.timedelta64(12,'h'), dtype='datetime64[h]'))
ax1.set_xticklabels(x1.Time[10::12].dt.strftime("%m/%d %H").values)
ax1.set_title(f'2m Potential Temperature Anomaly (NCRF-Control) Evolution\n Land: 0->{land_dist_thresh:.0f}km and Ocean: 0->{ocean_dist_thresh:.0f}km', fontsize=14)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Potential Temperature Anomaly [$K$]')
ax2.set_ylabel('Cloud Fraction')
# ax1.set_ylabel('Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(loc='upper left',ncol=1)
ax2.legend(loc='upper right',ncol=1)


# ##### Normal Wind

# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])
ax2 = ax1.twinx()

land_dist_thresh = -50	# km
ocean_dist_thresh = 50 # km

layer = 860 # hPa

## Normal Wind ##
# Normal Wind [00 UTC CRF Off]
    # select first 24 hours, 00 UTC CRF Off, a specific layer, average over Spread, then concat each 24 hour section together
da = da_d02_cross_NormalWind_CRFoff.isel(Time=np.arange(0,24,1)).sel(Lead=slice(0,18,2),bottom_top=layer).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='Normal Wind',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean('Distance')
# x2 = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')

# Normal Wind [control]
x1 = da_d02_cross_NormalWind_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).sel(bottom_top=layer).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean('Distance')
# x1 = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean('Distance')

## Rain Rate ##
# Rain Rate [00 UTC CRF Off]
    # remove first 12 hrs (spin-up), average over Spread, then concat each 24 hour section together
da = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x3 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<ocean_dist_thresh)).mean('Distance')
# x2 = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')

# Rain Rate [control]
x4 = da_d02_cross_RR_cntl[12:].sel(Time=slice(x3.Time[0],x3.Time[-1])).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<ocean_dist_thresh)).mean('Distance')
# x1 = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean('Distance')

# Normal Wind
    # Control
l1 = x1.plot.line(
    ax=ax1,
	x='Time',
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='Normal Wind Control'
)

    # CRF OFF 00UTC
l2 = x2.plot.line(
    ax=ax1,
	x='Time',
    color='k',
    linewidth=2,
    linestyle = '-',
	label='Normal Wind CRF Off 00 UTC'
)

# Rain Rate
    # Control
l4 = x4.plot.line(
    ax=ax2,
    x='Time',
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='RR Control'
)
    # CRF OFF 00UTC
l3 = x3.plot.line(
    ax=ax2,
    x='Time',
    color='k',
    linewidth=2,
    linestyle = '--',
	label='RR CRF Off 00 UTC'
)
ax1.axhline(y=0, color='k', linestyle='-')

ax1.set_xlim([x1.Time[0],x1.Time[-1]])
ax2.set_ylim([0,7])
ax1.set_ylim([-6,6])
ax1.set_xticks(np.arange('2015-11-23T12', '2015-12-02T12', np.timedelta64(12,'h'), dtype='datetime64[h]'))
ax1.set_xticklabels(x1.Time[11::12].dt.strftime("%m/%d %H").values)
ax1.set_title(f'Normal Wind @ {layer:.0f} hPa and Rain Rate Evolution between {ocean_dist_thresh:.0f}->{land_dist_thresh:.0f} km', fontsize=14)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Normal Wind [$m s^{-1}$]')
ax2.set_ylabel('Rain Rate [$mm d^{-1}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')


# ##### Dirunal Composite

# Rain Rate

# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

# Load Data
	# Control
x1 = da_d02_cross_RR_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))	# allow for 12-hr spin up
x1 = x1.groupby('Time.hour').mean()
		# Average
x1avg = x1.mean(['Distance','Spread'])
x1avg = xr.concat([x1avg,x1avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land only
x1land = x1.where(x1.Distance<0).mean(['Distance','Spread'])
x1land = xr.concat([x1land,x1land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x1ocean = x1.where(x1.Distance>0).mean(['Distance','Spread'])
x1ocean = xr.concat([x1ocean,x1ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# 00 UTC icloud off
x2 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
		# Average
x2avg = x2.mean('Distance')
x2avg = xr.concat([x2avg,x2avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land
x2land = x2.where(x2.Distance<0).mean('Distance')
x2land = xr.concat([x2land,x2land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
x2ocean = x2.where(x2.Distance>0).mean('Distance')
x2ocean = xr.concat([x2ocean,x2ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# 12 UTC icloud off
x3 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(1,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x3 = x3.mean('Spread')
da = x3.drop_vars(['Lead','Time','Times'])
x3 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x3 = x3.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
		# Average
x3avg = x3.mean('Distance')
x3avg = xr.concat([x3avg,x3avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land
x3land = x3.where(x3.Distance<0).mean('Distance')
x3land = xr.concat([x3land,x3land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
x3ocean = x3.where(x3.Distance>0).mean('Distance')
x3ocean = xr.concat([x3ocean,x3ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Control
avg1 = x1avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '-',
	label='$Control_{avg}$'
)
land1 = x1land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='$Control_{land}$'
)
ocean1 = x1ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='$Control_{ocean}$'
)
# 00UTC CRF off
avg2 = x2avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{avg}$'
)
land2 = x2land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{land}$'
)
ocean2 = x2ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='$00UTC CRF off_{ocean}$'
)
# 12UTC CRF off
avg3 = x3avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{avg}$'
)
land3 = x3land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{land}$'
)
ocean3 = x3ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = ':',
	label='$12UTC CRF off_{ocean}$'
)

ax1.set_xlim([0,24])
ax1.set_ylim([0,3])
ax1.set_xticks(np.arange(0,27,3))
title = f'Domain Averaged Diurnal Rain Rate: {cross_title:s}'
ax1.set_title(title, fontsize=fs-4)
ax1.set_xlabel('UTC')
ax1.set_ylabel('Rain Rate [$mm d^{-1}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncols=3)


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100
ocean_dist_thresh = 0

# Load Data
	# Control
x1 = da_d02_cross_RR_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
		# Average
x1avg = x1.mean(['Distance','Spread'])
x1avg = xr.concat([x1avg,x1avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land only
x1land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1land = xr.concat([x1land,x1land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x1ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1ocean = xr.concat([x1ocean,x1ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# 00 UTC icloud off
x2 = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='RR',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
		# Average
x2avg = x2.mean('Distance')
x2avg = xr.concat([x2avg,x2avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land
x2land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean('Distance')
x2land = xr.concat([x2land,x2land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean
x2ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')
x2ocean = xr.concat([x2ocean,x2ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Difference between CRFoff and Control!
x3diff = x2land-x1land

## Plotting
land1 = x1land.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='RR_control'
)
land2 = x2land.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='RR_CRFoff'
)
land3 = x3diff.plot.line(
    ax=ax1,
    color='red',
    linewidth=2,
    linestyle = ':',
	label='RR_Diff'
)

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
ax1.set_ylim([-0.5,5])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal Rain Rate between {ocean_dist_thresh:.0f}->{land_dist_thresh:.0f}km Western Central Sumatra Coast')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Rain Rate [$mm d^{-1}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncols=1)


# Normal Winds

# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -50	# km
ocean_dist_thresh = 50 # km

layer = 1000 # hPa

## Normal Wind ##
# Normal Wind [00 UTC CRF Off]
    # select first 24 hours, 00 UTC CRF Off, a specific layer, average over Spread, then concat each 24 hour section together
da = da_d02_cross_NormalWind_CRFoff.isel(Time=np.arange(0,24,1)).sel(Lead=slice(0,18,2)).interp(bottom_top=layer).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
da = da.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='Normal Wind',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
# Isolate over a specific region/time
x2avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean('Distance')
x2land = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Distance')
x2ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean('Distance')

# Normal Wind [control]
x1 = da_d02_cross_NormalWind_cntl[12:].sel(Time=slice(x2.Time[0],x2.Time[-1])).interp(bottom_top=layer).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# Isolate over a specific region/time
x1avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean('Distance')
x1land = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Distance')
x1ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean('Distance')

# ## Rain Rate ##
# # Rain Rate [00 UTC CRF Off]
#     # remove first 12 hrs (spin-up), average over Spread, then concat each 24 hour section together
# da = da_d02_cross_RR_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).mean('Spread').stack(Times=['Lead','Time']).transpose('Times','Distance')
# da = da.drop_vars(['Lead','Time','Times'])
# x3 = xr.DataArray(
# 				name='RR',
# 				data=da.values,
# 				dims=['Time','Distance'],
# 				coords=dict(
# 					Time = da.EnsTime.values,
# 					Distance = da.Distance.values,
# 					)
# 			)
# # Isolate over a specific region/time
# x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<ocean_dist_thresh)).mean('Distance')

# # Rain Rate [control]
# x4 = da_d02_cross_RR_cntl[12:].sel(Time=slice(x3.Time[0],x3.Time[-1])).mean('Spread') # remove first 12 hrs (spin-up) and average over Spread
# # Isolate over a specific region/time
# x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<ocean_dist_thresh)).mean('Distance')

# Diurnal Composite
x1avg = x1avg.groupby('Time.hour').mean()
x1avg = xr.concat([x1avg,x1avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1land = x1land.groupby('Time.hour').mean()
x1land = xr.concat([x1land,x1land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1ocean = x1ocean.groupby('Time.hour').mean()
x1ocean = xr.concat([x1ocean,x1ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2avg = x2avg.groupby('Time.hour').mean()
x2avg = xr.concat([x2avg,x2avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2land = x2land.groupby('Time.hour').mean()
x2land = xr.concat([x2land,x2land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2ocean = x2ocean.groupby('Time.hour').mean()
x2ocean = xr.concat([x2ocean,x2ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Normal Wind
    # Control Average
l1 = x1avg.plot.line(
    ax=ax1,
	x='hour',
    color='k',
    linewidth=2,
    linestyle = '-',
	label='Avg_Control'
)
    # Control Land
l2 = x1land.plot.line(
    ax=ax1,
	x='hour',
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Land_Control'
)
    # Control Ocean
l3 = x1ocean.plot.line(
    ax=ax1,
	x='hour',
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='Ocean_Control'
)

    # CRF OFF 00UTC Average
l4 = x2avg.plot.line(
    ax=ax1,
	x='hour',
    color='k',
    linewidth=2,
    linestyle = '--',
	label='Avg_CRF Off 00 UTC'
)
    # CRF OFF 00UTC Land
l5 = x2land.plot.line(
    ax=ax1,
	x='hour',
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='Land_Avg_CRF Off 00 UTC'
)
    # CRF OFF 00UTC Ocean
l6 = x2ocean.plot.line(
    ax=ax1,
	x='hour',
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='Ocean_Avg_CRF Off 00 UTC'
)

ax1.axhline(y=0, color='k', linestyle='-')
ax1.set_xlim([0,24])
# ax1.set_ylim([-16,-6])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal Normal Wind @ {layer:.0f}hPa over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Normal Wind [$m s^{-1}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncols=2,loc='upper right')


# Cloud Fraction

# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 100 # km
# land_dist_thresh = da_d02_cross_CLDFRA_cntl.Distance.min().values # km
# ocean_dist_thresh = da_d02_cross_CLDFRA_cntl.Distance.max().values # km

local_time = np.concatenate((np.arange(7,25,3),np.arange(1,10,3)))

# Load Data
	# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
		# Average
x1avg = x1.where((x1.Distance<ocean_dist_thresh)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1avg = xr.concat([x1avg,x1avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land only
x1land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1land = xr.concat([x1land,x1land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x1ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1ocean = xr.concat([x1ocean,x1ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# Mid Cloud Fraction
x2 = da_d02_cross_MidCLDFRA_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x2 = x2.groupby('Time.hour').mean()
		# Average
x2avg = x2.where((x2.Distance<ocean_dist_thresh)&(x2.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x2avg = xr.concat([x2avg,x2avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land only
x2land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x2land = xr.concat([x2land,x2land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x2ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x2ocean = xr.concat([x2ocean,x2ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Load Data
	# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x3 = x3.groupby('Time.hour').mean()
		# Average
x3avg = x3.where((x3.Distance<ocean_dist_thresh)&(x3.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x3avg = xr.concat([x3avg,x3avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Land only
x3land = x3.where((x3.Distance<0)&(x3.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x3land = xr.concat([x3land,x3land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x3ocean = x3.where((x3.Distance>0)&(x3.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x3ocean = xr.concat([x3ocean,x3ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Low Clouds
avg1 = x1avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '-',
	label='$Low_{avg}$'
)
land1 = x1land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='$Low_{land}$'
)
ocean1 = x1ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='$Low_{ocean}$'
)
# Mid Clouds
avg2 = x2avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '--',
	label='$Mid_{avg}$'
)
land2 = x2land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='$Mid_{land}$'
)
ocean2 = x2ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='$Mid_{ocean}$'
)
# Upper Clouds
avg3 = x3avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = ':',
	label='$Upper_{avg}$'
)
land3 = x3land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = ':',
	label='$Upper_{land}$'
)
ocean3 = x3ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = ':',
	label='$Upper_{ocean}$'
)


ax1.set_ylim([0,x3land.max()+.05])
if cross_title == 'Borneo Northwest': ax1.set_xticks(np.arange(0,27,3)-1)
else: ax1.set_xticks(np.arange(0,27,3))
ax1.set_xticklabels(local_time, fontsize=fs-6)
ax1.set_title('Domain Averaged Diurnal Cloud Fraction over Western Central Sumatra Coast Control\nLand:0-'+str(abs(land_dist_thresh))+'km Ocean:0-'+str(ocean_dist_thresh)+'km')
ax1.set_xlabel('Local Time')
ax1.set_xlim([0,24])
ax1.set_ylabel('Cloud Fraction')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=3)


# In[ ]:


fig = plt.figure(figsize=(15,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -241 # km
ocean_dist_thresh = 100 # km
# land_dist_thresh = da_d02_cross_CLDFRA_CRFoff.Distance.min().values # km
# ocean_dist_thresh = da_d02_cross_CLDFRA_CRFoff.Distance.max().values # km

# Load Data
	# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_CRFoff.sel(Lead=slice(0,18,2)).mean(['Lead']).rename({'Time':'hour'})
x1 = xr.concat([x1[23],x1[0:24]], dim='hour',data_vars='all').assign_coords(hour=np.arange(0,25,1))
		# Average
x1avg = x1.where((x1.Distance<ocean_dist_thresh)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1avg = x1avg.assign_coords(hour=np.arange(0,25,1))
		# Land only
x1land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1land = x1land.assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x1ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1ocean = x1ocean.assign_coords(hour=np.arange(0,25,1))

	# Mid Cloud Fraction
x2 = da_d02_cross_MidCLDFRA_CRFoff.sel(Lead=slice(0,18,2)).mean(['Lead']).rename({'Time':'hour'})
x2 = xr.concat([x2[23],x2[0:24]], dim='hour',data_vars='all').assign_coords(hour=np.arange(0,25,1))
		# Average
x2avg = x2.where((x2.Distance<ocean_dist_thresh)&(x2.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x2avg = x2avg.assign_coords(hour=np.arange(0,25,1))
		# Land only
x2land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x2land = x2land.assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x2ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x2ocean = x2ocean.assign_coords(hour=np.arange(0,25,1))

	# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_CRFoff.sel(Lead=slice(0,18,2)).mean(['Lead']).rename({'Time':'hour'})
x3 = xr.concat([x3[23],x3[0:24]], dim='hour',data_vars='all').assign_coords(hour=np.arange(0,25,1))
		# Average
x3avg = x3.where((x3.Distance<ocean_dist_thresh)&(x3.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x3avg = x3avg.assign_coords(hour=np.arange(0,25,1))
		# Land only
x3land = x3.where((x3.Distance<0)&(x3.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x3land = x3land.assign_coords(hour=np.arange(0,25,1))
		# Ocean only
x3ocean = x3.where((x3.Distance>0)&(x3.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x3ocean = x3ocean.assign_coords(hour=np.arange(0,25,1))

# Low Clouds
avg1 = x1avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '-',
	label='$Low_{avg}$'
)
land1 = x1land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='$Low_{land}$'
)
ocean1 = x1ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='$Low_{ocean}$'
)
# Mid Clouds
avg2 = x2avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = '--',
	label='$Mid_{avg}$'
)
land2 = x2land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='$Mid_{land}$'
)
ocean2 = x2ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='$Mid_{ocean}$'
)
# Upper Clouds
avg3 = x3avg.plot.line(
    ax=ax1,
	color='k',
    linewidth=2,
    linestyle = ':',
	label='$Upper_{avg}$'
)
land3 = x3land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = ':',
	label='$Upper_{land}$'
)
ocean3 = x3ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = ':',
	label='$Upper_{ocean}$'
)

ax1.set_ylim([0,x3land.max()+.05])
if cross_title == 'Borneo Northwest': ax1.set_xticks(np.arange(0,27,3)-1)
else: ax1.set_xticks(np.arange(0,27,3))
ax1.set_xticklabels(local_time, fontsize=fs-6)
ax1.set_title('Domain Averaged Diurnal Cloud Fraction over Western Central Sumatra Coast NCRF\nLand:0-'+str(abs(land_dist_thresh))+'km Ocean:0-'+str(ocean_dist_thresh)+'km')
ax1.set_xlabel('Local Time')
ax1.set_xlim([0,24])
ax1.set_ylabel('Cloud Fraction')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=3)


# HFX

# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

# Load Data
	# Control
x1 = da_d02_cross_HFX_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean().mean(['Distance','Spread'])
x1 = xr.concat([x1,x1[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

	# 00 UTC icloud off
x2 = da_d02_cross_HFX_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='HFX',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean().mean('Distance')
x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))


	# 12 UTC icloud off
x3 = da_d02_cross_HFX_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(1,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x3 = x3.mean('Spread')
da = x3.drop_vars(['Lead','Time','Times'])
x3 = xr.DataArray(
				name='HFX',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x3 = x3.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean().mean('Distance')
x3 = xr.concat([x3,x3[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Plotting
l1 = x1.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Control'
)
l2 = x2.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='00UTC CRF off'
)
l3 = x3.plot.line(
    ax=ax1,
    color='k',
    linewidth=2,
    linestyle = ':',
	label='12UTC CRF off'
)

ax1.set_xlim([0,24])
# ax1.set_ylim([.2,1.2])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title('Domain Averaged Diurnal Upward Surface Flux (HFX) over Western Central Sumatra Coast')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Upward Surface Flux [$W m^{-2}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend()


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 0 # km

## Load Data
# Control
x1 = da_d02_cross_HFX_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1 = x1.where((x1.Distance<ocean_dist_thresh)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1 = xr.concat([x1,x1[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_HFX_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='HFX',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
x2 = x2.where((x2.Distance<ocean_dist_thresh)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3 = x2 - x1

# Plotting
l1 = x1.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='HFX_control'
)
l2 = x2.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='HFX_CRFoff'
)
l3 = x3.plot.line(
    ax=ax1,
    color='red',
    linewidth=2,
    linestyle = ':',
	label='HFX_Diff'
)

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
ax1.set_ylim([-25,250])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal Upward Surface Heat Flux (HFX) between {ocean_dist_thresh:.0f}->{land_dist_thresh:.0f}km Western Central Sumatra Coast')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Upward Flux [$W/m^{2}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=1)


# Surface Net Flux Down

# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 0 # km

## Load Data
# Control
		# SW
x1_sw = da_d02_cross_SWSfc_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1_sw = x1_sw.groupby('Time.hour').mean()
x1_sw = x1_sw.where((x1_sw.Distance<ocean_dist_thresh)&(x1_sw.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_sw = xr.concat([x1_sw,x1_sw[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# LW
x1_lw = da_d02_cross_LWSfc_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1_lw = x1_lw.groupby('Time.hour').mean()
x1_lw = x1_lw.where((x1_lw.Distance<ocean_dist_thresh)&(x1_lw.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_lw = xr.concat([x1_lw,x1_lw[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# Net
x1_net = da_d02_cross_NetSfc_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1_net = x1_net.groupby('Time.hour').mean()
x1_net = x1_net.where((x1_net.Distance<ocean_dist_thresh)&(x1_net.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_net = xr.concat([x1_net,x1_net[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
		# SW
x2_sw = da_d02_cross_SWSfc_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2_sw = x2_sw.mean('Spread')
da = x2_sw.drop_vars(['Lead','Time','Times'])
x2_sw = xr.DataArray(
				name='SWSFC',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2_sw = x2_sw.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
x2_sw = x2_sw.where((x2_sw.Distance<ocean_dist_thresh)&(x2_sw.Distance>land_dist_thresh)).mean(['Distance'])
x2_sw = xr.concat([x2_sw,x2_sw[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
		# LW
x2_lw = da_d02_cross_LWSfc_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2_lw = x2_lw.mean('Spread')
da = x2_lw.drop_vars(['Lead','Time','Times'])
x2_lw = xr.DataArray(
				name='LWSFC',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2_lw = x2_lw.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
x2_lw = x2_lw.where((x2_lw.Distance<ocean_dist_thresh)&(x2_lw.Distance>land_dist_thresh)).mean(['Distance'])
x2_lw = xr.concat([x2_lw,x2_lw[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

		# Net
x2_net = da_d02_cross_NetSfc_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2_net = x2_net.mean('Spread')
da = x2_net.drop_vars(['Lead','Time','Times'])
x2_net = xr.DataArray(
				name='NetSFC',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2_net = x2_net.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
x2_net = x2_net.where((x2_net.Distance<ocean_dist_thresh)&(x2_net.Distance>land_dist_thresh)).mean(['Distance'])
x2_net = xr.concat([x2_net,x2_net[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_swdiff = x2_sw - x1_sw
x3_lwdiff = x2_lw - x1_lw
x3_netdiff = x2_net - x1_net

# Plotting
l1_sw = x1_sw.plot.line(
    ax=ax1,
    color='red',
    linewidth=2,
    linestyle = '-',
	label='SW_cntl'
)
l1_lw = x1_lw.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='LW_cntl'
)
l1_net = x1_net.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Net_cntl'
)
l2_sw = x2_sw.plot.line(
    ax=ax1,
    color='red',
    linewidth=2,
    linestyle = '--',
	label='SW_CRFoff'
)
l2_lw = x2_lw.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='LW_CRFoff'
)
l2_net = x2_net.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='Net_CRFoff'
)
l3_sw = x3_swdiff.plot.line(
    ax=ax1,
    color='red',
    linewidth=2,
    linestyle = ':',
	label='SW_Diff'
)
l3_lw = x3_lwdiff.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = ':',
	label='LW_Diff'
)
l3_net = x3_netdiff.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = ':',
	label='Net_Diff'
)

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
ax1.set_ylim([-200,1000])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal Surface Flux between {ocean_dist_thresh:.0f}->{land_dist_thresh:.0f}km Western Central Sumatra Coast')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Surface Flux [$W/m^{2}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=3)


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

# Load Data
	# Control
x1 = da_d02_cross_SWDownSfc_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean().mean(['Distance','Spread'])
x1 = xr.concat([x1,x1[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1 = np.cumsum(x1)

	# 00 UTC icloud off
x2 = da_d02_cross_SWDownSfc_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='SWDownSfc',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean().mean('Distance')
x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2 = np.cumsum(x2)

	# 12 UTC icloud off
x3 = da_d02_cross_SWDownSfc_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(1,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x3 = x3.mean('Spread')
da = x3.drop_vars(['Lead','Time','Times'])
x3 = xr.DataArray(
				name='SWDownSfc',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
			)
x3 = x3.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean().mean('Distance')
x3 = xr.concat([x3,x3[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x3 = np.cumsum(x3)

# Plotting
l1 = x1.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Control'
)
l2 = x2.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='00UTC CRF off'
)
l3 = x3.plot.line(
    ax=ax1,
    color='k',
    linewidth=2,
    linestyle = ':',
	label='12UTC CRF off'
)
ax1.set_xlim([0,24])
# ax1.set_ylim([.2,1.2])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title('Domain Averaged Diurnal Surface SW Flux Down over Western Central Sumatra Coast')
ax1.set_xlabel('UTC')
ax1.set_ylabel('SW SFC Flux Down [$W/m^{2}$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend()


# 2m Temperature

# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 100 # km

## Load Data
# Control
x1 = da_d02_cross_T2_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1_land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_land = xr.concat([x1_land,x1_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_ocean = xr.concat([x1_ocean,x1_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_avg = xr.concat([x1_avg,x1_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_T2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
# x2 = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
# x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2_land = xr.concat([x2_land,x2_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_ocean = xr.concat([x2_ocean,x2_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_avg = xr.concat([x2_avg,x2_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_land_diff = x2_land - x1_land
x3_ocean_diff = x2_ocean - x1_ocean
x3_avg_diff = x2_avg - x1_avg

# Calculate the difference between land & ocean
x4_cntl_diff = x1_land - x1_ocean
x4_NCRF_diff = x2_land - x2_ocean

l1_land = x1_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Control$_{land}$'
)
l1_ocean = x1_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='Control$_{ocean}$'
)
l1_avg = x1_avg.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Control$_{avg}$'
)

l2_land = x2_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{land}$'
)
l2_ocean = x2_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{ocean}$'
)
l2_avg = x2_avg.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{avg}$'
)

ax1.set_xlim([0,24])
# ax1.set_ylim([-1,1.5])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal 2m Temperature over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km, Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=2,loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 100 # km

## Load Data
# Control
x1 = da_d02_cross_T2_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1_land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_land = xr.concat([x1_land,x1_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_ocean = xr.concat([x1_ocean,x1_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_avg = xr.concat([x1_avg,x1_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_T2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
# x2 = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
# x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2_land = xr.concat([x2_land,x2_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_ocean = xr.concat([x2_ocean,x2_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_avg = xr.concat([x2_avg,x2_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_land_diff = x2_land - x1_land
x3_ocean_diff = x2_ocean - x1_ocean
x3_avg_diff = x2_avg - x1_avg

# Calculate the difference between land & ocean
x4_cntl_diff = x1_land - x1_ocean
x4_NCRF_diff = x2_land - x2_ocean

l1_land = x3_land_diff.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Difference$_{land}$'
)
l1_ocean = x3_ocean_diff.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='Difference$_{ocean}$'
)
l1_avg = x3_avg_diff.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Difference$_{avg}$'
)

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
ax1.set_ylim([-1,1.5])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal 2m Temperature Difference (NCRF-Control) over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km, Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=1,loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 100 # km

## Load Data
# Control
x1 = da_d02_cross_T2_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1_land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_land = xr.concat([x1_land,x1_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_ocean = xr.concat([x1_ocean,x1_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_avg = xr.concat([x1_avg,x1_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_T2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
# x2 = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
# x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2_land = xr.concat([x2_land,x2_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_ocean = xr.concat([x2_ocean,x2_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_avg = xr.concat([x2_avg,x2_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_land_diff = x2_land - x1_land
x3_ocean_diff = x2_ocean - x1_ocean
x3_avg_diff = x2_avg - x1_avg

# Calculate the difference between land & ocean
x4_cntl_diff = x1_land - x1_ocean
x4_NCRF_diff = x2_land - x2_ocean

# Control
l1_land = x1_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Control$_{land}$'
)
l1_ocean = x1_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='Control$_{ocean}$'
)
l1_avg = x1_avg.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Control$_{avg}$'
)
# NCRF
l2_land = x2_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{land}$'
)
l2_ocean = x2_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{ocean}$'
)
l2_avg = x2_avg.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{avg}$'
)

# ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
# ax1.set_ylim([-1,1.5])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal 2m Temperature Difference (NCRF-Control) over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km, Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=1,loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -50 # km
ocean_dist_thresh = 50 # km

## Load Data
# Control
x1 = da_d02_cross_T2_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1_land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_land = xr.concat([x1_land,x1_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_ocean = xr.concat([x1_ocean,x1_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_avg = xr.concat([x1_avg,x1_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_T2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
# x2 = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
# x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2_land = xr.concat([x2_land,x2_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_ocean = xr.concat([x2_ocean,x2_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_avg = xr.concat([x2_avg,x2_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_land_diff = x2_land - x1_land
x3_ocean_diff = x2_ocean - x1_ocean
x3_avg_diff = x2_avg - x1_avg

# Calculate the difference between land & ocean
x4_cntl_diff = x1_land - x1_ocean
x4_NCRF_diff = x2_land - x2_ocean

l2_gradient = x4_cntl_diff.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Gradient$_{control}$'
)
l2_gradient = x4_NCRF_diff.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='Gradient$_{NCRF}$'
)

# ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
ax1.set_ylim([-6,0])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal 2m Temperature Gradient (Land-Ocean) over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km, Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=1,loc='upper right')


# 2m Potential Temperature

# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 100 # km

## Load Data
# Control
x1 = da_d02_cross_Theta2_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1_land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_land = xr.concat([x1_land,x1_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_ocean = xr.concat([x1_ocean,x1_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_avg = xr.concat([x1_avg,x1_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_Theta2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='T2',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
# x2 = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
# x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2_land = xr.concat([x2_land,x2_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_ocean = xr.concat([x2_ocean,x2_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_avg = xr.concat([x2_avg,x2_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_land_diff = x2_land - x1_land
x3_ocean_diff = x2_ocean - x1_ocean
x3_avg_diff = x2_avg - x1_avg

# Calculate the difference between land & ocean
x4_cntl_diff = x1_land - x1_ocean
x4_NCRF_diff = x2_land - x2_ocean

l1_land = x1_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Control$_{land}$'
)
l1_ocean = x1_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='Control$_{ocean}$'
)
l1_avg = x1_avg.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Control$_{avg}$'
)

l2_land = x2_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{land}$'
)
l2_ocean = x2_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{ocean}$'
)
l2_avg = x2_avg.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{avg}$'
)

ax1.set_xlim([0,24])
# ax1.set_ylim([-1,1.5])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal 2m Potential Temperature over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km, Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Potential Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=2,loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 100 # km

## Load Data
# Control
x1 = da_d02_cross_Theta2_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1_land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_land = xr.concat([x1_land,x1_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_ocean = xr.concat([x1_ocean,x1_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_avg = xr.concat([x1_avg,x1_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_Theta2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='Theta 2m',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
# x2 = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
# x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2_land = xr.concat([x2_land,x2_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_ocean = xr.concat([x2_ocean,x2_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_avg = xr.concat([x2_avg,x2_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_land_diff = x2_land - x1_land
x3_ocean_diff = x2_ocean - x1_ocean
x3_avg_diff = x2_avg - x1_avg

# Calculate the difference between land & ocean
x4_cntl_diff = x1_land - x1_ocean
x4_NCRF_diff = x2_land - x2_ocean

l1_land = x3_land_diff.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Difference$_{land}$'
)
l1_ocean = x3_ocean_diff.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='Difference$_{ocean}$'
)
l1_avg = x3_avg_diff.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Difference$_{avg}$'
)

ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
ax1.set_ylim([-1,1.5])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal 2m Potential Temperature Difference (NCRF-Control) over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km, Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Potential Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=1,loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -100 # km
ocean_dist_thresh = 100 # km

## Load Data
# Control
x1 = da_d02_cross_Theta2_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1_land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_land = xr.concat([x1_land,x1_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_ocean = xr.concat([x1_ocean,x1_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_avg = xr.concat([x1_avg,x1_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_Theta2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='Theta 2m',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
# x2 = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
# x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2_land = xr.concat([x2_land,x2_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_ocean = xr.concat([x2_ocean,x2_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_avg = xr.concat([x2_avg,x2_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_land_diff = x2_land - x1_land
x3_ocean_diff = x2_ocean - x1_ocean
x3_avg_diff = x2_avg - x1_avg

# Calculate the difference between land & ocean
x4_cntl_diff = x1_land - x1_ocean
x4_NCRF_diff = x2_land - x2_ocean

# Control
l1_land = x1_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '-',
	label='Control$_{land}$'
)
l1_ocean = x1_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '-',
	label='Control$_{ocean}$'
)
l1_avg = x1_avg.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Control$_{avg}$'
)
# NCRF
l2_land = x2_land.plot.line(
    ax=ax1,
    color='peru',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{land}$'
)
l2_ocean = x2_ocean.plot.line(
    ax=ax1,
    color='dodgerblue',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{ocean}$'
)
l2_avg = x2_avg.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='NCRF$_{avg}$'
)

# ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
# ax1.set_ylim([-1,1.5])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal 2m Potential Temperature Difference (NCRF-Control) over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km, Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Potential Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=1,loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0,0])

land_dist_thresh = -50 # km
ocean_dist_thresh = 50 # km

## Load Data
# Control
x1 = da_d02_cross_Theta2_cntl.sel(Time=slice('2015-11-23T00','2015-12-02T00'))
x1 = x1.groupby('Time.hour').mean()
x1_land = x1.where((x1.Distance<0)&(x1.Distance>land_dist_thresh)).mean(['Distance','Spread'])
x1_land = xr.concat([x1_land,x1_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_ocean = x1.where((x1.Distance>0)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_ocean = xr.concat([x1_ocean,x1_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x1_avg = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<ocean_dist_thresh)).mean(['Distance','Spread'])
x1_avg = xr.concat([x1_avg,x1_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# 00 UTC icloud off
x2 = da_d02_cross_Theta2_CRFoff.isel(Time=np.arange(1,25,1)).sel(Lead=slice(0,18,2)).stack(Times=['Lead','Time']).transpose('Times','Distance','Spread')
x2 = x2.mean('Spread')
da = x2.drop_vars(['Lead','Time','Times'])
x2 = xr.DataArray(
				name='Theta 2m',
				data=da.values,
				dims=['Time','Distance'],
				coords=dict(
					Time = da.EnsTime.values,
					Distance = da.Distance.values,
					)
	)
x2 = x2.sel(Time=slice('2015-11-23T00','2015-12-02T00')).groupby('Time.hour').mean()
# x2 = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
# x2 = xr.concat([x2,x2[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_land = x2.where((x2.Distance<0)&(x2.Distance>land_dist_thresh)).mean(['Distance'])
x2_land = xr.concat([x2_land,x2_land[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_ocean = x2.where((x2.Distance>0)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_ocean = xr.concat([x2_ocean,x2_ocean[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))
x2_avg = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<ocean_dist_thresh)).mean(['Distance'])
x2_avg = xr.concat([x2_avg,x2_avg[0]],dim='hour').assign_coords(hour=np.arange(0,25,1))

# Calculate the difference between CRFoff & control
x3_land_diff = x2_land - x1_land
x3_ocean_diff = x2_ocean - x1_ocean
x3_avg_diff = x2_avg - x1_avg

# Calculate the difference between land & ocean
x4_cntl_diff = x1_land - x1_ocean
x4_NCRF_diff = x2_land - x2_ocean

l2_gradient = x4_cntl_diff.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '-',
	label='Gradient$_{control}$'
)
l2_gradient = x4_NCRF_diff.plot.line(
    ax=ax1,
    color='black',
    linewidth=2,
    linestyle = '--',
	label='Gradient$_{NCRF}$'
)

# ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax1.set_xlim([0,24])
# ax1.set_ylim([-6,0])
ax1.set_xticks(np.arange(0,27,3))
ax1.set_title(f'Domain Averaged Diurnal 2m Potential Temperature Gradient (Land-Ocean) over Western Central Sumatra Coast\nLand:0->{land_dist_thresh:.0f}km, Ocean:0->{ocean_dist_thresh:.0f}km')
ax1.set_xlabel('UTC')
ax1.set_ylabel('Potential Temperature [$K$]')
ax1.grid(linestyle='--', axis='x', linewidth=1)
ax1.grid(linestyle='--', axis='y', linewidth=1)
ax1.legend(ncol=1,loc='upper right')


# ### Diurnal Composite plots

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Control Normalized Diurnal Composite of Rain Rate over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_RR_CRFoff.sel(Lead=slice(0,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(2,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0:2,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_RR_CRFoff.sel(Lead=slice(1,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# # Switch to local time
# x3 = x3.assign_coords({'Time':x3.Time + np.timedelta64(7,'h')})
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(14,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:14,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3) / x1.std('hour')

# Normalize the data
x1 = (x1) / x1.std('hour')
# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,6.5,0.5),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,6.5,0.5),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,6.5,0.5),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Normalized Rain Rate [RR/$\sigma_{cntl}$]')


# #### Normal Wind

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 950
fig.suptitle(str(level)+'hPa Normal Wind Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_NormalWind_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_NormalWind_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_NormalWind_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-4,4.5,0.5),
	levels=np.arange(np.floor(x1.min()),(-np.floor(x1.min()))+1,1),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-4,4.5,0.5),
	levels=np.arange(np.floor(x1.min()),(-np.floor(x1.min()))+1,1),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-4,4.5,0.5),
	levels=np.arange(np.floor(x1.min()),(-np.floor(x1.min()))+1,1),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(-4,5,1))
cbar.set_ticks(np.arange(np.floor(x1.min()),(-np.floor(x1.min()))+2,2))
cbar.set_label('Normal wind @ ' + str(level) + 'hPa [$m/s$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 950
fig.suptitle(str(level)+'hPa Control Normalized Normal Wind Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_NormalWind_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_NormalWind_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = x2 / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_NormalWind_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = x3 / x1.std('hour')

# Normalize the data
x1 = x1 / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-4,5,1))
cbar.set_label('Normalized Normal wind @ ' + str(level) + 'hPa [$U_{Normal}/\sigma_{cntl}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 950
fig.suptitle(str(level)+'hPa Normalized Normal Wind Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_NormalWind_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_NormalWind_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_NormalWind_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-4,5,1))
cbar.set_label('Normal wind @ ' + str(level) + 'hPa [$m/s$]')

# Plot the colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(U_{Normal}-U_{Normal_{cntl}})/\sigma_{cntl}$]')


# #### Virtual Temperature

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 950

fig.suptitle(str(level)+'hPa Virtual Temperature Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_Tv_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_Tv_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_Tv_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(295,301.25,0.25),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(295,301.25,0.25),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(295,301.25,0.25),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(295,302,1))
cbar.set_label('Virtual Temperature @ ' + str(level) + 'hPa [$Tv$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 950
fig.suptitle(str(level)+'hPa Normalized Virtual Temperature Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_Tv_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_Tv_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_Tv_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(295,301.25,0.25),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(295,302,1))
cbar.set_label('Virtual Temperature @ ' + str(level) + 'hPa [$K$]')

# Plot the colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(Tv-Tv_{cntl})/\sigma_{cntl}$]')


# #### Vertical Wind Speed

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 200
fig.suptitle(str(level)+'hPa Vertical Wind Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_W_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_W_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_W_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-0.1,0.11,0.01),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-0.1,0.11,0.01),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-0.1,0.11,0.01),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-0.1,0.11,.02))
cbar.set_label('Vertical wind @ ' + str(level) + 'hPa [$m/s$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 950
fig.suptitle(str(level)+'hPa Control Normalized Vertical Wind Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_W_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_W_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = x2 / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_W_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = x3 / x1.std('hour')

# Normalize the data
x1 = x1 / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-4,5,1))
cbar.set_label('Normalized Vertical wind @ ' + str(level) + 'hPa [$W/\sigma_{cntl}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 950
fig.suptitle(str(level)+'hPa Normalized Vertical Wind Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_W_cntl[12:].sel(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_W_CRFoff.sel(bottom_top=level, Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_W_CRFoff.sel(bottom_top=level, Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-0.1,0.11,0.01),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-0.1,0.11,.02))
cbar.set_label('Vertical wind @ ' + str(level) + 'hPa [$m/s$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(W-W_{cntl})/\sigma_{cntl}$]')


# #### Water Vapor

# In[ ]:


def round_to_two_significant_figures(number):
    """
    Rounds a float value to two significant figures using np.ceil or np.floor.

    Args:
        number (float): The input float value.

    Returns:
        float: The rounded value.
    """
    # Calculate the order of magnitude (power of 10) for the input number
    order_of_magnitude = np.floor(np.log10(np.abs(number)))

    # Calculate the factor to round to two significant figures
    factor = 10**(2 - order_of_magnitude)

    # Round the number using np.ceil or np.floor
    rounded_value = np.ceil(number * factor) / factor

    return rounded_value

def round_to_one_significant_figures(number):
    """
    Rounds a float value to two significant figures using np.ceil or np.floor.

    Args:
        number (float): The input float value.

    Returns:
        float: The rounded value.
    """
    # Calculate the order of magnitude (power of 10) for the input number
    order_of_magnitude = np.floor(np.log10(np.abs(number)))

    # Calculate the factor to round to two significant figures
    factor = 10**(1 - order_of_magnitude)

    # Round the number using np.ceil or np.floor
    rounded_value = np.ceil(number * factor) / factor

    return rounded_value

round_to_one_significant_figures(0.00001427)


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 200
fig.suptitle(str(level)+'hPa Water Vapor Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QV_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QV_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QV_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10))
cbar.set_label('Water vapor mixing ratio @ ' + str(level) + 'hPa [$kg/kg$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 750
fig.suptitle(str(level)+'hPa Normalized Water Vapor Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QV_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QV_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QV_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the datasets
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-0.1,0.11,0.01),
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10))
cbar.set_label('Water vapor mixing ratio @ ' + str(level) + 'hPa [$kg/kg$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(QV-QV_{cntl})/\sigma_{cntl}$]')


# #### Total Q

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 850
fig.suptitle(str(level)+'hPa Total Q Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QTotal_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=10)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=10)
ax3.set_title('NCRF Sunset', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10))
cbar.set_label('Total Q mixing ratio @ ' + str(level) + 'hPa [$kg/kg$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 200
fig.suptitle(str(level)+'hPa Normalized Total Q Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QTotal_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the datasets
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-0.1,0.11,0.01),
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10))
cbar.set_label('Total Q mixing ratio @ ' + str(level) + 'hPa [$kg/kg$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(QTotal-QTotal_{cntl})/\sigma_{cntl}$]')


# #### Latent Heating

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 700
fig.suptitle(str(level)+'hPa Latent Heating Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_H_DIABATIC_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_H_DIABATIC_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_H_DIABATIC_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

if abs(x1.min())>abs(x1.max()): 
	x1_lims = [round_to_one_significant_figures(x1.min()),-round_to_one_significant_figures(x1.min())]
else:
	x1_lims = [-round_to_one_significant_figures(x1.max()),round_to_one_significant_figures(x1.max())]
x1_lims

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.linspace(round_to_two_significant_figures(x1_lims[0]),round_to_two_significant_figures(x1_lims[1]),10),
	levels=np.linspace(x1_lims[0],x1_lims[1],15),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.linspace(round_to_two_significant_figures(x1_lims[0]),round_to_two_significant_figures(x1_lims[1]),10),
	levels=np.linspace(x1_lims[0],x1_lims[1],15),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.linspace(round_to_two_significant_figures(x1_lims[0]),round_to_two_significant_figures(x1_lims[1]),10),
	levels=np.linspace(x1_lims[0],x1_lims[1],15),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.formatter.set_powerlimits((-2, 0))
cbar.set_ticks(np.linspace(x1_lims[0],x1_lims[1],15)[::2])
cbar.set_label('Latent Heating @ ' + str(level) + 'hPa [$K/s$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 700
fig.suptitle(str(level)+'hPa Control Normalized Latent Heating Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3
# Load Data
	# Control Data
x1 = da_d02_cross_H_DIABATIC_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_H_DIABATIC_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = x2 / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_H_DIABATIC_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = x3 / x1.std('hour')

# Normalize the data
x1 = x1 / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-4,5,1))
cbar.set_label('Normalized Latent Heating @ ' + str(level) + 'hPa [$W/\sigma_{cntl}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 500
fig.suptitle(str(level)+'hPa Normalized Latent Heating Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_H_DIABATIC_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_H_DIABATIC_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_H_DIABATIC_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

if abs(x1.min())>abs(x1.max()): 
	x1_lims = [round_to_one_significant_figures(x1.min()),-round_to_one_significant_figures(x1.min())]
else:
	x1_lims = [-round_to_one_significant_figures(x1.max()),round_to_one_significant_figures(x1.max())]
x1_lims

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.linspace(round_to_two_significant_figures(x1_lims[0]),round_to_two_significant_figures(x1_lims[1]),10),
	levels=np.linspace(x1_lims[0],x1_lims[1],15),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(x1_lims[0],x1_lims[1],15)[::2])
cbar.formatter.set_powerlimits((-2, 0))
cbar.set_label('Latent Heating @ ' + str(level) + 'hPa [$K/s$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(H_DIABATIC-H_DIABATIC_{cntl})/\sigma_{cntl}$]')


# #### Cloud Fraction

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 200
fig.suptitle(str(level)+'hPa Cloud Fraction Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_CLDFRA_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_CLDFRA_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_CLDFRA_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# if abs(x1.min())>abs(x1.max()): 
# 	x1_lims = [round_to_one_significant_figures(x1.min()),-round_to_one_significant_figures(x1.min())]
# else:
# 	x1_lims = [-round_to_one_significant_figures(x1.max()),round_to_one_significant_figures(x1.max())]
# x1_lims

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.linspace(round_to_two_significant_figures(x1_lims[0]),round_to_two_significant_figures(x1_lims[1]),10),
	# levels=np.linspace(x1_lims[0],x1_lims[1],15),
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.linspace(round_to_two_significant_figures(x1_lims[0]),round_to_two_significant_figures(x1_lims[1]),10),
	# levels=np.linspace(x1_lims[0],x1_lims[1],15),
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.linspace(round_to_two_significant_figures(x1_lims[0]),round_to_two_significant_figures(x1_lims[1]),10),
	# levels=np.linspace(x1_lims[0],x1_lims[1],15),
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.formatter.set_powerlimits((-2, 0))
# cbar.set_ticks(np.linspace(x1_lims[0],x1_lims[1],15)[::2])
cbar.set_ticks(np.arange(0,.3,.05))
cbar.set_label('Cloud Fraction @ ' + str(level) + 'hPa [$K/s$]')


# #### Rain Rate

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

title = 'Rain Rate Diurnal Composite over ' + cross_title

fig.suptitle(title, fontsize=26)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_RR_CRFoff.sel(Lead=slice(0,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(2,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0:2,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_RR_CRFoff.sel(Lead=slice(1,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# # Switch to local time
# x3 = x3.assign_coords({'Time':x3.Time + np.timedelta64(7,'h')})
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(14,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:14,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3[13:].plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,4.25,0.25),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Fontsize
fs = 20

# Plot phase speed lines
# ax1.plot([-45,117],[9,24] , color='r')
# ax1.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
# ax1.plot([-45,225],[9,24] , color='r')
# ax1.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
# ax1.plot([-45,603],[9,24] , color='r')
# ax1.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

# ax1.plot([-45,117],[33,48] , color='r')
# ax1.text(60, 43, '3 m/s', color='r', weight='bold',fontsize=fs-6)
# ax1.plot([-45,225],[33,48] , color='r')
# ax1.text(180, 39, '5 m/s', color='r', weight='bold',fontsize=fs-6)
# ax1.plot([-45,603],[33,48] , color='r')
# ax1.text(300, 37, '12 m/s', color='r', weight='bold',fontsize=fs-6)

# ax2.plot([-45,117],[9,24] , color='r')
# ax2.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
# ax2.plot([-45,225],[9,24] , color='r')
# ax2.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
# ax2.plot([-45,603],[9,24] , color='r')
# ax2.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

# ax3.plot([-45,117],[9,24] , color='r')
# ax3.text(60, 19, '3 m/s', color='r', weight='bold',fontsize=fs-6)
# ax3.plot([-45,225],[9,24] , color='r')
# ax3.text(180, 15, '5 m/s', color='r', weight='bold',fontsize=fs-6)
# ax3.plot([-45,603],[9,24] , color='r')
# ax3.text(300, 13, '12 m/s', color='r', weight='bold',fontsize=fs-6)

# ax3.plot([-45,117],[33,48] , color='r')
# ax3.text(60, 43, '3 m/s', color='r', weight='bold',fontsize=fs-6)
# ax3.plot([-45,225],[33,48] , color='r')
# ax3.text(180, 39, '5 m/s', color='r', weight='bold',fontsize=fs-6)
# ax3.plot([-45,603],[33,48] , color='r')
# ax3.text(300, 37, '12 m/s', color='r', weight='bold',fontsize=fs-6)

# Borneo phase speed lines
ax1.plot([-205,-43],[13.5,28.5] , color='r')
ax1.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
ax1.plot([-205,65],[13.5,28.5] , color='r')
ax1.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
ax1.plot([-205,530],[13.5,30.5] , color='r')
ax1.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

ax1.plot([-205,-43],[37.5,52.5] , color='r')
ax1.text(-145, 44, '3 m/s', color='r', weight='bold',fontsize=fs-6)
ax1.plot([-205,65],[37.5,52.5] , color='r')
ax1.text(25, 43.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
ax1.plot([-205,530],[37.5,54.5] , color='r')
ax1.text(145, 40, '12 m/s', color='r', weight='bold',fontsize=fs-6)

ax2.plot([-205,-43],[13.5,28.5] , color='r')
ax2.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
ax2.plot([-205,65],[13.5,28.5] , color='r')
ax2.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
ax2.plot([-205,530],[13.5,30.5] , color='r')
ax2.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

ax3.plot([-205,-43],[13.5,28.5] , color='r')
ax3.text(-145, 20, '3 m/s', color='r', weight='bold',fontsize=fs-6)
ax3.plot([-205,65],[13.5,28.5] , color='r')
ax3.text(25, 19.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
ax3.plot([-205,530],[13.5,30.5] , color='r')
ax3.text(145, 16, '12 m/s', color='r', weight='bold',fontsize=fs-6)

ax3.plot([-205,-43],[37.5,52.5] , color='r')
ax3.text(-145, 44, '3 m/s', color='r', weight='bold',fontsize=fs-6)
ax3.plot([-205,65],[37.5,52.5] , color='r')
ax3.text(25, 43.5, '5 m/s', color='r', weight='bold',fontsize=fs-6)
ax3.plot([-205,530],[37.5,54.5] , color='r')
ax3.text(145, 40, '12 m/s', color='r', weight='bold',fontsize=fs-6)


# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--', linewidth=2)
ax1.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax2.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax3.set_xlabel('Distance from coast [km]', fontsize=fs-6)
ax1.set_ylabel('Local Time', fontsize=fs)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('')
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_ylabel('Lead Time', fontsize=fs)
ax1.set_xticks(np.arange(-200,600,100))
ax2.set_xticks(np.arange(-200,600,100))
ax3.set_xticks(np.arange(-200,600,100))
ax1.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax2.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax3.set_xticklabels(np.arange(-200,600,100),fontsize=fs-6)
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(7,25,3),np.arange(1,25,3),np.arange(1,7,3))),fontsize=fs-6)
ax2.set_yticklabels(np.arange(0,48,3),fontsize=fs-6)
ax3.set_yticklabels(np.arange(-12,36,3),fontsize=fs-6)
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=fs)
ax1.set_title('a)', loc='left', fontsize=fs)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=fs)
ax2.set_title('b)', loc='left', fontsize=fs)
ax3.set_title('NCRF Sunset', loc='center', fontsize=fs)
ax3.set_title('c)', loc='left', fontsize=fs)

ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()


# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,5,1))
cbar.set_ticklabels(np.arange(0,5,1), fontsize=fs-6)
cbar.set_label('Rain Rate [$mm d^{-1}$]', fontsize=fs-6)


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Normalized Rain Rate Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_RR_CRFoff.sel(Lead=slice(0,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(2,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0:2,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_RR_CRFoff.sel(Lead=slice(1,18,2))[1:,...]		# Start from 1 instead of 0 because 0 is accumulated RR
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(14,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:14,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,4.5,0.5),
	cmap='gray_r',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,5,1))
cbar.set_label('Rain Rate [$mm/day$]')

cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label("[$(RR-RR_{cntl})/\sigma_{cntl}$]")


# #### CAPE

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('CAPE Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_CAPE_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_CAPE_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_CAPE_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,2100,100),
	cmap='Reds',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,2100,100),
	cmap='Reds',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,2100,100),
	cmap='Reds',
	# cmap='nipy_spectral',
	center=0,
	extend='max'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=10)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=10)
ax3.set_title('NCRF Sunset', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,2200,200))
cbar.set_label('CAPE [$J/kg$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('CAPE Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_CAPE_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_CAPE_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_CAPE_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,2100,100),
	cmap='Reds',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=10)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=10)
ax3.set_title('NCRF Sunset', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,2200,200))
cbar.set_label('CAPE [$J/kg$]')

cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label("[$(CAPE-CAPE{cntl})/\sigma_{cntl}$]")


# #### CIN

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('CIN Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_CIN_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_CIN_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_CIN_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,52,2),
	cmap='Reds',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,52,2),
	cmap='Reds',
	center=0,
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,52,2),
	cmap='Reds',
	center=0,
	extend='max'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=10)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=10)
ax3.set_title('NCRF Sunset', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,55,5))
cbar.set_label('CIN [$J/kg$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('CIN Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_CIN_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_CIN_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_CIN_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,52,2),
	cmap='Reds',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=10)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=10)
ax3.set_title('NCRF Sunset', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,55,5))
cbar.set_label('CIN [$J/kg$]')

cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label("[$(CIN-CIN{cntl})/\sigma_{cntl}$]")


# #### Surface SW Flux Down

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Shortwave Surface Flux Down Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_SWDownSfc_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_SWDownSfc_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_SWDownSfc_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,1050,50),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,1050,50),
	cmap='Reds'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,1050,50),
	cmap='Reds'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,1100,100))
cbar.set_label('SW SFC Flux Down [$W/m^{2}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Control Normalized Diurnal Composite of Shortwave Surface Flux Down over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_SWDownSfc_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_SWDownSfc_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_SWDownSfc_CRFoff.sel(Lead=slice(1,18,2))
# # Switch to local time
# x3 = x3.assign_coords({'Time':x3.Time + np.timedelta64(7,'h')})
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3) / x1.std('hour')

# Normalize the data
x1 = (x1) / x1.std('hour')
# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()


# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,5.5,0.5),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,5.5,0.5),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,5.5,0.5),
	cmap='Reds',
	extend='max'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,6,1))
cbar.set_label('Normalized SW SFC Flux Down [SwFlxDn/$\sigma_{cntl}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Normalized Shortwave Surface Flux Down Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_SWDownSfc_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_SWDownSfc_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_SWDownSfc_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,1050,50),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,1100,100))
cbar.set_label('SW SFC Flux Down [$W/m^{2}$]')

cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label("[$(SwFlxDn-SwFlxDn_{cntl})/\sigma_{cntl}$]")


# #### Surface Latent Heat Flux

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Latent Heat Flux at Surface Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_LH_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_LH_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_LH_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,550,50),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,550,50),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,550,50),
	cmap='Reds',
	extend='max'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,600,100))
cbar.set_label('Surface Latent Heat Flux [$W/m^{2}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Control Normalized Diurnal Composite of Latent Heat Flux at Surface over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_LH_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_LH_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_LH_CRFoff.sel(Lead=slice(1,18,2))
# # Switch to local time
# x3 = x3.assign_coords({'Time':x3.Time + np.timedelta64(7,'h')})
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3) / x1.std('hour')

# Normalize the data
x1 = (x1) / x1.std('hour')
# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()


# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,5.5,0.5),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,5.5,0.5),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,5.5,0.5),
	cmap='Reds',
	extend='max'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,6,1))
cbar.set_label('Surface Latent Heat Flux [LH/$\sigma_{cntl}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Normalized Latent Heat Flux at Surface Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_LH_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_LH_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_LH_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,550,50),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,600,100))
cbar.set_label('Surface Latent Heat Flux [$W/m^{2}$]')

cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label("[$(LH-LH_{cntl})/\sigma_{cntl}$]")


# #### Upward Heat Flux at Surface

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Upward Heat Flux at Surface Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_HFX_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_HFX_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-300,325,25),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-300,325,25),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-300,325,25),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-300,350,50))
cbar.set_label('Surface Heat Flux Up [$W/m^{2}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Control Normalized Upward Heat Flux at Surface over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_HFX_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_HFX_CRFoff.sel(Lead=slice(1,18,2))
# # Switch to local time
# x3 = x3.assign_coords({'Time':x3.Time + np.timedelta64(7,'h')})
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3) / x1.std('hour')

# Normalize the data
x1 = (x1) / x1.std('hour')
# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()


# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-4,4.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-4,5,1))
cbar.set_label('Normalized Surface Heat Flux Up [HFX/$\sigma_{cntl}$]')


# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Normalized Upward Heat Flux at Surface Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_HFX_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_HFX_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-300,325,25),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-300,350,50))
cbar.set_label('Surface Heat Flux Up [$W/m^{2}$]')

cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100, extend='both')
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label("[$(HFX-HFX{cntl})/\sigma_{cntl}$]")


# #### Upward Moisture Flux at Surface

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
fig.suptitle('Upward Moisture Flux at Surface Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

Lv=2.5*(10**6)	# [J/kg]	# liquid water
smoothing_num=3

# Load Data
	# Control Data
# x1 = da_d02_cross_QFX_cntl[12:]*Lv	# remove first 12 hrs (spin-up) and convert to W/m^2
x1 = da_d02_cross_QFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
# x2 = da_d02_cross_QFX_CRFoff.sel(Lead=slice(0,18,2))*Lv	# Convert to W/m^2
x2 = da_d02_cross_QFX_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))

	# Sim data CRF off @ 12UTC	[Evening]
# x3 = da_d02_cross_QFX_CRFoff.sel(Lead=slice(1,18,2))*Lv	# Convert to W/m^2
x3 = da_d02_cross_QFX_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(0,550,50),
	levels=np.arange(0,0.0002,0.00002),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(0,550,50),
	levels=np.arange(0,0.0002,0.00002),
	cmap='Reds',
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(0,550,50),
	levels=np.arange(0,0.0002,0.00002),
	cmap='Reds',
	extend='max'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,600,100))
# cbar.set_label('Surface Moisture Flux Up [$kg/m^{2}s$]')
cbar.set_label('Surface Moisture Flux Up [$W/m^{2}$]')


# ### Visualize at different levels below

# In[ ]:


# Set the level you want to look at
level = 200


# #### Normal Wind

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 200
fig.suptitle(str(level)+'hPa Normalized Normal Wind Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_NormalWind_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_NormalWind_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_NormalWind_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(np.floor(x1.min()),(-np.floor(x1.min()))+1,1),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(-4,5,1))
cbar.set_ticks(np.arange(np.floor(x1.min()),(-np.floor(x1.min()))+2,2))
cbar.set_label('Normal wind @ ' + str(level) + 'hPa [$m/s$]')

# Plot the colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(U_{Normal}-U_{Normal_{cntl}})/\sigma_{cntl}$]')


# ##### Vertically Averaged

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

levels = [250,200]

fig.suptitle(f'Vertically Averaged {levels[0]:.0f}-{levels[1]:.0f}hPa Normalized Normal Wind Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_NormalWind_cntl[12:].sel(bottom_top=slice(levels[0],levels[1])).mean('bottom_top')
# Vertically integrate
# x1 = (-1/9.8)*x1*(np.diff(levels))
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_NormalWind_CRFoff.sel(Lead=slice(0,18,2), bottom_top=slice(levels[0],levels[1])).mean('bottom_top')
# Vertically integrate
# x2 = (-1/9.8)*x2*(np.diff(levels))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_NormalWind_CRFoff.sel(Lead=slice(1,18,2), bottom_top=slice(levels[0],levels[1])).mean('bottom_top')
# Vertically integrate
# x3 = (-1/9.8)*x3*(np.diff(levels))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the datasets
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-14,15,1),
	levels=np.arange(np.floor(x1.min()),(-np.floor(x1.min()))+1,1),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(np.floor(x1.min()),(-np.floor(x1.min()))+2,2))
# cbar.set_ticks(np.arange(-14,16,2))
# cbar.set_ticks()
cbar.set_label('Vertically Averaged Normal Wind [$kg/ms$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(U_{Normal}-U_{Normal_{cntl}})/\sigma_{cntl}$]')


# #### Virtual Temperature

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
level = 980
fig.suptitle(str(level)+'hPa Normalized Virtual Temperature Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_Tv_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_Tv_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_Tv_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(np.floor(x1.min()),np.ceil(x1.max())+.25,.25),
	cmap='gray_r',
	# center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(np.floor(x1.min()),np.ceil(x1.max())+.5,.5))
cbar.set_label('Virtual Temperature @ ' + str(level) + 'hPa [$K$]')

# Plot the colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(Tv-Tv_{cntl})/\sigma_{cntl}$]')


# #### Vertical Wind Speed

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

fig.suptitle(str(level)+'hPa Normalized Vertical Wind Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_W_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_W_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_W_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-0.1,0.11,0.01),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-0.1,0.11,.02))
cbar.set_label('Vertical wind @ ' + str(level) + 'hPa [$m/s$]')

	# Plot colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(W-W_{cntl})/\sigma_{cntl}$]')


# #### Water Vapor

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

fig.suptitle(str(level)+'hPa Normalized Water Vapor Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QV_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QV_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QV_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the datasets
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-0.1,0.11,0.01),
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10))
cbar.set_label('Water vapor mixing ratio @ ' + str(level) + 'hPa [$kg/kg$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(QV-QV_{cntl})/\sigma_{cntl}$]')


# Vertically Integrated

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

levels = [950,200]

fig.suptitle(f'Vertically integrated {levels[0]:.0f}-{levels[1]:.0f}hPa Normalized Water Vapor Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QV_cntl[13:-12].sel(bottom_top=slice(levels[0],levels[1]))
# Vertically integrate
x1 = vertical_integration(x1)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QV_CRFoff.sel(Lead=slice(0,18,2), bottom_top=slice(levels[0],levels[1])).mean('Lead')
# Vertically integrate
x2 = vertical_integration(x2)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean('Spread')
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QV_CRFoff.sel(Lead=slice(1,18,2), bottom_top=slice(levels[0],levels[1])).mean('Lead')
# Vertically integrate
x3 = vertical_integration(x3)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean('Spread')
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the datasets
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-0.1,0.11,0.01),
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=10)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=10)
ax3.set_title('NCRF Sunset', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10))
cbar.set_label('Vertically integrated QV [$kg^{2}/kgm^{2}$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(QV-QV_{cntl})/\sigma_{cntl}$]')


# #### Total Q

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level=900

fig.suptitle(str(level)+'hPa Normalized Total Q Difference from Control over Western Central Sumatra Coast', fontsize=14)


smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QTotal_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the datasets
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-0.1,0.11,0.01),
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10))
cbar.set_label('Total Q mixing ratio @ ' + str(level) + 'hPa [$kg/kg$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(QTotal-QTotal_{cntl})/\sigma_{cntl}$]')


# ##### Vertically Integrated

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

levels = [500,200]

fig.suptitle(f'Vertically integrated {levels[0]:.0f}-{levels[1]:.0f}hPa Normalized Total Q Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_QTotal_cntl[13:-12].sel(bottom_top=slice(levels[0],levels[1]))
# Vertically integrate
x1 = vertical_integration(x1)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(0,18,2), bottom_top=slice(levels[0],levels[1])).mean('Lead')
# Vertically integrate
x2 = vertical_integration(x2)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean('Spread')
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_QTotal_CRFoff.sel(Lead=slice(1,18,2), bottom_top=slice(levels[0],levels[1])).mean('Lead')
# Vertically integrate
x3 = vertical_integration(x3)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean('Spread')
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Smooth the datasets
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.arange(-0.1,0.11,0.01),
	levels=np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10),
	cmap='gray_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control', loc='center', fontsize=10)
ax2.set_title('NCRF Sunrise', loc='center', fontsize=10)
ax3.set_title('NCRF Sunset', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(round_to_two_significant_figures(x1.min()),round_to_two_significant_figures(x1.max()),10))
cbar.set_label('Vertically integrated QTotal [$kg^{2}/kgm^{2}$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(QTotal-QTotal_{cntl})/\sigma_{cntl}$]')


# #### Latent Heating

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 200
fig.suptitle(str(level)+'hPa Normalized Latent Heating Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_H_DIABATIC_cntl[12:].interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_H_DIABATIC_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_H_DIABATIC_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

if abs(x1.min())>abs(x1.max()): 
	x1_lims = [round_to_one_significant_figures(x1.min()),-round_to_one_significant_figures(x1.min())]
else:
	x1_lims = [-round_to_one_significant_figures(x1.max()),round_to_one_significant_figures(x1.max())]
x1_lims

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	# levels=np.linspace(round_to_two_significant_figures(x1_lims[0]),round_to_two_significant_figures(x1_lims[1]),10),
	levels=np.linspace(x1_lims[0],x1_lims[1],15),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Rain rate colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.linspace(x1_lims[0],x1_lims[1],15)[::2])
cbar.formatter.set_powerlimits((-2, 0))
cbar.set_label('Latent Heating @ ' + str(level) + 'hPa [$K/s$]')

	# 
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(H_DIABATIC-H_DIABATIC_{cntl})/\sigma_{cntl}$]')


# #### Cloud Fraction

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

level = 200

fig.suptitle(str(level)+'hPa Normalized Cloud Fraction Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_CLDFRA_cntl[12:].interp(bottom_top=level)
# x1 = da_d02_cross_MidCLDFRA_cntl[12:]
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_CLDFRA_CRFoff.sel(Lead=slice(0,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_CLDFRA_CRFoff.sel(Lead=slice(1,18,2)).interp(bottom_top=level)
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	# center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,.3,.05))
cbar.set_label('Cloud Fraction @ ' + str(level) + 'hPa')

	# Plot colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(Cld-Cld_{cntl})/\sigma_{cntl}$]')


# ##### All Clouds

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

fig.suptitle('Cloud Fraction Diurnal Composite over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Low clouds
x1 = da_d02_cross_LowCLDFRA_cntl[12:]
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Mid clouds
x2 = da_d02_cross_MidCLDFRA_cntl[12:]
# Composite diurnally, and then average over each cross-section
x2 = x2.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x2 = xr.concat([x2,x2],dim='hour',data_vars='all')
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# High clouds
x3 = da_d02_cross_HighCLDFRA_cntl[12:]
# Composite diurnally, and then average over each cross-section
x3 = x3.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x3 = xr.concat([x3,x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

# Smooth the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	extend='max'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	extend='max'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax1.axhline(y=24, color='k', linestyle='--')
ax2.axhline(y=24, color='k', linestyle='--')
ax3.axhline(y=24, color='k', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax3.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
# Set titles/labels
ax1.set_title('Low Clouds [1000-750hPa]', loc='center', fontsize=10)
ax2.set_title('Mid Clouds [750-500hPa]', loc='center', fontsize=10)
ax3.set_title('High Clouds [500-200hPa]', loc='center', fontsize=10)

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,.3,.05))
cbar.set_label('Cloud Fraction')


# ##### Low Clouds

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

fig.suptitle('Normalized Low [1000-750hPa] Cloud Fraction Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_LowCLDFRA_cntl[12:]
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_LowCLDFRA_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_LowCLDFRA_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	# center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,.3,.05))
cbar.set_label('Low Cloud Fraction')

	# Plot colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(Cld-Cld_{cntl})/\sigma_{cntl}$]')


# ##### Mid Clouds

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

fig.suptitle('Normalized Mid [750-500hPa] Cloud Fraction Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_MidCLDFRA_cntl[12:]
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_MidCLDFRA_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_MidCLDFRA_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	# center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,.3,.05))
cbar.set_label('Mid Cloud Fraction')

	# Plot colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(Cld-Cld_{cntl})/\sigma_{cntl}$]')


# ##### High Clouds

# In[ ]:


fig = plt.figure(figsize=(19.5,8))
gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.25, height_ratios=[0.875,0.03], wspace=0.1, width_ratios=[.33,.33,.33])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

fig.suptitle('Normalized High [500-200hPa] Cloud Fraction Difference from Control over Western Central Sumatra Coast', fontsize=14)

smoothing_num=3

# Load Data
	# Control Data
x1 = da_d02_cross_HighCLDFRA_cntl[12:]
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Concat to make a more continuous plot
x1 = xr.concat([x1,x1],dim='hour',data_vars='all')
x1 = x1.assign_coords(hour=(['hour'],np.arange(-0.5,47.5)))

	# Sim data CRF off @ 00UTC	[Morning]
x2 = da_d02_cross_HighCLDFRA_CRFoff.sel(Lead=slice(0,18,2))
# Composite diurnally, and then average over each cross-section
x2 = x2.mean(['Lead','Spread'])
x2 = x2.rename({'Time':'hour'})
x2 = x2.assign_coords(hour=('hour',np.arange(1,37)))
# Concat to make a more continuous plot
x2 = xr.concat([x1[0,:], x2],dim='hour',data_vars='all').transpose()
x2 = x2.assign_coords(hour=(['hour'],np.arange(-0.5,36.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x2 = (x2-x1) / x1.std('hour')

	# Sim data CRF off @ 12UTC	[Evening]
x3 = da_d02_cross_HighCLDFRA_CRFoff.sel(Lead=slice(1,18,2))
# Composite diurnally, and then average over each cross-section
x3 = x3.mean(['Lead','Spread'])
x3 = x3.rename({'Time':'hour'})
x3 = x3.assign_coords(hour=('hour',np.arange(13,49)))
# Concat to make a more continuous plot
x3 = xr.concat([x1[0:13,:], x3],dim='hour',data_vars='all')
x3 = x3.assign_coords(hour=(['hour'],np.arange(-0.5,48.5)))
# Normalize the data
	# Subtract the cntl out and divide by the cntl standard deviation
x3 = (x3-x1) / x1.std('hour')

# Normalize the control data
x1 = x1.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x2 = x2.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
x3 = x3.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Plot the cross-sectional data
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(0,.275,0.025),
	cmap='gray_r',
	# center=0,
	extend='max'
)

# Plot the cross-sectional data
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot the cross-sectional data
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'hour',
	add_colorbar=False,
	levels=np.arange(-3,3.5,0.5),
	cmap='RdBu_r',
	center=0,
	extend='both'
)

# Plot phase speed lines
ax1.plot([-25,137],[9,24] , color='r')
ax1.text(60, 18, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[9,24] , color='r')
ax1.text(165, 15, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[9,24] , color='r')
ax1.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax1.plot([-25,137],[33,48] , color='r')
ax1.text(60, 42, '3 m/s', color='r', weight='bold')
ax1.plot([-25,245],[33,48] , color='r')
ax1.text(165, 39, '5 m/s', color='r', weight='bold')
ax1.plot([-25,623],[33,48] , color='r')
ax1.text(300, 37.5, '12 m/s', color='r', weight='bold')

ax2.plot([-25,137],[9,24] , color='r')
ax2.text(60, 18, '3 m/s', color='r', weight='bold')
ax2.plot([-25,245],[9,24] , color='r')
ax2.text(165, 15, '5 m/s', color='r', weight='bold')
ax2.plot([-25,623],[9,24] , color='r')
ax2.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[9,24] , color='r')
ax3.text(60, 18, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[9,24] , color='r')
ax3.text(165, 15, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[9,24] , color='r')
ax3.text(300, 13.5, '12 m/s', color='r', weight='bold')

ax3.plot([-25,137],[33,48] , color='r')
ax3.text(60, 42, '3 m/s', color='r', weight='bold')
ax3.plot([-25,245],[33,48] , color='r')
ax3.text(165, 39, '5 m/s', color='r', weight='bold')
ax3.plot([-25,623],[33,48] , color='r')
ax3.text(300, 37.5, '12 m/s', color='r', weight='bold')


ax1.set_xlim([x1.Distance[0],x1.Distance[-1]])
ax2.set_xlim([x2.Distance[0],x2.Distance[-1]])
ax3.set_xlim([x3.Distance[0],x3.Distance[-1]])
ax1.set_ylim([0,46.5])
ax2.set_ylim([0,46.5])
ax3.set_ylim([0,46.5])
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
ax3.axhline(y=12, color='r', linestyle='--')
ax1.set_xlabel('Distance from coast [km]')
ax2.set_xlabel('Distance from coast [km]')
ax3.set_xlabel('Distance from coast [km]')
ax1.set_ylabel('UTC')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.set_yticks(x1.hour[0::3].values+0.5)
ax2.set_yticks(x1.hour[0::3].values+0.5)
ax3.set_yticks(x1.hour[0::3].values+0.5)
ax1.set_yticklabels(np.concatenate((np.arange(0,24,3),np.arange(0,24,3))))
ax2.set_yticklabels(np.arange(0,48,3))
ax3.set_yticklabels(np.arange(-12,36,3))
# Set titles/labels
ax1.set_title('Control Simulation', loc='center', fontsize=10)
ax2.set_title('CRF off @ 00UTC/07LT', loc='center', fontsize=10)
ax3.set_title('CRF off @ 12UTC/19LT', loc='center', fontsize=10)
# ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
# ax1.set_title('', loc='center')

# Plot the colorbar
	# Vertical Wind colorbar
cax1 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=cax1, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,.3,.05))
cbar.set_label('High Cloud Fraction')

	# Plot colorbar
cax2 = fig.add_subplot(gs[1, 1:3])
cbar = plt.colorbar(cf2, cax=cax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-3,4,1))
cbar.set_label('[$(Cld-Cld_{cntl})/\sigma_{cntl}$]')


# ### Early Initiation Analysis

# #### Diurnal Composite Relationship

# In[ ]:


fig = plt.figure(figsize=(15,5))
gs = gridspec.GridSpec(nrows=1, ncols=3, hspace=0.075)

land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_cut = 9

fig.suptitle(f'Diurnal Relationship over 0->{land_dist_thresh:.0f} km between 0-{time_cut:.0f}UTC Cloud Fraction and SW Down CRF @ Surface', fontsize=14)

# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)&(x1.hour<=time_cut))

# Mid Cloud Fraction
x2 = da_d02_cross_MidCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x2 = x2.groupby('Time.hour').mean().mean('Spread')
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)&(x2.hour<=time_cut))

# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x3 = x3.groupby('Time.hour').mean().mean('Spread')
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)&(x3.hour<=time_cut))

# Shortwave Down @ Surface [W/m^2]
x4 = da_d02_cross_SWDownSfc_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x4 = x4.groupby('Time.hour').mean().mean('Spread')
x4_clear = da_d02_cross_SWDownSfcClear_cntl[12:].groupby('Time.hour').mean().mean('Spread')
# Cloud-radiative Forcing for SW Down @ SFC
x4 = x4 - x4_clear
# Normalize by Clear-sky SW Down @ Surface at each hour
# x4 = x4/(da_d02_cross_SWDownSfcClear_cntl[12:].groupby('Time.hour').mean().mean('Spread').max('Distance'))
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0)&(x4.hour<=time_cut))

## Plot & Calculate linear regressions	##
ax1 = fig.add_subplot(gs[0,0])
s1 = plt.scatter(x1,x4)
x = x1.values[~np.isnan(x1)]
y = x4.values[~np.isnan(x4)]
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
l1 = ax1.plot([0,1],[intercept,(slope+intercept)],'r',linestyle='--')
r2 = r_value**2
ax1.text(.1, -50, f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax2 = fig.add_subplot(gs[0,1])
s2 = plt.scatter(x2,x4)
x = x2.values[~np.isnan(x2)]
y = x4.values[~np.isnan(x4)]
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
l1 = ax2.plot([0,1],[intercept,(slope+intercept)],'r',linestyle='--')
r2 = r_value**2
ax2.text(.1, -50, f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax3 = fig.add_subplot(gs[0,2])
x = x3.values[~np.isnan(x3)]
y = x4.values[~np.isnan(x4)]
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
l1 = ax3.plot([0,1],[intercept,(slope+intercept)],'r',linestyle='--')
r2 = r_value**2
ax3.text(.1, -50, f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')
s3 = plt.scatter(x3,x4)


ax1.set_title('Low Cloud Fraction',fontsize=12)
ax1.set_xlabel('Cloud Fraction',fontsize=10)
ax1.set_ylabel('SW CRF Down',fontsize=10)
# ax1.set_ylabel('Normalized SW Down @ Surface',fontsize=10)
ax1.set_xlim([0,0.25])
# ax1.set_ylim([0,1])
ax1.set_ylim([-600,0])
ax1.grid(linestyle='--', axis='both', linewidth=1)

ax2.set_title('Mid Cloud Fraction',fontsize=12)
ax2.set_xlabel('Cloud Fraction',fontsize=10)
ax2.set_xlim([0,0.25])
# ax2.set_ylim([0,1])
ax2.set_ylim([-600,0])
ax2.grid(linestyle='--', axis='both', linewidth=1)

ax3.set_title('High Cloud Fraction',fontsize=12)
ax3.set_xlabel('Cloud Fraction',fontsize=10)
ax3.set_xlim([0,0.25])
# ax3.set_ylim([0,1])
ax3.set_ylim([-600,0])
ax3.grid(linestyle='--', axis='both', linewidth=1)


# In[ ]:


fig = plt.figure(figsize=(15,5))
gs = gridspec.GridSpec(nrows=1, ncols=3, hspace=0.075)

land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_cut = 9

fig.suptitle(f'Diurnal relationship over 0->{land_dist_thresh:.0f} km between 0-{time_cut:.0f}UTC between Cloud Fraction and Net CRF @ Surface', fontsize=14)
# f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^{2}={r2:.2f}$'
# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x1 = x1.groupby('Time.hour').mean().mean('Spread')
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)&(x1.hour<=time_cut))

# Mid Cloud Fraction
x2 = da_d02_cross_MidCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x2 = x2.groupby('Time.hour').mean().mean('Spread')
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)&(x2.hour<=time_cut))

# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x3 = x3.groupby('Time.hour').mean().mean('Spread')
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)&(x3.hour<=time_cut))

# Net Down @ Surface [W/m^2]
x4 = da_d02_cross_NetSfcCRF_cntl[12:]	# remove first 12 hrs (spin-up)
# Composite diurnally, and then average over each cross-section
x4 = x4.groupby('Time.hour').mean().mean('Spread')
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0)&(x4.hour<=time_cut))

## Plot & Calculate linear regressions	##
ax1 = fig.add_subplot(gs[0,0])
s1 = plt.scatter(x1,x4)
x = x1.values[~np.isnan(x1)]
y = x4.values[~np.isnan(x4)]
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
l1 = ax1.plot([0,1],[intercept,(slope+intercept)],'r',linestyle='--')
r2 = r_value**2
ax1.text(.1, -50, f'$y = {slope:.2f}x {intercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax2 = fig.add_subplot(gs[0,1])
s2 = plt.scatter(x2,x4)
x = x2.values[~np.isnan(x2)]
y = x4.values[~np.isnan(x4)]
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
l1 = ax2.plot([0,1],[intercept,(slope+intercept)],'r',linestyle='--')
r2 = r_value**2
ax2.text(.1, -50, f'$y = {slope:.2f}x {intercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

ax3 = fig.add_subplot(gs[0,2])
x = x3.values[~np.isnan(x3)]
y = x4.values[~np.isnan(x4)]
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
l1 = ax3.plot([0,1],[intercept,(slope+intercept)],'r',linestyle='--')
r2 = r_value**2
ax3.text(.1, -50, f'$y = {slope:.2f}x {intercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')
s3 = plt.scatter(x3,x4)


ax1.set_title('Low Cloud Fraction',fontsize=12)
ax1.set_xlabel('Cloud Fraction',fontsize=10)
ax1.set_ylabel('Net Surface CRF [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([0,0.25])
# ax1.set_ylim([0,1])
ax1.set_ylim([-600,0])
ax1.grid(linestyle='--', axis='both', linewidth=1)

ax2.set_title('Mid Cloud Fraction',fontsize=12)
ax2.set_xlabel('Cloud Fraction',fontsize=10)
ax2.set_xlim([0,0.25])
# ax2.set_ylim([0,1])
ax2.set_ylim([-600,0])
ax2.grid(linestyle='--', axis='both', linewidth=1)

ax3.set_title('High Cloud Fraction',fontsize=12)
ax3.set_xlabel('Cloud Fraction',fontsize=10)
ax3.set_xlim([0,0.25])
# ax3.set_ylim([0,1])
ax3.set_ylim([-600,0])
ax3.grid(linestyle='--', axis='both', linewidth=1)


# #### Time Series Relationship

# In[ ]:


fig = plt.figure(figsize=(15,12))
gs = gridspec.GridSpec(nrows=3, ncols=3, hspace=0.35, height_ratios=[0.45,0.275,0.275])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,:])
ax5 = fig.add_subplot(gs[2,:])

# Set thresholds
land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_start = 0
time_end = 9
lag_plot = 0				# The lag relationship plotted on the first row

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1 = x1.sel(Time=x1.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Spread')

# Mid Cloud Fraction
x2 = da_d02_cross_MidCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x2 = x2.sel(Time=x2.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Spread')

# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x3 = x3.sel(Time=x3.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Spread')

# Net Down @ Surface [W/m^2]
x4 = da_d02_cross_SWSfcCRF_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x4 = x4.sel(Time=x4.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0)).mean('Spread')

## Calculate linear regressions	##
min_lag = -12
max_lag = 12

da1 = linreg(x1.values[~np.isnan(x1)], x4.values[~np.isnan(x1)], min_lag, max_lag)
da2 = linreg(x2.values[~np.isnan(x2)], x4.values[~np.isnan(x2)], min_lag, max_lag)
da3 = linreg(x3.values[~np.isnan(x3)], x4.values[~np.isnan(x3)], min_lag, max_lag)

## Plot linear regressions	##
s1 = ax1.scatter(x1,x4)
l1 = ax1.plot([0,1],[da1.where(da1==lag_plot,drop=True).yintercept.values, da1.where(da1==lag_plot,drop=True).yintercept.values+da1.where(da1==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da1.where(da1==lag_plot,drop=True).slope.values[0]
yintercept = da1.where(da1==lag_plot,drop=True).yintercept.values[0]
r2 = da1.where(da1==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(.20, -95, f'$y = {slope:.2f}x {yintercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s2 = ax2.scatter(x2,x4)
l2 = ax2.plot([0,1],[da2.where(da2==lag_plot,drop=True).yintercept.values, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax2.text(.20, -95, f'$y = {slope:.2f}x {yintercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s3 = ax3.scatter(x3,x4)
l3 = ax3.plot([0,1],[da3.where(da3==lag_plot,drop=True).yintercept.values, da3.where(da3==lag_plot,drop=True).yintercept.values+da3.where(da3==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da3.where(da3==lag_plot,drop=True).slope.values[0]
yintercept = da3.where(da3==lag_plot,drop=True).yintercept.values[0]
r2 = da3.where(da3==lag_plot,drop=True).rvalue.values[0]**2
ax3.text(.20, -95, f'$y = {slope:.2f}x {yintercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

l1_slope = ax4.plot(da1.values,da1.slope.values,'r',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_r2 = ax5.plot(da1.values,da1.rvalue.values**2,'r',linestyle='-',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

l2_slope = ax4.plot(da2.values,da2.slope.values,'b',linestyle='-',linewidth=1.5,label='Mid Cloud')	# plot slope (m)
l2_r2 = ax5.plot(da2.values,da2.rvalue.values**2,'b',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l2_p = ax5.plot(da2.values,da2.pvalue.values,'b',linestyle='--',linewidth=1.5)						# plot p-value

l3_slope = ax4.plot(da3.values,da3.slope.values,'c',linestyle='-',linewidth=1.5,label='High Cloud')	# plot slope (m)
l3_r2 = ax5.plot(da3.values,da3.rvalue.values**2,'c',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l3_p = ax5.plot(da3.values,da3.pvalue.values,'c',linestyle='--',linewidth=1.5)						# plot p-value


ax1.set_title('Low Cloud',fontsize=12)
ax1.set_xlabel('Cloud Fraction',fontsize=10)
ax1.set_ylabel('SW Surface CRF [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([0,0.45])
ax1.set_ylim([-800,0])
ax1.grid(linestyle='--', axis='both', linewidth=1)

ax2.set_title(f'Shortwave Surface CRF vs Cloud Fraction relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end-1:.0f}UTC\nLag = {lag_plot:.0f}\nMid Cloud',fontsize=12)
ax2.set_xlabel('Cloud Fraction',fontsize=10)
ax2.set_xlim([0,0.45])
ax2.set_ylim([-800,0])
ax2.grid(linestyle='--', axis='both', linewidth=1)

ax3.set_title('High Cloud',fontsize=12)
ax3.set_xlabel('Cloud Fraction',fontsize=10)
ax3.set_xlim([0,0.45])
ax3.set_ylim([-800,0])
ax3.grid(linestyle='--', axis='both', linewidth=1)

ax4.set_title('Lag Relationship\n Slope ($m$)',fontsize=12)
ax4.set_ylabel('Slope [$W/m^{2} / CldFrac$]',fontsize=10)
ax4.set_xlabel('Lag-step [hour]',fontsize=10)
ax4.set_xlim([min_lag,max_lag])
ax4.set_ylim([-2000,0])
ax4.set_xticks(np.arange(min_lag,max_lag+1,2))
ax4.grid(linestyle='--', axis='both', linewidth=1)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax4.legend(loc='upper right')

ax5.set_title('$R^{2}$ and p-value',fontsize=12)
ax5.set_xlabel('Lag-step [hour]',fontsize=10)
ax5.set_xlim([min_lag,max_lag])
ax5.set_ylim([0,1])
ax5.set_yticks(np.arange(0,1.2,.2))
ax5.set_xticks(np.arange(min_lag,max_lag+1,2))
ax5.grid(linestyle='--', axis='both', linewidth=1)
ax5.plot(da1.values,np.repeat(.05,len(da1.values)),color='k',linestyle='--')	# 95th percentile
ax5.text(0.075, 0.075, '$95^{th}$ Percentile', color='k')
ax5.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax5.legend(loc='upper right')


# In[ ]:


da2.stderror


# In[ ]:


fig = plt.figure(figsize=(15,12))
gs = gridspec.GridSpec(nrows=3, ncols=3, hspace=0.35, height_ratios=[0.45,0.275,0.275])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,:])
ax5 = fig.add_subplot(gs[2,:])

# Set thresholds
land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_start = 0
time_end = 9
lag_plot = 0				# The lag relationship plotted on the first row

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1 = x1.sel(Time=x1.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Spread')

# Mid Cloud Fraction
x2 = da_d02_cross_MidCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x2 = x2.sel(Time=x2.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Spread')

# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x3 = x3.sel(Time=x3.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Spread')

# Net Down @ Surface [W/m^2]
x4 = da_d02_cross_NetSfcCRF_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x4 = x4.sel(Time=x4.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0)).mean('Spread')

## Calculate linear regressions	##
min_lag = -12
max_lag = 12

da1 = linreg(x1.values[~np.isnan(x1)], x4.values[~np.isnan(x1)], min_lag, max_lag)
da2 = linreg(x2.values[~np.isnan(x2)], x4.values[~np.isnan(x2)], min_lag, max_lag)
da3 = linreg(x3.values[~np.isnan(x3)], x4.values[~np.isnan(x3)], min_lag, max_lag)

## Plot linear regressions	##
s1 = ax1.scatter(x1,x4)
l1 = ax1.plot([0,1],[da1.where(da1==lag_plot,drop=True).yintercept.values, da1.where(da1==lag_plot,drop=True).yintercept.values+da1.where(da1==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da1.where(da1==lag_plot,drop=True).slope.values[0]
yintercept = da1.where(da1==lag_plot,drop=True).yintercept.values[0]
r2 = da1.where(da1==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(.20, -95, f'$y = {slope:.2f}x {yintercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s2 = ax2.scatter(x2,x4)
l2 = ax2.plot([0,1],[da2.where(da2==lag_plot,drop=True).yintercept.values, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax2.text(.20, -95, f'$y = {slope:.2f}x {yintercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s3 = ax3.scatter(x3,x4)
l3 = ax3.plot([0,1],[da3.where(da3==lag_plot,drop=True).yintercept.values, da3.where(da3==lag_plot,drop=True).yintercept.values+da3.where(da3==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da3.where(da3==lag_plot,drop=True).slope.values[0]
yintercept = da3.where(da3==lag_plot,drop=True).yintercept.values[0]
r2 = da3.where(da3==lag_plot,drop=True).rvalue.values[0]**2
ax3.text(.20, -95, f'$y = {slope:.2f}x {yintercept:.2f}$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

l1_slope = ax4.plot(da1.values,da1.slope.values,'r',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_slope_err = ax4.fill_between(da1.values,da1.slope.values-da1.stderror.values,da1.slope.values+da1.stderror.values, alpha=0.15, color='r')
l1_r2 = ax5.plot(da1.values,da1.rvalue.values**2,'r',linestyle='-',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

l2_slope = ax4.plot(da2.values,da2.slope.values,'b',linestyle='-',linewidth=1.5,label='Mid Cloud')	# plot slope (m)
l2_slope_err = ax4.fill_between(da2.values,da2.slope.values-da2.stderror.values,da2.slope.values+da1.stderror.values, alpha=0.15, color='b')
l2_r2 = ax5.plot(da2.values,da2.rvalue.values**2,'b',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l2_p = ax5.plot(da2.values,da2.pvalue.values,'b',linestyle='--',linewidth=1.5)						# plot p-value

l3_slope = ax4.plot(da3.values,da3.slope.values,'c',linestyle='-',linewidth=1.5,label='High Cloud')	# plot slope (m)
l3_slope_err = ax4.fill_between(da3.values,da3.slope.values-da3.stderror.values,da3.slope.values+da3.stderror.values, alpha=0.15, color='c')
l3_r2 = ax5.plot(da3.values,da3.rvalue.values**2,'c',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l3_p = ax5.plot(da3.values,da3.pvalue.values,'c',linestyle='--',linewidth=1.5)						# plot p-value


ax1.set_title('Low Cloud',fontsize=12)
ax1.set_xlabel('Cloud Fraction',fontsize=10)
ax1.set_ylabel('Net Surface CRF [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([0,0.45])
ax1.set_ylim([-800,0])
ax1.grid(linestyle='--', axis='both', linewidth=1)

ax2.set_title(f'Net Surface CRF vs Cloud Fraction relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end-1:.0f}UTC\nLag = {lag_plot:.0f}\nMid Cloud',fontsize=12)
ax2.set_xlabel('Cloud Fraction',fontsize=10)
ax2.set_xlim([0,0.45])
ax2.set_ylim([-800,0])
ax2.grid(linestyle='--', axis='both', linewidth=1)

ax3.set_title('High Cloud',fontsize=12)
ax3.set_xlabel('Cloud Fraction',fontsize=10)
ax3.set_xlim([0,0.45])
ax3.set_ylim([-800,0])
ax3.grid(linestyle='--', axis='both', linewidth=1)

ax4.set_title('Lag Relationship\n Slope ($m$)',fontsize=12)
ax4.set_xlabel('Lag-step [hour]',fontsize=10)
ax4.set_ylabel('Slope [$W/m^{2} / CldFrac$]',fontsize=10)
ax4.set_xlim([min_lag,max_lag])
ax4.set_ylim([-2000,0])
ax4.set_xticks(np.arange(min_lag,max_lag+1,2))
ax4.grid(linestyle='--', axis='both', linewidth=1)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax4.legend(loc='upper right')

ax5.set_title('$R^{2}$ and p-value',fontsize=12)
ax5.set_xlabel('Lag-step [hour]',fontsize=10)
ax5.set_xlim([min_lag,max_lag])
ax5.set_ylim([0,1])
ax5.set_yticks(np.arange(0,1.2,.2))
ax5.set_xticks(np.arange(min_lag,max_lag+1,2))
ax5.grid(linestyle='--', axis='both', linewidth=1)
ax5.plot(da1.values,np.repeat(.05,len(da1.values)),color='k',linestyle='--')	# 95th percentile
ax5.text(0.075, 0.075, '$95^{th}$ Percentile', color='k')
ax5.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax5.legend(loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(15,12))
gs = gridspec.GridSpec(nrows=3, ncols=3, hspace=0.35, height_ratios=[0.45,0.275,0.275])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,:])
ax5 = fig.add_subplot(gs[2,:])

# Set thresholds
land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_start = 0
time_end = 9
lag_plot = 0				# The lag relationship plotted on the first row

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1 = x1.sel(Time=x1.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Spread')

# Mid Cloud Fraction
x2 = da_d02_cross_MidCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x2 = x2.sel(Time=x2.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Spread')

# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x3 = x3.sel(Time=x3.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Spread')

# Rain Rate [mm/day]
x4 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x4 = x4.sel(Time=x4.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0)).mean('Spread')

## Calculate linear regressions	##
min_lag = -12
max_lag = 12

da1 = linreg(x1.values[~np.isnan(x1)], x4.values[~np.isnan(x1)], min_lag, max_lag)
da2 = linreg(x2.values[~np.isnan(x2)], x4.values[~np.isnan(x2)], min_lag, max_lag)
da3 = linreg(x3.values[~np.isnan(x3)], x4.values[~np.isnan(x3)], min_lag, max_lag)

## Plot linear regressions	##
s1 = ax1.scatter(x1,x4)
l1 = ax1.plot([0,1],[da1.where(da1==lag_plot,drop=True).yintercept.values, da1.where(da1==lag_plot,drop=True).yintercept.values+da1.where(da1==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da1.where(da1==lag_plot,drop=True).slope.values[0]
yintercept = da1.where(da1==lag_plot,drop=True).yintercept.values[0]
r2 = da1.where(da1==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(.25, 14.5, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s2 = ax2.scatter(x2,x4)
l2 = ax2.plot([0,1],[da2.where(da2==lag_plot,drop=True).yintercept.values, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax2.text(.25, 14.5, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s3 = ax3.scatter(x3,x4)
l3 = ax3.plot([0,1],[da3.where(da3==lag_plot,drop=True).yintercept.values, da3.where(da3==lag_plot,drop=True).yintercept.values+da3.where(da3==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da3.where(da3==lag_plot,drop=True).slope.values[0]
yintercept = da3.where(da3==lag_plot,drop=True).yintercept.values[0]
r2 = da3.where(da3==lag_plot,drop=True).rvalue.values[0]**2
ax3.text(.25, 14.5, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

l1_slope = ax4.plot(da1.values,da1.slope.values,'r',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_r2 = ax5.plot(da1.values,da1.rvalue.values**2,'r',linestyle='-',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

l2_slope = ax4.plot(da2.values,da2.slope.values,'b',linestyle='-',linewidth=1.5,label='Mid Cloud')	# plot slope (m)
l2_r2 = ax5.plot(da2.values,da2.rvalue.values**2,'b',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l2_p = ax5.plot(da2.values,da2.pvalue.values,'b',linestyle='--',linewidth=1.5)						# plot p-value

l3_slope = ax4.plot(da3.values,da3.slope.values,'c',linestyle='-',linewidth=1.5,label='High Cloud')	# plot slope (m)
l3_r2 = ax5.plot(da3.values,da3.rvalue.values**2,'c',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l3_p = ax5.plot(da3.values,da3.pvalue.values,'c',linestyle='--',linewidth=1.5)						# plot p-value


ax1.set_title('Low Cloud',fontsize=12)
ax1.set_xlabel('Cloud Fraction',fontsize=10)
ax1.set_ylabel('Rain Rate [$mm/day$]',fontsize=10)
ax1.set_xlim([0,0.45])
ax1.set_ylim([0,16])
ax1.grid(linestyle='--', axis='both', linewidth=1)

ax2.set_title(f'Rain Rate vs Cloud Fraction relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end-1:.0f}UTC @ Lag = {lag_plot:.0f}\nMid Cloud',fontsize=12)
ax2.set_xlabel('Cloud Fraction',fontsize=10)
ax2.set_xlim([0,0.45])
ax2.set_ylim([0,16])
ax2.grid(linestyle='--', axis='both', linewidth=1)

ax3.set_title('High Cloud',fontsize=12)
ax3.set_xlabel('Cloud Fraction',fontsize=10)
ax3.set_xlim([0,0.45])
ax3.set_ylim([0,16])
ax3.grid(linestyle='--', axis='both', linewidth=1)

ax4.set_title('Lag Relationship\n Slope ($m$)',fontsize=12)
ax4.set_xlabel('Lag-step [hour]',fontsize=10)
ax4.set_ylabel('Slope [$mm/day / CldFrac$]',fontsize=10)
ax4.set_xlim([min_lag,max_lag])
ax4.set_ylim([0,15])
ax4.set_xticks(np.arange(min_lag,max_lag+1,2))
ax4.grid(linestyle='--', axis='both', linewidth=1)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax4.legend(loc='upper right')

ax5.set_title('$R^{2}$ and p-value',fontsize=12)
ax5.set_xlabel('Lag-step [hour]',fontsize=10)
ax5.set_xlim([min_lag,max_lag])
ax5.set_ylim([0,1])
ax5.set_yticks(np.arange(0,1.2,.2))
ax5.set_xticks(np.arange(min_lag,max_lag+1,2))
ax5.grid(linestyle='--', axis='both', linewidth=1)
ax5.plot(da1.values,np.repeat(.05,len(da1.values)),color='k',linestyle='--')	# 95th percentile
ax5.text(0.075, 0.075, '$95^{th}$ Percentile', color='k')
ax5.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax5.legend(loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(15,12))
gs = gridspec.GridSpec(nrows=3, ncols=3, hspace=0.35, height_ratios=[0.45,0.275,0.275])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,:])
ax5 = fig.add_subplot(gs[2,:])

# Set thresholds
land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_start = 0
time_end = 9
lag_plot = 0				# The lag relationship plotted on the first row

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1 = x1.sel(Time=x1.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0)).mean('Spread')

# Mid Cloud Fraction
x2 = da_d02_cross_MidCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x2 = x2.sel(Time=x2.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0)).mean('Spread')

# High Cloud Fraction
x3 = da_d02_cross_HighCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x3 = x3.sel(Time=x3.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0)).mean('Spread')

# Upward Sensible Heat Flux @ Surface [W/m^2] 
x4 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x4 = x4.sel(Time=x4.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0)).mean('Spread')

## Calculate linear regressions	##
min_lag = -12
max_lag = 12

da1 = linreg(x1.values[~np.isnan(x1)], x4.values[~np.isnan(x1)], min_lag, max_lag)
da2 = linreg(x2.values[~np.isnan(x2)], x4.values[~np.isnan(x2)], min_lag, max_lag)
da3 = linreg(x3.values[~np.isnan(x3)], x4.values[~np.isnan(x3)], min_lag, max_lag)

## Plot linear regressions	##
s1 = ax1.scatter(x1,x4)
l1 = ax1.plot([0,1],[da1.where(da1==lag_plot,drop=True).yintercept.values, da1.where(da1==lag_plot,drop=True).yintercept.values+da1.where(da1==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da1.where(da1==lag_plot,drop=True).slope.values[0]
yintercept = da1.where(da1==lag_plot,drop=True).yintercept.values[0]
r2 = da1.where(da1==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(.2, 270, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s2 = ax2.scatter(x2,x4)
l2 = ax2.plot([0,1],[da2.where(da2==lag_plot,drop=True).yintercept.values, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax2.text(.2, 270, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s3 = ax3.scatter(x3,x4)
l3 = ax3.plot([0,1],[da3.where(da3==lag_plot,drop=True).yintercept.values, da3.where(da3==lag_plot,drop=True).yintercept.values+da3.where(da3==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da3.where(da3==lag_plot,drop=True).slope.values[0]
yintercept = da3.where(da3==lag_plot,drop=True).yintercept.values[0]
r2 = da3.where(da3==lag_plot,drop=True).rvalue.values[0]**2
ax3.text(.2, 270, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

l1_slope = ax4.plot(da1.values,da1.slope.values,'r',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_r2 = ax5.plot(da1.values,da1.rvalue.values**2,'r',linestyle='-',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

l2_slope = ax4.plot(da2.values,da2.slope.values,'b',linestyle='-',linewidth=1.5,label='Mid Cloud')	# plot slope (m)
l2_r2 = ax5.plot(da2.values,da2.rvalue.values**2,'b',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l2_p = ax5.plot(da2.values,da2.pvalue.values,'b',linestyle='--',linewidth=1.5)						# plot p-value

l3_slope = ax4.plot(da3.values,da3.slope.values,'c',linestyle='-',linewidth=1.5,label='High Cloud')	# plot slope (m)
l3_r2 = ax5.plot(da3.values,da3.rvalue.values**2,'c',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l3_p = ax5.plot(da3.values,da3.pvalue.values,'c',linestyle='--',linewidth=1.5)						# plot p-value


ax1.set_title('Low Cloud',fontsize=12)
ax1.set_xlabel('Cloud Fraction',fontsize=10)
ax1.set_ylabel('Sensible Heat Flux [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([0,0.45])
ax1.set_ylim([0,300])
ax1.grid(linestyle='--', axis='both', linewidth=1)

ax2.set_title(f'Sensible Heat Flux vs Cloud Fraction relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end-1:.0f}UTC @ Lag = {lag_plot:.0f}\nMid Cloud',fontsize=12)
ax2.set_xlabel('Cloud Fraction',fontsize=10)
ax2.set_xlim([0,0.45])
ax2.set_ylim([0,300])
ax2.grid(linestyle='--', axis='both', linewidth=1)

ax3.set_title('High Cloud',fontsize=12)
ax3.set_xlabel('Cloud Fraction',fontsize=10)
ax3.set_xlim([0,0.45])
ax3.set_ylim([0,300])
ax3.grid(linestyle='--', axis='both', linewidth=1)

ax4.set_title('Lag Relationship\n Slope ($m$)',fontsize=12)
ax4.set_xlabel('Lag-step [hour]',fontsize=10)
ax4.set_ylabel('Slope [$W/m^{2} / CldFrac$]',fontsize=10)
ax4.set_xlim([min_lag,max_lag])
ax4.set_ylim([-300,300])
ax4.set_xticks(np.arange(min_lag,max_lag+1,2))
ax4.grid(linestyle='--', axis='both', linewidth=1)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.75)
ax4.legend(loc='upper right')

ax5.set_title('$R^{2}$ and p-value',fontsize=12)
ax5.set_xlabel('Lag-step [hour]',fontsize=10)
ax5.set_xlim([min_lag,max_lag])
ax5.set_ylim([0,1])
ax5.set_yticks(np.arange(0,1.2,.2))
ax5.set_xticks(np.arange(min_lag,max_lag+1,2))
ax5.grid(linestyle='--', axis='both', linewidth=1)
ax5.plot(da1.values,np.repeat(.05,len(da1.values)),color='k',linestyle='--')	# 95th percentile
ax5.text(0.075, 0.075, '$95^{th}$ Percentile', color='k')
ax5.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax5.legend(loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(15,12))
gs = gridspec.GridSpec(nrows=3, ncols=3, hspace=0.35, height_ratios=[0.45,0.275,0.275])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,:])
ax5 = fig.add_subplot(gs[2,:])

# Set thresholds
land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_start = 0
time_end = 9
lag_plot = 0				# The lag relationship plotted on the first row

# fig.suptitle(f'Time series relationship over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end:.0f}UTC between Cloud Fraction and SW CRF @ Surface', fontsize=14)

# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1 = x1.sel(Time=x1.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0),drop=True).mean('Spread')
# x1mean = x1.groupby('Time.hour').mean()
# for i in np.arange(time_start,time_end):
# 	x1[x1.Time.dt.hour.isin(i)] = x1[x1.Time.dt.hour.isin(i)]-x1mean.sel(hour=i)	# Divide by the mean recorded at each hour to remove diurnal variations

# Upward Sensible Heat Flux @ Surface [W/m^2] 
x2 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x2 = x2.sel(Time=x2.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0),drop=True).mean('Spread')
# x2mean = x2.groupby('Time.hour').mean()
# for i in np.arange(time_start,time_end):
# 	x2[x2.Time.dt.hour.isin(i)] = x2[x2.Time.dt.hour.isin(i)]-x2mean.sel(hour=i)	# Divide by the mean recorded at each hour to remove diurnal variations

# Rain Rate [mm/day]
x3 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x3 = x3.sel(Time=x3.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0),drop=True).mean('Spread')
# x3mean = x3.groupby('Time.hour').mean()
# for i in np.arange(time_start,time_end):
# 	x3[x3.Time.dt.hour.isin(i)] = x3[x3.Time.dt.hour.isin(i)]-x3mean.sel(hour=i)	# Divide by the mean recorded at each hour to remove diurnal variations

# Net Surface CRF [W/m^2] 
x4 = da_d02_cross_NetSfcCRF_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x4 = x4.sel(Time=x4.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0),drop=True).mean('Spread')

## Calculate linear regressions	##
min_lag = -12
max_lag = 12

da1 = linreg(x1.values[~np.isnan(x1)], x4.values[~np.isnan(x1)], min_lag, max_lag)
da2 = linreg(x2.values[~np.isnan(x2)], x4.values[~np.isnan(x2)], min_lag, max_lag)
da3 = linreg(x3.values[~np.isnan(x3)], x4.values[~np.isnan(x3)], min_lag, max_lag)

## Plot linear regressions	##
s1 = ax1.scatter(x1,x4)
l1 = ax1.plot([0,1],[da1.where(da1==lag_plot,drop=True).yintercept.values, da1.where(da1==lag_plot,drop=True).yintercept.values+da1.where(da1==lag_plot,drop=True).slope.values],'r',linestyle='--')
slope = da1.where(da1==lag_plot,drop=True).slope.values[0]
yintercept = da1.where(da1==lag_plot,drop=True).yintercept.values[0]
r2 = da1.where(da1==lag_plot,drop=True).rvalue.values[0]**2
ax1.text(.2, -95, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s2 = ax2.scatter(x2,x4)
l2 = ax2.plot([0,300],[da2.where(da2==lag_plot,drop=True).yintercept.values, da2.where(da2==lag_plot,drop=True).yintercept.values+da2.where(da2==lag_plot,drop=True).slope.values*300],'r',linestyle='--')
slope = da2.where(da2==lag_plot,drop=True).slope.values[0]
yintercept = da2.where(da2==lag_plot,drop=True).yintercept.values[0]
r2 = da2.where(da2==lag_plot,drop=True).rvalue.values[0]**2
ax2.text(150, -775, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

s3 = ax3.scatter(x3,x4)
l3 = ax3.plot([0,16],[da3.where(da3==lag_plot,drop=True).yintercept.values, da3.where(da3==lag_plot,drop=True).yintercept.values+da3.where(da3==lag_plot,drop=True).slope.values*16],'r',linestyle='--')
slope = da3.where(da3==lag_plot,drop=True).slope.values[0]
yintercept = da3.where(da3==lag_plot,drop=True).yintercept.values[0]
r2 = da3.where(da3==lag_plot,drop=True).rvalue.values[0]**2
ax3.text(7.5, -95, f'$y = {slope:.2f}x + ({yintercept:.2f})$\n$R^{2}={r2:.2f}$', color='k', weight='bold')

l1_slope = ax4.plot(da1.values,da1.slope.values/da1.slope.std().values,'r',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_r2 = ax5.plot(da1.values,da1.rvalue.values**2,'r',linestyle='-',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

l2_slope = ax4.plot(da2.values,da2.slope.values/da2.slope.std().values,'b',linestyle='-',linewidth=1.5,label='HFX')		# plot slope (m)
l2_r2 = ax5.plot(da2.values,da2.rvalue.values**2,'b',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l2_p = ax5.plot(da2.values,da2.pvalue.values,'b',linestyle='--',linewidth=1.5)						# plot p-value

l3_slope = ax4.plot(da3.values,da3.slope.values/da3.slope.std().values,'c',linestyle='-',linewidth=1.5,label='RR')			# plot slope (m)
l3_r2 = ax5.plot(da3.values,da3.rvalue.values**2,'c',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
l3_p = ax5.plot(da3.values,da3.pvalue.values,'c',linestyle='--',linewidth=1.5)						# plot p-value


ax1.set_title('Low Cloud',fontsize=12)
ax1.set_xlabel('Cloud Fraction',fontsize=10)
ax1.set_ylabel('Net Surface CRF [$W/m^{2}$]',fontsize=10)
ax1.set_xlim([0,0.45])
ax1.set_ylim([-800,0])
ax1.grid(linestyle='--', axis='both', linewidth=1)

ax2.set_title(f'Net Surface CRF relationships over 0->{land_dist_thresh:.0f} km during {time_start:.0f}-{time_end-1:.0f}UTC\nLag = {lag_plot:.0f}\nSensible Heat Flux Up',fontsize=12)
ax2.set_xlabel('HFX [$W/m^{2}$]',fontsize=10)
ax2.set_xlim([0,300])
ax2.set_ylim([-800,0])
ax2.grid(linestyle='--', axis='both', linewidth=1)

ax3.set_title('Rain Rate',fontsize=12)
ax3.set_xlabel('RR [mm/day]',fontsize=10)
ax3.set_xlim([0,16])
ax3.set_ylim([-800,0])
ax3.grid(linestyle='--', axis='both', linewidth=1)

ax4.set_title('Lag Relationship\n Standardized Slope ($m$)',fontsize=12)
ax4.set_xlabel('Lag-step [hour]',fontsize=10)
ax4.set_ylabel('Standardized Slope [$W/m^{2} / variable$]',fontsize=10)
ax4.set_xlim([min_lag,max_lag])
# ax4.set_ylim([-300,300])
ax4.set_xticks(np.arange(min_lag,max_lag+1,2))
ax4.grid(linestyle='--', axis='both', linewidth=1)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.75)
ax4.legend(loc='upper right')

ax5.set_title('$R^{2}$ and p-value',fontsize=12)
ax5.set_xlabel('Lag-step [hour]',fontsize=10)
ax5.set_xlim([min_lag,max_lag])
ax5.set_ylim([0,1])
ax5.set_yticks(np.arange(0,1.2,.2))
ax5.set_xticks(np.arange(min_lag,max_lag+1,2))
ax5.grid(linestyle='--', axis='both', linewidth=1)
ax5.plot(da1.values,np.repeat(.05,len(da1.values)),color='k',linestyle='--')	# 95th percentile
ax5.text(0.075, 0.075, '$95^{th}$ Percentile', color='k')
ax5.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax5.legend(loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(nrows=2, ncols=1, hspace=0.4)
# ax1 = fig.add_subplot(gs[0,0])
# ax2 = fig.add_subplot(gs[0,1])
# ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,:])
ax5 = fig.add_subplot(gs[1,:])

# Set thresholds
land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_start = 0
time_end = 24	# MAKE SURE you +1 whatever UTC time you want to end on i.e., if you want to end at 9utc, time_end = 10


# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1 = x1.sel(Time=x1.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0),drop=True).mean('Spread')

# Mid Cloud Fraction
x1_mid = da_d02_cross_MidCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1_mid = x1_mid.sel(Time=x1_mid.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1_mid = x1_mid.where((x1_mid.Distance>land_dist_thresh)&(x1_mid.Distance<0),drop=True).mean('Spread')

# High Cloud Fraction
x1_high = da_d02_cross_HighCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1_high = x1_high.sel(Time=x1_high.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1_high = x1_high.where((x1_high.Distance>land_dist_thresh)&(x1_high.Distance<0),drop=True).mean('Spread')

# Upward Sensible Heat Flux @ Surface [W/m^2] 
x2 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x2 = x2.sel(Time=x2.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0),drop=True).mean('Spread')

# Rain Rate [mm/day]
x3 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# x3 = da_d02_cross_NormalWind_cntl[12:].sel(bottom_top=950)	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x3 = x3.sel(Time=x3.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0),drop=True).mean('Spread')

# Net Surface CRF [W/m^2] 
x4 = da_d02_cross_NetSfcCRF_cntl[12:]	# remove first 12 hrs (spin-up)
# x4 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x4 = x4.sel(Time=x4.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0),drop=True).mean('Spread')

## Calculate linear regressions	##
	# Here we will find the linear regressions at lag=0 over each hour of the day to observe how the relationship evolves diurnally
min_lag = 0
max_lag = 0

# Initialize with the 0UTC and change dim name from 'lag' to 'Time'
da1 = linreg(x1.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})
da1_mid = linreg(x1_mid.sel(Time=x1_mid.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1_mid.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})
da1_high = linreg(x1_high.sel(Time=x1_high.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1_high.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})
da2 = linreg(x2.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})
da3 = linreg(x3.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})

# Loop through the times
for i in np.arange(time_start+1,time_end):
	# Concat and then adjust the values to correspond to the hour/UTC
	da1 = xr.concat([da1,linreg(x1.sel(Time=x1.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x1.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	da1_mid = xr.concat([da1_mid,linreg(x1_mid.sel(Time=x1_mid.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x1_mid.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	da1_high = xr.concat([da1_high,linreg(x1_high.sel(Time=x1_high.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x1_high.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	
	da2 = xr.concat([da2,linreg(x2.sel(Time=x2.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x2.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	da3 = xr.concat([da3,linreg(x3.sel(Time=x3.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x3.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	
da1.values=np.arange(time_start,time_end)
da1_mid.values=np.arange(time_start,time_end)
da1_high.values=np.arange(time_start,time_end)
da2.values=np.arange(time_start,time_end)
da3.values=np.arange(time_start,time_end)


# l1_slope = ax4.plot(da1.values,da1.slope.values/da1.slope.std().values,'r',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_slope = ax4.plot(da1.values,da1.slope.values,'r',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_r2 = ax5.plot(da1.values,da1.rvalue.values**2,'r',linestyle='-',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
# l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

# l1_mid_slope = ax4.plot(da1_mid.values,da1_mid.slope.values/da1_mid.slope.std().values,'g',linestyle='-',linewidth=1.5,label='Mid Cloud')	# plot slope (m)
l1_mid_slope = ax4.plot(da1_mid.values,da1_mid.slope.values,'r',linestyle='--',linewidth=1.5,label='Mid Cloud')	# plot slope (m)
l1_mid_r2 = ax5.plot(da1_mid.values,da1_mid.rvalue.values**2,'r',linestyle='--',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
# l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

# l1_high_slope = ax4.plot(da1_high.values,da1_high.slope.values/da1_high.slope.std().values,'m',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_high_slope = ax4.plot(da1_high.values,da1_high.slope.values,'r',linestyle=':',linewidth=1.5,label='High Cloud')	# plot slope (m)
l1_high_r2 = ax5.plot(da1_high.values,da1_high.rvalue.values**2,'r',linestyle=':',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
# l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

# l2_slope = ax4.plot(da2.values,da2.slope.values/da2.slope.std().values,'b',linestyle='-',linewidth=1.5,label='HFX')		# plot slope (m)
l2_slope = ax4.plot(da2.values,da2.slope.values,'b',linestyle='-',linewidth=1.5,label='HFX')		# plot slope (m)
l2_r2 = ax5.plot(da2.values,da2.rvalue.values**2,'b',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
# l2_p = ax5.plot(da2.values,da2.pvalue.values,'b',linestyle='--',linewidth=1.5)						# plot p-value

# l3_slope = ax4.plot(da3.values,da3.slope.values/da3.slope.std().values,'c',linestyle='-',linewidth=1.5,label='RR')			# plot slope (m)
l3_slope = ax4.plot(da3.values,da3.slope.values,'c',linestyle='-',linewidth=1.5,label='RR')			# plot slope (m)
l3_r2 = ax5.plot(da3.values,da3.rvalue.values**2,'c',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
# l3_p = ax5.plot(da3.values,da3.pvalue.values,'c',linestyle='--',linewidth=1.5)						# plot p-value

ax4.set_title(f'Diurnal Evolution of Net Surface CRF linear relationship over 0->{land_dist_thresh:.0f} km\nSlope ($m$)',fontsize=12)
ax4.set_xlabel('Time [UTC]',fontsize=10)
# ax4.set_ylabel('Standardized Slope [$m / std(m)$]',fontsize=10)
ax4.set_ylabel('Slope',fontsize=10)
ax4.set_xlim([time_start,time_end])
# ax4.set_ylim([-4,4])
# ax4.set_ylim([-500,500])
ax4.set_xticks(np.arange(time_start,time_end+1,3))
ax4.grid(linestyle='--', axis='both', linewidth=1)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.75)
ax4.legend(loc='lower right')

ax5.set_title('Varience Explained [$R^{2}$]',fontsize=12)
ax5.set_xlabel('Time [UTC]',fontsize=10)
ax5.set_ylabel('$R^{2}$',fontsize=10)
ax5.set_xlim([time_start,time_end])
ax5.set_ylim([0,1])
ax5.set_yticks(np.arange(0,1.2,.2))
ax5.set_xticks(np.arange(time_start,time_end+1,3))
ax5.grid(linestyle='--', axis='both', linewidth=1)
# ax5.legend(loc='upper right')


# In[ ]:


fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(nrows=2, ncols=1, hspace=0.4)
# ax1 = fig.add_subplot(gs[0,0])
# ax2 = fig.add_subplot(gs[0,1])
# ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,:])
ax5 = fig.add_subplot(gs[1,:])

# Set thresholds
land_dist_thresh = -100		# km
# ocean_dist_thresh = 100	# km
time_start = 0
time_end = 24	# MAKE SURE you +1 whatever UTC time you want to end on i.e., if you want to end at 9utc, time_end = 10


# Low Cloud Fraction
x1 = da_d02_cross_LowCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1 = x1.sel(Time=x1.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1 = x1.where((x1.Distance>land_dist_thresh)&(x1.Distance<0),drop=True).mean('Spread')

# Mid Cloud Fraction
x1_mid = da_d02_cross_MidCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1_mid = x1_mid.sel(Time=x1_mid.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1_mid = x1_mid.where((x1_mid.Distance>land_dist_thresh)&(x1_mid.Distance<0),drop=True).mean('Spread')

# High Cloud Fraction
x1_high = da_d02_cross_HighCLDFRA_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x1_high = x1_high.sel(Time=x1_high.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x1_high = x1_high.where((x1_high.Distance>land_dist_thresh)&(x1_high.Distance<0),drop=True).mean('Spread')

# Upward Sensible Heat Flux @ Surface [W/m^2] 
x2 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x2 = x2.sel(Time=x2.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x2 = x2.where((x2.Distance>land_dist_thresh)&(x2.Distance<0),drop=True).mean('Spread')

# Rain Rate [mm/day]
x3 = da_d02_cross_RR_cntl[12:]	# remove first 12 hrs (spin-up)
# x3 = da_d02_cross_NormalWind_cntl[12:].sel(bottom_top=950)	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x3 = x3.sel(Time=x3.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x3 = x3.where((x3.Distance>land_dist_thresh)&(x3.Distance<0),drop=True).mean('Spread')

# Net Surface CRF [W/m^2] 
x4 = da_d02_cross_NetSfcCRF_cntl[12:]	# remove first 12 hrs (spin-up)
# x4 = da_d02_cross_HFX_cntl[12:]	# remove first 12 hrs (spin-up)
# Only select the times you want to look at
x4 = x4.sel(Time=x4.Time.dt.hour.isin(np.arange(time_start,time_end)))
# Isolate over a specific region/time
x4 = x4.where((x4.Distance>land_dist_thresh)&(x4.Distance<0),drop=True).mean('Spread')

## Calculate linear regressions	##
	# Here we will find the linear regressions at lag=0 over each hour of the day to observe how the relationship evolves diurnally
min_lag = 0
max_lag = 0

# Initialize with the 0UTC and change dim name from 'lag' to 'Time'
da1 = linreg(x1.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})
da1_mid = linreg(x1_mid.sel(Time=x1_mid.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1_mid.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})
da1_high = linreg(x1_high.sel(Time=x1_high.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1_high.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})
da2 = linreg(x2.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})
da3 = linreg(x3.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), x4.sel(Time=x1.Time.dt.hour.isin(0)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})

# Loop through the times
for i in np.arange(time_start+1,time_end):
	# Concat and then adjust the values to correspond to the hour/UTC
	da1 = xr.concat([da1,linreg(x1.sel(Time=x1.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x1.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	da1_mid = xr.concat([da1_mid,linreg(x1_mid.sel(Time=x1_mid.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x1_mid.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	da1_high = xr.concat([da1_high,linreg(x1_high.sel(Time=x1_high.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x1_high.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	
	da2 = xr.concat([da2,linreg(x2.sel(Time=x2.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x2.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	da3 = xr.concat([da3,linreg(x3.sel(Time=x3.Time.dt.hour.isin(i)).values.flatten(), x4.sel(Time=x3.Time.dt.hour.isin(i)).values.flatten(), min_lag, max_lag).rename({'lag':'Time'})],dim='Time')
	
da1.values=np.arange(time_start,time_end)
da1_mid.values=np.arange(time_start,time_end)
da1_high.values=np.arange(time_start,time_end)
da2.values=np.arange(time_start,time_end)
da3.values=np.arange(time_start,time_end)


l1_slope = ax4.plot(da1.values,da1.slope.values/da1.slope.std().values,'r',linestyle='-',linewidth=1.5,label='Low Cloud')	# plot slope (m)
l1_r2 = ax5.plot(da1.values,da1.rvalue.values**2,'r',linestyle='-',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
# l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

l1_mid_slope = ax4.plot(da1_mid.values,da1_mid.slope.values/da1_mid.slope.std().values,'r',linestyle='--',linewidth=1.5,label='Mid Cloud')	# plot slope (m)
l1_mid_r2 = ax5.plot(da1_mid.values,da1_mid.rvalue.values**2,'r',linestyle='--',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
# l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

l1_high_slope = ax4.plot(da1_high.values,da1_high.slope.values/da1_high.slope.std().values,'r',linestyle=':',linewidth=1.5,label='High Cloud')	# plot slope (m)
l1_high_r2 = ax5.plot(da1_high.values,da1_high.rvalue.values**2,'r',linestyle=':',linewidth=1.5,label='$R^{2}$')	# calculate r^2 and plot r^2
# l1_p = ax5.plot(da1.values,da1.pvalue.values,'r',linestyle='--',linewidth=1.5,label='p-value')		# plot p-value

l2_slope = ax4.plot(da2.values,da2.slope.values/da2.slope.std().values,'b',linestyle='-',linewidth=1.5,label='HFX')		# plot slope (m)
l2_r2 = ax5.plot(da2.values,da2.rvalue.values**2,'b',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
# l2_p = ax5.plot(da2.values,da2.pvalue.values,'b',linestyle='--',linewidth=1.5)						# plot p-value

l3_slope = ax4.plot(da3.values,da3.slope.values/da3.slope.std().values,'c',linestyle='-',linewidth=1.5,label='RR')			# plot slope (m)
l3_r2 = ax5.plot(da3.values,da3.rvalue.values**2,'c',linestyle='-',linewidth=1.5)					# calculate r^2 and plot r^2
# l3_p = ax5.plot(da3.values,da3.pvalue.values,'c',linestyle='--',linewidth=1.5)						# plot p-value

ax4.set_title(f'Diurnal Evolution of Net Surface CRF linear relationship over 0->{land_dist_thresh:.0f} km\n Standardized Slope',fontsize=12)
ax4.set_xlabel('Time [UTC]',fontsize=10)
ax4.set_ylabel('Standardized Slope [$m / std(m)$]',fontsize=10)
ax4.set_xlim([time_start,time_end])
ax4.set_ylim([-4,4])
ax4.set_xticks(np.arange(time_start,time_end+1,3))
ax4.grid(linestyle='--', axis='both', linewidth=1)
ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.75)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.75)
ax4.legend(loc='lower right')

ax5.set_title('Varience Explained',fontsize=12)
ax5.set_xlabel('Time [UTC]',fontsize=10)
ax5.set_ylabel('$R^{2}$',fontsize=10)
ax5.set_xlim([time_start,time_end])
ax5.set_ylim([0,1])
ax5.set_yticks(np.arange(0,1.2,.2))
ax5.set_xticks(np.arange(time_start,time_end+1,3))
ax5.grid(linestyle='--', axis='both', linewidth=1)
# ax5.legend(loc='upper right')


# ### Cross-Cectional Analysis

# #### Doppler Shift Analysis

# In[ ]:


start_LT = 7
end_LT = 16

# Control
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
NormalWind_cntl = da_d02_cross_NormalWind_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
NormalWind_cntl = NormalWind_cntl.sel(hour=slice(start_LT,end_LT)).mean('hour')

# NCRF Sunrise
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
NormalWind_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
NormalWind_NCRF = NormalWind_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
NormalWind_NCRF = NormalWind_NCRF.sel(hour=slice(start_LT,end_LT)).mean('hour')

NormalWind_Diff = NormalWind_NCRF - NormalWind_cntl

fig = plt.figure(figsize=(6.5,4.5))
gs = gridspec.GridSpec(nrows=1, ncols=2, wspace= 0.1, width_ratios=[.96,.04])
ax1 = fig.add_subplot(gs[0,0])

# Plot terrains
y = d02_cross_PSFC.max(axis=(0,2))
plt.plot(NormalWind_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
y = d02_cross_PSFC.min(axis=(0,2))
plt.plot(NormalWind_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
y = d02_cross_PSFC.mean(axis=(0,2))
plt.plot(NormalWind_cntl.Distance,y,color='black')

# Plot the cross-sectional data
cf1 = NormalWind_Diff.plot.contourf(
	ax=ax1,
	add_colorbar=False,
	# levels=np.arange(-10,10.5,.5),
	levels=np.arange(-4,4.25,.25),
	cmap='RdBu_r',
	yscale='log',
	ylim=[200,1000],
	extend='both'
)
# Plot the vertical line at approximate coastline
plt.axvline(x=0, color='k', linestyle='--')
ax1.set_ylabel('Pressure Levels [hPa]')
ax1.set_xlabel('Distance from coast [km]')
ax1.invert_yaxis()
ax1.invert_xaxis()

string = f'Daily Averaged Normal Wind Difference from Control\n {start_LT:.0f}-{end_LT:.0f}LT over Central Western Sumatra'
ax1.set_title(string)
yticks = np.linspace(1000,100,10)
ax1.set_yticks(yticks)
ax1.set_xticks([])
ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.set_ylim([1000,200])

ax3 = fig.add_subplot(gs[:, 1])
cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
cbar.set_ticks(np.arange(-4,5,1))
cbar.set_label('Normal Wind [$m/s$]')
cbar.minorticks_off()


# #### Normal Wind

# Control/NCRF

# In[ ]:


# Load in the data
	## Data plotted on the contour

# Control
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
NormalWind_cntl = da_d02_cross_NormalWind_cntl[13:-24].groupby('Time.hour').mean().mean(['Spread'])

# NCRF Sunrise
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
NormalWind_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
NormalWind_NCRF = NormalWind_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})

# for i in range(NormalWind_cntl.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=1, ncols=2, wspace= 0.1, width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	plt.plot(NormalWind_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	plt.plot(NormalWind_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	plt.plot(NormalWind_cntl.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = NormalWind_NCRF[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-10,10.5,.5),
		# levels=np.arange(-4,4.25,.25),
		# levels=np.append(0,np.logspace(-2,0,25)),
		# vmax=0.5,
		cmap='RdBu_r',
		yscale='log',
		# center=0,
		ylim=[200,1000],
		# alpha=0.4
		extend='both'
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('Distance from coast [km]')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# local_date = NormalWind_cntl.Time[i] + np.timedelta64(7,'h')
	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = 'NCRF Normal Wind at ' + str(LT) + 'LT\nCentral Western Sumatra'
	# string = 'Normal Wind at ' + str(LT) + 'LT\nCentral Western Sumatra'
	# In UTC
	# string = 'Cloud Fraction at ' + str(i) + 'UTC between\n (' + str(abs(start_coord[0])) + '\N{DEGREE SIGN}S,' + str(abs(start_coord[1])) + '\N{DEGREE SIGN}S) and (' + str(abs(end_coord[0])) + '\N{DEGREE SIGN}E,' + str(abs(end_coord[1])) + '\N{DEGREE SIGN}E)'
	
	ax1.set_title(string)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,200])
	
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(-10,12,2))
	cbar.set_label('Normal Wind [$m/s$]')
	cbar.minorticks_off()

	# plt.savefig('/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_NormalWind_NCRF_'+str(LT)+'.png',dpi=300)
	# mpl.pyplot.close()


# Difference

# In[ ]:


# Load in the data
	## Data plotted on the contour

# Control
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
NormalWind_cntl = da_d02_cross_NormalWind_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')

# NCRF Sunrise
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
NormalWind_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
NormalWind_NCRF = NormalWind_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})

# NCRF difference from Control
NormalWind_Diff = NormalWind_NCRF-NormalWind_cntl

# for i in range(NormalWind_cntl.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=1, ncols=2, wspace= 0.1, width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	plt.plot(NormalWind_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	plt.plot(NormalWind_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	plt.plot(NormalWind_cntl.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = NormalWind_Diff[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		# levels=np.arange(-10,11,1),
		levels=np.arange(-4,4.25,.25),
		# levels=np.append(0,np.logspace(-2,0,25)),
		# vmax=0.5,
		cmap='RdBu_r',
		yscale='log',
		# center=0,
		ylim=[200,1000],
		# alpha=0.4
		extend='both'
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('Distance from coast [km]')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# local_date = NormalWind_cntl.Time[i] + np.timedelta64(7,'h')
	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = 'Normal Wind Difference from Control at ' + str(LT) + 'LT\nCentral Western Sumatra'
	# string = 'Normal Wind at ' + str(LT) + 'LT\nCentral Western Sumatra'
	# In UTC
	# string = 'Cloud Fraction at ' + str(i) + 'UTC between\n (' + str(abs(start_coord[0])) + '\N{DEGREE SIGN}S,' + str(abs(start_coord[1])) + '\N{DEGREE SIGN}S) and (' + str(abs(end_coord[0])) + '\N{DEGREE SIGN}E,' + str(abs(end_coord[1])) + '\N{DEGREE SIGN}E)'
	
	ax1.set_title(string)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,200])
	
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_label('Normal Wind Anomaly [$m/s$]')
	cbar.minorticks_off()

	# plt.savefig('/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_NormalWind_Anomaly_'+str(LT)+'.png',dpi=300)
	# mpl.pyplot.close()


# #### Virtual Temperature

# Control & NCRF

# In[ ]:


# Load in the data

## Control
	# Virtual Temperature
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
Tv_cntl = da_d02_cross_Tv_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# Remove the layer average of Virtual Temperature
Tv_cntl = Tv_cntl - Tv_cntl.mean('Distance')	

	# QTotal
# QTotal_cntl = da_d02_cross_QTotal_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
QTotal_cntl = da_d02_cross_NormalWind_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
QTotal_cntl = QTotal_cntl - QTotal_cntl.mean('Distance')	

	# Rain Rate
RainRate_cntl = da_d02_cross_RR_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')

# Run a smoother over distance to make it less noisy
smoothing_num = 3
Tv_cntl = Tv_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_cntl = QTotal_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_cntl = RainRate_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

## NCRF Sunrise
	# Virtual Temperature
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
Tv_NCRF = da_d02_cross_Tv_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
Tv_NCRF = Tv_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
Tv_NCRF = Tv_NCRF - Tv_NCRF.mean('Distance')

	# QTotal
# QTotal_NCRF = da_d02_cross_QTotal_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
QTotal_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
QTotal_NCRF = QTotal_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
QTotal_NCRF = QTotal_NCRF - QTotal_NCRF.mean('Distance')

	# Rain Rate
RainRate_NCRF = da_d02_cross_RR_CRFoff.sel(Time=slice(1,25),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(2,24),[0,1]))))
RainRate_NCRF = RainRate_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})

# Run a smoother over distance to make it less noisy
Tv_NCRF = Tv_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_NCRF = QTotal_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_NCRF = RainRate_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()


Ptop = 600
for i in range(Tv_NCRF.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(13,5.5))
	# fig.suptitle('Layer Anomalous Tv and Total Q over Central Western Sumatra', fontsize=14)
	fig.suptitle('Layer Anomalous Tv and Normal Wind over Central Western Sumatra', fontsize=14)
	gs = gridspec.GridSpec(nrows=2, ncols=3, wspace= 0.1, hspace=0.1, width_ratios=[.48,.48,.02], height_ratios=[0.85,0.15])
	
	# Right subplot (NCRF)
	ax1 = fig.add_subplot(gs[0,1])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax1.plot(Tv_NCRF.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax1.plot(Tv_NCRF.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax1.plot(Tv_NCRF.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Tv_NCRF[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-2,2.25,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_NCRF[i,...].plot.contour(
		ax=ax1,
		add_colorbar=False,
		# levels=np.arange(-0.002,.003,.0005),
		levels=np.arange(-3,4,1),	# Normal Wind
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both',
	)
	ax1.clabel(cf2, inline=True, fontsize=10)
	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'NCRF Sunrise {LT:.0f}LT'
	ax1.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.tick_params(labelbottom=False, labelleft=False)    
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,1])
	l1 = RainRate_NCRF[i,...].plot(
		ax=ax2,
		xlim=[RainRate_NCRF.Distance[0],RainRate_NCRF.Distance[-1]],
		ylim=[0,5],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	
	ax3 = fig.add_subplot(gs[0,0])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax3.plot(Tv_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax3.plot(Tv_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax3.plot(Tv_cntl.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Tv_cntl[i,...].plot.contourf(
		ax=ax3,
		add_colorbar=False,
		levels=np.arange(-2,2.25,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_cntl[i,...].plot.contour(
		ax=ax3,
		add_colorbar=False,
		# levels=np.arange(-0.002,.003,.0005),
		levels=np.arange(-3,4,1),	# Normal Wind
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both',
	)
	ax3.clabel(cf2, inline=True, fontsize=10)
	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax3.set_ylabel('')
	ax3.set_xlabel('')
	ax3.invert_yaxis()
	ax3.invert_xaxis()

	# local_date = Tv_cntl.Time[i] + np.timedelta64(7,'h')
	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'Control {LT:.0f}LT'
	ax3.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax3.set_yticks(yticks)
	ax3.set_xticks([])
	ax3.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax3.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax4 = fig.add_subplot(gs[1,0])
	l1 = RainRate_cntl[i,...].plot(
		ax=ax4,
		xlim=[RainRate_cntl.Distance[0],RainRate_cntl.Distance[-1]],
		ylim=[0,5],
	)
	ax4.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax4.set_xlabel('Distance from coast [km]')
	ax4.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax4.invert_xaxis()
	ax4.set_yticks([0,2,4])
	ax4.set_title('')

	ax5 = fig.add_subplot(gs[0, 2])
	cbar = plt.colorbar(cf1, cax=ax5, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(-2,3,1))
	cbar.set_label('Tv [$K$]')
	cbar.minorticks_off()

	plt.savefig(f'/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_Tv_and_NormalWind_NCRF_and_Control_{LT:.0f}LT.png',dpi=300)
	mpl.pyplot.close()


# #### Theta

# Control

# In[ ]:


# Load in the data
	## Data plotted on the contour

# Control
	# Theta
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
Theta_cntl = da_d02_cross_Theta_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# Remove the layer average of theta
Theta_cntl = Theta_cntl - Theta_cntl.mean('Distance')	

	# QTotal
QTotal_cntl = da_d02_cross_QTotal_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
QTotal_cntl = QTotal_cntl - QTotal_cntl.mean('Distance')	

	# Rain Rate
RainRate_cntl = da_d02_cross_RR_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')

# Run a smoother over distance to make it less noisy
smoothing_num = 3
Theta_cntl = Theta_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_cntl = QTotal_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_cntl = RainRate_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()


for i in range(Theta_cntl.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, wspace= 0.1, hspace=0.1, width_ratios=[.96,.04], height_ratios=[0.85,0.15])
	
	ax1 = fig.add_subplot(gs[0,0])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax1.plot(Theta_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax1.plot(Theta_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax1.plot(Theta_cntl.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Theta_cntl[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-2,2.25,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[200,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_cntl[i,...].plot.contour(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-0.002,.003,.0005),
		# levels=np.arange(0,.03,.005),
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[200,1000],
		extend='both',
	)
	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# local_date = Theta_cntl.Time[i] + np.timedelta64(7,'h')
	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = 'Control Theta and QTotal at ' + str(LT) + 'LT\nCentral Western Sumatra'

	ax1.set_title(string)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,200])

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = RainRate_cntl[i,...].plot(
		ax=ax2,
		xlim=[RainRate_cntl.Distance[0],RainRate_cntl.Distance[-1]],
		ylim=[0,5],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(-2,3,1))
	cbar.set_label('Theta [$K$]')
	cbar.minorticks_off()

	# plt.savefig('/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_Theta_Control_'+str(LT)+'.png',dpi=300)
	# mpl.pyplot.close()


# NCRF

# In[ ]:


# Load in the data
	## Data plotted on the contour

# NCRF Sunrise
	# Potential Temperature
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
Theta_NCRF = da_d02_cross_Theta_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
Theta_NCRF = Theta_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
Theta_NCRF = Theta_NCRF - Theta_NCRF.mean('Distance')

	# QTotal
QTotal_NCRF = da_d02_cross_QTotal_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
QTotal_NCRF = QTotal_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
QTotal_NCRF = QTotal_NCRF - QTotal_NCRF.mean('Distance')

	# Rain Rate
RainRate_NCRF = da_d02_cross_RR_CRFoff.sel(Time=slice(1,25),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(2,24),[0,1]))))
RainRate_NCRF = RainRate_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})

# Run a smoother over distance to make it less noisy
smoothing_num = 3
Theta_NCRF = Theta_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_NCRF = QTotal_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_NCRF = RainRate_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

for i in range(Theta_NCRF.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, wspace= 0.1, hspace=0.1, width_ratios=[.96,.04], height_ratios=[0.85,0.15])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Theta_NCRF[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-2,2.25,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[200,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_NCRF[i,...].plot.contour(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-0.002,.003,.0005),
		# levels=np.arange(0,.03,.005),
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[200,1000],
		extend='both',
	)
	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = 'NCRF Theta and QTotal at ' + str(LT) + 'LT\nCentral Western Sumatra'

	ax1.set_title(string)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,200])

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = RainRate_NCRF[i,...].plot(
		ax=ax2,
		xlim=[RainRate_NCRF.Distance[0],RainRate_NCRF.Distance[-1]],
		ylim=[0,5],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(-2,3,1))
	cbar.set_label('Theta [$K$]')
	cbar.minorticks_off()

	# plt.savefig('/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_Theta_NCRF_'+str(LT)+'.png',dpi=300)
	# mpl.pyplot.close()


# Control & NCRF

# In[ ]:


# Load in the data

## Control
	# Theta
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
Theta_cntl = da_d02_cross_Theta_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# Remove the layer average of theta
Theta_cntl = Theta_cntl - Theta_cntl.mean('Distance')	

	# QTotal
# QTotal_cntl = da_d02_cross_QTotal_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
QTotal_cntl = da_d02_cross_NormalWind_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
QTotal_cntl = QTotal_cntl - QTotal_cntl.mean('Distance')	

	# Rain Rate
RainRate_cntl = da_d02_cross_RR_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')

# Run a smoother over distance to make it less noisy
smoothing_num = 3
Theta_cntl = Theta_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_cntl = QTotal_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_cntl = RainRate_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

## NCRF Sunrise
	# Potential Temperature
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
Theta_NCRF = da_d02_cross_Theta_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
Theta_NCRF = Theta_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
Theta_NCRF = Theta_NCRF - Theta_NCRF.mean('Distance')

	# QTotal
# QTotal_NCRF = da_d02_cross_QTotal_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
QTotal_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
QTotal_NCRF = QTotal_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
QTotal_NCRF = QTotal_NCRF - QTotal_NCRF.mean('Distance')

	# Rain Rate
RainRate_NCRF = da_d02_cross_RR_CRFoff.sel(Time=slice(1,25),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(2,24),[0,1]))))
RainRate_NCRF = RainRate_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})

# Run a smoother over distance to make it less noisy
smoothing_num = 3
Theta_NCRF = Theta_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_NCRF = QTotal_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_NCRF = RainRate_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()


Ptop = 200
for i in range(Theta_NCRF.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(13,5.5))
	# fig.suptitle('Layer Anomalous Theta and Total Q over Central Western Sumatra', fontsize=14)
	fig.suptitle('Layer Anomalous Theta and Normal Wind over Central Western Sumatra', fontsize=14)
	gs = gridspec.GridSpec(nrows=2, ncols=3, wspace= 0.1, hspace=0.1, width_ratios=[.48,.48,.02], height_ratios=[0.85,0.15])
	
	# Right subplot (NCRF)
	ax1 = fig.add_subplot(gs[0,1])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Theta_NCRF[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-2,2.25,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_NCRF[i,...].plot.contour(
		ax=ax1,
		add_colorbar=False,
		# levels=np.arange(-0.002,.003,.0005),
		levels=np.arange(-3,4,1),	# Normal Wind
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both',
	)
	ax1.clabel(cf2, inline=True, fontsize=10)
	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'NCRF Sunrise {LT:.0f}LT'
	ax1.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.tick_params(labelbottom=False, labelleft=False)    
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,1])
	l1 = RainRate_NCRF[i,...].plot(
		ax=ax2,
		xlim=[RainRate_NCRF.Distance[0],RainRate_NCRF.Distance[-1]],
		ylim=[0,5],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	
	ax3 = fig.add_subplot(gs[0,0])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax3.plot(Theta_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax3.plot(Theta_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax3.plot(Theta_cntl.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Theta_cntl[i,...].plot.contourf(
		ax=ax3,
		add_colorbar=False,
		levels=np.arange(-2,2.25,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_cntl[i,...].plot.contour(
		ax=ax3,
		add_colorbar=False,
		# levels=np.arange(-0.002,.003,.0005),
		levels=np.arange(-3,4,1),	# Normal Wind
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both',
	)
	ax3.clabel(cf2, inline=True, fontsize=10)
	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax3.set_ylabel('')
	ax3.set_xlabel('')
	ax3.invert_yaxis()
	ax3.invert_xaxis()

	# local_date = Theta_cntl.Time[i] + np.timedelta64(7,'h')
	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'Control {LT:.0f}LT'
	ax3.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax3.set_yticks(yticks)
	ax3.set_xticks([])
	ax3.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax3.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax4 = fig.add_subplot(gs[1,0])
	l1 = RainRate_cntl[i,...].plot(
		ax=ax4,
		xlim=[RainRate_cntl.Distance[0],RainRate_cntl.Distance[-1]],
		ylim=[0,5],
	)
	ax4.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax4.set_xlabel('Distance from coast [km]')
	ax4.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax4.invert_xaxis()
	ax4.set_yticks([0,2,4])
	ax4.set_title('')

	ax5 = fig.add_subplot(gs[0, 2])
	cbar = plt.colorbar(cf1, cax=ax5, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(-2,3,1))
	cbar.set_label('Theta [$K$]')
	cbar.minorticks_off()

	plt.savefig(f'/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_Theta_and_QTotal_NCRF_and_Control_{LT:.0f}LT.png',dpi=300)
	mpl.pyplot.close()


# In[ ]:


# Load in the data

## Control
	# Theta
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
Theta_cntl = da_d02_cross_Theta_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# Remove daily average at each layer 
Theta_cntl = Theta_cntl - Theta_cntl.mean(['Distance','hour'])

	# QTotal
QTotal_cntl = da_d02_cross_QTotal_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# QTotal_cntl = da_d02_cross_NormalWind_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
QTotal_cntl = QTotal_cntl - QTotal_cntl.mean(['Distance','hour'])

	# Rain Rate
RainRate_cntl = da_d02_cross_RR_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')

# Run a smoother over distance to make it less noisy
smoothing_num = 3
Theta_cntl = Theta_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_cntl = QTotal_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_cntl = RainRate_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

## NCRF Sunrise
	# Potential Temperature
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
Theta_NCRF = da_d02_cross_Theta_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
Theta_NCRF = Theta_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
# Remove daily average at each layer 
Theta_NCRF = Theta_NCRF - Theta_NCRF.mean(['Distance','hour'])

	# QTotal
QTotal_NCRF = da_d02_cross_QTotal_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# QTotal_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
QTotal_NCRF = QTotal_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
QTotal_NCRF = QTotal_NCRF - QTotal_NCRF.mean(['Distance','hour'])

	# Rain Rate
RainRate_NCRF = da_d02_cross_RR_CRFoff.sel(Time=slice(1,25),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(2,24),[0,1]))))
RainRate_NCRF = RainRate_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})

# Run a smoother over distance to make it less noisy
smoothing_num = 3
Theta_NCRF = Theta_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_NCRF = QTotal_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()*10**3
RainRate_NCRF = RainRate_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()


Ptop = 100
for i in range(Theta_NCRF.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(13,5.5))
	# fig.suptitle('Layer Anomalous Theta and Total Q over Central Western Sumatra', fontsize=14)
	fig.suptitle('Daily Layer Anomalous Theta and Total Q over Central Western Sumatra', fontsize=14)
	gs = gridspec.GridSpec(nrows=2, ncols=3, wspace= 0.1, hspace=0.1, width_ratios=[.48,.48,.02], height_ratios=[0.85,0.15])
	
	# Right subplot (NCRF)
	ax1 = fig.add_subplot(gs[0,1])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Theta_NCRF[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-2,2.25,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_NCRF[i,...].plot.contour(
		ax=ax1,
		add_colorbar=False,
		# levels=np.arange(-0.002,.0025,.0005),	# Qtotal
		levels=np.arange(-2,2.5,0.5),			# Qtotal
		# levels=np.arange(-3,4,1),				# Normal Wind
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both',
	)
	ax1.clabel(cf2, inline=True, fontsize=10)

	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'NCRF Sunrise {LT:.0f}LT'
	ax1.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.tick_params(labelbottom=False, labelleft=False)    
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,1])
	l1 = RainRate_NCRF[i,...].plot(
		ax=ax2,
		xlim=[RainRate_NCRF.Distance[0],RainRate_NCRF.Distance[-1]],
		ylim=[0,5],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	
	ax3 = fig.add_subplot(gs[0,0])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax3.plot(Theta_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax3.plot(Theta_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax3.plot(Theta_cntl.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Theta_cntl[i,...].plot.contourf(
		ax=ax3,
		add_colorbar=False,
		levels=np.arange(-2,2.25,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_cntl[i,...].plot.contour(
		ax=ax3,
		add_colorbar=False,
		# levels=np.arange(-0.002,.0025,.0005),	# Qtotal
		levels=np.arange(-2,2.5,0.5),			# Qtotal
		# levels=np.arange(-3,4,1),	# Normal Wind
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both',
	)
	ax3.clabel(cf2, inline=True, fontsize=10)

	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax3.set_ylabel('')
	ax3.set_xlabel('')
	ax3.invert_yaxis()
	ax3.invert_xaxis()

	# local_date = Theta_cntl.Time[i] + np.timedelta64(7,'h')
	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'Control {LT:.0f}LT'
	ax3.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax3.set_yticks(yticks)
	ax3.set_xticks([])
	ax3.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax3.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax4 = fig.add_subplot(gs[1,0])
	l1 = RainRate_cntl[i,...].plot(
		ax=ax4,
		xlim=[RainRate_cntl.Distance[0],RainRate_cntl.Distance[-1]],
		ylim=[0,5],
	)
	ax4.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax4.set_xlabel('Distance from coast [km]')
	ax4.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax4.invert_xaxis()
	ax4.set_yticks([0,2,4])
	ax4.set_title('')
	
	ax5 = fig.add_subplot(gs[0, 2])
	cbar = plt.colorbar(cf1, cax=ax5, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(-2,3,1))
	cbar.set_label("Theta' [$K$]")
	cbar.minorticks_off()

	# plt.savefig(f'/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_Theta_and_NormalWind_NCRF_and_Control_{LT:.0f}LT.png',dpi=300)
	# mpl.pyplot.close()


# Difference

# In[ ]:


# Load in the data
	## Data plotted on the contour

# Control
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
Theta_cntl = da_d02_cross_Theta_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# Remove the layer average of theta
Theta_cntl = Theta_cntl - Theta_cntl.mean('Distance')	

# NCRF Sunrise
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
Theta_NCRF = da_d02_cross_Theta_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
Theta_NCRF = Theta_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
Theta_NCRF = Theta_NCRF - Theta_NCRF.mean('Distance')

Theta_Diff = Theta_NCRF - Theta_cntl

for i in range(Theta_NCRF.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=1, ncols=2, wspace= 0.1, width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	plt.plot(Theta_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	plt.plot(Theta_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	plt.plot(Theta_cntl.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Theta_Diff[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-2.5,2.75,.25),
		cmap='RdBu_r',
		yscale='log',
		ylim=[200,1000],
		extend='both'
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('Distance from coast [km]')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# local_date = Theta_cntl.Time[i] + np.timedelta64(7,'h')
	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = 'Theta Difference from Control at ' + str(LT) + 'LT\nCentral Western Sumatra'

	ax1.set_title(string)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,200])
	
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(-2,3,1))
	cbar.set_label('Theta Anomaly [$K$]')
	cbar.minorticks_off()

	plt.savefig('/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_Theta_Anomaly_'+str(LT)+'.png',dpi=300)
	mpl.pyplot.close()


# In[ ]:


# Load in the data

## Control
	# Theta
# [13:-24] ensures that the times we are comparing with NCRF is the same, average over hours, and then over all cross-sections
Theta_cntl = da_d02_cross_Theta_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# Remove the layer average of theta
# Theta_cntl = Theta_cntl - Theta_cntl.mean('Distance')	

	# QTotal
QTotal_cntl = da_d02_cross_QTotal_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# QTotal_cntl = da_d02_cross_NormalWind_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# QTotal_cntl = da_d02_cross_W_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# QTotal_cntl = QTotal_cntl - QTotal_cntl.mean('Distance')	

	# Rain Rate
RainRate_cntl = da_d02_cross_RR_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')

# Run a smoother over distance to make it less noisy
# smoothing_num = 3
# Theta_cntl = Theta_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
# QTotal_cntl = QTotal_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
# RainRate_cntl = RainRate_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

## NCRF Sunrise
	# Potential Temperature
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
Theta_NCRF = da_d02_cross_Theta_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
Theta_NCRF = Theta_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
# Theta_NCRF = Theta_NCRF - Theta_NCRF.mean('Distance')

	# QTotal
QTotal_NCRF = da_d02_cross_QTotal_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# QTotal_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# QTotal_NCRF = da_d02_cross_W_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
QTotal_NCRF = QTotal_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
# QTotal_NCRF = QTotal_NCRF - QTotal_NCRF.mean('Distance')

	# Rain Rate
RainRate_NCRF = da_d02_cross_RR_CRFoff.sel(Time=slice(1,25),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(2,24),[0,1]))))
RainRate_NCRF = RainRate_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})

# Run a smoother over distance to make it less noisy
# smoothing_num = 3
# Theta_NCRF = Theta_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
# QTotal_NCRF = QTotal_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
# RainRate_NCRF = RainRate_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

# Difference (NCRF - Control)
Theta_Diff = Theta_NCRF - Theta_cntl
QTotal_Diff = QTotal_NCRF - QTotal_cntl
RainRate_Diff = RainRate_NCRF - RainRate_cntl

smoothing_num = 3
Theta_Diff = Theta_Diff.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
QTotal_Diff = QTotal_Diff.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()*(10**3)
RainRate_Diff = RainRate_Diff.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

Ptop = 600

for i in range(Theta_NCRF.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	# fig.suptitle('Diurnal Evolution of Layer Anomalous Theta and Total Q over Central Western Sumatra', fontsize=14)
	gs = gridspec.GridSpec(nrows=2, ncols=2, wspace= 0.1, hspace=0.1, width_ratios=[.96,.04], height_ratios=[0.85,0.15])
	
	# Difference (NCRF - Control)
	ax1 = fig.add_subplot(gs[0,0])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax1.plot(Theta_NCRF.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = Theta_Diff[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-1,1.1,.1),
		cmap='RdBu_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both'
	)
	# Plot the cross-sectional data
	cf2 = QTotal_Diff[i,...].plot.contour(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-0.002,.002,.001),	# Q Total
		# levels=np.arange(-0.2,0.4,0.2),	# W or Vertical Velocity
		# levels=np.arange(-3,4,1),			# Normal Wind
		# cmap='PuOr',
		colors='black',
		yscale='log',
		ylim=[Ptop,1000],
		extend='both',
	)
	# Plot the vertical line at approximate coastline
	# plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'Theta and Total Q Difference from Control\nover Central Western Sumatra at {LT:.0f}LT'
	# string = f'Theta and Normal Wind Difference from Control\nover Central Western Sumatra at {LT:.0f}LT'
	# string = f'Theta and W Difference from Control\nover Central Western Sumatra at {LT:.0f}LT'
	ax1.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = RainRate_Diff[i,...].plot(
		ax=ax2,
		xlim=[RainRate_Diff.Distance[0],RainRate_Diff.Distance[-1]],
		ylim=[-2,2]
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	ax2.axvline(x=0, color='k', linestyle='--')
	ax2.axhline(y=0, color='k', linestyle='-')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel("Rain Rate'\n[$mm day^{-1}$]\n",fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([-2,0,2])
	ax2.set_title('')
	
	ax3 = fig.add_subplot(gs[0, 1])
	ax3.clabel(cf2, inline=True, fontsize=10)
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(-1,1.5,0.5))
	cbar.set_label("Theta' [$K$]")
	cbar.minorticks_off()

	# plt.savefig(f'/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_Theta_QTotal_Difference_{LT:.0f}LT.png',dpi=300)
	# mpl.pyplot.close()


# #### Cloud Fraction

# Control & NCRF

# In[ ]:


# Load in the data

## Control
	# Theta
# [13:-24] ensures that the times we are comparing with NCRF Sunrise is the same, average over hours, and then over all cross-sections
CLDFRA_cntl = da_d02_cross_CLDFRA_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')

	# QTotal
QTotal_cntl = da_d02_cross_QTotal_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
# QTotal_cntl = da_d02_cross_NormalWind_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')
QTotal_cntl = QTotal_cntl - QTotal_cntl.mean('Distance')	

	# Rain Rate
RainRate_cntl = da_d02_cross_RR_cntl[13:-24].groupby('Time.hour').mean().mean('Spread')

# Run a smoother over distance to make it less noisy
smoothing_num = 3
CLDFRA_cntl = CLDFRA_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
# QTotal_cntl = QTotal_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_cntl = RainRate_cntl.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()

## NCRF Sunrise
	# Potential Temperature
# Include the first 24 hrs of the CRF Sunrise case, then create a coordinate named Time that corresspond to the hours that are included (starts at 01UTC -> 00 UTC)
CLDFRA_NCRF = da_d02_cross_CLDFRA_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# Average over all the simulations and cross-sections, group by Time and then average 
CLDFRA_NCRF = CLDFRA_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
# CLDFRA_NCRF = CLDFRA_NCRF - CLDFRA_NCRF.mean('Distance')

	# QTotal
QTotal_NCRF = da_d02_cross_QTotal_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# QTotal_NCRF = da_d02_cross_NormalWind_CRFoff.sel(Time=slice(0,24),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(1,24),[0]))))
# QTotal_NCRF = QTotal_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})
QTotal_NCRF = QTotal_NCRF - QTotal_NCRF.mean('Distance')

	# Rain Rate
RainRate_NCRF = da_d02_cross_RR_CRFoff.sel(Time=slice(1,25),Lead=slice(0,18,2)).assign_coords(Time=('Time',np.concatenate((np.arange(2,24),[0,1]))))
RainRate_NCRF = RainRate_NCRF.mean(['Lead','Spread']).groupby('Time').mean().rename({'Time':'hour'})

# Run a smoother over distance to make it less noisy
smoothing_num = 3
CLDFRA_NCRF = CLDFRA_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
# QTotal_NCRF = QTotal_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()
RainRate_NCRF = RainRate_NCRF.rolling(Distance=smoothing_num, min_periods=1, center=True).mean()


Ptop = 200

for i in range(CLDFRA_NCRF.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(11,5.5))
	# fig.suptitle('Layer Anomalous Theta and Total Q over Central Western Sumatra', fontsize=14)
	fig.suptitle('Cloud Fraction over Central Western Sumatra', fontsize=14)
	gs = gridspec.GridSpec(nrows=2, ncols=3, wspace= 0.1, hspace=0.1, width_ratios=[.48,.48,.02], height_ratios=[0.85,0.15])
	
	# Right subplot (NCRF)
	ax1 = fig.add_subplot(gs[0,1])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax1.plot(CLDFRA_NCRF.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax1.plot(CLDFRA_NCRF.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax1.plot(CLDFRA_NCRF.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = CLDFRA_NCRF[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(0,.525,.025),
		cmap='gray_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='max'
	)

	# Plot the vertical line at approximate coastline
	ax1.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()

	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'NCRF Sunrise {LT:.0f}LT'
	ax1.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.tick_params(labelbottom=False, labelleft=False)    
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,1])
	l1 = RainRate_NCRF[i,...].plot(
		ax=ax2,
		xlim=[RainRate_NCRF.Distance[0],RainRate_NCRF.Distance[-1]],
		ylim=[0,5],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	ax2.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	
	ax3 = fig.add_subplot(gs[0,0])
	# Plot terrains
	y = d02_cross_PSFC.max(axis=(0,2))
	ax3.plot(CLDFRA_cntl.Distance,y,color='blue',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.min(axis=(0,2))
	ax3.plot(CLDFRA_cntl.Distance,y,color='red',linewidth=1,alpha=0.5)
	y = d02_cross_PSFC.mean(axis=(0,2))
	ax3.plot(CLDFRA_cntl.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = CLDFRA_cntl[i,...].plot.contourf(
		ax=ax3,
		add_colorbar=False,
		levels=np.arange(0,.525,.025),
		cmap='gray_r',
		yscale='log',
		ylim=[Ptop,1000],
		extend='max'
	)

	# Plot the vertical line at approximate coastline
	ax3.axvline(x=0, color='k', linestyle='--')
	ax3.set_ylabel('')
	ax3.set_xlabel('')
	ax3.invert_yaxis()
	ax3.invert_xaxis()

	# local_date = CLDFRA_cntl.Time[i] + np.timedelta64(7,'h')
	# Calculate LT
	if i > 16:
		LT = i-17
	else:
		LT = i+7
	# In Local Time
	string = f'Control {LT:.0f}LT'
	ax3.set_title(string)

	yticks = np.linspace(1000,100,10)
	ax3.set_yticks(yticks)
	ax3.set_xticks([])
	ax3.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax3.set_ylim([1000,Ptop])

	# Plot the rain rates
	ax4 = fig.add_subplot(gs[1,0])
	l1 = RainRate_cntl[i,...].plot(
		ax=ax4,
		xlim=[RainRate_cntl.Distance[0],RainRate_cntl.Distance[-1]],
		ylim=[0,5],
	)
	ax4.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	ax4.axvline(x=0, color='k', linestyle='--')
	ax4.set_xlabel('Distance from coast [km]')
	ax4.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax4.invert_xaxis()
	ax4.set_yticks([0,2,4])
	ax4.set_title('')

	ax5 = fig.add_subplot(gs[0, 2])
	cbar = plt.colorbar(cf1, cax=ax5, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_ticks(np.arange(0,.55,.05))
	cbar.set_label('Cloud Fraction')
	cbar.minorticks_off()

	# plt.savefig(f'/home/hragnajarian/PhD/temp_plots/WRF_DC_Cross_Sumatra_Theta_and_QTotal_NCRF_and_Control_{LT:.0f}LT.png',dpi=300)
	# mpl.pyplot.close()


# ### CRF Off 3-hrly sensitivity tests

# Rain Rate

# In[ ]:


fig = plt.figure(figsize=(56,10))
gs = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.1, hspace=0.25, height_ratios=[0.875,0.03])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])
ax5 = fig.add_subplot(gs[0,4])
ax6 = fig.add_subplot(gs[0,5])

# Data
	# Average over each cross-section
x1 = da_d02_RR_cross_cntl.mean('Spread')
# Data
	# Average over each cross-section
x2 = da_d02_RR_cross_CRFoff[...,0].mean('Spread')	# starts at 11-25-03UTC
# This concats 11-25 00-04UTC with 11-25 05UTC to 11-26 12UTC
x2 = xr.concat([x1.isel(Time=slice(0,5)),x2[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x3 = da_d02_RR_cross_CRFoff[3:,...,1].mean('Spread')	# starts at 11-25-06UTC
# This concats 11-25 00-07UTC with 11-25 08UTC to 11-26 12UTC
x3 = xr.concat([x1.isel(Time=slice(0,8)),x3[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x4 = da_d02_RR_cross_CRFoff[6:,...,2].mean('Spread')	# starts at 11-25-09UTC
# This concats 11-25 00-10UTC with 11-25 11UTC to 11-26 12UTC
x4 = xr.concat([x1.isel(Time=slice(0,11)),x4[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x5 = da_d02_RR_cross_CRFoff[9:,...,3].mean('Spread')	# starts at 11-25-12UTC
# This concats 11-25 00-13UTC with 11-25 14UTC to 11-26 12UTC
x5 = xr.concat([x1.isel(Time=slice(0,14)),x5[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x6 = da_d02_RR_cross_CRFoff[12:,...,4].mean('Spread')	# starts at 11-25-15UTC
# This concats 11-25 00-16UTC with 11-25 17UTC to 11-26 12UTC
x6= xr.concat([x1.isel(Time=slice(0,17)),x6[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point

# Plot the cross-sectional data
	# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,6.25,0.25),
	cmap='gray_r',
	center=0,
)
	# CRF off @ 03 UTC
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,6.25,0.25),
	cmap='gray_r',
	center=0,
)
	# CRF off @ 06 UTC
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,6.25,0.25),
	cmap='gray_r',
	center=0,
)
	# CRF off @ 09 UTC
cf4 = x4.plot.contourf(
	ax=ax4,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,6.25,0.25),
	cmap='gray_r',
	center=0,
)
	# CRF off @ 12 UTC
cf5 = x5.plot.contourf(
	ax=ax5,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,6.25,0.25),
	cmap='gray_r',
	center=0,
)
	# CRF off @ 15 UTC
cf6 = x6.plot.contourf(
	ax=ax6,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,6.25,0.25),
	cmap='gray_r',
	center=0,
)
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax4.axvline(x=0, color='k', linestyle='--')
ax5.axvline(x=0, color='k', linestyle='--')
ax6.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=x1.Time[3].values, color='r', linestyle='-')
ax3.axhline(y=x1.Time[6].values, color='r', linestyle='-')
ax4.axhline(y=x1.Time[9].values, color='r', linestyle='-')
ax5.axhline(y=x1.Time[12].values, color='r', linestyle='-')
ax6.axhline(y=x1.Time[15].values, color='r', linestyle='-')

ax1.set_xlabel('Distance from coast [km]', weight='bold')
ax2.set_xlabel('Distance from coast [km]', weight='bold')
ax3.set_xlabel('Distance from coast [km]', weight='bold')
ax4.set_xlabel('Distance from coast [km]', weight='bold')
ax5.set_xlabel('Distance from coast [km]', weight='bold')
ax6.set_xlabel('Distance from coast [km]', weight='bold')
ax1.set_ylabel('UTC', weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax4.invert_xaxis()
ax5.invert_xaxis()
ax6.invert_xaxis()
# ax1.set_yticks(x1.Time[0::12].values)
# ax1.set_yticklabels(x1.Time[0::12].dt.strftime("%m-%d %H").values)
ax1.set_yticks(x1.Time[::3].values)
ax2.set_yticks(x1.Time[::3].values)
ax3.set_yticks(x1.Time[::3].values)
ax4.set_yticks(x1.Time[::3].values)
ax5.set_yticks(x1.Time[::3].values)
ax6.set_yticks(x1.Time[::3].values)
ax1.set_yticklabels(x1.Time[::3].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
# Set titles/labels
ax1.set_title('Rain Rate Evolution - d02 [CRF On]', loc='left', fontsize=10)
ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax2.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 03]', loc='left', fontsize=10)
ax2.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax3.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 06]', loc='left', fontsize=10)
ax3.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax4.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 09]', loc='left', fontsize=10)
ax4.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax5.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 12]', loc='left', fontsize=10)
ax5.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax6.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 15]', loc='left', fontsize=10)
ax6.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax1.set_title('', loc='center')
ax2.set_title('', loc='center')
ax3.set_title('', loc='center')
ax4.set_title('', loc='center')
ax5.set_title('', loc='center')
ax6.set_title('', loc='center')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
ax4.grid(linestyle='--', axis='y', linewidth=1.5)
ax5.grid(linestyle='--', axis='y', linewidth=1.5)
ax6.grid(linestyle='--', axis='y', linewidth=1.5)

# Plot the colorbar
	# Rain rate colorbar
ax2 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Rain Rate [$mm d^{-1}$]', weight='bold')


# Rain Rate Anomalies

# In[ ]:


fig = plt.figure(figsize=(56,10))
gs = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.1, hspace=0.25, height_ratios=[0.875,0.03])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])
ax5 = fig.add_subplot(gs[0,4])
ax6 = fig.add_subplot(gs[0,5])

# Data
	# Average over each cross-section
x1 = da_d02_RR_cross_cntl.mean('Spread')
# Data
	# Average over each cross-section
x2 = da_d02_RR_cross_CRFoff[...,0].mean('Spread')	# starts at 11-25-03UTC
# This concats 11-25 00-04UTC with 11-25 05UTC to 11-26 12UTC
x2 = xr.concat([x1.isel(Time=slice(0,5)),x2[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x2 = x2-x1	# Calculate anomalies from Control
x3 = da_d02_RR_cross_CRFoff[3:,...,1].mean('Spread')	# starts at 11-25-06UTC
# This concats 11-25 00-07UTC with 11-25 08UTC to 11-26 12UTC
x3 = xr.concat([x1.isel(Time=slice(0,8)),x3[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x3 = x3-x1	# Calculate anomalies from Control
x4 = da_d02_RR_cross_CRFoff[6:,...,2].mean('Spread')	# starts at 11-25-09UTC
# This concats 11-25 00-10UTC with 11-25 11UTC to 11-26 12UTC
x4 = xr.concat([x1.isel(Time=slice(0,11)),x4[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x4 = x4-x1	# Calculate anomalies from Control
x5 = da_d02_RR_cross_CRFoff[9:,...,3].mean('Spread')	# starts at 11-25-12UTC
# This concats 11-25 00-13UTC with 11-25 14UTC to 11-26 12UTC
x5 = xr.concat([x1.isel(Time=slice(0,14)),x5[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x5 = x5-x1	# Calculate anomalies from Control
x6 = da_d02_RR_cross_CRFoff[12:,...,4].mean('Spread')	# starts at 11-25-15UTC
# This concats 11-25 00-16UTC with 11-25 17UTC to 11-26 12UTC
x6= xr.concat([x1.isel(Time=slice(0,17)),x6[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x6 = x6-x1	# Calculate anomalies from Control

# Plot the cross-sectional data
	# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,6.25,0.25),
	cmap='gray_r',
	center=0,
)
	# CRF off @ 03 UTC
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-2,2,0.25),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 06 UTC
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-2,2,0.25),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 09 UTC
cf4 = x4.plot.contourf(
	ax=ax4,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-2,2,0.25),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 12 UTC
cf5 = x5.plot.contourf(
	ax=ax5,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-2,2,0.25),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 15 UTC
cf6 = x6.plot.contourf(
	ax=ax6,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-2,2,0.25),
	cmap='RdBu_r',
	center=0,
)
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax4.axvline(x=0, color='k', linestyle='--')
ax5.axvline(x=0, color='k', linestyle='--')
ax6.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=x1.Time[3].values, color='r', linestyle='-')
ax3.axhline(y=x1.Time[6].values, color='r', linestyle='-')
ax4.axhline(y=x1.Time[9].values, color='r', linestyle='-')
ax5.axhline(y=x1.Time[12].values, color='r', linestyle='-')
ax6.axhline(y=x1.Time[15].values, color='r', linestyle='-')

ax1.set_xlabel('Distance from coast [km]', weight='bold')
ax2.set_xlabel('Distance from coast [km]', weight='bold')
ax3.set_xlabel('Distance from coast [km]', weight='bold')
ax4.set_xlabel('Distance from coast [km]', weight='bold')
ax5.set_xlabel('Distance from coast [km]', weight='bold')
ax6.set_xlabel('Distance from coast [km]', weight='bold')
ax1.set_ylabel('UTC', weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax4.invert_xaxis()
ax5.invert_xaxis()
ax6.invert_xaxis()
# ax1.set_yticks(x1.Time[0::12].values)
# ax1.set_yticklabels(x1.Time[0::12].dt.strftime("%m-%d %H").values)
ax1.set_yticks(x1.Time[::3].values)
ax2.set_yticks(x1.Time[::3].values)
ax3.set_yticks(x1.Time[::3].values)
ax4.set_yticks(x1.Time[::3].values)
ax5.set_yticks(x1.Time[::3].values)
ax6.set_yticks(x1.Time[::3].values)
ax1.set_yticklabels(x1.Time[::3].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
# Set titles/labels
ax1.set_title('Rain Rate Evolution - d02 [CRF On]', loc='left', fontsize=10)
ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax2.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 03]', loc='left', fontsize=10)
ax2.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax3.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 06]', loc='left', fontsize=10)
ax3.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax4.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 09]', loc='left', fontsize=10)
ax4.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax5.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 12]', loc='left', fontsize=10)
ax5.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax6.set_title('Rain Rate Evolution - d02 [CRF Off @ 11-25 15]', loc='left', fontsize=10)
ax6.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax1.set_title('', loc='center')
ax2.set_title('', loc='center')
ax3.set_title('', loc='center')
ax4.set_title('', loc='center')
ax5.set_title('', loc='center')
ax6.set_title('', loc='center')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
ax4.grid(linestyle='--', axis='y', linewidth=1.5)
ax5.grid(linestyle='--', axis='y', linewidth=1.5)
ax6.grid(linestyle='--', axis='y', linewidth=1.5)

# Plot the colorbar
	# Control rain rate colorbar
ax2 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Rain Rate [$mm d^{-1}$]', weight='bold')
	# CRF Off rain rate colorbar
ax2 = fig.add_subplot(gs[1, 1:])
cbar = plt.colorbar(cf2, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
cbar.set_ticks(np.arange(-2,3,1))
cbar.set_label('Rain Rate Anomlaies [$mm d^{-1}$]', weight='bold')


# Normal Wind

# In[ ]:


fig = plt.figure(figsize=(56,10))
gs = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.1, hspace=0.25, height_ratios=[0.875,0.03])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])
ax5 = fig.add_subplot(gs[0,4])
ax6 = fig.add_subplot(gs[0,5])

# Data
	# Average over each cross-section
x1 = da_d02_cross_NormalWind_cntl[:,:6,:,:].mean('Spread').mean('bottom_top')
# Data
	# Average over each cross-section
x2 = da_d02_NormalWind_cross_CRFoff[:,:6,:,:,0].mean('Spread').mean('bottom_top')	# starts at 11-25-03UTC
# This concats 11-25 00-04UTC with 11-25 05UTC to 11-26 12UTC
x2 = xr.concat([x1.isel(Time=slice(0,5)),x2[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x3 = da_d02_NormalWind_cross_CRFoff[3:,:6,:,:,1].mean('Spread').mean('bottom_top')	# starts at 11-25-06UTC
# This concats 11-25 00-07UTC with 11-25 08UTC to 11-26 12UTC
x3 = xr.concat([x1.isel(Time=slice(0,8)),x3[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x4 = da_d02_NormalWind_cross_CRFoff[6:,:6,:,:,2].mean('Spread').mean('bottom_top')	# starts at 11-25-09UTC
# This concats 11-25 00-10UTC with 11-25 11UTC to 11-26 12UTC
x4 = xr.concat([x1.isel(Time=slice(0,11)),x4[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x5 = da_d02_NormalWind_cross_CRFoff[9:,:6,:,:,3].mean('Spread').mean('bottom_top')	# starts at 11-25-12UTC
# This concats 11-25 00-13UTC with 11-25 14UTC to 11-26 12UTC
x5 = xr.concat([x1.isel(Time=slice(0,14)),x5[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x6 = da_d02_NormalWind_cross_CRFoff[12:,:6,:,:,4].mean('Spread').mean('bottom_top')	# starts at 11-25-15UTC
# This concats 11-25 00-16UTC with 11-25 17UTC to 11-26 12UTC
x6 = xr.concat([x1.isel(Time=slice(0,17)),x6[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point

# Plot the cross-sectional data
	# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 03 UTC
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 06 UTC
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 09 UTC
cf4 = x4.plot.contourf(
	ax=ax4,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 12 UTC
cf5 = x5.plot.contourf(
	ax=ax5,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 15 UTC
cf6 = x6.plot.contourf(
	ax=ax6,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax4.axvline(x=0, color='k', linestyle='--')
ax5.axvline(x=0, color='k', linestyle='--')
ax6.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=x1.Time[3].values, color='r', linestyle='-')
ax3.axhline(y=x1.Time[6].values, color='r', linestyle='-')
ax4.axhline(y=x1.Time[9].values, color='r', linestyle='-')
ax5.axhline(y=x1.Time[12].values, color='r', linestyle='-')
ax6.axhline(y=x1.Time[15].values, color='r', linestyle='-')

ax1.set_xlabel('Distance from coast [km]', weight='bold')
ax2.set_xlabel('Distance from coast [km]', weight='bold')
ax3.set_xlabel('Distance from coast [km]', weight='bold')
ax4.set_xlabel('Distance from coast [km]', weight='bold')
ax5.set_xlabel('Distance from coast [km]', weight='bold')
ax6.set_xlabel('Distance from coast [km]', weight='bold')
ax1.set_ylabel('UTC', weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax4.invert_xaxis()
ax5.invert_xaxis()
ax6.invert_xaxis()
# ax1.set_yticks(x1.Time[0::12].values)
# ax1.set_yticklabels(x1.Time[0::12].dt.strftime("%m-%d %H").values)
ax1.set_yticks(x1.Time[::3].values)
ax2.set_yticks(x1.Time[::3].values)
ax3.set_yticks(x1.Time[::3].values)
ax4.set_yticks(x1.Time[::3].values)
ax5.set_yticks(x1.Time[::3].values)
ax6.set_yticks(x1.Time[::3].values)
ax1.set_yticklabels(x1.Time[::3].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
# Set titles/labels
ax1.set_title('Normal Wind Evolution - d02 [CRF On]', loc='left', fontsize=10)
ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax2.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 03]', loc='left', fontsize=10)
ax2.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax3.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 06]', loc='left', fontsize=10)
ax3.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax4.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 09]', loc='left', fontsize=10)
ax4.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax5.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 12]', loc='left', fontsize=10)
ax5.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax6.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 15]', loc='left', fontsize=10)
ax6.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax1.set_title('', loc='center')
ax2.set_title('', loc='center')
ax3.set_title('', loc='center')
ax4.set_title('', loc='center')
ax5.set_title('', loc='center')
ax6.set_title('', loc='center')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
ax4.grid(linestyle='--', axis='y', linewidth=1.5)
ax5.grid(linestyle='--', axis='y', linewidth=1.5)
ax6.grid(linestyle='--', axis='y', linewidth=1.5)

# Plot the colorbar
	# Normal Wind colorbar
ax2 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Wind Speed [$m s^{-1}$]', weight='bold')


# Normal Wind Anomalies

# In[ ]:


fig = plt.figure(figsize=(56,10))
gs = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.1, hspace=0.25, height_ratios=[0.875,0.03])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])
ax5 = fig.add_subplot(gs[0,4])
ax6 = fig.add_subplot(gs[0,5])

# Data
	# Average over each cross-section
x1 = da_d02_cross_NormalWind_cntl[:,:6,:,:].mean('Spread').mean('bottom_top')
# Data
	# Average over each cross-section
x2 = da_d02_NormalWind_cross_CRFoff[:,:6,:,:,0].mean('Spread').mean('bottom_top')	# starts at 11-25-03UTC
# This concats 11-25 00-04UTC with 11-25 05UTC to 11-26 12UTC
x2 = xr.concat([x1.isel(Time=slice(0,5)),x2[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x2 = x2-x1	# Calculate anomalies from Control
x3 = da_d02_NormalWind_cross_CRFoff[3:,:6,:,:,1].mean('Spread').mean('bottom_top')	# starts at 11-25-06UTC
# This concats 11-25 00-07UTC with 11-25 08UTC to 11-26 12UTC
x3 = xr.concat([x1.isel(Time=slice(0,8)),x3[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x3 = x3-x1	# Calculate anomalies from Control
x4 = da_d02_NormalWind_cross_CRFoff[6:,:6,:,:,2].mean('Spread').mean('bottom_top')	# starts at 11-25-09UTC
# This concats 11-25 00-10UTC with 11-25 11UTC to 11-26 12UTC
x4 = xr.concat([x1.isel(Time=slice(0,11)),x4[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x4 = x4-x1	# Calculate anomalies from Control
x5 = da_d02_NormalWind_cross_CRFoff[9:,:6,:,:,3].mean('Spread').mean('bottom_top')	# starts at 11-25-12UTC
# This concats 11-25 00-13UTC with 11-25 14UTC to 11-26 12UTC
x5 = xr.concat([x1.isel(Time=slice(0,14)),x5[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x5 = x5-x1	# Calculate anomalies from Control
x6 = da_d02_NormalWind_cross_CRFoff[12:,:6,:,:,4].mean('Spread').mean('bottom_top')	# starts at 11-25-15UTC
# This concats 11-25 00-16UTC with 11-25 17UTC to 11-26 12UTC
x6 = xr.concat([x1.isel(Time=slice(0,17)),x6[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x6 = x6-x1	# Calculate anomalies from Control

# Plot the cross-sectional data
	# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 03 UTC
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 06 UTC
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 09 UTC
cf4 = x4.plot.contourf(
	ax=ax4,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 12 UTC
cf5 = x5.plot.contourf(
	ax=ax5,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 15 UTC
cf6 = x6.plot.contourf(
	ax=ax6,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-6,7,1),
	cmap='RdBu_r',
	center=0,
)
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax4.axvline(x=0, color='k', linestyle='--')
ax5.axvline(x=0, color='k', linestyle='--')
ax6.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=x1.Time[3].values, color='r', linestyle='-')
ax3.axhline(y=x1.Time[6].values, color='r', linestyle='-')
ax4.axhline(y=x1.Time[9].values, color='r', linestyle='-')
ax5.axhline(y=x1.Time[12].values, color='r', linestyle='-')
ax6.axhline(y=x1.Time[15].values, color='r', linestyle='-')

ax1.set_xlabel('Distance from coast [km]', weight='bold')
ax2.set_xlabel('Distance from coast [km]', weight='bold')
ax3.set_xlabel('Distance from coast [km]', weight='bold')
ax4.set_xlabel('Distance from coast [km]', weight='bold')
ax5.set_xlabel('Distance from coast [km]', weight='bold')
ax6.set_xlabel('Distance from coast [km]', weight='bold')
ax1.set_ylabel('UTC', weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax4.invert_xaxis()
ax5.invert_xaxis()
ax6.invert_xaxis()
# ax1.set_yticks(x1.Time[0::12].values)
# ax1.set_yticklabels(x1.Time[0::12].dt.strftime("%m-%d %H").values)
ax1.set_yticks(x1.Time[::3].values)
ax2.set_yticks(x1.Time[::3].values)
ax3.set_yticks(x1.Time[::3].values)
ax4.set_yticks(x1.Time[::3].values)
ax5.set_yticks(x1.Time[::3].values)
ax6.set_yticks(x1.Time[::3].values)
ax1.set_yticklabels(x1.Time[::3].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
# Set titles/labels
ax1.set_title('Normal Wind Evolution - d02 [CRF On]', loc='left', fontsize=10)
ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax2.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 03]', loc='left', fontsize=10)
ax2.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax3.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 06]', loc='left', fontsize=10)
ax3.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax4.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 09]', loc='left', fontsize=10)
ax4.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax5.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 12]', loc='left', fontsize=10)
ax5.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax6.set_title('Normal Wind Evolution - d02 [CRF Off @ 11-25 15]', loc='left', fontsize=10)
ax6.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax1.set_title('', loc='center')
ax2.set_title('', loc='center')
ax3.set_title('', loc='center')
ax4.set_title('', loc='center')
ax5.set_title('', loc='center')
ax6.set_title('', loc='center')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
ax4.grid(linestyle='--', axis='y', linewidth=1.5)
ax5.grid(linestyle='--', axis='y', linewidth=1.5)
ax6.grid(linestyle='--', axis='y', linewidth=1.5)

# Plot the colorbar
	# Normal Wind colorbar
ax2 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Wind Speed [$m s^{-1}$]', weight='bold')
	# CRF Off Normal Wind colorbar
ax2 = fig.add_subplot(gs[1, 1:])
cbar = plt.colorbar(cf2, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Wind Speed Anomalies[$m s^{-1}$]', weight='bold')


# Total Moisture Evolution

# In[ ]:


fig = plt.figure(figsize=(56,10))
gs = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.1, hspace=0.25, height_ratios=[0.875,0.03])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])
ax5 = fig.add_subplot(gs[0,4])
ax6 = fig.add_subplot(gs[0,5])

# Data
	# Average over each cross-section
x1 = da_d02_QTotal_cross_cntl[:,:6,:,:].mean('Spread').mean('bottom_top')
# Data
	# Average over each cross-section
x2 = da_d02_QTotal_cross_CRFoff[:,:6,:,:,0].mean('Spread').mean('bottom_top')	# starts at 11-25-03UTC
# This concats 11-25 00-04UTC with 11-25 05UTC to 11-26 12UTC
x2 = xr.concat([x1.isel(Time=slice(0,5)),x2[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x3 = da_d02_QTotal_cross_CRFoff[3:,:6,:,:,1].mean('Spread').mean('bottom_top')	# starts at 11-25-06UTC
# This concats 11-25 00-07UTC with 11-25 08UTC to 11-26 12UTC
x3 = xr.concat([x1.isel(Time=slice(0,8)),x3[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x4 = da_d02_QTotal_cross_CRFoff[6:,:6,:,:,2].mean('Spread').mean('bottom_top')	# starts at 11-25-09UTC
# This concats 11-25 00-10UTC with 11-25 11UTC to 11-26 12UTC
x4 = xr.concat([x1.isel(Time=slice(0,11)),x4[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x5 = da_d02_QTotal_cross_CRFoff[9:,:6,:,:,3].mean('Spread').mean('bottom_top')	# starts at 11-25-12UTC
# This concats 11-25 00-13UTC with 11-25 14UTC to 11-26 12UTC
x5 = xr.concat([x1.isel(Time=slice(0,14)),x5[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x6 = da_d02_QTotal_cross_CRFoff[12:,:6,:,:,4].mean('Spread').mean('bottom_top')	# starts at 11-25-15UTC
# This concats 11-25 00-16UTC with 11-25 17UTC to 11-26 12UTC
x6 = xr.concat([x1.isel(Time=slice(0,17)),x6[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point

# Plot the cross-sectional data
	# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,.0255,.0005),
	cmap='Blues',
	center=0,
)
	# CRF off @ 03 UTC
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,.0255,.0005),
	cmap='Blues',
	center=0,
)
	# CRF off @ 06 UTC
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,.0255,.0005),
	cmap='Blues',
	center=0,
)
	# CRF off @ 09 UTC
cf4 = x4.plot.contourf(
	ax=ax4,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,.0255,.0005),
	cmap='Blues',
	center=0,
)
	# CRF off @ 12 UTC
cf5 = x5.plot.contourf(
	ax=ax5,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,.0255,.0005),
	cmap='Blues',
	center=0,
)
	# CRF off @ 15 UTC
cf6 = x6.plot.contourf(
	ax=ax6,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,.0255,.0005),
	cmap='Blues',
	center=0,
)
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax4.axvline(x=0, color='k', linestyle='--')
ax5.axvline(x=0, color='k', linestyle='--')
ax6.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=x1.Time[3].values, color='r', linestyle='-')
ax3.axhline(y=x1.Time[6].values, color='r', linestyle='-')
ax4.axhline(y=x1.Time[9].values, color='r', linestyle='-')
ax5.axhline(y=x1.Time[12].values, color='r', linestyle='-')
ax6.axhline(y=x1.Time[15].values, color='r', linestyle='-')

ax1.set_xlabel('Distance from coast [km]', weight='bold')
ax2.set_xlabel('Distance from coast [km]', weight='bold')
ax3.set_xlabel('Distance from coast [km]', weight='bold')
ax4.set_xlabel('Distance from coast [km]', weight='bold')
ax5.set_xlabel('Distance from coast [km]', weight='bold')
ax6.set_xlabel('Distance from coast [km]', weight='bold')
ax1.set_ylabel('UTC', weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax4.invert_xaxis()
ax5.invert_xaxis()
ax6.invert_xaxis()
# ax1.set_yticks(x1.Time[0::12].values)
# ax1.set_yticklabels(x1.Time[0::12].dt.strftime("%m-%d %H").values)
ax1.set_yticks(x1.Time[::3].values)
ax2.set_yticks(x1.Time[::3].values)
ax3.set_yticks(x1.Time[::3].values)
ax4.set_yticks(x1.Time[::3].values)
ax5.set_yticks(x1.Time[::3].values)
ax6.set_yticks(x1.Time[::3].values)
ax1.set_yticklabels(x1.Time[::3].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
# Set titles/labels
ax1.set_title('Total Moisture Evolution - d02 [CRF On]', loc='left', fontsize=10)
ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax2.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 03]', loc='left', fontsize=10)
ax2.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax3.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 06]', loc='left', fontsize=10)
ax3.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax4.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 09]', loc='left', fontsize=10)
ax4.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax5.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 12]', loc='left', fontsize=10)
ax5.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax6.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 15]', loc='left', fontsize=10)
ax6.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax1.set_title('', loc='center')
ax2.set_title('', loc='center')
ax3.set_title('', loc='center')
ax4.set_title('', loc='center')
ax5.set_title('', loc='center')
ax6.set_title('', loc='center')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
ax4.grid(linestyle='--', axis='y', linewidth=1.5)
ax5.grid(linestyle='--', axis='y', linewidth=1.5)
ax6.grid(linestyle='--', axis='y', linewidth=1.5)

# Plot the colorbar
	# Total Moisture colorbar
ax2 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Total Moisture [$kg/kg$]', weight='bold')


# Total Moisture Anomaly

# In[ ]:


fig = plt.figure(figsize=(56,10))
gs = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.1, hspace=0.25, height_ratios=[0.875,0.03])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])
ax5 = fig.add_subplot(gs[0,4])
ax6 = fig.add_subplot(gs[0,5])

# Data
	# Average over each cross-section
x1 = da_d02_QTotal_cross_cntl[:,:6,:,:].mean('Spread').mean('bottom_top')
# Data
	# Average over each cross-section
x2 = da_d02_QTotal_cross_CRFoff[:,:6,:,:,0].mean('Spread').mean('bottom_top')	# starts at 11-25-03UTC
# This concats 11-25 00-04UTC with 11-25 05UTC to 11-26 12UTC
x2 = xr.concat([x1.isel(Time=slice(0,5)),x2[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x2 = x2-x1	# Calculate anomalies from Control
x3 = da_d02_QTotal_cross_CRFoff[3:,:6,:,:,1].mean('Spread').mean('bottom_top')	# starts at 11-25-06UTC
# This concats 11-25 00-07UTC with 11-25 08UTC to 11-26 12UTC
x3 = xr.concat([x1.isel(Time=slice(0,8)),x3[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x3 = x3-x1	# Calculate anomalies from Control
x4 = da_d02_QTotal_cross_CRFoff[6:,:6,:,:,2].mean('Spread').mean('bottom_top')	# starts at 11-25-09UTC
# This concats 11-25 00-10UTC with 11-25 11UTC to 11-26 12UTC
x4 = xr.concat([x1.isel(Time=slice(0,11)),x4[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x4 = x4-x1	# Calculate anomalies from Control
x5 = da_d02_QTotal_cross_CRFoff[9:,:6,:,:,3].mean('Spread').mean('bottom_top')	# starts at 11-25-12UTC
# This concats 11-25 00-13UTC with 11-25 14UTC to 11-26 12UTC
x5 = xr.concat([x1.isel(Time=slice(0,14)),x5[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x5 = x5-x1	# Calculate anomalies from Control
x6 = da_d02_QTotal_cross_CRFoff[12:,:6,:,:,4].mean('Spread').mean('bottom_top')	# starts at 11-25-15UTC
# This concats 11-25 00-16UTC with 11-25 17UTC to 11-26 12UTC
x6 = xr.concat([x1.isel(Time=slice(0,17)),x6[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x6 = x6-x1	# Calculate anomalies from Control

# Plot the cross-sectional data
	# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(0,0.0255,0.0005),
	cmap='Blues',
	center=0,
)
	# CRF off @ 03 UTC
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-0.005,0.0055,0.0005),
	cmap='BrBG',
	center=0,
)
	# CRF off @ 06 UTC
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-0.005,0.0055,0.0005),
	cmap='BrBG',
	center=0,
)
	# CRF off @ 09 UTC
cf4 = x4.plot.contourf(
	ax=ax4,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-0.005,0.0055,0.0005),
	cmap='BrBG',
	center=0,
)
	# CRF off @ 12 UTC
cf5 = x5.plot.contourf(
	ax=ax5,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-0.005,0.0055,0.0005),
	cmap='BrBG',
	center=0,
)
	# CRF off @ 15 UTC
cf6 = x6.plot.contourf(
	ax=ax6,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-0.005,0.0055,0.0005),
	cmap='BrBG',
	center=0,
)
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax4.axvline(x=0, color='k', linestyle='--')
ax5.axvline(x=0, color='k', linestyle='--')
ax6.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=x1.Time[3].values, color='r', linestyle='-')
ax3.axhline(y=x1.Time[6].values, color='r', linestyle='-')
ax4.axhline(y=x1.Time[9].values, color='r', linestyle='-')
ax5.axhline(y=x1.Time[12].values, color='r', linestyle='-')
ax6.axhline(y=x1.Time[15].values, color='r', linestyle='-')

ax1.set_xlabel('Distance from coast [km]', weight='bold')
ax2.set_xlabel('Distance from coast [km]', weight='bold')
ax3.set_xlabel('Distance from coast [km]', weight='bold')
ax4.set_xlabel('Distance from coast [km]', weight='bold')
ax5.set_xlabel('Distance from coast [km]', weight='bold')
ax6.set_xlabel('Distance from coast [km]', weight='bold')
ax1.set_ylabel('UTC', weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax4.invert_xaxis()
ax5.invert_xaxis()
ax6.invert_xaxis()
# ax1.set_yticks(x1.Time[0::12].values)
# ax1.set_yticklabels(x1.Time[0::12].dt.strftime("%m-%d %H").values)
ax1.set_yticks(x1.Time[::3].values)
ax2.set_yticks(x1.Time[::3].values)
ax3.set_yticks(x1.Time[::3].values)
ax4.set_yticks(x1.Time[::3].values)
ax5.set_yticks(x1.Time[::3].values)
ax6.set_yticks(x1.Time[::3].values)
ax1.set_yticklabels(x1.Time[::3].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
# Set titles/labels
ax1.set_title('Total Moisture Evolution - d02 [CRF On]', loc='left', fontsize=10)
ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax2.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 03]', loc='left', fontsize=10)
ax2.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax3.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 06]', loc='left', fontsize=10)
ax3.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax4.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 09]', loc='left', fontsize=10)
ax4.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax5.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 12]', loc='left', fontsize=10)
ax5.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax6.set_title('Total Moisture Evolution - d02 [CRF Off @ 11-25 15]', loc='left', fontsize=10)
ax6.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax1.set_title('', loc='center')
ax2.set_title('', loc='center')
ax3.set_title('', loc='center')
ax4.set_title('', loc='center')
ax5.set_title('', loc='center')
ax6.set_title('', loc='center')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
ax4.grid(linestyle='--', axis='y', linewidth=1.5)
ax5.grid(linestyle='--', axis='y', linewidth=1.5)
ax6.grid(linestyle='--', axis='y', linewidth=1.5)

# Plot the colorbar
	# Total Moisture colorbar
ax2 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Total Moisture [$kg/kg$]', weight='bold')
	# CRF Off Total Moisture colorbar
ax2 = fig.add_subplot(gs[1, 1:])
cbar = plt.colorbar(cf2, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Total Moisture Anomalies [$kg/kg$]', weight='bold')


# Theta or Potential Temperature Evolution

# In[ ]:


fig = plt.figure(figsize=(56,10))
gs = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.1, hspace=0.25, height_ratios=[0.875,0.03])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])
ax5 = fig.add_subplot(gs[0,4])
ax6 = fig.add_subplot(gs[0,5])

# Data
	# Average over each cross-section
x1 = da_d02_Theta_cross_cntl[:,:6,:,:].mean('Spread').mean('bottom_top')
# Data
	# Average over each cross-section
x2 = da_d02_Theta_cross_CRFoff[:,:6,:,:,0].mean('Spread').mean('bottom_top')	# starts at 11-25-03UTC
# This concats 11-25 00-04UTC with 11-25 05UTC to 11-26 12UTC
x2 = xr.concat([x1.isel(Time=slice(0,5)),x2[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x3 = da_d02_Theta_cross_CRFoff[3:,:6,:,:,1].mean('Spread').mean('bottom_top')	# starts at 11-25-06UTC
# This concats 11-25 00-07UTC with 11-25 08UTC to 11-26 12UTC
x3 = xr.concat([x1.isel(Time=slice(0,8)),x3[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x4 = da_d02_Theta_cross_CRFoff[6:,:6,:,:,2].mean('Spread').mean('bottom_top')	# starts at 11-25-09UTC
# This concats 11-25 00-10UTC with 11-25 11UTC to 11-26 12UTC
x4 = xr.concat([x1.isel(Time=slice(0,11)),x4[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x5 = da_d02_Theta_cross_CRFoff[9:,:6,:,:,3].mean('Spread').mean('bottom_top')	# starts at 11-25-12UTC
# This concats 11-25 00-13UTC with 11-25 14UTC to 11-26 12UTC
x5 = xr.concat([x1.isel(Time=slice(0,14)),x5[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x6 = da_d02_Theta_cross_CRFoff[12:,:6,:,:,4].mean('Spread').mean('bottom_top')	# starts at 11-25-15UTC
# This concats 11-25 00-16UTC with 11-25 17UTC to 11-26 12UTC
x6 = xr.concat([x1.isel(Time=slice(0,17)),x6[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point

# Plot the cross-sectional data
	# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(297,306,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 03 UTC
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(297,306,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 06 UTC
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(297,306,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 09 UTC
cf4 = x4.plot.contourf(
	ax=ax4,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(297,306,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 12 UTC
cf5 = x5.plot.contourf(
	ax=ax5,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(297,306,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 15 UTC
cf6 = x6.plot.contourf(
	ax=ax6,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(297,306,0.5),
	cmap='RdBu_r',
	center=0,
)
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax4.axvline(x=0, color='k', linestyle='--')
ax5.axvline(x=0, color='k', linestyle='--')
ax6.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=x1.Time[3].values, color='r', linestyle='-')
ax3.axhline(y=x1.Time[6].values, color='r', linestyle='-')
ax4.axhline(y=x1.Time[9].values, color='r', linestyle='-')
ax5.axhline(y=x1.Time[12].values, color='r', linestyle='-')
ax6.axhline(y=x1.Time[15].values, color='r', linestyle='-')

ax1.set_xlabel('Distance from coast [km]', weight='bold')
ax2.set_xlabel('Distance from coast [km]', weight='bold')
ax3.set_xlabel('Distance from coast [km]', weight='bold')
ax4.set_xlabel('Distance from coast [km]', weight='bold')
ax5.set_xlabel('Distance from coast [km]', weight='bold')
ax6.set_xlabel('Distance from coast [km]', weight='bold')
ax1.set_ylabel('UTC', weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax4.invert_xaxis()
ax5.invert_xaxis()
ax6.invert_xaxis()
# ax1.set_yticks(x1.Time[0::12].values)
# ax1.set_yticklabels(x1.Time[0::12].dt.strftime("%m-%d %H").values)
ax1.set_yticks(x1.Time[::3].values)
ax2.set_yticks(x1.Time[::3].values)
ax3.set_yticks(x1.Time[::3].values)
ax4.set_yticks(x1.Time[::3].values)
ax5.set_yticks(x1.Time[::3].values)
ax6.set_yticks(x1.Time[::3].values)
ax1.set_yticklabels(x1.Time[::3].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
# Set titles/labels
ax1.set_title('Potential Temperature Evolution - d02 [CRF On]', loc='left', fontsize=10)
ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax2.set_title('Potential Temperature Evolution - d02 [CRF Off @ 11-25 03]', loc='left', fontsize=10)
ax2.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax3.set_title('Potential Temperature Evolution - d02 [CRF Off @ 11-25 06]', loc='left', fontsize=10)
ax3.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax4.set_title('Potential Temperature Evolution - d02 [CRF Off @ 11-25 09]', loc='left', fontsize=10)
ax4.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax5.set_title('Potential Temperature Evolution - d02 [CRF Off @ 11-25 12]', loc='left', fontsize=10)
ax5.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax6.set_title('Potential Temperature Evolution - d02 [CRF Off @ 11-25 15]', loc='left', fontsize=10)
ax6.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax1.set_title('', loc='center')
ax2.set_title('', loc='center')
ax3.set_title('', loc='center')
ax4.set_title('', loc='center')
ax5.set_title('', loc='center')
ax6.set_title('', loc='center')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
ax4.grid(linestyle='--', axis='y', linewidth=1.5)
ax5.grid(linestyle='--', axis='y', linewidth=1.5)
ax6.grid(linestyle='--', axis='y', linewidth=1.5)

# Plot the colorbar
	# Potential Temperature colorbar
ax2 = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Potential Temperature [$K$]', weight='bold')


# Anomalous Potential Temperature Evolution

# In[ ]:


fig = plt.figure(figsize=(56,10))
gs = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.1, hspace=0.25, height_ratios=[0.875,0.03])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[0,3])
ax5 = fig.add_subplot(gs[0,4])
ax6 = fig.add_subplot(gs[0,5])

# Data
	# Average over each cross-section
x1 = da_d02_Theta_cross_cntl[:,:6,:,:].mean('Spread').mean('bottom_top')
# Data
	# Average over each cross-section
x2 = da_d02_Theta_cross_CRFoff[:,:6,:,:,0].mean('Spread').mean('bottom_top')	# starts at 11-25-03UTC
# This concats 11-25 00-04UTC with 11-25 05UTC to 11-26 12UTC
x2 = xr.concat([x1.isel(Time=slice(0,5)),x2[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x2 = x2-x1	# Calculate anomalies from Control
x3 = da_d02_Theta_cross_CRFoff[3:,:6,:,:,1].mean('Spread').mean('bottom_top')	# starts at 11-25-06UTC
# This concats 11-25 00-07UTC with 11-25 08UTC to 11-26 12UTC
x3 = xr.concat([x1.isel(Time=slice(0,8)),x3[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x3 = x3-x1	# Calculate anomalies from Control
x4 = da_d02_Theta_cross_CRFoff[6:,:6,:,:,2].mean('Spread').mean('bottom_top')	# starts at 11-25-09UTC
# This concats 11-25 00-10UTC with 11-25 11UTC to 11-26 12UTC
x4 = xr.concat([x1.isel(Time=slice(0,11)),x4[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x4 = x4-x1	# Calculate anomalies from Control
x5 = da_d02_Theta_cross_CRFoff[9:,:6,:,:,3].mean('Spread').mean('bottom_top')	# starts at 11-25-12UTC
# This concats 11-25 00-13UTC with 11-25 14UTC to 11-26 12UTC
x5 = xr.concat([x1.isel(Time=slice(0,14)),x5[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x5 = x5-x1	# Calculate anomalies from Control
x6 = da_d02_Theta_cross_CRFoff[12:,:6,:,:,4].mean('Spread').mean('bottom_top')	# starts at 11-25-15UTC
# This concats 11-25 00-16UTC with 11-25 17UTC to 11-26 12UTC
x6 = xr.concat([x1.isel(Time=slice(0,17)),x6[1:]],dim='Time',data_vars='all') # Must start at x2[1:] since the oth index is accumulated RR from start of cntl sim to the experiment point
x6 = x6-x1	# Calculate anomalies from Control

# Plot the cross-sectional data
	# Control
cf1 = x1.plot.contourf(
	ax=ax1,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(297,306,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 03 UTC
cf2 = x2.plot.contourf(
	ax=ax2,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-4,4,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 06 UTC
cf3 = x3.plot.contourf(
	ax=ax3,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-4,4,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 09 UTC
cf4 = x4.plot.contourf(
	ax=ax4,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-4,4,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 12 UTC
cf5 = x5.plot.contourf(
	ax=ax5,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-4,4,0.5),
	cmap='RdBu_r',
	center=0,
)
	# CRF off @ 15 UTC
cf6 = x6.plot.contourf(
	ax=ax6,
	x = 'Distance',
    y = 'Time',
	add_colorbar=False,
	levels=np.arange(-4,4,0.5),
	cmap='RdBu_r',
	center=0,
)
# Plot the vertical line at approximate coastline
ax1.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')
ax4.axvline(x=0, color='k', linestyle='--')
ax5.axvline(x=0, color='k', linestyle='--')
ax6.axvline(x=0, color='k', linestyle='--')
ax2.axhline(y=x1.Time[3].values, color='r', linestyle='-')
ax3.axhline(y=x1.Time[6].values, color='r', linestyle='-')
ax4.axhline(y=x1.Time[9].values, color='r', linestyle='-')
ax5.axhline(y=x1.Time[12].values, color='r', linestyle='-')
ax6.axhline(y=x1.Time[15].values, color='r', linestyle='-')

ax1.set_xlabel('Distance from coast [km]', weight='bold')
ax2.set_xlabel('Distance from coast [km]', weight='bold')
ax3.set_xlabel('Distance from coast [km]', weight='bold')
ax4.set_xlabel('Distance from coast [km]', weight='bold')
ax5.set_xlabel('Distance from coast [km]', weight='bold')
ax6.set_xlabel('Distance from coast [km]', weight='bold')
ax1.set_ylabel('UTC', weight='bold')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax4.invert_xaxis()
ax5.invert_xaxis()
ax6.invert_xaxis()
# ax1.set_yticks(x1.Time[0::12].values)
# ax1.set_yticklabels(x1.Time[0::12].dt.strftime("%m-%d %H").values)
ax1.set_yticks(x1.Time[::3].values)
ax2.set_yticks(x1.Time[::3].values)
ax3.set_yticks(x1.Time[::3].values)
ax4.set_yticks(x1.Time[::3].values)
ax5.set_yticks(x1.Time[::3].values)
ax6.set_yticks(x1.Time[::3].values)
ax1.set_yticklabels(x1.Time[::3].dt.strftime("%m-%d %H").values)
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
# Set titles/labels
ax1.set_title('Theta Evolution - d02 [CRF On]', loc='left', fontsize=10)
ax1.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax2.set_title('Theta Evolution - d02 [CRF Off @ 11-25 03]', loc='left', fontsize=10)
ax2.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax3.set_title('Theta Evolution - d02 [CRF Off @ 11-25 06]', loc='left', fontsize=10)
ax3.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax4.set_title('Theta Evolution - d02 [CRF Off @ 11-25 09]', loc='left', fontsize=10)
ax4.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax5.set_title('Theta Evolution - d02 [CRF Off @ 11-25 12]', loc='left', fontsize=10)
ax5.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax6.set_title('Theta Evolution - d02 [CRF Off @ 11-25 15]', loc='left', fontsize=10)
ax6.set_title('Western Central Coast of Sumatra', loc='right', fontsize=10)
ax1.set_title('', loc='center')
ax2.set_title('', loc='center')
ax3.set_title('', loc='center')
ax4.set_title('', loc='center')
ax5.set_title('', loc='center')
ax6.set_title('', loc='center')
# Create grids
ax1.grid(linestyle='--', axis='y', linewidth=1.5)
ax2.grid(linestyle='--', axis='y', linewidth=1.5)
ax3.grid(linestyle='--', axis='y', linewidth=1.5)
ax4.grid(linestyle='--', axis='y', linewidth=1.5)
ax5.grid(linestyle='--', axis='y', linewidth=1.5)
ax6.grid(linestyle='--', axis='y', linewidth=1.5)

# Plot the colorbar
	# Theta colorbar
ax2 = fig.add_subplot(gs[1, 0])
cbar = plt.colorbar(cf1, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Theta [$K$]', weight='bold')
	# CRF Off Theta colorbar
ax2 = fig.add_subplot(gs[1, 1:])
cbar = plt.colorbar(cf2, cax=ax2, orientation='horizontal', pad=0 , aspect=100)
# cbar.set_ticks(np.arange(0,7,1))
cbar.set_label('Theta Anomalies [$K$]', weight='bold')


# #### Spatial Analysis

# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)

# Assign variable
x = da_d02_RR.sel(
	Time=slice(da_d02_RR.Time[60],da_d02_RR_cross_preconv.Time[-1])
)

# Set the lat lon bounds
lat = [da_d02_cross_NormalWind_cntl.Lat.max()+0.5,da_d02_cross_NormalWind_cntl.Lat.min()-0.5]
lon = [da_d02_cross_NormalWind_cntl.Lon.min()-0.5,da_d02_cross_NormalWind_cntl.Lon.max()+0.5]

# Yokoi et al. 2017-2019 domain:
x = x.sel(
	south_north=slice(lat[1],lat[0]),
	west_east=slice(lon[0],lon[1]))

x_ticks = np.array([100,102,104])
x_tick_labels = [u'100\N{DEGREE SIGN}E',
                 u'102\N{DEGREE SIGN}E', u'104\N{DEGREE SIGN}E']
y_ticks = np.array([-6,-4,-2])
y_tick_labels = [u'6\N{DEGREE SIGN}S',
                 u'4\N{DEGREE SIGN}S', u'2\N{DEGREE SIGN}S']
# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(9.75,6.75))
	gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)
	ax1 = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree(central_longitude=0))
	ax1.coastlines(linewidth=1.5, color='k', resolution='50m')  # cartopy function

	cf1 = x[i].plot.contourf(
		ax=ax1,
		levels=np.arange(0,6.5,0.5),
		cmap='gray_r',
	)

	plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,0],da_d02_cross_NormalWind_cntl.Lat[:,0],'r',linewidth=0.5)
	plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,-1],da_d02_cross_NormalWind_cntl.Lat[:,-1],'r',linewidth=0.5)
	# Plot the center line
	plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],da_d02_cross_NormalWind_cntl.Lat[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],'r',linewidth=1)
	# Plot the off-shore radar (R/V Mirai of JAMSTEC)
	plt.scatter(101.90,-4.07,s=100,marker='*',c='r')
	# Plot the on-shore observatory in Bengkulu city (BMKG observatory)
	plt.scatter(102.34,-3.86,s=100,marker='o',c='r')

	cbar=cf1.colorbar
	cbar.set_label('Rain Rate [$mm d^{-1}$]',fontsize=16)
	ax1.set_xlabel('Longitude',fontsize=16)
	ax1.set_ylabel('Latitude',fontsize=16)
	# In UTC
	local_date = x.Time[i]
	string = 'Rain rate at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF On', loc='left')
	# ax1.set_title('Central Sumatra\nRain Rate Evolution',fontsize=22)
	ax1.set_xticks(x_ticks)
	ax1.set_xticklabels(x_tick_labels,fontsize=14)
	ax1.set_yticks(y_ticks)
	ax1.set_yticklabels(y_tick_labels,fontsize=14)

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Sumatra_RR_cntl_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# In[ ]:


fig = plt.figure(figsize=(9.75,6.75))
gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)

# Assign variable
file = parent_dir +'/CRFoff/pre_conv_2015-11-25-03--11-26-12'+'/L1/d02_RR'				# [mm/dt]
ds = open_ds(file,time_ind_d02,lat_ind_d02,lon_ind_d02)
x = ds['RR'].compute()
d02_coords = dict(
    XLAT=(('Time','south_north','west_east'),ds_d02.XLAT.values[64:97,...]),
    XLONG=(('Time','south_north','west_east'),ds_d02.XLONG.values[64:97,...]),
    bottom_top=(('bottom_top'),interp_P_levels),
    Time=('Time',ds_d02.XTIME.values[64:97]),
    south_north=(('south_north'),ds_d02.XLAT[0,:,0].values),
    west_east=(('west_east'),ds_d02.XLONG[0,0,:].values)
    )
x = x.assign_coords(without_keys(d02_coords,'bottom_top'))
	# Average over each cross-section
y = da_d02_RR
x = xr.concat([y.isel(Time=slice(60,65)),x[1:]],dim='Time',data_vars='all')


# Set the lat lon bounds
lat = [da_d02_cross_NormalWind_cntl.Lat.max()+0.5,da_d02_cross_NormalWind_cntl.Lat.min()-0.5]
lon = [da_d02_cross_NormalWind_cntl.Lon.min()-0.5,da_d02_cross_NormalWind_cntl.Lon.max()+0.5]

# Yokoi et al. 2017-2019 domain:
x = x.sel(
	south_north=slice(lat[1],lat[0]),
	west_east=slice(lon[0],lon[1]))

x_ticks = np.array([100,102,104])
x_tick_labels = [u'100\N{DEGREE SIGN}E',
                 u'102\N{DEGREE SIGN}E', u'104\N{DEGREE SIGN}E']
y_ticks = np.array([-6,-4,-2])
y_tick_labels = [u'6\N{DEGREE SIGN}S',
                 u'4\N{DEGREE SIGN}S', u'2\N{DEGREE SIGN}S']
# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(9.75,6.75))
	gs = gridspec.GridSpec(nrows=1, ncols=1, hspace=0.075)
	ax1 = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree(central_longitude=0))
	ax1.coastlines(linewidth=1.5, color='k', resolution='50m')  # cartopy function

	cf1 = x[i].plot.contourf(
		ax=ax1,
		levels=np.arange(0,6.5,0.5),
		cmap='gray_r',
	)

	plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,0],da_d02_cross_NormalWind_cntl.Lat[:,0],'r',linewidth=0.5)
	plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,-1],da_d02_cross_NormalWind_cntl.Lat[:,-1],'r',linewidth=0.5)
	# Plot the center line
	plt.plot(da_d02_cross_NormalWind_cntl.Lon[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],da_d02_cross_NormalWind_cntl.Lat[:,int(da_d02_cross_NormalWind_cntl.shape[3]/2)],'r',linewidth=1)
	# Plot the off-shore radar (R/V Mirai of JAMSTEC)
	plt.scatter(101.90,-4.07,s=100,marker='*',c='r')
	# Plot the on-shore observatory in Bengkulu city (BMKG observatory)
	plt.scatter(102.34,-3.86,s=100,marker='o',c='r')

	cbar=cf1.colorbar
	cbar.set_label('Rain Rate [$mm d^{-1}$]',fontsize=16)
	ax1.set_xlabel('Longitude',fontsize=16)
	ax1.set_ylabel('Latitude',fontsize=16)
	# In UTC
	local_date = x.Time[i]
	string = 'Rain rate at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF Off @ 11-25 03UTC', loc='left', fontsize=10)
	# ax1.set_title('Central Sumatra\nRain Rate Evolution',fontsize=22)
	ax1.set_xticks(x_ticks)
	ax1.set_xticklabels(x_tick_labels,fontsize=14)
	ax1.set_yticks(y_ticks)
	ax1.set_yticklabels(y_tick_labels,fontsize=14)

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Sumatra_RR_cntl_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# #### Cross-sectional Analysis

# In[ ]:


# Data
x = da_d02_cross_NormalWind_cntl.mean('Spread')[60:97]
z = da_d02_RR_cross_cntl.mean('Spread')[60:97]

# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-6,7,1),
		cmap='RdBu_r',
		yscale='log',
		center=0,
		ylim=[200,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Normal Wind at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF On', loc='left', fontsize=10)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Wind Speed [$m s^{-1}$]')

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_NormalWind_cntl_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_NormalWind_cross_preconv.mean('Spread')
y = da_d02_cross_NormalWind_cntl.mean('Spread')
x = xr.concat([y.isel(Time=slice(60,64)),x[:]],dim='Time',data_vars='all')

z = da_d02_RR_cross_preconv.mean('Spread')
y = da_d02_RR_cross_cntl.mean('Spread')
z = xr.concat([y.isel(Time=slice(60,65)),z[1:]],dim='Time',data_vars='all')

# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-6,7,1),
		cmap='RdBu_r',
		yscale='log',
		center=0,
		ylim=[200,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Normal Wind at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF Off @ '+ str(x.Time[3].dt.strftime("%m-%d %H").to_numpy()) + 'UTC', loc='left', fontsize=8)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Wind Speed [$m s^{-1}$]')

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_NormalWind_preconv_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_CLDFRA_cross_cntl.mean('Spread')[60:97]
z = da_d02_RR_cross_cntl.mean('Spread')[60:97]

# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		# levels=np.arange(0,0.525,0.025),
		levels=np.append(0,np.logspace(-2,0,25)),
		# vmax=0.5,
		cmap='Greys',
		yscale='log',
		# center=0,
		ylim=[100,1000],
		# alpha=0.4
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Cloud Fraction at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF On', loc='left', fontsize=10)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_label('Cloud Fraction')
	cbar.set_ticks([0,0.1,0.2,0.4,0.6,0.8,1])
	cbar.minorticks_off()
	cbar.set_ticklabels([0,0.1,0.2,0.4,0.6,0.8,1])

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_CLDFRA_cntl_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_CLDFRA_cross_preconv.mean('Spread')
y = da_d02_CLDFRA_cross_cntl.mean('Spread')
x = xr.concat([y.isel(Time=slice(60,64)),x[:]],dim='Time',data_vars='all')

z = da_d02_RR_cross_preconv.mean('Spread')
y = da_d02_RR_cross_cntl.mean('Spread')
z = xr.concat([y.isel(Time=slice(60,65)),z[1:]],dim='Time',data_vars='all')

# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')

	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		# levels=np.arange(0,0.525,0.025),
		levels=np.append(0,np.logspace(-2,0,25)),
		# vmax=0.5,
		cmap='Greys',
		yscale='log',
		# center=0,
		ylim=[100,1000],
		# alpha=0.4
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Cloud Fraction at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF Off @ '+ str(x.Time[3].dt.strftime("%m-%d %H").to_numpy()) + 'UTC', loc='left', fontsize=8)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	cbar.set_label('Cloud Fraction')
	cbar.set_ticks([0,0.1,0.2,0.4,0.6,0.8,1])
	cbar.minorticks_off()
	cbar.set_ticklabels([0,0.1,0.2,0.4,0.6,0.8,1])

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_CLDFRA_preconv_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_RH_cross_cntl.mean('Spread')[60:97]
z = da_d02_RR_cross_cntl.mean('Spread')[60:97]

for i in range(x.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(0,100,5),
		extend='max',
		cmap='BrBG',
		yscale='log',
		center=0,
		ylim=[100,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Relative Humidity at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF On', loc='left', fontsize=10)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Relative Humidity [%]')

	plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_RH_cntl_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	mpl.pyplot.close()


# In[ ]:


# Data

x = da_d02_RH_cross_preconv.mean('Spread')
y = da_d02_RH_cross_cntl.mean('Spread')
x = xr.concat([y.isel(Time=slice(60,64)),x[:]],dim='Time',data_vars='all')

z = da_d02_RR_cross_preconv.mean('Spread')
y = da_d02_RR_cross_cntl.mean('Spread')
z = xr.concat([y.isel(Time=slice(60,65)),z[1:]],dim='Time',data_vars='all')


for i in range(x.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(0,100,5),
		extend='max',
		cmap='BrBG',
		yscale='log',
		center=0,
		ylim=[100,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Relative Humidity at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF Off @ '+ str(x.Time[3].dt.strftime("%m-%d %H").to_numpy()) + 'UTC', loc='left', fontsize=8)
	yticks = np.linspace(1000,100,10)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, extend='max')
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Relative Humidity [%]')

	plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_RH_preconv_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_QTotal_cross_cntl.mean('Spread')[60:97]
z = da_d02_RR_cross_cntl.mean('Spread')[60:97]

# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(0,.021,.0005),
		cmap='BrBG',
		yscale='log',
		center=0,
		ylim=[200,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Total Moisture at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF On', loc='left', fontsize=10)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Total Moisture [$kg/kg$]')

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_QTotal_cntl_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_QTotal_cross_preconv.mean('Spread')
y = da_d02_QTotal_cross_cntl.mean('Spread')
x = xr.concat([y.isel(Time=slice(60,64)),x[:]],dim='Time',data_vars='all')

z = da_d02_RR_cross_preconv.mean('Spread')
y = da_d02_RR_cross_cntl.mean('Spread')
z = xr.concat([y.isel(Time=slice(60,65)),z[1:]],dim='Time',data_vars='all')

for i in range(x.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(0,.021,.0005),
		cmap='BrBG',
		yscale='log',
		center=0,
		ylim=[200,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Total Moisture at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF Off @ '+ str(x.Time[3].dt.strftime("%m-%d %H").to_numpy()) + 'UTC', loc='left', fontsize=8)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Total Moisture [$kg/kg$]')

	plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_QTotal_preconv_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_QI_cross_cntl.mean('Spread')[60:97]
z = da_d02_RR_cross_cntl.mean('Spread')[60:97]

for i in range(x.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(0,.00008,.000005),
		cmap='BrBG',
		yscale='log',
		center=0,
		ylim=[200,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Ice Mixing Ratio at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF On', loc='left', fontsize=10)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Ice mixing ratio [$kg/kg$]')

	plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_QIce_cntl_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_QI_cross_preconv.mean('Spread')
y = da_d02_QI_cross_cntl.mean('Spread')
x = xr.concat([y.isel(Time=slice(60,64)),x[:]],dim='Time',data_vars='all')

z = da_d02_RR_cross_preconv.mean('Spread')
y = da_d02_RR_cross_cntl.mean('Spread')
z = xr.concat([y.isel(Time=slice(60,65)),z[1:]],dim='Time',data_vars='all')

for i in range(x.shape[0]):
# for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(0,.000085,.000005),
		cmap='BrBG',
		yscale='log',
		center=0,
		ylim=[200,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Ice Mixing Ratio at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF Off @ '+ str(x.Time[3].dt.strftime("%m-%d %H").to_numpy()) + 'UTC', loc='left', fontsize=8)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Ice mixing ratio [$kg/kg$]')

	plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_QIce_preconv_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	mpl.pyplot.close()


# In[ ]:


# Data
x = da_d02_NormalWind_cross_preconv.mean('Spread')
y = da_d02_cross_NormalWind_cntl.mean('Spread')
x = xr.concat([y.isel(Time=slice(60,64)),x[:]],dim='Time',data_vars='all')

z = da_d02_RR_cross_preconv.mean('Spread')
y = da_d02_RR_cross_cntl.mean('Spread')
z = xr.concat([y.isel(Time=slice(60,65)),z[1:]],dim='Time',data_vars='all')

# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-6,7,1),
		cmap='RdBu_r',
		yscale='log',
		center=0,
		ylim=[200,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Normal Wind at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF Off @ '+ str(x.Time[3].dt.strftime("%m-%d %H").to_numpy()) + 'UTC', loc='left', fontsize=8)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Wind Speed [$m s^{-1}$]')

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_NormalWind_preconv_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# #### Sanity Check

# In[ ]:


# Data
	# Composite diurnally, and then average over each cross-section
x = da_d02_NetCRF_cross_preconv.mean('Spread')
y = da_d02_NetCRF_cross_cntl.mean('Spread')
x = xr.concat([y.isel(Time=slice(60,64)),x[:]],dim='Time',data_vars='all')

z = da_d02_RR_cross_preconv.mean('Spread')
y = da_d02_RR_cross_cntl.mean('Spread')
z = xr.concat([y.isel(Time=slice(60,65)),z[1:]],dim='Time',data_vars='all')

# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-3*10**-5,3.25*10**-5,0.25*10**-5),
		extend='both',
		vmax=3*10**-5,
		vmin=-3*10**-5,
		# norm=MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0),
		cmap='RdBu_r',
		yscale='log',
		ylim=[100,1000],
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Net CRF at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF Off @ '+ str(x.Time[3].dt.strftime("%m-%d %H").to_numpy()) + 'UTC', loc='left', fontsize=8)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100, ticks=np.arange(-3*10**-5,4*10**-5,1*10**-5))
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Net CRF [$K s^{-1}$]')

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_NetCRF_preconv_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()


# In[ ]:


# Calculate temperature
x = (1000/np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50))))**0.286
da_d02_Temp_cross_cntl = xr.DataArray.copy(da_d02_Theta_cross_cntl)
for i in range(len(x)):
	da_d02_Temp_cross_cntl[:,i,:,:] = da_d02_Theta_cross_cntl[:,i,:,:]/x[i]


# In[ ]:


# Data
x = da_d02_Theta_cross_cntl.mean('Spread')[60:97]
z = da_d02_RR_cross_cntl.mean('Spread')[60:97]

# for i in range(x.shape[0]):
for i in range(1):
	fig = plt.figure(figsize=(6.5,4.5))
	gs = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.1, wspace= 0.1, height_ratios=[0.85,0.15], width_ratios=[.96,.04])
	ax1 = fig.add_subplot(gs[0,0])

	# Plot terrains
	y = np.max(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='blue',linewidth=1)
	y = np.min(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='red',linewidth=1)
	y = np.average(terrain_height_d02,axis=(0,2))
	plt.plot(x.Distance,y,color='black')
	# Plot the cross-sectional data
	cf1 = x[i,...].plot.contourf(
		ax=ax1,
		add_colorbar=False,
		levels=np.arange(-6,7,1),
		cmap='RdBu_r',
		yscale='log',
		center=0,
		ylim=[200,1000]
	)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax1.set_ylabel('Pressure Levels [hPa]')
	ax1.set_xlabel('')
	ax1.invert_yaxis()
	ax1.invert_xaxis()
	# In UTC
	local_date = x.Time[i]
	string = 'Normal Wind at ' + str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy()) + 'UTC'
	ax1.set_title('',loc='center')
	ax1.set_title(string,loc='right')
	ax1.set_title('CRF On', loc='left', fontsize=10)
	yticks = np.linspace(1000,200,9)
	ax1.set_yticks(yticks)
	ax1.set_xticks([])
	ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Plot the rain rates
	ax2 = fig.add_subplot(gs[1,0])
	l1 = z[i,...].plot(
		ax=ax2,
		xlim=[z.Distance[0],z.Distance[-1]],
		ylim=[0,4],
	)
	ax2.grid(linestyle='--', axis='y', linewidth=0.5)
	# Plot the vertical line at approximate coastline
	plt.axvline(x=0, color='k', linestyle='--')
	ax2.set_xlabel('Distance from coast [km]')
	ax2.set_ylabel('Rain Rate\n[$mm day^{-1}$]\n',fontsize=8)
	ax2.invert_xaxis()
	ax2.set_yticks([0,2,4])
	ax2.set_title('')
	# Plot the color bar
	ax3 = fig.add_subplot(gs[:, 1])
	cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	# cbar = plt.colorbar(cf1, cax=ax3, orientation='vertical', pad=0 , aspect=100)
	cbar.set_label('Wind Speed [$m s^{-1}$]')

	# plt.savefig('/home/hragnajarian/analysis/temp_plots/WRF_Cross_Sumatra_NormalWind_cntl_d02_'+str(local_date.dt.strftime("%Y-%m-%d %H").to_numpy())+'UTC.png',dpi=300)
	# mpl.pyplot.close()

