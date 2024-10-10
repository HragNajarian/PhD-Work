#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" 
Author: Hrag Najarian
Date made: September 9, 2024
Purpose: Concat CRFOff wrfout files without overlap
"""


# In[ ]:


# Run this command on the command line to create a .py script instead of .ipynb
	# jupyter nbconvert concat_CRFOff.ipynb --to python


# In[6]:


import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import wrf
import sys
import glob
import xarray as xr


# In[20]:


# Declare parent_dir where all the directories are located 
parent_dir = sys.argv[1]
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'

# List files
sunset_raw_d02 = sorted(glob.glob(parent_dir + '/*00/raw/d02'))			# Sunset case since it ends with 00UTC for a 36hr run, meaning sim started at 12UTC
sunrise_raw_d02 = sorted(glob.glob(parent_dir + '/*12/raw/d02'))		# Sunrise case



# In[21]:


def time_slice(ds):
	ds = ds.isel(Time=slice(0,24))
	return ds

ds_sunrise_d02 = xr.open_mfdataset(sunrise_raw_d02, concat_dim='Time', combine='nested', data_vars='all', coords='all', preprocess=time_slice)
ds_sunset_d02 = xr.open_mfdataset(sunset_raw_d02, concat_dim='Time', combine='nested', data_vars='all', coords='all', preprocess=time_slice)


# In[ ]:


ds_sunrise_d02.to_netcdf(path=parent_dir+'/d02_sunrise', mode='w', format='NETCDF4', unlimited_dims='Time')
ds_sunset_d02.to_netcdf(path=parent_dir+'/d02_sunset', mode='w', format='NETCDF4', unlimited_dims='Time')

