#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" 
Author: Hrag Najarian
Date made: September 9, 2024
Purpose: Stitch CRFoff wrfout files
"""


# In[ ]:


# Run this command on the command line to create a .py script instead of .ipynb
	# jupyter nbconvert concat_CRFOff.ipynb --to python


# In[19]:


import glob
import xarray as xr


# In[22]:


# Declare parent_dir where all the directories are located 
# parent_dir = sys.argv[1]
parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00/CRFoff'
timeseries = True
# List files
if timeseries==True:
	# sunset_raw_d02 = sorted(glob.glob(parent_dir + '/*00/raw/d02'))			# Sunset case since it ends with 00UTC for a 36hr run, meaning sim started at 12UTC
	sunrise_raw_d02 = sorted(glob.glob(parent_dir + '/*12/raw/d02'))		# Sunrise case
else:
	NCRF_raw_d02 = sorted(glob.glob(parent_dir + '/2015*/raw/d02'))					# both Sunset and Sunrise 


# In[5]:


def time_slice(ds):
	ds = ds.isel(Time=slice(0,24))
	return ds

if timeseries==True:
	# Save the first 24-hours and concat with next simulation 24-hrs
	ds_sunrise_d02 = xr.open_mfdataset(sunrise_raw_d02, concat_dim='Time', combine='nested', data_vars='all', coords='all', preprocess=time_slice)
	# ds_sunset_d02 = xr.open_mfdataset(sunset_raw_d02, concat_dim='Time', combine='nested', data_vars='all', coords='all', preprocess=time_slice)
	# Save file
	ds_sunrise_d02.to_netcdf(path=parent_dir+'/d02_sunrise', mode='w', format='NETCDF4', unlimited_dims='Time')
	# ds_sunset_d02.to_netcdf(path=parent_dir+'/d02_sunset', mode='w', format='NETCDF4', unlimited_dims='Time')
else:
	# Save the entire 36-hours and concat over 'Lead' dimension
	ds_NCRF_d02 = xr.open_mfdataset(NCRF_raw_d02, concat_dim='Lead', combine='nested', data_vars='all', coords='all', parallel=True)
	# Save file
	ds_NCRF_d02.to_netcdf(path=parent_dir+'/d02_NCRF', mode='w', format='NETCDF4', unlimited_dims='Time')

