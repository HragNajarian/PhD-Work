#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Author: Hrag Najarian
Date: June 2023
'''


# In[ ]:


# Run this command on the command line to create a .py script instead of .ipynb
	# jupyter nbconvert extract_variable.ipynb --to python


# In[2]:


# Purpose: Extract important variables from a stitched WRFout file and 
        # create a variable specific .nc file from it.

# Input:
	# input_file: The stitched WRFout file path. This can be done through the 'ncrcat' function
    # variable_name: A list of variables that you are interested in calculating
			# 3D variables
		# P == Pressure 								[hPa]
		# U == Zonal wind 								[m/s],	destaggered
		# V == Meridional wind 							[m/s],	destaggered
		# QV == Water vapor mixing ratio 				[kg/kg]
		# QC == Cloud water mixing ratio 				[kg/kg]
		# QR == Rain water mixing ratio 				[kg/kg]
		# QI == Ice mixing ratio 						[kg/kg]
		# QS == Snow mixing ratio 						[kg/kg]
		# QG == Graupel mixing ratio 					[kg/kg]
		# CLDFRA == Cloud Fraction
		# Theta == Potential Temperature 				[K]
		# H_DIABATIC == Microphysics Latent heating 	[K/s]
		# CAPE == Convective available potential energy	[J/kg]
		# CIN == Convective Inhibition 					[J/kg]
		# RTHRATSWC == SW Radiative heating CLEAR SKY 	[K/s]
		# RTHRATSW == SW Radiative heating 				[K/s]
		# RTHRATLWC == LW Radiative heating CLEAR SKY 	[K/s]
		# RTHRATLW == LW Radiative heating 				[K/s]
			# 2D variables
        # RR == Rain rate 						[mm/dt], where dt is your timestep
        # HFX == Upward Heat Flux at Surface	[W/m^2]
		# QFX == Upward Moisture Flux at Surface[kg/(m^2s^1)]
		# LH == Latent Heat Flux at Surface		[W/m^2]
		# TSK == Surface Skin Temperature		[K]
		# T2 == Temperature at 2m 				[K]
		# Q2 == Water vapor mixing ratio at 2m	[kg/kg]
		# U10 == Zonal wind at 10m 				[m/s]
		# V10 == Meridonal wind at 10m 			[m/s]
		# PSFC == Pressure at surface 			[hPa]
		# HGT == Terrain Height					[m]
		# CAPE_CIN_2D == CAPE and CIN calculation		[J/kg]
			# All sky
		# LWUPT == INSTANTANEOUS UPWELLING LONGWAVE FLUX AT TOP , [W/m^2]
		# LWDNT == INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT TOP , [W/m^2]
		# LWUPB == INSTANTANEOUS UPWELLING LONGWAVE FLUX AT BOTTOM , [W/m^2]
		# LWDNB == INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT BOTTOM , [W/m^2]
		# SWUPT == INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT TOP , [W/m^2]
		# SWDNT == INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT TOP , [W/m^2]
		# SWUPB == INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT BOTTOM , [W/m^2]
		# SWDNB == INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT BOTTOM , [W/m^2]
			# Clear sky
		# LWUPTC == INSTANTANEOUS UPWELLING LONGWAVE FLUX AT TOP CLEAR, [W/m^2]
		# LWDNTC == INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT TOP CLEAR, [W/m^2]
		# LWUPBC == INSTANTANEOUS UPWELLING LONGWAVE FLUX AT BOTTOM CLEAR, [W/m^2]
		# LWDNBC == INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT BOTTOM CLEAR, [W/m^2]
		# SWUPTC == INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT TOP CLEAR, [W/m^2]
		# SWDNTC == INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT TOP CLEAR, [W/m^2]
		# SWUPBC == INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT BOTTOM CLEAR, [W/m^2]
		# SWDNBC == INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT BOTTOM CLEAR, [W/m^2]
    # output_dir: The path to a directory where you'd like the new .nc files to be located
# Output:
    # .nc files for specific variables
# Process:
    # Open the stitched wrfout file and then loop through the list of variables
    # Going down the line, it will calculate your variable of interest,
    	# create a new .nc file, copy important attributes over and edit certain ones,
    	# and then copy the calculate variable over and voilà, you have a variable specific .nc file.
# Tip:
	# You'd want to do this for each domain file you have, input_file currently only holds one path.
######## EXAMPLE ########
# i.e. I want to extract pressure, zonal winds, and rain rate from my raw WRFout files in d01 and d02: 
# parent_dir = '/this/is/where/my/data/lives'
# input_file_d01 = parent_dir + '/raw/d01'  # Path to the raw input netCDF file
# input_file_d02 = parent_dir + '/raw/d02'  # Path to the raw input netCDF file
# output_dir = parent_dir + '/L1/'          # Path to the directory with variable specific files
# variable_name = ['P','RR','U']            # Declare the variables write into .nc files
# Call the function:
# extract_variable(input_file_d01, variable_name, output_dir)
##############################################################################

import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import wrf
import sys

##############################################################################

def extract_variable(input_file, variable_name, output_dir, file_name):
    # Open the input netCDF file
	dataset = nc.Dataset(input_file, 'r')	# 'r' is just to read the dataset, we do NOT want write privledges

	for i in variable_name:

		##############################################################################################
		######################################## 3-D Variables #######################################
		##############################################################################################

		# Pressure
		if i == 'P':
			# Create new .nc file we can write to and name it appropriately
			output_dataset = nc.Dataset(output_dir + file_name + '_P', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions based on global dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable and set/edit attributes
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['P'].dimensions)	# 'f4' == float32
			temp_atts = dataset.variables['P'].__dict__
			temp_atts.update({'description':'Pressure', 'units':'hPa'})
			output_variable.setncatts(temp_atts)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				variable = wrf.getvar(dataset, 'pressure', timeidx=t, meta=False)
				output_variable[t,...] = variable[:]
			output_dataset.close()	# Make sure you close the .nc file

		# Zonal Wind
		elif i == 'U':
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_U', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Since dimension west_east_stag doesn't apply here anymore, change it to west_east
			temp = list(dataset.variables['U'].dimensions)
			temp[-1] = 'west_east'
			temp = tuple(temp)
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['P'].dimensions)	# It's 'P' here because 'U' would be staggered
			temp_atts = dataset.variables['U'].__dict__
			temp_atts.update({'stagger': ''})
			output_variable.setncatts(temp_atts)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				variable = wrf.getvar(dataset, 'ua', timeidx=t, meta=False)
				output_variable[t,...] = variable[:]
			output_dataset.close()

		# Meridional Wind
		elif i == 'V':
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_V', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Since dimension south_north_stag doesn't apply here anymore, change it to south_north
			temp = list(dataset.variables['V'].dimensions)
			temp[-2] = 'south_north'
			temp = tuple(temp)
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['P'].dimensions)	# It's 'P' here because 'V' would be staggered
			temp_atts = dataset.variables['V'].__dict__
			temp_atts.update({'stagger': ''})
			output_variable.setncatts(temp_atts)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				variable = wrf.getvar(dataset, 'va', timeidx=t, meta=False)
				output_variable[t,...] = variable[:]
			output_dataset.close()
		
		# Water Vapor mixing ratio
		elif i == 'QV':
			variable = dataset.variables['QVAPOR']    # Water vapor mixing ratio [kg/kg]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_QV', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)
			output_variable.setncatts(dataset.variables['QVAPOR'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		# Cloud water mixing ratio
		elif i == 'QC':
			variable = dataset.variables['QCLOUD']    # Cloud water mixing ratio [kg/kg]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_QC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QCLOUD'].dimensions)
			output_variable.setncatts(dataset.variables['QCLOUD'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		# Rain water mixing ratio
		elif i == 'QR':
			variable = dataset.variables['QRAIN']    # Rain water mixing ratio [kg/kg]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_QR', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QRAIN'].dimensions)
			output_variable.setncatts(dataset.variables['QRAIN'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		# Ice mixing ratio
		elif i == 'QI':
			variable = dataset.variables['QICE']    # Ice mixing ratio [kg/kg]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_QI', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QICE'].dimensions)
			output_variable.setncatts(dataset.variables['QICE'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		# Snow mixing ratio
		elif i == 'QS':
			variable = dataset.variables['QSNOW']    # Snow mixing ratio [kg/kg]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_QS', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QSNOW'].dimensions)
			output_variable.setncatts(dataset.variables['QSNOW'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		# Graupel mixing ratio
		elif i == 'QG':
			variable = dataset.variables['QGRAUP']    # Graupel mixing ratio [kg/kg]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_QG', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QGRAUP'].dimensions)
			output_variable.setncatts(dataset.variables['QGRAUP'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		# Cloud Fraction
		elif i == 'CLDFRA':
			variable = dataset.variables['CLDFRA']    # Cloud Fraction
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_CLDFRA', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['CLDFRA'].dimensions)
			output_variable.setncatts(dataset.variables['CLDFRA'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()
		
		# Potential Temperature [K]
		elif i == 'Theta':	# The variable provided by WRF is actually perturbation potential temperature
						# so we will have to convert it to potential temperature by adding 300
							# Source: https://mailman.ucar.edu/pipermail/wrf-users/2010/001896.html
			variable = dataset.variables['T']    # Potential Temperature [K]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_Theta', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['T'].dimensions)
			output_variable.setncatts(dataset.variables['T'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...] + 300	# Add 300 to convert perturb theta to theta
			output_dataset.close()

		# Latent Heating [K/s]
		elif i == 'H_DIABATIC':
			variable = dataset.variables['H_DIABATIC']    # Latent Heating [K/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_H_DIABATIC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['H_DIABATIC'].dimensions)
			output_variable.setncatts(dataset.variables['H_DIABATIC'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		# CAPE and CIN calculation [J/kg]
		elif i == 'CAPE_CIN_3D':
			variable = dataset.variables['CAPE_CIN']    # Latent Heating [K/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_CAPE_CIN', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['CAPE_CIN'].dimensions)
			temp_atts = dataset.variables['P'].__dict__
			temp_atts.update({'description':'CAPE_CIN', 'units':'J/kg'})
			output_variable.setncatts(temp_atts)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				variable = wrf.getvar(dataset, 'cape_3d', timeidx=t, meta=False)
				output_variable[t,...] = variable[:]
			output_dataset.close()	# Make sure you close the .nc file

		# SW Radiative heating CLEAR SKY [K/s]
		elif i == 'SWClear':
			variable = dataset.variables['RTHRATSWC']    # SW Radiative heating CLEAR SKY [K/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWClear', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATSWC'].dimensions)
			output_variable.setncatts(dataset.variables['RTHRATSWC'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()
		
		# SW Radiative heating ALL SKY [K/s]
		elif i == 'SWAll':
			variable = dataset.variables['RTHRATSW']    # SW Radiative heating CLEAR SKY [K/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWAll', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATSW'].dimensions)
			output_variable.setncatts(dataset.variables['RTHRATSW'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		# LW Radiative heating CLEAR SKY [K/s]
		elif i == 'LWClear':
			variable = dataset.variables['RTHRATLWC']    # LW Radiative heating CLEAR SKY [K/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWClear', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATLWC'].dimensions)
			output_variable.setncatts(dataset.variables['RTHRATLWC'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()
		
		# LW Radiative heating ALL SKY [K/s]
		elif i == 'LWAll':
			variable = dataset.variables['RTHRATLW']    # LW Radiative heating ALL SKY [K/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWAll', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create the dimensions
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into the new nc file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATLW'].dimensions)
			output_variable.setncatts(dataset.variables['RTHRATLW'].__dict__)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				output_variable[t,...] = variable[t,...]
			output_dataset.close()

		###############################################################################################
		######################################## 2-D Variables ########################################
		###############################################################################################
		
		# Rain Rate
		elif (i == 'RR'):
			if (input_file[-3:]=='d02' or input_file[-3:]=='d01'):
				R_accum = dataset.variables['RAINNC']    # ACCUMULATED TOTAL GRID SCALE PRECIPITATION [mm]
				RR = R_accum[1:] - R_accum[:-1]		     # Take the difference to make it rain rate per timestep [mm/dt]	
				# Append the first timestep to rain rate (all zeros) to keep the same shape
				variable = np.ma.append(np.expand_dims(R_accum[0], axis=0), RR, axis=0)
				# Create new .nc file
				output_dataset = nc.Dataset(output_dir + file_name + '_RR', 'w', clobber=True)
				output_dataset.setncatts(dataset.__dict__)
				# Create dimensions in the output file
				for dim_name, dim in dataset.dimensions.items():
					output_dataset.createDimension(dim_name, len(dim))
				# Create the variable, set attributes, and copy the variable into new file
				output_variable = output_dataset.createVariable(i, 'f4', R_accum.dimensions)
				temp_atts = R_accum.__dict__
				temp_atts.update({'description':'Rain Rate', 'units':'mm/dt'})
				output_variable.setncatts(temp_atts)
				output_variable[:] = variable[:]	# not a large variable so no need to loop
				output_dataset.close()

			# You are most likely looking at a concatenated ensemble of simulations from restart files
			elif (input_file[-3:]=='ise' or input_file[-3:]=='set'):# d02_sunr(ise) or d02_sun(set)
				R_accum = dataset.variables['RAINNC']    			# ACCUMULATED TOTAL GRID SCALE PRECIPITATION [mm]
				## In order to account for the difference in rain rates between NCRF simulations, control data needs to be subsituted in for the first time steps
					# Load in the control data 
				cntl_file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/raw/d02'
				dataset_cntl = nc.Dataset(cntl_file, 'r')			# 'r' is just to read the dataset, we do NOT want write privledges
				R_accum_cntl = dataset_cntl.variables['RAINNC']		# ACCUMULATED TOTAL GRID SCALE PRECIPITATION [mm]
				# Select datasets over the same times
				start_ind, end_ind = np.where(dataset_cntl.variables['XTIME'] == dataset.variables['XTIME'][0])[0][0], np.where(dataset_cntl.variables['XTIME'] == dataset.variables['XTIME'][-1])[0][0]
				R_accum_cntl = R_accum_cntl[start_ind:end_ind+1]
				# Take the differences
				RR = R_accum[1:] - R_accum[:-1]		     			# Take the difference to make it rain rate per timestep [mm/dt]	
				# Taking the difference between simulations at the 24 hr mark tends to create negative rain rates since the states of each simulation is different
					# To resolve this issue, we calculate the rain rate at the first time step of each simulation from the control since it initiates from the control.
				replacement_inds = np.argwhere(np.mean(RR,axis=(1,2))<0).squeeze()
				RR[replacement_inds,:,:] = R_accum[replacement_inds,:,:] - R_accum_cntl[replacement_inds-1,:,:]
				# Append zeros to the first timestep to rain rate to say the first timestep is zero
				variable = np.ma.append(np.zeros((1,RR[0].shape[0],RR[0].shape[1])), RR, axis=0)
				
				# Create new .nc file
				output_dataset = nc.Dataset(output_dir + file_name + '_RR', 'w', clobber=True)
				output_dataset.setncatts(dataset.__dict__)
				# Create dimensions in the output file
				for dim_name, dim in dataset.dimensions.items():
					output_dataset.createDimension(dim_name, len(dim))
				# Create the variable, set attributes, and copy the variable into new file
				output_variable = output_dataset.createVariable(i, 'f4', R_accum.dimensions)
				temp_atts = R_accum.__dict__
				temp_atts.update({'description':'Rain Rate', 'units':'mm/dt'})
				output_variable.setncatts(temp_atts)
				output_variable[:] = variable[:]	# not a large variable so no need to loop
				output_dataset.close()
				dataset_cntl.close()

			# This file retains the entire simulation period
			elif (input_file[-3:]=='ens' or input_file[-3:]=='set'):
				R_accum = dataset.variables['RAINNC']				# ACCUMULATED TOTAL GRID SCALE PRECIPITATION [mm]
				RR = R_accum[:,1:] - R_accum[:,:-1]					# Take the difference to make it rain rate [mm/dt]
				## Make RR values at the start of each simulation the same as the control
					# Load in the control data 
				cntl_file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/raw/d02'
				dataset_cntl = nc.Dataset(cntl_file, 'r')			# 'r' is just to read the dataset, we do NOT want write privledges
				R_accum_cntl = dataset_cntl.variables['RAINNC']		# ACCUMULATED TOTAL GRID SCALE PRECIPITATION [mm]
				# Find out the matching start times of the simulations with control
				itter = 0
				replacement_ind = []
				for t in range(R_accum_cntl.shape[0]):	# Loop over time
					# Does the cntl time and the first time step of the simulation match
					match = np.where(dataset_cntl.variables['Times'][t] == dataset.variables['Times'][itter,0], True, False)
					if np.all(match == True):
						replacement_ind.append(t)	# Note the index if it matches
						itter = itter + 1			# Go to the next simulation run
					else:
						continue					# Keep going
					# Break loop once all simulation start indicies are accounted for
					if itter == RR.shape[0]:
						break
				replacement_ind = np.array(replacement_ind)
				# Find the rain rate of those timesteps/indices
				RR_replacement = R_accum_cntl[replacement_ind] - R_accum_cntl[replacement_ind-1]
				# Concat them to the start of the simulations
				variable = np.concatenate((np.expand_dims(RR_replacement, axis=1), RR), axis=1, dtype=np.float32)

				## Make RR values at the start of each simulation 0 mm/hr
				# RR = np.append(np.zeros((RR.shape[0],1,RR.shape[2],RR.shape[3])), RR, axis=1)

				# Create new .nc file
				output_dataset = nc.Dataset(output_dir + file_name + '_RR', 'w', clobber=True)
				output_dataset.setncatts(dataset.__dict__)
				# Create dimensions in the output file
				for dim_name, dim in dataset.dimensions.items():
					output_dataset.createDimension(dim_name, len(dim))
				# Create the variable, set attributes, and copy the variable into new file
				output_variable = output_dataset.createVariable(i, 'f4', R_accum.dimensions)
				temp_atts = R_accum.__dict__
				temp_atts.update({'description':'Rain Rate', 'units':'mm/dt'})
				output_variable.setncatts(temp_atts)
				output_variable[:] = variable[:]	# not a large variable so no need to loop
				output_dataset.close()
				dataset_cntl.close()

		# Upward Heat Flux at Surface (W/m^2)
		elif i == 'HFX':
			variable = dataset.variables['HFX']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_HFX', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# Upward Moisture Flux at Surface[kg/(m^2s^1)]
		elif i == 'QFX':
			variable = dataset.variables['QFX']	# [kg/(m^2s^1)]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_QFX', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# Latent Heat Flux at Surface		[W/m^2]
		elif i == 'LH':
			variable = dataset.variables['LH']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LH', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# Surface Skin Temperature (@ Surface)
		elif i == 'TSK':
			variable = dataset.variables['TSK']	# [K]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_TSK', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# Surface temperature (@ 2 meters)
		elif i == 'T2':
			variable = dataset.variables['T2']	# [K]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_T2', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# Surface water vapor (@ 2 meters)
		elif i == 'Q2':
			variable = dataset.variables['Q2']	# [kg/kg]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_Q2', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()
		
		# Surface Zonal Wind (@ 10 meters)
		elif i == 'U10':
			variable = dataset.variables['U10']	# [m/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_U10', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# Surface Meridional Wind (@ 10 meters) [m/s]
		elif i == 'V10':
			variable = dataset.variables['V10']	# [m/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_V10', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# Surface Pressure [Pa]
		elif i == 'PSFC':
			variable = dataset.variables['PSFC']	# [Pa]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_PSFC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			# Convert Pa into hPa with /100
			output_variable[:] = variable[:]/100	# not a large variable so no need to loop
			output_dataset.close()
		
		# Terrain Height [m]
		elif i == 'HGT':
			variable = dataset.variables['HGT']	# [m]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_HGT', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# CAPE 2-D space [J/kg]
		elif i == 'CAPE':
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_CAPE', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['XLAT'].dimensions)	# 'f4' == float32
			temp_atts = dataset.variables['XLAT'].__dict__
			temp_atts.update({'description':'CAPE', 'units':'J/kg'})
			output_variable.setncatts(temp_atts)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				variable = wrf.getvar(dataset, 'cape_2d', timeidx=t, meta=False)[0,...]	# Only include CAPE with the [0,...]
				output_variable[t,...] = variable[:]								
			output_dataset.close()	# Make sure you close the .nc file

		# CIN 2-D space [J/kg]
		elif i == 'CIN':
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_CIN', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['XLAT'].dimensions)	# 'f4' == float32
			temp_atts = dataset.variables['XLAT'].__dict__
			temp_atts.update({'description':'CIN', 'units':'J/kg'})
			output_variable.setncatts(temp_atts)
			for t in range(dataset.dimensions['Time'].size):	# loop through time for large variables
				variable = wrf.getvar(dataset, 'cape_2d', timeidx=t, meta=False)[1,...]	# Only include CIN with the [0,...]
				output_variable[t,...] = variable[:]								
			output_dataset.close()	# Make sure you close the .nc file

		#######################################################################
		############################## RADIATION ##############################
		#######################################################################
			
		# INSTANTANEOUS UPWELLING LONGWAVE FLUX AT TOP [W/m^2]
		elif i == 'LWUPT':
			variable = dataset.variables['LWUPT']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWUPT', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT TOP [W/m^2]
		elif i == 'LWDNT':
			variable = dataset.variables['LWDNT']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWDNT', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS UPWELLING LONGWAVE FLUX AT BOTTOM [W/m^2]
		elif i == 'LWUPB':
			variable = dataset.variables['LWUPB']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWUPB', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT BOTTOM [W/m^2]
		elif i == 'LWDNB':
			variable = dataset.variables['LWDNB']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWDNB', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT TOP [W/m^2]
		elif i == 'SWUPT':
			variable = dataset.variables['SWUPT']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWUPT', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT TOP [W/m^2]
		elif i == 'SWDNT':
			variable = dataset.variables['SWDNT']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWDNT', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT BOTTOM [W/m^2]
		elif i == 'SWUPB':
			variable = dataset.variables['SWUPB']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWUPB', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT BOTTOM [W/m^2]
		elif i == 'SWDNB':
			variable = dataset.variables['SWDNB']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWDNB', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS UPWELLING LONGWAVE FLUX AT TOP CLEAR [W/m^2]
		elif i == 'LWUPTC':
			variable = dataset.variables['LWUPTC']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWUPTC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT TOP CLEAR [W/m^2]
		elif i == 'LWDNTC':
			variable = dataset.variables['LWDNTC']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWDNTC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS UPWELLING LONGWAVE FLUX AT BOTTOM CLEAR [W/m^2]
		elif i == 'LWUPBC':
			variable = dataset.variables['LWUPBC']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWUPBC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT BOTTOM CLEAR [W/m^2]
		elif i == 'LWDNBC':
			variable = dataset.variables['LWDNBC']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_LWDNBC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT TOP CLEAR [W/m^2]
		elif i == 'SWUPTC':
			variable = dataset.variables['SWUPTC']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWUPTC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT TOP CLEAR [W/m^2]
		elif i == 'SWDNTC':
			variable = dataset.variables['SWDNTC']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWDNTC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT BOTTOM CLEAR [W/m^2]
		elif i == 'SWUPBC':
			variable = dataset.variables['SWUPBC']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWUPBC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT BOTTOM CLEAR [W/m^2]
		elif i == 'SWDNBC':
			variable = dataset.variables['SWDNBC']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + file_name + '_SWDNBC', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, variable.dimensions)
			temp_atts = variable.__dict__
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

	# Close the output files
	dataset.close()
	return


# In[ ]:


# Pick the main folder:
# parent_dir = '/where/your/wrfoutfiles/exist'
parent_dir = sys.argv[1]

# Control where icloud=1
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00'
# NCRF where icloud=0
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'

# Pick the raw folders:
	# Control
# raw_folder_d01 = '/raw/d01'
# input_file_d01 = parent_dir + raw_folder_d01  # Path to the raw input netCDF file
# raw_folder_d02 = '/raw/d02'
# input_file_d02 = parent_dir + raw_folder_d02  # Path to the raw input netCDF file
	# CRF Off
raw_folder_d02 = '/raw/d02_sunrise'
input_file_d02 = parent_dir + raw_folder_d02  # Path to the raw input netCDF file

	# CRF Off Ensemble
# raw_folder_d02 = '/raw_ens/d02_sunrise_ens'
# input_file_d02 = parent_dir + raw_folder_d02  # Path to the raw input netCDF file

# Output to level 1 directory:
output_dir = parent_dir + '/L1/'  # Path to the input netCDF file
# Declare variables needed: 'P', 'U', 'V', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'Theta', 'H_DIABATIC', 'SWClear', 'SWAll', 'LWClear', 'LWAll', 'RR', 'HFX', 'QFX', 'LH', 'T2', 'U10', 'V10', 'PSFC', 'LWUPT', 'LWUPB', 'LWDNT', 'LWDNB', 'SWUPT', 'SWUPB', 'SWDNT', 'SWDNB', 'LWUPTC', 'LWUPBC', 'LWDNTC', 'LWDNBC', 'SWUPTC', 'SWUPBC', 'SWDNTC', 'SWDNBC' 
# variable_name = ['P', 'PSFC', 'RR', 'HFX', 'QFX', 'LH', 'TSK', 'T2', 'Q2' 'U10', 'V10','HGT', 'CAPE', 'CIN', 'LWUPT', 'LWUPB', 'LWDNT', 'LWDNB', 'SWUPT', 'SWUPB', 'SWDNT', 'SWDNB', 'LWUPTC', 'LWUPBC', 'LWDNTC', 'LWDNBC', 'SWUPTC', 'SWUPBC', 'SWDNTC', 'SWDNBC']
variable_name = ['TSK']

# Call on your function:
# extract_variable(input_file_d01, variable_name, output_dir, file_name=raw_folder_d01[5:])
extract_variable(input_file_d02, variable_name, output_dir, file_name=raw_folder_d02[5:])


# In[4]:


# import netCDF4 as nc
# import numpy as np
# import numpy.ma as ma
# import wrf
# import sys

# # Control where icloud=1
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'

# # Pick the raw folders:
# 	# CRF Off
# raw_folder_d02 = '/raw_ens/d02_sunrise_ens'
# input_file_d02 = parent_dir + raw_folder_d02  # Path to the raw input netCDF file

# # Output to level 1 directory:
# output_dir = parent_dir + '/L1_ens/'  # Path to the input netCDF file
# # Declare variables needed: 'P', 'U', 'V', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'Theta', 'H_DIABATIC', 'SWClear', 'SWAll', 'LWClear', 'LWAll', 'RR', 'HFX', 'QFX', 'LH', 'T2', 'U10', 'V10', 'PSFC', 'LWUPT', 'LWUPB', 'LWDNT', 'LWDNB', 'SWUPT', 'SWUPB', 'SWDNT', 'SWDNB', 'LWUPTC', 'LWUPBC', 'LWDNTC', 'LWDNBC', 'SWUPTC', 'SWUPBC', 'SWDNTC', 'SWDNBC' 
# variable_name = ['RR']
# file_name=raw_folder_d02[9:]
# i='RR'

# dataset = nc.Dataset(input_file_d02, 'r')			# 'r' is just to read the dataset, we do NOT want write privledges
# R_accum = dataset.variables['RAINNC']				# ACCUMULATED TOTAL GRID SCALE PRECIPITATION [mm]
# RR = R_accum[:,1:] - R_accum[:,:-1]					# Take the difference to make it rain rate [mm/dt]
# ## Make RR values at the start of each simulation the same as the control
# 	# Load in the control data 
# cntl_file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/raw/d02'
# dataset_cntl = nc.Dataset(cntl_file, 'r')			# 'r' is just to read the dataset, we do NOT want write privledges
# R_accum_cntl = dataset_cntl.variables['RAINNC']		# ACCUMULATED TOTAL GRID SCALE PRECIPITATION [mm]

# itter = 0
# replacement_ind = []
# for i in range(R_accum_cntl.shape[0]):	# Loop over time
# 	# Does the cntl time and the first time step of the simulation match
# 	match = np.where(dataset_cntl.variables['Times'][i] == dataset.variables['Times'][itter,0], True, False)
# 	if np.all(match == True):
# 		replacement_ind.append(i)	# Note the index if it matches
# 		itter = itter + 1			# Go to the next simulation run
# 	else:
# 		continue					# Keep going
# 	# Break loop once all simulation start indicies are accounted for
# 	if itter == RR.shape[0]:
# 		break
# replacement_ind = np.array(replacement_ind)
# # Find the rain rate of those timesteps/indices
# RR_replacement = R_accum_cntl[replacement_ind] - R_accum_cntl[replacement_ind-1]
# # Concat them to the start of the simulations
# variable = np.concatenate((np.expand_dims(RR_replacement, axis=1), RR), axis=1, dtype=np.float32)

