#!/usr/bin/env python
# coding: utf-8

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
		# P == Pressure 		[hPa]
		# U == Zonal wind 		[m/s],	destaggered
		# V == Meridional wind 	[m/s],	destaggered
		# QV == Water vapor mixing ratio 				[kg/kg]
		# QC == Cloud water mixing ratio 				[kg/kg]
		# QR == Rain water mixing ratio 				[kg/kg]
		# QI == Ice mixing ratio 						[kg/kg]
		# QS == Snow mixing ratio 						[kg/kg]
		# QG == Graupel mixing ratio 					[kg/kg]
		# CLDFRA == Cloud Fraction
		# Theta == Potential Temperature 				[K]
		# H_DIABATIC == Microphysics Latent heating 	[K/s]
		# RTHRATSWC == SW Radiative heating CLEAR SKY 	[K/s]
		# RTHRATSW == SW Radiative heating 				[K/s]
		# RTHRATLWC == LW Radiative heating CLEAR SKY 	[K/s]
		# RTHRATLW == LW Radiative heating 				[K/s]
			# 2D variables
        # RR == Rain rate 				[mm/dt], where dt is your timestep
		# T2 == Temperature at 2m 		[K]
		# U10 == Zonal wind at 10m 		[m/s]
		# V10 == Meridonal wind at 10m 	[m/s]
		# PSFC == Pressure at surface 	[hPa]
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

def extract_variable(input_file, variable_name, output_dir):
    # Open the input netCDF file
	dataset = nc.Dataset(input_file, 'r')	# 'r' is just to read the dataset, we do NOT want write privledges

	for i in variable_name:

		######################################## 3-D Variables #######################################
		
		# Pressure
		if i == 'P':
			# Create new .nc file we can write to and name it appropriately
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_P', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_U', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_V', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_QV', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_QC', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_QR', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_QI', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_QS', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_QG', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_CLDFRA', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_Theta', 'w', clobber=True)
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
		elif i == 'LH':
			variable = dataset.variables['H_DIABATIC']    # Latent Heating [K/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LH', 'w', clobber=True)
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

		# SW Radiative heating CLEAR SKY [K/s]
		elif i == 'SWClear':
			variable = dataset.variables['RTHRATSWC']    # SW Radiative heating CLEAR SKY [K/s]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWClear', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWAll', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWClear', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWAll', 'w', clobber=True)
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

		######################################## 2-D Variables ########################################

		# Rain Rate
		elif i == 'RR':
			R_accum = dataset.variables['RAINNC']    # ACCUMULATED TOTAL GRID SCALE PRECIPITATION [mm]
			RR = R_accum[1:] - R_accum[:-1]		     # Take the difference to make it rain rate per timestep [mm/dt]	
			# Append the first timestep to rain rate (all zeros) to keep the same shape
			variable = np.ma.append(np.expand_dims(R_accum[0], axis=0), RR, axis=0)
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_RR', 'w', clobber=True)
			output_dataset.setncatts(dataset.__dict__)
			# Create dimensions in the output file
			for dim_name, dim in dataset.dimensions.items():
				output_dataset.createDimension(dim_name, len(dim))
			# Create the variable, set attributes, and copy the variable into new file
			output_variable = output_dataset.createVariable(i, variable.dtype, R_accum.dimensions)
			temp_atts = R_accum.__dict__
			temp_atts.update({'description':'Rain Rate', 'units':'mm/dt'})
			output_variable.setncatts(temp_atts)
			output_variable[:] = variable[:]	# not a large variable so no need to loop
			output_dataset.close()

		# Surface temperature (@ 2 meters)
		elif i == 'T2':
			variable = dataset.variables['T2']	# [K]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_T2', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_U10', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_V10', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_PSFC', 'w', clobber=True)
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

		#######################################################################
		############################## RADIATION ##############################
		#######################################################################
			
		# INSTANTANEOUS UPWELLING LONGWAVE FLUX AT TOP [W/m^2]
		elif i == 'LWUPT':
			variable = dataset.variables['LWUPT']	# [W/m^2]
			# Create new .nc file
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWUPT', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWDNT', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWUPB', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWDNB', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWUPT', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWDNT', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWUPB', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWDNB', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWUPTC', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWDNTC', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWUPBC', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_LWDNBC', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWUPTC', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWDNTC', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWUPBC', 'w', clobber=True)
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
			output_dataset = nc.Dataset(output_dir + input_file[-3:] + '_SWDNBC', 'w', clobber=True)
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
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00'

# Tests where icloud=0
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26/2015-11-25-03--11-26-12'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26/2015-11-25-06--11-26-12'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26/2015-11-25-09--11-26-12'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26/2015-11-25-12--11-26-12'
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-11-22-12--12-03-00/CRFoff/MC_Sumatra_2015-11-25--26/2015-11-25-15--11-26-12'

# Pick the raw folders:
input_file_d01 = parent_dir + '/raw/d01'  # Path to the raw input netCDF file
input_file_d02 = parent_dir + '/raw/d02'  # Path to the raw input netCDF file

# Output to level 1 directory:
output_dir = parent_dir + '/L1/'  # Path to the input netCDF file
# Declare variables needed: 'P', 'U', 'V', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'Theta', 'LH', 'SWClear', 'SWAll', 'LWClear', 'LWAll', 'RR', 'T2', 'U10', 'V10', 'PSFC', 'LWUPT', 'LWUPB', 'LWDNT', 'LWDNB', 'SWUPT', 'SWUPB', 'SWDNT', 'SWDNB', 'LWUPTC', 'LWUPBC', 'LWDNTC', 'LWDNBC', 'SWUPTC', 'SWUPBC', 'SWDNTC', 'SWDNBC' 
variable_name = ['P', 'PSFC', 'RR', 'T2', 'U10', 'V10', 'LWUPT', 'LWUPB', 'LWDNT', 'LWDNB', 'SWUPT', 'SWUPB', 'SWDNT', 'SWDNB', 'LWUPTC', 'LWUPBC', 'LWDNTC', 'LWDNBC', 'SWUPTC', 'SWUPBC', 'SWDNTC', 'SWDNBC']
# variable_name = ['LWUPT', 'LWUPB', 'LWDNT', 'LWDNB', 'SWUPT', 'SWUPB', 'SWDNT', 'SWDNB', 'LWUPTC', 'LWUPBC', 'LWDNTC', 'LWDNBC', 'SWUPTC', 'SWUPBC', 'SWDNTC', 'SWDNBC']

# Call on your function:
extract_variable(input_file_d01, variable_name, output_dir)
extract_variable(input_file_d02, variable_name, output_dir)

