#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Run this command on the command line to create a .py script instead of .ipynb
	# jupyter nbconvert interp_variable.ipynb --to python


# In[1]:


# Purpose: Vertically interpolate your WRF datasets and then make it into its own .nc file!

# Input:
    # input_file: The stitched WRFout file path
    # pressure_file: This would be a WRFout file that only has pressure data.
        # Use extract_variable function for variable 'P' to get this file
        # Using a file that only holds pressure speeds up computing, BUT if you don't have it,
            # Just put your input_file in and it will work the same way.
    # variable_name: A list of variables that you are interested in calculating
        # U == Zonal wind                               [m/s]
        # V == Meridional wind                          [m/s]
        # W == Vertical wind                            [m/s]
		# QV == Water vapor mixing ratio 				[kg/kg]
		# QC == Cloud water mixing ratio 				[kg/kg]
		# QR == Rain water mixing ratio 				[kg/kg]
		# QI == Ice mixing ratio 						[kg/kg]
		# QS == Snow mixing ratio 						[kg/kg]
		# QG == Graupel mixing ratio 					[kg/kg]
        # CLDFRA == Cloud Fraction                      [0-1]
		# Theta == Potential Temperature                [K]
		# H_DIABATIC == Microphysics Latent heating     [K/s]
		# SWClear / RTHRATSWC == SW Radiative heating CLEAR SKY 	[K/s]
		# SWAll / RTHRATSW == SW Radiative heating 				    [K/s]
		# LWClear / RTHRATLWC == LW Radiative heating CLEAR SKY 	[K/s]
		# LWAll / RTHRATLW == LW Radiative heating 				    [K/s]


    # output_dir: The path to a directory where you'd like the new .nc files to be located
    # vertical_levels: An np.array() of pressure level(s) (in hPa) to interpolate
# Output:
    # .nc files for specific variables
# Process:
    # Open the stitched wrfout file
    # Figure out if the user wants 1 level or multiple levels, then loop through the variables
    # Create the new .nc file, copy global attributes over, and edit certain dims
    # Create the home where the variable will live then loop through each timestep
        # and fill it with the interpolated variable. This loop is necessary for 
        # variables that are too big to load into one variable.
# Tip:
    # You'd want to run this function for each domain file you have because input_file currently takes one path.
######## EXAMPLE ########
# i.e. if I want to interpolate zonal winds on pressure coordinates on 50hPa , I would run this: 
# parent_dir = '/this/is/where/my/data/lives'
# input_file_d01 = parent_dir + '/raw/d01'  # Path to the raw input netCDF file
# input_file_d02 = parent_dir + '/raw/d02'  # Path to the raw input netCDF file
# output_dir = parent_dir + '/L2/'          # Path to the directory with interpolated files
# variable_name = ['U']                     # Declare the variables you want to interpolate
# vertical_levels = np.arange(1000,0,-50)   # Pressure heights you want to interpolate at
# # or
# vertical_levels = np.array(850)
# Call the function:
# interp_variable(input_file_d01, variable_name, output_dir, vertical_levels)

##############################################################################

import netCDF4 as nc
import numpy as np
import wrf
import sys

##############################################################################

def interp_variable(input_file, pressure_file, variable_name, output_dir, vertical_levels, file_name):
    # Open the input netCDF file
    dataset = nc.Dataset(input_file, 'r')   # 'r' is just to read the dataset, we do NOT want write privledges
    # Load in the dataset with the pressure variable to interpolate from
    pressure_dataset = nc.Dataset(pressure_file, 'r')
    P = pressure_dataset.variables['P']    # Pressure [hPa]

    if vertical_levels.shape == (): levels = 1
    else: levels = len(vertical_levels)

    for i in variable_name:
        
        # Zonal Wind [m/s]
        if i == 'U':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'U' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_U', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)  # 'f4' == float32, 'QVAPOR' because 'U' is staggered
            temp_atts = dataset.variables['U'].__dict__
            temp_atts.update({'stagger': '','coordinates': 'XLONG XLAT XTIME'})
            output_variable.setncatts(temp_atts)
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                variable = wrf.getvar(dataset, 'ua', timeidx=t, meta=False)
                variable.set_fill_value(wrf.default_fill(np.float32))
                interp_variable = wrf.interplevel(variable, P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Meridional Wind [m/s]
        if i == 'V':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'V' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_V', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)  # 'f4' == float32, 'QVAPOR' because 'V' is staggered
            temp_atts = dataset.variables['V'].__dict__
            temp_atts.update({'stagger': '','coordinates': 'XLONG XLAT XTIME'})
            output_variable.setncatts(temp_atts)
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                variable = wrf.getvar(dataset, 'va', timeidx=t, meta=False)
                variable.set_fill_value(wrf.default_fill(np.float32))
                interp_variable = wrf.interplevel(variable, P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Vertical Wind [m/s]
        if i == 'W':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'W' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_W', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)  # 'f4' == float32, 'QVAPOR' because 'V' is staggered
            temp_atts = dataset.variables['W'].__dict__
            temp_atts.update({'stagger': '','coordinates': 'XLONG XLAT XTIME'})
            output_variable.setncatts(temp_atts)
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                variable = wrf.getvar(dataset, 'wa', timeidx=t, meta=False)
                variable.set_fill_value(wrf.default_fill(np.float32))
                interp_variable = wrf.interplevel(variable, P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Water vapor mixing ratio [kg/kg]
        elif i == 'QV':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QV' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_QV', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['QVAPOR'].__dict__)
            # Dataset variable to read from
            QV = dataset.variables['QVAPOR']    # Water vapor mixing ratio [kg/kg]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(QV[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Cloud water mixing ratio [kg/kg]
        elif i == 'QC':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QC' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_QC', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QCLOUD'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['QCLOUD'].__dict__)
            # Dataset variable to read from
            QC = dataset.variables['QCLOUD']    # Water vapor mixing ratio [kg/kg]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(QC[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Rain water mixing ratio [kg/kg]
        elif i == 'QR':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QR' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_QR', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QRAIN'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['QRAIN'].__dict__)
            # Dataset variable to read from
            QR = dataset.variables['QRAIN']    # Water vapor mixing ratio [kg/kg]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(QR[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Ice mixing ratio [kg/kg]
        elif i == 'QI':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QI' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_QI', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QICE'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['QICE'].__dict__)
            # Dataset variable to read from
            QI = dataset.variables['QICE']    # Water vapor mixing ratio [kg/kg]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(QI[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Snow mixing ratio [kg/kg]
        elif i == 'QS':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QS' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_QS', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QSNOW'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['QSNOW'].__dict__)
            # Dataset variable to read from
            QS = dataset.variables['QSNOW']    # Water vapor mixing ratio [kg/kg]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(QS[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Graupel mixing ratio [kg/kg]
        elif i == 'QG':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QG' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_QG', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QGRAUP'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['QGRAUP'].__dict__)
            # Dataset variable to read from
            QG = dataset.variables['QGRAUP']    # Water vapor mixing ratio [kg/kg]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(QG[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Cloud Fraction
        elif i == 'CLDFRA':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'CLDFRA' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_CLDFRA', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['CLDFRA'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['CLDFRA'].__dict__)
            # Dataset variable to read from
            CLDFRA = dataset.variables['CLDFRA']    # Cloud Fraction
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(CLDFRA[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Potential Temperature [K]
        elif i == 'Theta':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'Theta' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_Theta', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['T'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['T'].__dict__)
            # Dataset variable to read from
            Theta = dataset.variables['T']    # Potential Temperature [K]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(Theta[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:] + 300   # Add 300 to convert perturb theta to theta
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # Latent Heating [K/s]
        elif i == 'H_DIABATIC':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'H_DIABATIC' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_H_DIABATIC', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['H_DIABATIC'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['H_DIABATIC'].__dict__)
            # Dataset variable to read from
            H_DIABATIC = dataset.variables['H_DIABATIC']    # Latent Heating [K/s]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(H_DIABATIC[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # SW Radiative heating CLEAR SKY [K/s]
        elif i == 'SWClear':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'SWClear' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_SWClear', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATSWC'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['RTHRATSWC'].__dict__)
            # Dataset variable to read from
            SWClear = dataset.variables['RTHRATSWC']    # SW Radiative heating CLEAR SKY [K/s]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(SWClear[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # SW Radiative heating ALL SKY [K/s]
        elif i == 'SWAll':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'SWAll' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_SWAll', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATSW'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['RTHRATSW'].__dict__)
            # Dataset variable to read from
            SWAll = dataset.variables['RTHRATSW']    # SW Radiative heating ALL SKY [K/s]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(SWAll[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # LW Radiative heating CLEAR SKY [K/s]
        elif i == 'LWClear':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'LWClear' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_LWClear', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATLWC'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['RTHRATLWC'].__dict__)
            # Dataset variable to read from
            LWClear = dataset.variables['RTHRATLWC']    # LW Radiative heating CLEAR SKY [K/s]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(LWClear[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

        # LW Radiative heating ALL SKY [K/s]
        elif i == 'LWAll':
            # Create new .nc file we can write to and name it appropriately
            if levels == 1:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'LWAll' + str(vertical_levels), 'w', clobber=True)
            else:
                output_dataset = nc.Dataset(output_dir + file_name + '_interp_LWAll', 'w', clobber=True)
            output_dataset.setncatts(dataset.__dict__)
            # Create the dimensions based on global dimensions, with exception to bottom_top
            for dim_name, dim in dataset.dimensions.items():
                if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
                else:   output_dataset.createDimension(dim_name, len(dim))
            # Create the variable, set attributes, and start filling the variable into the new nc file
            output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATLW'].dimensions)  # 'f4' == float32
            output_variable.setncatts(dataset.variables['RTHRATLW'].__dict__)
            # Dataset variable to read from
            LWAll = dataset.variables['RTHRATLW']    # LW Radiative heating ALL SKY [K/s]
            # Make sure the fill value is consistent as you move forward
                # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
                # wrf.interp => 'f8' fill value (64-bit float)
                # default netCDF4 => 'f4' fill value (32-bit float)
            for t in range(dataset.dimensions['Time'].size):
                interp_variable = wrf.interplevel(LWAll[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
                output_variable[t,...] = interp_variable[:]
            # Make sure you close the input and output files at the end
            output_dataset.close()

    dataset.close()
    return


# In[ ]:


## Pick the main folder:
# parent_dir = '/where/your/wrfoutfiles/exist'
parent_dir = sys.argv[1]

# Pick the raw folders:
# input_file_d01 = parent_dir + '/raw/d01'  # Path to the raw input netCDF file
# input_file_d02 = parent_dir + '/raw/d02'  # Path to the raw input netCDF file
raw_folder_d02 = '/raw/d02_sunrise'
input_file_d02 = parent_dir + raw_folder_d02

# Where does your 3-D pressure file live
# pressure_file_d01 = parent_dir + '/L1/d01_P'
# pressure_file_d02 = parent_dir + '/L1/d02_P'

pressure_file_d02 = parent_dir + '/L1/d02_sunrise_P'


# Output to level 2 directory:
output_dir = parent_dir + '/L2/'  # Path to the input netCDF file
# Declare variables needed: 'U', 'V', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'Theta', 'H_DIABATIC', 'SWClear', 'SWAll', 'LWClear', 'LWAll'
# variable_name = ['U', 'V', 'W', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'Theta', 'H_DIABATIC', 'SWClear', 'SWAll', 'LWClear', 'LWAll']
variable_name = ['QV', 'QC', 'QR', 'QI', 'QS', 'QG']

# Declare the vertial levels you want to interpolate:
# vertical_levels = np.array(1000)
# vertical_levels = np.arange(1000,0,-50)
vertical_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))
# interp_variable(input_file_d01, pressure_file_d01, variable_name, output_dir, vertical_levels)
interp_variable(input_file_d02, pressure_file_d02, variable_name, output_dir, vertical_levels, file_name=raw_folder_d02[5:])

