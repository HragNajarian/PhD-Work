#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Run this command on the command line to create a .py script instead of .ipynb
	# jupyter nbconvert interp_variable.ipynb --to python


# In[7]:


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
import time

##############################################################################

## Pick the main folder:
	# parent_dir = '/where/your/wrfoutfiles/exist'
parent_dir = sys.argv[1]
# Examples:
	# Control where icloud=1
	# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00'
	# NCRF where icloud=0
	# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'


## Pick the raw folders:
# raw_folder_d01 = '/raw/d01'  # Path to the raw input netCDF file
# raw_folder_d02 = '/raw/d02'  # Path to the raw input netCDF file
raw_folder_d02 = '/raw/d02_sunrise'		# Path to stitched raw CRFoff files
input_file_d02 = parent_dir + raw_folder_d02


## Where does your 3-D pressure file live
# pressure_file_d01 = parent_dir + '/L1/d01_P'
# pressure_file_d02 = parent_dir + '/L1/d02_P'
pressure_file_d02 = parent_dir + '/L1/d02_sunrise_P'


## Output to level 2 directory:
output_dir = parent_dir + '/L2/'  # Path to the input netCDF file


## Declare the vertial levels you want to interpolate:
# vertical_levels = np.array(1000)
# vertical_levels = np.arange(1000,0,-50)
vertical_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))


var_info = {
    'U':            {'wrf_name': 'U',          'wrfpy_name': 'ua',     'ref_dim': 'QVAPOR'},
    'V':            {'wrf_name': 'V',          'wrfpy_name': 'va',     'ref_dim': 'QVAPOR'},
    'W':            {'wrf_name': 'W',          'wrfpy_name': 'wa',     'ref_dim': 'QVAPOR'},
    'QV':           {'wrf_name': 'QVAPOR',     'wrfpy_name': None,     'ref_dim': 'QVAPOR'},
    'QC':           {'wrf_name': 'QCLOUD',     'wrfpy_name': None,     'ref_dim': 'QCLOUD'},
    'QR':           {'wrf_name': 'QRAIN',      'wrfpy_name': None,     'ref_dim': 'QRAIN'},
    'QI':           {'wrf_name': 'QICE',       'wrfpy_name': None,     'ref_dim': 'QICE'},
    'QS':           {'wrf_name': 'QSNOW',      'wrfpy_name': None,     'ref_dim': 'QSNOW'},
    'QG':           {'wrf_name': 'QGRAUP',     'wrfpy_name': None,     'ref_dim': 'QGRAUP'},
    'CLDFRA':       {'wrf_name': 'CLDFRA',     'wrfpy_name': None,     'ref_dim': 'CLDFRA'},
    'Theta':        {'wrf_name': 'T',          'wrfpy_name': None,     'ref_dim': 'T'},
    'H_DIABATIC':   {'wrf_name': 'H_DIABATIC', 'wrfpy_name': None,     'ref_dim': 'H_DIABATIC'},
    'SWClear':      {'wrf_name': 'RTHRATSWC',  'wrfpy_name': None,     'ref_dim': 'RTHRATSWC'},
    'SWAll':        {'wrf_name': 'RTHRATSW',   'wrfpy_name': None,     'ref_dim': 'RTHRATSW'},
    'LWClear':      {'wrf_name': 'RTHRATLWC',  'wrfpy_name': None,     'ref_dim': 'RTHRATLWC'},
    'LWAll':        {'wrf_name': 'RTHRATLW',   'wrfpy_name': None,     'ref_dim': 'RTHRATLW'}
}

# Open the input netCDF file
dataset = nc.Dataset(input_file_d02, 'r')   # 'r' is just to read the dataset, we do NOT want write privledges
# Load in the dataset with the pressure variable to interpolate from
pressure_dataset = nc.Dataset(pressure_file_d02, 'r')
P_var = pressure_dataset.variables['P']    # Pressure [hPa]

# Declare variables to interpolate (they must exist in 'var_info')
# variables_to_process = ['U', 'V', 'W', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'Theta', 'H_DIABATIC', 'SWClear', 'SWAll', 'LWClear', 'LWAll']
variables_to_process = ['QV']


###############################################################################################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ USER INPUTS ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
###############################################################################################################


def interp_variable(dataset, P_var, variable_name, output_dir, vertical_levels, file_name, var_info):

    # Determine number of vertical levels
    levels = 1 if vertical_levels.shape == () else len(vertical_levels)

    if variable_name not in var_info:
        print(f"Variable {variable_name} is not recognized.")
        return

    wrf_name, wrfpy_name, ref_var  = var_info[variable_name].values()
    
    # Output file naming
    level_suffix = str(vertical_levels) if levels == 1 else ''
    output_path = f"{output_dir}{file_name}_interp_{variable_name}{level_suffix}"
    output_dataset = nc.Dataset(output_path, 'w', clobber=True)
    output_dataset.setncatts(dataset.__dict__)

    # Create dimensions
    for dim_name, dim in dataset.dimensions.items():
        size = levels if dim_name == 'bottom_top' else len(dim)
        output_dataset.createDimension(dim_name, size)

    # Create variable and copy attributes
    output_variable = output_dataset.createVariable(variable_name, 'f4', dataset.variables[ref_var].dimensions)  # 'f4' == float32
    # Update staggered attributes to unstaggered
    if variable_name in ['U','V','W']:
        template_atts = dataset.variables[wrf_name].__dict__
        template_atts.update({'stagger': '', 'coordinates': 'XLONG XLAT XTIME'})
        output_variable.setncatts(template_atts)
    else:
        output_variable.setncatts(dataset.variables[wrf_name].__dict__)

    # Start popoulating interpolated data into output_variable
    
    step_start = time.perf_counter()

    n_times = dataset.dimensions['Time'].size
    fill_value = wrf.default_fill(np.float32)
    if variable_name in ['U', 'V', 'W']:
        for t in range(n_times):
            var = wrf.getvar(dataset, wrfpy_name, timeidx=t, meta=False)
            var.set_fill_value(fill_value)
            interp = wrf.interplevel(var, P_var[t,...], vertical_levels, meta=False, missing=fill_value)
            output_variable[t,...] = interp
            print(f'Time index {t} stored')
    else:
        for t in range(n_times):
            var = dataset.variables[wrf_name][t, ...]
            interp = wrf.interplevel(var, P_var[t,...], vertical_levels, meta=False, missing=fill_value)
            output_variable[t, ...] = interp
            print(f'Time index {t} stored')
    
    print(f'{variable_name} saved ✔ ', round(time.perf_counter() - step_start, 2), ' seconds')
    
    output_dataset.close()

    return

# Loop through variables
for var in variables_to_process:
    interp_variable(
        dataset=dataset,
        P_var=P_var,
        variable_name=var,
        output_dir=output_dir,
        vertical_levels=vertical_levels,
        file_name=raw_folder_d02[5:],
        var_info=var_info
    )

# Close dataset and pressure_dataset AFTER all variables are processed
dataset.close()
pressure_dataset.close()


# In[ ]:


# ################################################################################################################################



# def interp_variable(input_file, pressure_file, variable_name, output_dir, vertical_levels, file_name):
#     # Open the input netCDF file
#     dataset = nc.Dataset(input_file, 'r')   # 'r' is just to read the dataset, we do NOT want write privledges
#     # Load in the dataset with the pressure variable to interpolate from
#     pressure_dataset = nc.Dataset(pressure_file, 'r')
#     P = pressure_dataset.variables['P']    # Pressure [hPa]

#     if vertical_levels.shape == (): levels = 1
#     else: levels = len(vertical_levels)

#     for i in variable_name:
        
#         # Zonal Wind [m/s]
#         if i == 'U':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'U' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_U', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)  # 'f4' == float32, 'QVAPOR' because 'U' is staggered
#             temp_atts = dataset.variables['U'].__dict__
#             temp_atts.update({'stagger': '','coordinates': 'XLONG XLAT XTIME'})
#             output_variable.setncatts(temp_atts)
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 variable = wrf.getvar(dataset, 'ua', timeidx=t, meta=False)
#                 variable.set_fill_value(wrf.default_fill(np.float32))
#                 interp_variable = wrf.interplevel(variable, P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Meridional Wind [m/s]
#         if i == 'V':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'V' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_V', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)  # 'f4' == float32, 'QVAPOR' because 'V' is staggered
#             temp_atts = dataset.variables['V'].__dict__
#             temp_atts.update({'stagger': '','coordinates': 'XLONG XLAT XTIME'})
#             output_variable.setncatts(temp_atts)
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 variable = wrf.getvar(dataset, 'va', timeidx=t, meta=False)
#                 variable.set_fill_value(wrf.default_fill(np.float32))
#                 interp_variable = wrf.interplevel(variable, P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Vertical Wind [m/s]
#         if i == 'W':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'W' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_W', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)  # 'f4' == float32, 'QVAPOR' because 'V' is staggered
#             temp_atts = dataset.variables['W'].__dict__
#             temp_atts.update({'stagger': '','coordinates': 'XLONG XLAT XTIME'})
#             output_variable.setncatts(temp_atts)
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 variable = wrf.getvar(dataset, 'wa', timeidx=t, meta=False)
#                 variable.set_fill_value(wrf.default_fill(np.float32))
#                 interp_variable = wrf.interplevel(variable, P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Water vapor mixing ratio [kg/kg]
#         elif i == 'QV':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QV' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_QV', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QVAPOR'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['QVAPOR'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['QVAPOR']    # Water vapor mixing ratio [kg/kg]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Cloud water mixing ratio [kg/kg]
#         elif i == 'QC':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QC' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_QC', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QCLOUD'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['QCLOUD'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['QCLOUD']    # Water vapor mixing ratio [kg/kg]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Rain water mixing ratio [kg/kg]
#         elif i == 'QR':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QR' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_QR', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QRAIN'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['QRAIN'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['QRAIN']    # Water vapor mixing ratio [kg/kg]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Ice mixing ratio [kg/kg]
#         elif i == 'QI':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QI' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_QI', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QICE'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['QICE'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['QICE']    # Water vapor mixing ratio [kg/kg]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Snow mixing ratio [kg/kg]
#         elif i == 'QS':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QS' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_QS', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QSNOW'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['QSNOW'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['QSNOW']    # Water vapor mixing ratio [kg/kg]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Graupel mixing ratio [kg/kg]
#         elif i == 'QG':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'QG' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_QG', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['QGRAUP'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['QGRAUP'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['QGRAUP']    # Water vapor mixing ratio [kg/kg]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Cloud Fraction
#         elif i == 'CLDFRA':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'CLDFRA' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_CLDFRA', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['CLDFRA'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['CLDFRA'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['CLDFRA']    # Cloud Fraction
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Potential Temperature [K]
#         elif i == 'Theta':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'Theta' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_Theta', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['T'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['T'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['T']    # Potential Temperature [K]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:] + 300   # Add 300 to convert perturb theta to theta
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Latent Heating [K/s]
#         elif i == 'H_DIABATIC':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'H_DIABATIC' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_H_DIABATIC', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['H_DIABATIC'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['H_DIABATIC'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['H_DIABATIC']    # Latent Heating [K/s]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # SW Radiative heating CLEAR SKY [K/s]
#         elif i == 'SWClear':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'SWClear' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_SWClear', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATSWC'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['RTHRATSWC'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['RTHRATSWC']    # SW Radiative heating CLEAR SKY [K/s]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # SW Radiative heating ALL SKY [K/s]
#         elif i == 'SWAll':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'SWAll' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_SWAll', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATSW'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['RTHRATSW'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['RTHRATSW']    # SW Radiative heating ALL SKY [K/s]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # LW Radiative heating CLEAR SKY [K/s]
#         elif i == 'LWClear':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'LWClear' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_LWClear', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATLWC'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['RTHRATLWC'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['RTHRATLWC']    # LW Radiative heating CLEAR SKY [K/s]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # LW Radiative heating ALL SKY [K/s]
#         elif i == 'LWAll':
#             # Create new .nc file we can write to and name it appropriately
#             if levels == 1:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_' + 'LWAll' + str(vertical_levels), 'w', clobber=True)
#             else:
#                 output_dataset = nc.Dataset(output_dir + file_name + '_interp_LWAll', 'w', clobber=True)
#             output_dataset.setncatts(dataset.__dict__)
#             # Create the dimensions based on global dimensions, with exception to bottom_top
#             for dim_name, dim in dataset.dimensions.items():
#                 if dim_name == 'bottom_top':    output_dataset.createDimension(dim_name, levels)
#                 else:   output_dataset.createDimension(dim_name, len(dim))
#             # Create the variable, set attributes, and start filling the variable into the new nc file
#             output_variable = output_dataset.createVariable(i, 'f4', dataset.variables['RTHRATLW'].dimensions)  # 'f4' == float32
#             output_variable.setncatts(dataset.variables['RTHRATLW'].__dict__)
#             # Dataset variable to read from
#             variable = dataset.variables['RTHRATLW']    # LW Radiative heating ALL SKY [K/s]
#             # Make sure the fill value is consistent as you move forward
#                 # wrf.getvar => 'u8' fill value (8-bit unisgned integer)
#                 # wrf.interp => 'f8' fill value (64-bit float)
#                 # default netCDF4 => 'f4' fill value (32-bit float)
#             for t in range(dataset.dimensions['Time'].size):
#                 interp_variable = wrf.interplevel(variable[t,...], P[t,...], vertical_levels, meta=False, missing=wrf.default_fill(np.float32))
#                 output_variable[t,...] = interp_variable[:]
#             # Make sure you close the input and output files at the end
#             output_dataset.close()

#         # Delete variable to make space
#         del variable, interp_variable
        
#     dataset.close()
#     return


# In[3]:


# ## Pick the main folder:
# 	# parent_dir = '/where/your/wrfoutfiles/exist'
# parent_dir = sys.argv[1]
# # Examples:
# 	# Control where icloud=1
# 	# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00'
# 	# NCRF where icloud=0
# 	# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'

# ## Pick the raw folders:
# # raw_folder_d01 = '/raw/d01'  # Path to the raw input netCDF file
# # raw_folder_d02 = '/raw/d02'  # Path to the raw input netCDF file
# raw_folder_d02 = '/raw/d02_sunrise'		# Path to stitched raw CRFoff files
# input_file_d02 = parent_dir + raw_folder_d02

# ## Where does your 3-D pressure file live
# # pressure_file_d01 = parent_dir + '/L1/d01_P'
# pressure_file_d02 = parent_dir + '/L1/d02_P'
# # pressure_file_d02 = parent_dir + '/L1/d02_sunrise_P'


# ## Output to level 2 directory:
# output_dir = parent_dir + '/L2/'  # Path to the input netCDF file
# # Declare variables needed: 'U', 'V', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'Theta', 'H_DIABATIC', 'SWClear', 'SWAll', 'LWClear', 'LWAll'
# variable_name = ['U', 'V', 'W', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG', 'CLDFRA', 'Theta', 'H_DIABATIC', 'SWClear', 'SWAll', 'LWClear', 'LWAll']


# ## Declare the vertial levels you want to interpolate:
# # vertical_levels = np.array(1000)
# # vertical_levels = np.arange(1000,0,-50)
# vertical_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))

# # interp_variable(input_file_d01, pressure_file_d01, variable_name, output_dir, vertical_levels, file_name=raw_folder_d01[5:])
# interp_variable(input_file_d02, pressure_file_d02, variable_name, output_dir, vertical_levels, file_name=raw_folder_d02[5:])


# In[59]:


# # Function to layer weighted average cloud fraction due to uneven vertical grid spacing.
# def layer_weighted_average(ds_CLDFRA, lower_layer, upper_layer, vertical_levels):
	
# 	# Create a mask that include the layers between, turns np.arrays into a np.ma.array
# 	mask = ((vertical_levels<=lower_layer)&(vertical_levels>=upper_layer))
# 	# Isolate those layers
# 	variable = np.array(ds_CLDFRA[:5,(mask),:,:])	# 2nd dimension must be the P/z dimension
# 	# Remove values that are >1 (CLDFRA cannot be >1)
# 	variable = np.where(variable>1, np.nan ,variable)
# 	# Remove the upper layer
# 	variable = variable[:,:-1,:,:]
# 	# Calculate the pressure differences between layers
# 	dp = np.array(np.diff(vertical_levels[mask], n=1, axis=0))
# 	# Expand dp into the shape of variable
# 	dp = dp.reshape((1,-1,1,1))
# 	dp = np.broadcast_to(dp, variable.shape)
# 	# Remove dp values if variable is nan at that pressure layer. This means np.nansum(dp, axis=1) is not constant over all grid points
# 	dp = np.where(np.isnan(variable), np.nan, dp)
# 	# Multiple the variable based on the pressure differences, then divide by the total pressure layer difference, and then sum over the P/z layer
# 	variable = np.nansum((variable * dp), axis=1) / (np.nansum(dp, axis=1))
# 	# Convert masked array into just array
# 	variable = np.array(variable)

# 	return variable, dp


# # Declare vertical levels that you've used when interpolating
# vertical_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))

# 	# Control where icloud=1
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00'
# raw_folder_d02 = '/L2/d02_interp_CLDFRA'
# file_name = 'd02'
# 	# NCRF where icloud=0
# # parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'
# # raw_folder_d02 = '/L2/d02_sunrise_interp_CLDFRA'
# # file_name = 'd02_sunrise'

# output_dir = parent_dir + '/L1/'

# input_file_d02 = parent_dir + raw_folder_d02
# dataset = nc.Dataset(input_file_d02, 'r')
# # Assign the variable
# ds_CLDFRA = dataset.variables['CLDFRA']


# In[5]:


# # Function to layer weighted average cloud fraction due to uneven vertical grid spacing.
# def layer_weighted_average(ds_CLDFRA, lower_layer, upper_layer, vertical_levels):
	
# 	# Create a mask that include the layers between, turns np.arrays into a np.ma.array
# 	mask = ((vertical_levels<=lower_layer)&(vertical_levels>=upper_layer))
# 	# Isolate those layers
# 	variable = np.array(ds_CLDFRA[:,(mask),:,:])	# 2nd dimension must be the P/z dimension
# 	# Remove values that are >1 (CLDFRA cannot be >1)
# 	variable = np.where(variable>1, np.nan ,variable)
# 	# Remove the upper layer
# 	variable = variable[:,:-1,:,:]
# 	# Calculate the pressure differences between layers
# 	dp = np.array(np.diff(vertical_levels[mask], n=1, axis=0))
# 	# Expand dp into the shape of variable
# 	dp = dp.reshape((1,-1,1,1))
# 	dp = np.broadcast_to(dp, variable.shape)
# 	# Remove dp values if variable is nan at that pressure layer. This means np.nansum(dp, axis=1) is not constant over all grid points
# 	dp = np.where(np.isnan(variable), np.nan, dp)
# 	# Multiple the variable based on the pressure differences, then divide by the total pressure layer difference, and then sum over the P/z layer
# 	variable = np.nansum((variable * dp), axis=1) / (np.nansum(dp, axis=1))
# 	# Convert masked array into just array
# 	variable = np.array(variable)

# 	return variable

# # Declare vertical levels that you've used when interpolating
# vertical_levels = np.concatenate((np.arange(1000,950,-10),np.arange(950,350,-30),np.arange(350,0,-50)))

# 	# Control where icloud=1
# parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00'
# raw_folder_d02 = '/L2/d02_interp_CLDFRA'
# file_name = 'd02'
# 	# NCRF where icloud=0
# # parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff'
# # raw_folder_d02 = '/L2/d02_sunrise_interp_CLDFRA'
# # file_name = 'd02_sunrise'

# output_dir = parent_dir + '/L1/'
# input_file_d02 = parent_dir + raw_folder_d02
# dataset = nc.Dataset(input_file_d02, 'r')
# # Assign the variable
# ds_CLDFRA = dataset.variables['CLDFRA']


# ## Low Cloud Fraction [1000-700 hPa]
# variable = layer_weighted_average(ds_CLDFRA=ds_CLDFRA, lower_layer=1000, upper_layer=680, vertical_levels=vertical_levels)
# # Create new .nc file
# output_dataset = nc.Dataset(output_dir + file_name + '_LowCLDFRA', 'w', clobber=True)
# output_dataset.setncatts(dataset.__dict__)
# # Create the dimensions
# for dim_name, dim in dataset.dimensions.items():
# 	output_dataset.createDimension(dim_name, len(dim))
# # Create the variable, set attributes, and copy the variable into the new nc file
# temp_dimensions = list(dataset.variables['CLDFRA'].dimensions)	# For some reason variable.dimensions wasn't working, this is a work around.
# temp_dimensions.remove("bottom_top")
# temp_dimensions = tuple(temp_dimensions)
# output_variable = output_dataset.createVariable('CLDFRA', 'f4', temp_dimensions)
# output_variable[:] = variable[:]	# not a large variable so no need to loop
# output_dataset.close()
# print('Low cloud fraction uploaded')


# ## Mid Cloud Fraction [700-450 hPa]
# variable = layer_weighted_average(ds_CLDFRA=ds_CLDFRA, lower_layer=680, upper_layer=440, vertical_levels=vertical_levels)
# # Create new .nc file
# output_dataset = nc.Dataset(output_dir + file_name + '_MidCLDFRA', 'w', clobber=True)
# output_dataset.setncatts(dataset.__dict__)
# # Create the dimensions
# for dim_name, dim in dataset.dimensions.items():
# 	output_dataset.createDimension(dim_name, len(dim))
# # Create the variable, set attributes, and copy the variable into the new nc file
# temp_dimensions = list(dataset.variables['CLDFRA'].dimensions)	# For some reason variable.dimensions wasn't working, this is a work around.
# temp_dimensions.remove("bottom_top")
# temp_dimensions = tuple(temp_dimensions)
# output_variable = output_dataset.createVariable('CLDFRA', 'f4', temp_dimensions)
# output_variable[:] = variable[:]	# not a large variable so no need to loop
# output_dataset.close()
# print('Mid cloud fraction uploaded')


# ## High Cloud Fraction [450-200 hPa]
# variable = layer_weighted_average(ds_CLDFRA=ds_CLDFRA, lower_layer=440, upper_layer=150, vertical_levels=vertical_levels)
# # Create new .nc file
# output_dataset = nc.Dataset(output_dir + file_name + '_HighCLDFRA', 'w', clobber=True)
# output_dataset.setncatts(dataset.__dict__)
# # Create the dimensions
# for dim_name, dim in dataset.dimensions.items():
# 	output_dataset.createDimension(dim_name, len(dim))
# # Create the variable, set attributes, and copy the variable into the new nc file
# temp_dimensions = list(dataset.variables['CLDFRA'].dimensions)	# For some reason variable.dimensions wasn't working, this is a work around.
# temp_dimensions.remove("bottom_top")
# temp_dimensions = tuple(temp_dimensions)
# output_variable = output_dataset.createVariable('CLDFRA', 'f4', temp_dimensions)
# output_variable[:] = variable[:]	# not a large variable so no need to loop
# output_dataset.close()
# print('High cloud fraction uploaded')

