#!/bin/bash

# First get source bashrc_wrf
source bashrc_wrf

# Run getera5.sh script within your WPS directory so that you get the .grib files
./WPS/run_getera5.sh

# Make sure Vtable is symbolically linked 
ln -sf WPS/ungrib/Variable_Tables/Vtable.ERA-interim.pl WPS/Vtable

# This will make files ready (not entirely sure what this does) 
./WPS/link_grib.csh WPS/ERA5*

# Execute ungrib.exe which puts the ERA5 data into an intermediate file format
./WPS/ungrib.exe

# We need another table to tell Metgrid which WRF version we're using
ln -sf ./WPS/metgrid/METGRID.TBL.ARW ./WPS/METGRID.TBL

# Execute metgrid.exe which puts the intermediate data into NetCDF format on the model grid, i.e., merging it with the output from Geogrid.
./WPS/metgrid.exe



# Error messages and their meaning:

# If you get an error message when trying to run metgrid.exe that looks like WARNING: Field PRES has 
  # missing values, this is related to the domain of the ERA5 data not being far enough beyond the bound 
  # metgrid is looking for.
# If you get an error ./metgrid.exe: error while loading shared libraries:..., make sure you source 
 # ../bashrc_wrf since you deactivated it earlier

