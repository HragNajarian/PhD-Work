#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Run this command on the command line to create a .py script instead of .ipynb
	# jupyter nbconvert mkdir_subdir_namelists.ipynb --to python


# In[ ]:


# This code will create directories, subdirectories, and namelists dependening on your start time and end time.


# In[1]:


# import sys	# utilize sys.argv[1] to get the parent directory of where you want these directories to go
import os
import numpy as np
# from datetime import datetime, timedelta
import shutil

def create_dir(parent_dir, start_time, end_time, model_length_hr, ensemble_2_ensemble_length_hr):
	# convert to times to datetime
	start_date = np.datetime64(start_time)
	end_date = np.datetime64(end_time)
	# Create intervals of times and then convert to string
	model_start_times = np.datetime_as_string(np.arange(start_date, end_date - ensemble_2_ensemble_length_hr, ensemble_2_ensemble_length_hr))
	model_end_times = np.datetime_as_string(np.arange(start_date + model_length_hr, end_date + model_length_hr + ensemble_2_ensemble_length_hr, ensemble_2_ensemble_length_hr))
	# Create the directory_name matrix and then mkdir the names.
	directory_name = np.empty_like(model_end_times)
	created_directories = []
	for i in range(len(model_start_times)): 
		model_start_times[i] = model_start_times[i].replace('T','-')
		model_end_times[i] = model_end_times[i].replace('T','-')
		directory_name[i] = model_start_times[i] + str('--') + model_end_times[i]
		os.mkdir(parent_dir+directory_name[i])
		created_directories.append(parent_dir+directory_name[i])
	
	# Create stitched NCRF raw, L1, L2, L3, L4 directories
	os.mkdir(parent_dir + 'raw')	# Raw WRF output
	os.mkdir(parent_dir + 'L1')	# Extracted variables from raw
	os.mkdir(parent_dir + 'L2')	# Interpolate 3-D vars into pressure coordinates
	os.mkdir(parent_dir + 'L3')	# Cross-sectional analysis
	os.mkdir(parent_dir + 'L3/Sumatra_mid_central')
	os.mkdir(parent_dir + 'L3/Sumatra_northwest')
	os.mkdir(parent_dir + 'L3/Borneo_northwest')
	os.mkdir(parent_dir + 'L4')	# Diurnal composites

	return created_directories

def create_subdir(created_directories):
	for i in sorted(created_directories):
		os.mkdir(i + '/raw')	# Raw WRF output
		os.mkdir(i + '/L1')		# Extracted variables from raw
		os.mkdir(i + '/L2')		# Interpolate 3-D vars into pressure coordinates
		os.mkdir(i + '/L3')		# Cross-sectional analysis
		os.mkdir(i + '/L4')		# Diurnal composites

def create_cross_subdir(created_directories):
	for i in sorted(created_directories):
		os.mkdir(i + '/L3/Sumatra_mid_central')
		os.mkdir(i + '/L3/Sumatra_northwest')
		os.mkdir(i + '/L3/Borneo_northwest')

def create_namelists(control_namelist_path, model_length_hr, created_directories):
	# Loop through simulation directories
	for destination in created_directories:
		# List the directories from the destination path
		directory = destination[-28:]
		# Copy the control namelist to the simulation directory
		shutil.copy(control_namelist_path, destination)
		# Create a new namelist.input path for the simulation directory
		new_namelist_path = destination + '/namelist.input'
		# Open the control namelist and copy it's content
		with open(control_namelist_path, 'r') as input_file:
			# Read the content of the input file
			content = input_file.read()
		# Edit the start times, end times, restart, and icloud variable within namelist.input
		# Replace the desired text (for example, replace 'old_text' with 'new_text')
			# This will be done by reading the time information from the directories that were created.
			# start_times:
		content = content.replace('start_year = 2015, 2015', 'start_year = ' + directory[0:4] + ', ' + directory[0:4])
		content = content.replace('start_month = 12, 12', 'start_month = ' + directory[5:7] + ', ' + directory[5:7])
		content = content.replace('start_day = 09, 09', 'start_day = ' + directory[8:10] + ', ' + directory[8:10])
		content = content.replace('start_hour = 12, 12', 'start_hour = ' + directory[11:13] + ', ' + directory[11:13])
			# end_times:
		content = content.replace('end_year = 2015, 2015', 'end_year = ' + directory[15:19] + ', ' + directory[15:19])
		content = content.replace('end_month = 12, 12', 'end_month = ' + directory[20:22] + ', ' + directory[20:22])
		content = content.replace('end_day = 20, 20', 'end_day = ' + directory[23:25] + ', ' + directory[23:25])
		content = content.replace('end_hour = 00, 00', 'end_hour = ' + directory[26:28] + ', ' + directory[26:28])
			# restart
		content = content.replace('restart = .false.', 'restart = .true.')
			# restart_interval
				# Make it greater than the model run length i.e., > model_length_hr*60
		content = content.replace('restart_interval = 720', 'restart_interval = ' + str((model_length_hr*60)+1))
			# icloud
		content = content.replace('icloud = 1', 'icloud = 0')
		
		# Save the edited namelist.input into the simulation directory.
		with open(new_namelist_path, 'w') as output_file:
			output_file.write(content)


# In[2]:


# The directory you want to fill with more directories
parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00/CRFoff/'
# start_time/end_time: YYYY-MM-DD HH
start_time = '2015-12-10 00'
end_time = '2015-12-20 12'
# How many hours does the model run for?
model_length_hr = 36
# How many hours until a new model is run?
ensemble_2_ensemble_length_hr = 24

created_directories = create_dir(parent_dir, start_time, end_time, model_length_hr, ensemble_2_ensemble_length_hr)
create_subdir(created_directories)		# L1,L2,L3,L4,raw
create_cross_subdir(created_directories)	# L3/<cross_sections>
# Where is the control namelist you want to edit for your simulations
control_namelist_path = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/10day-2015-12-09-12--12-20-00/namelist.input'
create_namelists(control_namelist_path, model_length_hr, created_directories)

