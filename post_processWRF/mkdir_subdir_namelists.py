#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Run this command on the command line to create a .py script instead of .ipynb
	# jupyter nbconvert mkdir_subdir_namelists.ipynb --to python


# In[ ]:


# This code will create directories, subdirectories, and namelists dependening on your start time and end time.


# In[4]:


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
	model_start_times = np.datetime_as_string(np.arange(start_date, end_date + ensemble_2_ensemble_length_hr, ensemble_2_ensemble_length_hr))
	model_end_times = np.datetime_as_string(np.arange(start_date+model_length_hr, end_date + ensemble_2_ensemble_length_hr + model_length_hr, ensemble_2_ensemble_length_hr))
	# Create the directory_name matrix and then mkdir the names.
	directory_name = np.empty_like(model_end_times)
	for i in range(len(model_start_times)): 
		model_start_times[i] = model_start_times[i].replace('T','-')
		model_end_times[i] = model_end_times[i].replace('T','-')
		directory_name[i] = model_start_times[i] + str('--') + model_end_times[i]
		os.mkdir(parent_dir+directory_name[i])

def create_subdir(parent_dir):
	for i in sorted(os.listdir(parent_dir)):
		os.mkdir(parent_dir + i + '/raw')
		os.mkdir(parent_dir + i + '/L1')
		os.mkdir(parent_dir + i + '/L2')
		os.mkdir(parent_dir + i + '/L3')

def create_namelists(parent_dir,control_namelist_path, model_length_hr):
	# Loop through simulation directories
	for i in sorted(os.listdir(parent_dir)):
		destination = parent_dir + i
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
		content = content.replace('start_year = 2015, 2015', 'start_year = ' + i[0:4] + ', ' + i[0:4])
		content = content.replace('start_month = 11, 11', 'start_month = ' + i[5:7] + ', ' + i[5:7])
		content = content.replace('start_day = 22, 22', 'start_day = ' + i[8:10] + ', ' + i[8:10])
		content = content.replace('start_hour = 12, 12', 'start_hour = ' + i[11:13] + ', ' + i[11:13])
			# end_times:
		content = content.replace('end_year = 2015, 2015', 'end_year = ' + i[15:19] + ', ' + i[15:19])
		content = content.replace('end_month = 12, 12', 'end_month = ' + i[20:22] + ', ' + i[20:22])
		content = content.replace('end_day = 03, 03', 'end_day = ' + i[23:25] + ', ' + i[23:25])
		content = content.replace('end_hour = 00, 00', 'end_hour = ' + i[26:28] + ', ' + i[26:28])
			# restart
		content = content.replace('restart = .false.', 'restart = .true.')
			# restart_interval
				# Make it greater than the model run length i.e., > model_length_hr*60
		content = content.replace('restart_interval = 360', 'restart_interval = ' + str(model_length_hr*60+1))
			# icloud
		content = content.replace('icloud = 1', 'icloud = 0')
		
		# Save the edited namelist.input into the simulation directory.
		with open(new_namelist_path, 'w') as output_file:
			output_file.write(content)


# In[5]:


# The directory you want to fill with more directories
parent_dir = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/CRFoff/'
# start_time/end_time: YYYY-MM-DD HH
start_time = '2015-11-23 00'
end_time = '2015-12-01 12'
# How many hours does the model run for?
model_length_hr = 36
# How many hours until a new model is run?
ensemble_2_ensemble_length_hr = 12

create_dir(parent_dir, start_time, end_time, model_length_hr, ensemble_2_ensemble_length_hr)
create_subdir(parent_dir)
# Where is the control namelist you want to edit for your simulations
control_namelist_path = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/wrfout.files/new10day-2015-11-22-12--12-03-00/namelist.input'
create_namelists(parent_dir, control_namelist_path, model_length_hr)

