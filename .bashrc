# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi


## Modules
module purge
# modeule load Miniconda3/4.9.2
# modeule load CDO
# modeule load NCL
# modeule load ncview

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/oscer/software/Mamba/23.1.0-4/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/oscer/software/Mamba/23.1.0-4/etc/profile.d/conda.sh" ]; then
        . "/opt/oscer/software/Mamba/23.1.0-4/etc/profile.d/conda.sh"
    else
        export PATH="/opt/oscer/software/Mamba/23.1.0-4/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/opt/oscer/software/Mamba/23.1.0-4/etc/profile.d/mamba.sh" ]; then
    . "/opt/oscer/software/Mamba/23.1.0-4/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

export PATH
export SACCT_FORMAT=JobID,JobName,Partition,AllocCPUS,Start,End,Elapsed,State,ExitCode

# Aliases
# Establishing custom commands below
alias del="rm -rf rsl*"
alias topme="top -u hragnajarian"
alias squeueme="squeue -u hragnajarian"
alias scancelall="scancel -u hragnajarian"
alias ta5="tail -n 500"
alias he5="head -n 500"
alias sinfome="sinfo -p radclouds"
alias lsgb="ls -l --block-size=GB"
alias lsmb="ls -l --block-size=MB"
# alias general="cd /ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/tc_ens/haiyan/memb_01/ctl"
alias home="cd /ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian"

mamba activate WRF_Xarray


