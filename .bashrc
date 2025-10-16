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
alias ta5="tail -n 500"
alias he5="head -n 500"
alias sinfome="sinfo -p radclouds"
alias lsgb="ls -l --block-size=GB"
alias lsmb="ls -l --block-size=MB"
# alias general="cd /ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/tc_ens/haiyan/memb_01/ctl"
alias home="cd /ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian"

# Function to check users running Python on nodes
    # Please try to avoid running jobs on nodes that people are actively on. If cannot be avoided, notfiy the user <3
check_active_users() {
    nodes=("c793.oscer.ou.edu" "c795.oscer.ou.edu" "c797.oscer.ou.edu" "c798.oscer.ou.edu" "c799.oscer.ou.edu" "c855.oscer.ou.edu" "c965.oscer.ou.edu" "c966.oscer.ou.edu" "c968.oscer.ou.edu" "c969.oscer.ou.edu")  # Add your actual node names here

    for node in "${nodes[@]}"; do
        echo "Active Python users on $node:"
        ssh $node "ps aux | grep python | grep -v grep | grep -v root | awk '{print \$1}' | sort -u"
    done
}
# Alias to call the function easily
alias checkusers="check_active_users"

mamba activate WRF_Xarray
