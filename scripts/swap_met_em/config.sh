## Directory where met_files are located
NC_DIR="/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/met_em_files/og_files"
SWAP_DIR="/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/hragnajarian/met_em_files/swapped_files/rh"

## Variables to swap, i.e., "UU,VV,RH"
VARS="RH"

## Date ranges (inclusive, assumes hourly timesteps, can change it in the generate_hours function)
START1="2015-12-10_00:00:00"
END1="2015-12-10_03:00:00"

START2="2015-12-19_00:00:00"
END2="2015-12-19_03:00:00"

## Domains to process i.e., ("d01" "d02")
DOMAINS=("d01" "d02")